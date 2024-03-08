import math
import torch
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast
from microxcaling import mx
import functools
from einops import rearrange, einsum
from dataclasses import dataclass, field
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from typing import Dict, Union

# linear = nn.Linear

def exists(val):
    return val is not None




def make_causal_mask(bsz, tgt_len, device, dtype):
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=1024, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class MLP(nn.Module):
    def __init__(self, hidden_size, mx_specs):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size * 2.6875)

        if mx_specs is not None:
            mk_linear = functools.partial(mx.Linear, mx_specs=mx_specs)
        else:
            mk_linear = nn.Linear

        self.gate_proj = mk_linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj   = mk_linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = mk_linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        gate_proj = nn.functional.silu(self.gate_proj(x))
        up_proj = self.up_proj(x)
        down_proj = self.down_proj(gate_proj * up_proj)
        return down_proj

class Attention(nn.Module):
    def __init__(self, config, attention_mask, position_ids, mx_specs, mx_attn=True):
        super(Attention, self).__init__()
        self.mx_attn=mx_attn
        self.mx_specs=mx_specs
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.max_position_embeddings = config.sequence_length
        self.attention_mask = attention_mask
        
        if mx_specs is not None:
            mk_linear = functools.partial(mx.Linear, mx_specs=mx_specs)
        else:
            mk_linear = nn.Linear
        self.q_proj = mk_linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = mk_linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = mk_linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = mk_linear(config.hidden_size, config.hidden_size, bias=False)

        self.position_ids = position_ids
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(self, hidden_states, masks):
        bsz, seq_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        position_ids = self.position_ids[:, :seq_len]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        if not self.mx_attn:
            if masks is None:
                output = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
            else:
                attn_mask = self.attention_mask[:bsz, :, :seq_len, :seq_len] + masks.view(bsz, 1, seq_len, seq_len)
                output = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        else:
            att = mx.matmul(query / math.sqrt(self.head_dim), key.transpose(2, 3), mx_specs=self.mx_specs) 
            assert att.size() == (bsz, self.num_heads, seq_len, seq_len), "Attention weights shape error"
            if masks is None:
                att = att + self.attention_mask[:bsz, :, :seq_len, :seq_len]
            else:
                att = att + self.attention_mask[:bsz, :, :seq_len, :seq_len] + masks.view(bsz, 1, seq_len, seq_len)
            att = nn.functional.softmax(att, dim=-1)
            output = mx.matmul(att, value, mx_specs=self.mx_specs)

        output = output.transpose(1, 2).contiguous()
        assert output.size() == (bsz, seq_len, self.num_heads, self.head_dim), "Attention output shape error"
        output = output.reshape(bsz, seq_len, self.hidden_size)

        output = self.o_proj(output)
        return output

class DecoderLayer(nn.Module):
    def __init__(self, config, attention_mask, position_ids, mx_specs, mx_skip=False, mx_attn=False):
        super().__init__()
        self.mx_specs = mx_specs
        self.mx_skip = mx_skip
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.self_attn = Attention(config, attention_mask, position_ids, mx_specs, mx_attn=mx_attn)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = MLP(config.hidden_size, mx_specs=mx_specs)

    def forward(self, x, masks):
        if self.mx_specs is None or not self.mx_skip:
            x = x + self.self_attn(self.input_layernorm(x), masks)
            x = x + self.mlp(self.post_attention_layernorm(x))
        else:
            x, residual = mx.simd_split(x)
            x = mx.simd_add(residual, self.self_attn(self.input_layernorm(x), masks))
            x, residual = mx.simd_split(x)
            x = mx.simd_add(residual, self.mlp(self.post_attention_layernorm(x)))
        return x

@dataclass
class RetrieverParseConfig:
    vocab_size: int = 50257
    num_layers: Optional[int] = field(default=6)
    hidden_size: Optional[int] = field(default=512)
    num_heads: Optional[int] = field(default=8)
    mx_skip : Optional[bool] = field(default=False)
    mx_attn : Optional[bool] = field(default=False)
    using_mx : Optional[bool] = field(default=False)
    mx_lm_head : Optional[bool] = field(default=False)
    sequence_length: Optional[int] = field(default=1024)
    mx_specs: Optional[dict] = field(default=None)

class RetrieverConfig(PretrainedConfig):
    model_type = "retriever"

    def __init__(self, vocab_size=50257, num_layers=6, hidden_size=768, num_heads=12, mx_skip=False, mx_attn=False, using_mx=False, mx_lm_head=False, sequence_length=1024, mx_specs=None, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mx_skip = mx_skip
        self.mx_attn = mx_attn
        self.using_mx = using_mx
        self.mx_lm_head = mx_lm_head
        self.sequence_length = sequence_length
        self.mx_specs = mx_specs
        self.device = device
        self.ptdtype = torch.bfloat16 if self.device == "cuda" else torch.float32

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "mx_skip": self.mx_skip,
            "mx_attn": self.mx_attn,
            "using_mx": self.using_mx,
            "mx_lm_head": self.mx_lm_head,
            "sequence_length": self.sequence_length,
            "mx_specs": self.mx_specs,
            "device": self.device,
            "ptdtype": self.ptdtype
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = cls()
        return config

    def update(self, config: Union[Dict, "PretrainedConfig"]):
        if isinstance(config, dict):
            for k, v in config.items():
                setattr(self, k, v)
        else:
            for k, v in config.to_dict().items():
                setattr(self, k, v)
        return self

    def __eq__(self, other):
        if not isinstance(other, Retriever):
            return False
        return self.to_dict() == other.to_dict()

    def __repr__(self):
        return f"RetrieverConfig(vocab_size={self.vocab_size}, num_layers={self.num_layers}, hidden_size={self.hidden_size}, num_heads={self.num_heads}, mx_skip={self.mx_skip}, mx_attn={self.mx_attn}, using_mx={self.using_mx}, mx_lm_head={self.mx_lm_head}, sequence_length={self.sequence_length}, mx_specs={self.mx_specs})"

class Retriever(nn.Module):
    def __init__(self, device, ptdtype, config, mx_specs, mx_skip=False, mx_attn=False, mx_lm_head=False):
        super(Retriever, self).__init__()
        self.device = device
        self.ptdtype = ptdtype
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        position_ids = torch.arange(0, config.sequence_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, config.sequence_length)
        attention_mask = make_causal_mask(config.batch_size, config.sequence_length, device, ptdtype)
        self.layers = nn.ModuleList([DecoderLayer(config, attention_mask, position_ids, mx_specs, mx_skip=mx_skip, mx_attn=mx_attn) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        if mx_specs and mx_lm_head:
            self.lm_head = mx.Linear(config.hidden_size, config.vocab_size, bias=False, mx_specs=mx_specs)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.embed_tokens.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('d_proj.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, mx.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def configure_optimizers(self, weight_decay, learning_rate, betas, is_master):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if is_master:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)
        return optimizer

    def forward(self, input_ids, labels=None, masks=None):
        tok_emb = self.embed_tokens(input_ids)
        x = tok_emb

        if masks is not None:
            masks.masked_fill_(masks == -1, torch.finfo(self.ptdtype).min)

        for decoder_layer in self.layers:
            x = decoder_layer(x, masks)
        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is None:
            return logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = nn.functional.cross_entropy(shift_logits, shift_labels)

        return logits, loss
    
    @torch.no_grad()
    def chat(self, tokenizer, message, history, temperature=1.0, top_k=1, max_new_tokens=256):
        # prompt = "you are a helpful assistant\n"
        # for q, r in history:
        #     prompt += f"user: {q}\nassistant: {r}\n"
        # prompt += f"user: {message}\nassistant: "
        prompt = message + '\n' + 'Response:'
        print("prompt: ", prompt)
        input_ids = tokenizer.encode(prompt)
        output_ids = []
        input = torch.from_numpy(np.array([input_ids])).to(torch.long)

        for i in range(max_new_tokens):
            input_trunc = input if input.shape[1] <= self.config.sequence_length else input[:, -self.config.sequence_length:]
            input_trunc = input_trunc.to(self.device)
            with autocast(dtype=self.ptdtype):
                logits = self(input_trunc)
                logits = logits[:, -1].cpu()
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = nn.functional.softmax(logits, dim=-1)
            id_next = torch.multinomial(probs, num_samples=1)
            input = torch.cat((input, id_next), dim=1)
            output_ids.append(int(id_next[0][0]))

            if output_ids[-1] == 2:
                break
        
        return tokenizer.decode(output_ids).split('</s>')[0]

def retriever(device, ptdtype, config, pretrained=False, model_path=None, flash=True):
    model = Retriever(device, ptdtype, config, flash)
    if pretrained:
        state_dict = torch.load(model_path)['state_dict']
        replacer = '_orig_mod.'
        if "module" == list(state_dict.keys())[0][:6]:
            replacer = 'module._orig_mod.'
        model.load_state_dict({k.replace(replacer, ''): v for k, v in state_dict.items()})
    return model


class RetrieverModel(PreTrainedModel):
    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, pretrained=False,):
        super(RetrieverModel, self).__init__(config)
        if config.using_mx:
            dtype = torch.float32
        else:
            dtype = torch.bfloat16
        self.model = Retriever(device="cuda", ptdtype=dtype, config=config, 
                    mx_specs=config.mx_specs, mx_skip=config.mx_skip, 
                    mx_attn=config.mx_attn,
                    mx_lm_head=config.mx_lm_head, )
        self.config = config

    def forward(self, input_ids, labels=None, masks=None):
        logits, loss = self.model(input_ids, labels, masks)
        return CausalLMOutput(logits=logits, loss=loss)

