import torch
from torch import nn
from microxcaling import mx

def replace_llama(model, mx_specs):
    def recursive_replace_module(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print("replacing", name, "with microxcaling")
                weight = child.weight
                setattr(module, name, mx.Linear(child.in_features, child.out_features, mx_specs))
                getattr(module, name).weight = weight
            else:
                recursive_replace_module(child)
    recursive_replace_module(model)
    return model

