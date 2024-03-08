# --resume_from_checkpoint output/llama_int8_scale5_mx/checkpoint-100 \

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2 \
accelerate launch --config_file deepspeed_offload.yaml finetune_llama.py \
    --model_name_or_path meta-llama/llama-2-7b-hf \
    --dataset_name "TokenBender/code_instructions_122k_alpaca_style" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 64 \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --save_steps 100 \
    --output_dir output/llama_int8_scale5 \
    --block_size 1024 \
    --do_train \
    --logging_steps 1 \
    --lr_scheduler_type cosine \
    --using_mx yes \
    --scale_bits 5 \
    --w_elem_format int8 \
    --a_elem_format int8 \
    --quantize_backprop yes \
    --bf16 yes \
    --learning_rate=5e-5 \



   
