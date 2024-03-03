#  accelerate launch run_clm_no_trainer.py \
# --model_name_or_path gpt2 \
# --dataset_name wikitext \
# --dataset_config_name wikitext-2-raw-v1 \
# --per_device_train_batch_size $BS 
# --per_device_eval_batch_size $BS 
# --num_train_epochs 1 
# --block_size 12
# --report_to wandb
# --output_dir output
# --overwrite_output_dir
# --do_train
# --learning_rate 5e-5
# --
    # per_device_train_batch_size=32, # batch size for training
    # per_device_eval_batch_size=64,  # batch size for evaluation
    # eval_steps = 400, # Number of update steps between two evaluations.
    # save_steps=800, # after # steps model is saved

    # warmup_steps=500,# number of warmup steps for learning rate scheduler


TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2 \
accelerate launch --config_file fsdp_config.yaml train.py \
    --tokenizer_name gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-v1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 64 \
    --streaming \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --save_steps 800 \
    --output_dir output/0.1b_int8_mx_attn \
    --block_size 1024 \
    --do_train \
    --max_steps 40000 \
    --logging_steps 10 \
    --lr_scheduler_type cosine \
    --using_mx yes \
    --w_elem_format int8 \
    --a_elem_format int8 \
    --num_layers 18 \
    --hidden_size 1280 \
    --num_heads 16 \
    --mx_attn yes \
    --learning_rate=5e-4 \



   

