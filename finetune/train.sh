#!/bin/bash

# run `accelerate config` first. pass --deepspeed to finetune.py if using DeepSpeed
export WANDB_MODE=offline
accelerate launch finetune.py \
    --output-dir output/qwen3-0.6b-arxiv \
    --model Qwen/Qwen3-0.6B \
    --max-train-steps 10 \
    --scaling-factor 8.0 \
    --save-steps 5 \
    --checkpointing-steps 5 \
    --dataset /home/ubuntu/tiansheng/26_ICLR_Sparse_attn/LongPPL_sparse_attn/finetune/dataset/pile_arxiv \
    --gradient-accumulate-every 4 \
    --wandb "qwen3-longce-experiment" \
    --eval-batches 10 \
    --topk 8 \

# accelerate launch finetune.py \
#     --output-dir output/llama2-pg19-eabf-longce \
#     --model Llama-2-7b-hf \
#     --deepspeed \
#     --max-train-steps 200 \
#     --use-eabf \
#     --dataset datasets/pg19 \
#     --save-steps 50 \
#     --gradient-accumulate-every 8 \

# accelerate launch finetune.py \
#     --output-dir output/llama2-pg19-arxiv-longce \
#     --model Llama-2-7b-hf \
#     --deepspeed \
#     --max-train-steps 200 \
#     --use-eabf \
#     --dataset datasets/pile_arxiv \
#     --save-steps 50 \
#     --gradient-accumulate-every 1 \
