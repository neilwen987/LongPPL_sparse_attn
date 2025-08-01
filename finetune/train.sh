#!/bin/bash

# run `accelerate config` first. pass --deepspeed to finetune.py if using DeepSpeed
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,7
accelerate launch finetune.py \
    --output-dir output/qwen3-0.6b-arxiv \
    --model Qwen/Qwen3-0.6B \
    --deepspeed \
    --max-train-steps 200 \
    --scaling-factor 8.0 \
    --save-steps 50 \
    --dataset /home/ubuntu/tiansheng/26_ICLR_Sparse_attn/LongPPL_sparse_attn/finetune/dataset/pile_arxiv \
    --gradient-accumulate-every 4 \

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
