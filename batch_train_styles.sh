#!/bin/bash

# Batch training and testing script for all style directories
set -e

STYLE_ROOT="./Styles"

# Training loop
for dir in "$STYLE_ROOT"/*/; do
    [ -d "$dir" ] || continue
    data_name=$(basename "$dir")
    for num in 1 2; do
        prompt=$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 5)
        (
            cd "$dir"
            output_dir="./output/$data_name/$num"
            mkdir -p "$output_dir"
            echo "$prompt" > "$output_dir/prompt.txt"
            accelerate launch ../../train_dreambooth_ot-lora_sdxl.py \
                --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
                --instance_data_dir="$data_name" \
                --output_dir="$output_dir" \
                --instance_prompt="a $prompt" \
                --resolution=1024 \
                --rank=64 \
                --train_batch_size=1 \
                --learning_rate=5e-8 \
                --lr_scheduler="cosine" \
                --lr_warmup_steps=0 \
                --max_train_steps=1000 \
                --checkpointing_steps=200 \
                --seed="0" \
                --gradient_checkpointing \
                --use_8bit_adam \
                --mixed_precision="fp16"
        )
    done
done

# Testing loop
for dir in "$STYLE_ROOT"/*/; do
    [ -d "$dir" ] || continue
    data_name=$(basename "$dir")
    for num in 1 2; do
        (
            cd "$dir"
            output_dir="./output/$data_name/$num"
            prompt=$(cat "$output_dir/prompt.txt")
            python ../../inference.py \
                --prompt="A [$prompt] made of gold" \
                --content_B_LoRA="$output_dir/checkpoint-500" \
                --output_path="$output_dir/content" \
                --num_images_per_prompt=10

            python ../../inference.py \
                --prompt="A dog in [$prompt]" \
                --content_B_LoRA="$output_dir/checkpoint-500" \
                --output_path="$output_dir/style" \
                --num_images_per_prompt=10
        )
    done
done
