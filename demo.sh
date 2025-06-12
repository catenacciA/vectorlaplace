#!/bin/bash

mkdir -p results_detail
mkdir -p results_hdr

python3 script.py \
    --input input_images/input_png/flower.png \
    --output results_detail/flower_detail.png \
    --sigma_r 0.4 \
    --alpha 0.25 \
    --beta 1 \
    --mode rgb \
    --domain lin

python3 script.py \
    --input input_images/input_hdr/doll.hdr \
    --output results_hdr/doll_gamma.png \
    --sigma_r 0 \
    --alpha 1 \
    --beta 1 \
    --mode lum \
    --domain log

python3 script.py \
    --input input_images/input_hdr/doll.hdr \
    --output results_hdr/doll_hdr.png \
    --sigma_r 2.5 \
    --alpha 1 \
    --beta 0 \
    --mode lum \
    --domain log

for input_file in input_images/input_png/*.png; do
    base=$(basename "$input_file" .png)
    sigma_r=0.2
    alpha=0.5
    output_file="results_detail/${base}_rgb_lin_s${sigma_r}_a${alpha}_b1.png"
    python3 script.py \
        --input "$input_file" \
        --output "$output_file" \
        --sigma_r "$sigma_r" \
        --alpha "$alpha" \
        --beta 1 \
        --mode rgb \
        --domain lin
done

# ────────────────────────────────────────────────────────────────────────────────
# Reduced Supplementary: Tone Manipulation
# ────────────────────────────────────────────────────────────────────────────────

for input_file in input_images/input_hdr/*.hdr; do
    base=$(basename "$input_file" .hdr)
    alpha=0.75
    beta=0.3
    output_file="results_hdr/${base}_lum_log_s2.5_a${alpha}_b${beta}.png"
    python3 script.py \
        --input "$input_file" \
        --output "$output_file" \
        --sigma_r 2.5 \
        --alpha "$alpha" \
        --beta "$beta" \
        --mode lum \
        --domain log
    
    python3 script.py \
        --input "$input_file" \
        --output "results_hdr/${base}.png" \
        --sigma_r 2.5 \
        --alpha 1 \
        --beta 1 \
        --mode lum \
        --domain log
done
