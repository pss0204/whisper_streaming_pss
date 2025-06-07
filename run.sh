#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

# for i in {0.1,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0}; do
#     python src/main.py \
#         --beam_size=5 \
#         --min_chunk_size=$i
# done

for i in {1..5}; do
    python src/main.py \
        --beam_size=$i \
        --min_chunk_size=0.1
done