#!/bin/bash

for i in {1..5}; do
    python src/main.py \
        --beam_size=$i \
        --min_chunk_size=0.1 \
done