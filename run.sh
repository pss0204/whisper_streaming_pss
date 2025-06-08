#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1

# # for i in {0.1,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0}; do
# #     python src/main.py \
# #         --beam_size=5 \
# #         --min_chunk_size=$i
# # done

# for i in {1..5}; do
#     python src/main.py \
#         --beam_size=$i \
#         --min_chunk_size=0.1
# # done

# #  python3 whisper_online.py harvard.wav --language en --min-chunk-size 0.1 -- > out.txt

# python whisper_online.py "dataset:disco-eth/EuroSpeech:uk:train:0" --language en --min-chunk-size 0.1 --buffer_trimming sentence > out.txt

# Basic test
python whisper_online.py ted_16k_mono.wav \
    --language en \
    --min-chunk-size 1.0 \
    --buffer_trimming segment \
    --buffer_trimming_sec 30 \
    > out.txt


# # Basic test
# python whisper_online.py ted_16k_mono.wav \
#     --language en \
#     --min-chunk-size 1.0 \
#     --buffer_trimming segment \
#     --buffer_trimming_sec 30 \
#     --target-latency 3.0 \
#     --max-chunk-size 10.0 \
#     --adaptation-factor 0.1 \
#     --adaptive-chunk  \
#     > out.txt



echo "Extracting clean text..."
python extract_text.py out.txt clean_output.txt

echo "Calculating WER..."
python calculate_wer.py ted.txt clean_output.txt