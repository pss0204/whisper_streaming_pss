import sys
sys.path.append('/home/pss/whisper_streaming')

import soundfile as sf
from whisper_online import *
import librosa
import numpy as np
from datasets import load_dataset
from run import init_asr, process_audio_with_asr

def main():
    # ASR 모델 초기화
    print("Initializing ASR model...")
    asr = init_asr("en", "base")
    
    # EuroSpeech 데이터셋 로드 (UK 설정)
    print("Loading EuroSpeech dataset...")
    dataset = load_dataset("disco-eth/EuroSpeech", "uk", split="train")
    
    # 데이터셋 크기 제한 (테스트용으로 처음 5개만 사용)
    dataset = dataset.select(range(min(5, len(dataset))))
    
    print(f"Processing {len(dataset)} samples...")
    
    # dataset.map을 사용하여 ASR 적용
    dataset_with_pred = dataset.map(
        process_audio_with_asr,
        desc="Running ASR on audio samples",
        load_from_cache_file=False,
    )
    
    # 데이터셋 정보 출력
    print(f"\nDataset columns: {dataset_with_pred.column_names}")
    print(f"Dataset size: {len(dataset_with_pred)}")
    # 결과 출력
    print("\n=== ASR Results ===")
    for i, sample in enumerate(dataset_with_pred):
        print(f"\nSample {i+1}:")
        print(f"  Ground Truth: {sample['human_transcript']}")
        print(f"  Prediction: {sample['pred']}")
        print(f"  WER: {sample['whisper_wer']}")
        print(f"  Latency: {sample['avg_latency']:.2f} seconds")
        print(f"  Beam Size: {sample['beam_size']}")

    print(f" \n\nAvg_wer: {np.mean([s['whisper_wer'] for s in dataset_with_pred]):.2f}")
    print(f"\n\n Avg_latency: {np.mean([s['avg_latency'] for s in dataset_with_pred]):.2f} seconds")

if __name__ == "__main__":
    main()

