import sys
sys.path.append('/home/pss/whisper_streaming')

import soundfile as sf
from whisper_online import *
import librosa
import numpy as np
from datasets import load_dataset
from run import init_asr, process_audio_with_asr
import json
import os
from datetime import datetime
from src.config import Config
import argparse

def main():
    # ASR 모델 초기화
    print("Initializing ASR model...")
    asr = init_asr("en", "base")

    parser = argparse.ArgumentParser()
    parser.add_argument("--beam_size", type=int, default=Config.beam_size)
    parser.add_argument("--min_chunk_size", type=float, default=Config.min_chunk_size)
    args = parser.parse_args()
    
    config = Config()
    config.beam_size = args.beam_size
    config.min_chunk_size = args.min_chunk_size

    # EuroSpeech 데이터셋 로드 (UK 설정)
    print("Loading EuroSpeech dataset...")
    dataset = load_dataset("disco-eth/EuroSpeech", "uk", split="train")
    
    # 데이터셋 크기 제한 (테스트용으로 처음 5개만 사용)
    dataset = dataset.select(range(min(100, len(dataset))))
    
    print(f"Processing {len(dataset)} samples...")
    
    # dataset.map을 사용하여 ASR 적용
    dataset_with_pred = dataset.map(
        process_audio_with_asr,
        desc="Running ASR on audio samples",
        fn_kwargs={"config": config},
        load_from_cache_file=False,
    )
    
    # 데이터셋 정보 출력
    print(f"\nDataset columns: {dataset_with_pred.column_names}")
    print(f"Dataset size: {len(dataset_with_pred)}")
    # 결과 출력
    print("\n=== ASR Results ===")
    
    # JSON에 저장할 데이터 구조 생성
    results_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_size": len(dataset_with_pred),
            "model_info": {
                "language": "en",
                "model_size": "base"
            }
        },
        "samples": [],
        "summary": {}
    }
    
    # # 각 샘플 결과를 JSON 데이터에 추가
    # for i, sample in enumerate(dataset_with_pred):
    #     print(f"\nSample {i+1}:")
    #     print(f"  Ground Truth: {sample['human_transcript']}")
    #     print(f"  Prediction: {sample['pred']}")
    #     print(f"  WER: {sample['whisper_wer']}")
    #     print(f"  Latency: {sample['avg_latency']:.2f} seconds")
    #     print(f"  Beam Size: {sample['beam_size']}")
        
    #     # JSON 데이터에 샘플 정보 추가
    #     sample_data = {
    #         "sample_id": i + 1,
    #         "ground_truth": sample['human_transcript'],
    #         "prediction": sample['pred'],
    #         "wer": sample['whisper_wer'],
    #         "avg_latency": sample['avg_latency'],
    #         "beam_size": sample['beam_size']
    #     }
    #     results_data["samples"].append(sample_data)

    # 평균값 계산 및 출력
    avg_wer = np.mean([s['whisper_wer'] for s in dataset_with_pred])
    avg_latency = np.mean([s['avg_latency'] for s in dataset_with_pred])
    Beam_sizes = []
    Latencies = []

    for s in dataset_with_pred:
        Beam_sizes.append(s['beam_size'])
        Latencies.append(s['avg_latency'])
    avg_beam_size = np.mean(Beam_sizes)

    print(f" \n\nAvg_wer: {avg_wer:.2f}")
    print(f"\n\n Avg_latency: {avg_latency:.2f} seconds")
    
    # 요약 정보를 JSON 데이터에 추가
    results_data["summary"] = {
        "avg_wer": round(avg_wer, 4),
        "Latency_list": Latencies,
        "avg_latency": round(avg_latency, 4),
        "avg_beam_size": round(avg_beam_size, 4),
        "total_samples": len(dataset_with_pred),
        "Beam_size_list": Beam_sizes
    }
    
    # JSON 파일로 저장
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{output_dir}/asr_results_{timestamp}.json"
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과가 {json_filename}에 저장되었습니다.")

if __name__ == "__main__":
    main()

