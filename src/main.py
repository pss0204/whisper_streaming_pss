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
    # Initialize ASR model
    print("Initializing ASR model...")
    asr = init_asr("en", "large-v2")

    parser = argparse.ArgumentParser()
    parser.add_argument("--beam_size", type=int, default=Config.beam_size)
    parser.add_argument("--min_chunk_size", type=float, default=Config.min_chunk_size)
    args = parser.parse_args()
    
    config = Config()
    config.beam_size = args.beam_size
    config.min_chunk_size = args.min_chunk_size

    # Load EuroSpeech dataset (UK configuration)
    print("Loading EuroSpeech dataset...")
    dataset = load_dataset("disco-eth/EuroSpeech", "uk", split="train")
    
    
    dataset = dataset.select(range(min(100, len(dataset))))
    
    print(f"Processing {len(dataset)} samples...")
    
    # Apply ASR using dataset.map
    dataset_with_pred = dataset.map(
        process_audio_with_asr,
        desc="Running ASR on audio samples",
        fn_kwargs={"config": config},
        load_from_cache_file=False,
    )
    
    # Output dataset information
    print(f"\nDataset columns: {dataset_with_pred.column_names}")
    print(f"Dataset size: {len(dataset_with_pred)}")
    # Output results
    print("\n=== ASR Results ===")
    
    # Create data structure for saving to JSON
    results_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_size": len(dataset_with_pred),
            "model_info": {
                "language": "en",
                "model_size": "large-v2"
            }
        },
        "samples": [],
        "summary": {}
    }
    
    # # Add each sample result to JSON data
    # for i, sample in enumerate(dataset_with_pred):
    #     print(f"\nSample {i+1}:")
    #     print(f"  Ground Truth: {sample['human_transcript']}")
    #     print(f"  Prediction: {sample['pred']}")
    #     print(f"  WER: {sample['whisper_wer']}")
    #     print(f"  Latency: {sample['avg_latency']:.2f} seconds")
    #     print(f"  Beam Size: {sample['beam_size']}")
        
    #     # Add sample information to JSON data
    #     sample_data = {
    #         "sample_id": i + 1,
    #         "ground_truth": sample['human_transcript'],
    #         "prediction": sample['pred'],
    #         "wer": sample['whisper_wer'],
    #         "avg_latency": sample['avg_latency'],
    #         "beam_size": sample['beam_size']
    #     }
    #     results_data["samples"].append(sample_data)

    # Calculate and output average values
    avg_wer = np.mean([s['whisper_wer'] for s in dataset_with_pred])
    avg_latency = np.mean([s['avg_latency'] for s in dataset_with_pred])
    Beam_sizes = []
    Latencies = []

    for s in dataset_with_pred:
        Beam_sizes.append(s['beam_size'])
        Latencies.append(s['avg_latency'])
    avg_beam_size = np.mean(Beam_sizes)
    avg_min_chunk = np.mean([s['min_chunk'] for s in dataset_with_pred])

    print(f" \n\nAvg_wer: {avg_wer:.2f}")
    print(f"\n\n Avg_latency: {avg_latency:.2f} seconds")
    
    # Add summary information to JSON data
    results_data["summary"] = {
        "avg_wer": round(avg_wer, 4),
        "Latency_list": Latencies,
        "avg_latency": round(avg_latency, 4),
        "avg_beam_size": round(avg_beam_size, 4),
        "total_samples": len(dataset_with_pred),
        "min_chunk_size": round(avg_min_chunk, 4),
        "Beam_size_list": Beam_sizes
    }
    
    # Save to JSON file
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{output_dir}/asr_results_{timestamp}.json"
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {json_filename}.")

if __name__ == "__main__":
    main()

