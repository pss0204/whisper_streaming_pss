import sys
sys.path.append('/home/pss/whisper_streaming')

import soundfile as sf
from whisper_online import *
import librosa
import numpy as np
from datasets import load_dataset


def run_asr(audio, asr):
    asr.beam_size = 1
    processor = OnlineASRProcessor(asr)
    
    # 청크 크기 설정 (초 단위)
    chunk_size_sec = 1.0
    chunk_size_samples = int(chunk_size_sec * 16000)
    
    results = []
    
    # 스트리밍 시뮬레이션
    for i in range(0, len(audio), chunk_size_samples):
        chunk = audio[i:i+chunk_size_samples]
        
        processor.insert_audio_chunk(chunk)
        result = processor.process_iter()
        
        # 결과가 있으면 저장
        if result and result[0] is not None:
            results.append(result)
    
    # 마지막 처리
    final_result = processor.finish()
    if final_result and final_result[0] is not None:
        results.append(final_result)
    
    return results

