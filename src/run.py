import sys
sys.path.append('/home/pss/whisper_streaming')

import soundfile as sf
from whisper_online import *
import librosa
import numpy as np
from datasets import load_dataset
from jiwer import wer

# 전역 ASR 객체
asr_model = None

def init_asr(language="en", model_size="base"):
    """ASR 모델 초기화"""
    global asr_model
    if asr_model is None:
        asr_model = CustomFasterWhisperASR(language, model_size)
        asr_model.beam_size = 1
    return asr_model

def run_asr(audio_array):
    """오디오 배열에 대해 ASR 실행"""
    global asr_model
    if asr_model is None:
        raise ValueError("ASR model not initialized. Call init_asr() first.")
    
    processor = OnlineASRProcessor(asr_model)
    
    # 청크 크기 설정 (초 단위)
    chunk_size_sec = 1.0
    chunk_size_samples = int(chunk_size_sec * 16000)
    
    results = []
    
    # 스트리밍 시뮬레이션
    for i in range(0, len(audio_array), chunk_size_samples):
        chunk = audio_array[i:i+chunk_size_samples]
        
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

def process_audio_with_asr(sample):
    """dataset.map에서 사용할 함수 - 오디오 전처리 및 ASR 실행"""
    print(f"Processing sample...")  # 디버깅 출력 추가
    try:
        # 오디오 데이터 추출
        audio_array = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']
        
        # 16kHz로 리샘플링 (whisper 요구사항)
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        
        # numpy array로 변환
        audio_array = np.array(audio_array, dtype=np.float32)
        
        # ASR 실행
        asr_results = run_asr(audio_array)
        
        # 결과를 텍스트로 변환
        pred_text = ""
        if asr_results:
            # 모든 결과를 합쳐서 하나의 텍스트로 만들기
            text_parts = []
            for result in asr_results:
                if result and len(result) >= 3 and result[2]:
                    text_parts.append(result[2].strip())
            pred_text = " ".join(text_parts)

        # 실제 텍스트 (ground truth) 추출
        real_text = sample["human_transcript"] 
        
        # 기존 샘플에 pred 추가
        sample['pred'] = pred_text

        wer_value = wer(real_text, pred_text)
        sample['whisper_wer'] = wer_value
        
        return sample
        
    except Exception as e:
        print(f"Error processing audio sample: {e}")
        import traceback
        traceback.print_exc()  # 전체 스택 트레이스 출력
        # 에러 발생 시 빈 예측 텍스트와 기본 WER 반환
        sample['pred'] = ""
        sample['whisper_wer'] = 1.0  # 최대 WER 값
        return sample

