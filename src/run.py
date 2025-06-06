import sys
sys.path.append('/home/pss/whisper_streaming')

import soundfile as sf
from whisper_online import *
import librosa
import numpy as np
from datasets import load_dataset
from jiwer import wer
from src.config import Config

# 전역 ASR 객체
asr_model = None

def init_asr(language="en", model_size="base"):
    """ASR 모델 초기화"""
    global asr_model
    if asr_model is None:
        asr_model = CustomFasterWhisperASR(language, model_size)
        #asr_model.beam_size = 5
    return asr_model

def run_asr(audio_array, config=Config):
    """오디오 배열에 대해 ASR 실행 (실시간 시뮬레이션)"""
    global asr_model
    if asr_model is None:
        raise ValueError("ASR model not initialized. Call init_asr() first.")
    
    
    processor = VACOnlineASRProcessor(
        online_chunk_size=config.min_chunk_size,
        asr=asr_model,
        tokenizer=None,
        buffer_trimming=("segment", 15)
    )


    asr_model.beam_size = config.beam_size
    
    start_time = time.time()
    audio_start_time = time.time()  # 오디오 시작 시간
    results = []
    word_latencies = []
    
    vac_chunk_size = 0.04
    vac_chunk_samples = int(vac_chunk_size * 16000)
    
    # 실시간 시뮬레이션: 실제 오디오 재생 시간에 맞춰 처리
    for i in range(0, len(audio_array), vac_chunk_samples):
        chunk = audio_array[i:i+vac_chunk_samples]
        
        # 현재 오디오 위치의 실제 시간
        audio_time = i / 16000.0
        
        # 실시간 대기: 실제 오디오 재생 시간까지 기다리기
        elapsed_real_time = time.time() - audio_start_time
        if audio_time > elapsed_real_time:
            time.sleep(audio_time - elapsed_real_time)
        
        processor.insert_audio_chunk(chunk)
        result = processor.process_iter()
        
        # 현재 실제 시간과 오디오 진행 시간
        current_real_time = time.time() - audio_start_time
        audio_progress_time = (i + len(chunk)) / 16000.0
        
        if result and result[0] is not None:
            results.append(result)
            
            word_start_time = result[0]
            word_end_time = result[1]
            
            if word_end_time is not None:
                # 실시간 시뮬레이션에서의 레이턴시 계산
                # 레이턴시 = 출력된 실제 시간 - 단어가 끝난 오디오 시간
                latency = current_real_time - word_end_time
                
                word_latencies.append(latency)
                print(f"Word/phrase at {word_start_time:.2f}-{word_end_time:.2f}s, "
                      f"output at real time {current_real_time:.2f}s, latency: {latency:.2f}s")
    
    final_result = processor.finish()
    if final_result and final_result[0] is not None:
        results.append(final_result)

    avg_latency = np.mean(word_latencies) if word_latencies else 0.0
    beam_size = asr_model.beam_size

    return results, avg_latency, beam_size

def process_audio_with_asr(sample,config=Config):
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
        asr_results, avg_latency, beam_size = run_asr(audio_array, config=config)

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
        sample["avg_latency"] = avg_latency
        
        # 기존 샘플에 pred 추가
        sample['pred'] = pred_text
        sample['beam_size'] = beam_size

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

