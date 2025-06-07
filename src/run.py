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

def init_asr(language="en", model_size="large-v2"):
    """ASR 모델 초기화"""
    global asr_model
    if asr_model is None:
        asr_model = CustomFasterWhisperASR(language, model_size)
        # 모델 정보 확인 - 올바른 속성 사용
        print(f"✅ Model loaded: {model_size}")
        try:
            # faster-whisper의 경우 model_size_or_path 대신 다른 방식으로 접근
            print(f"✅ Device: {asr_model.model.device}")
            print(f"✅ Compute type: {asr_model.model.compute_type}")
            # model_size_or_path는 제거하거나 다른 방식으로 접근
            print(f"✅ Model size: {model_size}")
        except AttributeError as e:
            print(f"⚠️  Model info access error: {e}")
    return asr_model

def run_asr(audio_array, config=Config):
    """오디오 배열에 대해 ASR 실행 (실시간 시뮬레이션)"""
    global asr_model
    if asr_model is None:
        raise ValueError("ASR model not initialized. Call init_asr() first.")
    
    processor = OnlineASRProcessor(
        asr=asr_model,
        tokenizer=None,
        buffer_trimming=("sentence", 30)
    )

    asr_model.beam_size = config.beam_size
    
    # whisper_online.py 스타일의 처리 루프 적용
    start_time = time.time()
    results = []
    word_latencies = []
    
    # min_chunk_size 사용 (config에서 가져옴)
    min_chunk = config.min_chunk_size
    
    # 오디오를 메모리에서 시뮬레이션하기 위한 설정
    total_duration = len(audio_array) / 16000.0
    beg = 0.0  # 시작 시간 (초)
    end = 0.0  # 종료 시간 (초)
    
    # whisper_online.py의 핵심 처리 루프 적용
    while end < total_duration:
        now = time.time() - start_time
        
        # min_chunk만큼 대기 (실시간 시뮬레이션)
        if now < end + min_chunk:
            time.sleep(min_chunk + end - now)
        
        # 새로운 end 시간 계산
        end = time.time() - start_time
        if end > total_duration:
            end = total_duration
        
        # 오디오 청크 로드 (beg부터 end까지)
        beg_samples = int(beg * 16000)
        end_samples = int(end * 16000)
        
        if end_samples > len(audio_array):
            end_samples = len(audio_array)
            
        chunk = audio_array[beg_samples:end_samples]
        
        if len(chunk) > 0:
            # 청크 삽입
            processor.insert_audio_chunk(chunk)
            
            try:
                # ASR 처리
                result = processor.process_iter()
                
                # 결과 처리
                current_real_time = time.time() - start_time
                
                if result and result[0] is not None:
                    results.append(result)
                    
                    word_start_time = result[0]
                    word_end_time = result[1]
                    
                    if word_end_time is not None:
                        # 레이턴시 계산
                        latency = current_real_time - word_end_time
                        word_latencies.append(latency)
                        print(f"Word/phrase at {word_start_time:.2f}-{word_end_time:.2f}s, "
                              f"output at real time {current_real_time:.2f}s, latency: {latency:.2f}s")
                        
            except AssertionError as e:
                print(f"Assertion error: {e}")
        
        # 다음 반복을 위한 beg 업데이트
        beg = end
        
        # 무한 루프 방지
        if beg >= total_duration:
            break
    
    # 마지막 결과 처리
    final_result = processor.finish()
    if final_result and final_result[0] is not None:
        results.append(final_result)

    avg_latency = np.mean(word_latencies) if word_latencies else 0.0
    beam_size = asr_model.beam_size

    return results, avg_latency, beam_size, min_chunk

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
        asr_results, avg_latency, beam_size, min_chunk = run_asr(audio_array, config=config)

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
        sample['min_chunk'] = min_chunk

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

