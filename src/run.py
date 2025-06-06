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

def run_asr(audio_array, min_chunk_size=1.0):
    """오디오 배열에 대해 ASR 실행"""
    global asr_model
    if asr_model is None:
        raise ValueError("ASR model not initialized. Call init_asr() first.")
    
    # VACOnlineASRProcessor 사용 (VAD 기능 포함)
    processor = VACOnlineASRProcessor(
        online_chunk_size=min_chunk_size,  # min_chunk_size가 여기서 사용됨
        asr=asr_model,
        tokenizer=None,
        buffer_trimming=("segment", 15)
    )
    
    start_time = time.time()
    results = []
    word_latencies = []
    
    # VAC는 더 작은 청크 (0.04초)로 VAD 처리
    vac_chunk_size = 0.04  # VAD 분석용 작은 청크
    vac_chunk_samples = int(vac_chunk_size * 16000)
    
    # VAC 방식으로 스트리밍 시뮬레이션
    for i in range(0, len(audio_array), vac_chunk_samples):
        chunk = audio_array[i:i+vac_chunk_samples]
        
        processor.insert_audio_chunk(chunk)
        result = processor.process_iter()
        
        current_time = time.time() - start_time
        # 현재 오디오 진행 시간 (실제 오디오에서의 시간)
        audio_progress_time = (i + len(chunk)) / 16000.0
        
        # 결과가 있으면 저장
        if result and result[0] is not None:
            results.append(result)
            
            # 확정된 단어들의 레이턴시 계산
            # result 형식: (beg_timestamp, end_timestamp, "text")
            word_start_time = result[0]  # 단어/구문의 시작 시간
            word_end_time = result[1]    # 단어/구문의 끝 시간
            
            # 올바른 레이턴시 계산:
            # 레이턴시 = (단어가 출력된 시점의 실제 시간) - (단어가 실제 끝난 시점의 실제 시간)
            # 실시간 스트리밍에서는 단어가 끝난 시점 = 스트리밍 시작 + 오디오 내 단어 끝 시간
            if word_end_time is not None:
                # 실제 시나리오: 단어가 끝난 시점에 해당하는 실시간 
                word_actual_end_time = word_end_time  # 오디오 시작 기준
                # 단어가 출력된 시점에서의 오디오 진행 시간
                output_time_in_audio = audio_progress_time
                
                # 레이턴시 = 출력 시점 - 단어가 실제 끝난 시점
                latency = output_time_in_audio - word_end_time
                
                word_latencies.append(latency)
                print(f"Word/phrase at {word_start_time:.2f}-{word_end_time:.2f}s, "
                      f"output when audio at {output_time_in_audio:.2f}s, latency: {latency:.2f}s")  # 디버깅용
    
    # 마지막 처리
    final_result = processor.finish()
    if final_result and final_result[0] is not None:
        results.append(final_result)

    # 지연시간이 있을 때만 평균 계산
    avg_latency = np.mean(word_latencies) if word_latencies else 0.0
    
    return results, avg_latency

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
        asr_results, avg_latency = run_asr(audio_array)

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

