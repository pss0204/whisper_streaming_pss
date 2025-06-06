# Whisper Streaming - 모든 조절 가능한 파라미터 완전 가이드

## 1. 핵심 모델 설정 파라미터

### 1.1 Whisper 모델 선택 (`--model`)
```bash
--model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo}
```
- **기본값**: `large-v2`
- **설명**: 사용할 Whisper 모델의 크기와 버전
- **성능 vs 속도 트레이드오프**:
  - `tiny` (39MB): 가장 빠름, 품질 낮음
  - `base` (74MB): 빠름, 기본 품질
  - `small` (244MB): 중간 속도, 좋은 품질
  - `medium` (769MB): 느림, 매우 좋은 품질
  - `large-v3` (1550MB): 가장 느림, 최고 품질
  - `large-v3-turbo`: 최신 버전, large-v3보다 8배 빠름
- **언어별 모델**: `.en` 접미사는 영어 전용 모델 (영어에서 더 좋은 성능)

### 1.2 백엔드 선택 (`--backend`)
```bash
--backend {faster-whisper,whisper_timestamped,mlx-whisper,openai-api}
```
- **기본값**: `faster-whisper`
- **각 백엔드 특징**:
  - `faster-whisper`: 가장 빠름 (GPU 필요), CUDA 지원
  - `whisper_timestamped`: 중간 속도, GPU 설치가 쉬움
  - `mlx-whisper`: Apple Silicon 최적화 (M1, M2 등)
  - `openai-api`: 클라우드 기반, GPU 불필요, 비용 발생

### 1.3 언어 설정 (`--lan`, `--language`)
```bash
--lan {auto,en,ko,ja,zh,de,fr,es,...}
```
- **기본값**: `auto`
- **지원 언어**: 80+ 언어 지원
- **한국어**: `ko`
- **자동 감지**: `auto` (첫 30초 기반으로 언어 감지)

### 1.4 작업 유형 (`--task`)
```bash
--task {transcribe,translate}
```
- **기본값**: `transcribe`
- `transcribe`: 원본 언어로 전사
- `translate`: 영어로 번역

## 2. 오디오 처리 및 청킹 파라미터

### 2.1 최소 청크 크기 (`--min-chunk-size`)
```bash
--min-chunk-size FLOAT
```
- **기본값**: `1.0` (초)
- **설명**: 처리할 최소 오디오 청크 크기
- **영향**: 
  - 작을수록: 낮은 지연시간, 높은 CPU 사용량
  - 클수록: 높은 지연시간, 낮은 CPU 사용량, 더 정확한 전사
- **권장값**: 
  - 실시간성 중요: `0.5-1.0`초
  - 품질 중요: `2.0-3.0`초

### 2.2 버퍼 트리밍 방식 (`--buffer_trimming`)
```bash
--buffer_trimming {sentence,segment}
```
- **기본값**: `segment`
- **segment**: Whisper가 반환하는 세그먼트 단위로 트리밍
- **sentence**: 문장 부호 기반 문장 단위로 트리밍 (문장 분할기 필요)

### 2.3 버퍼 트리밍 임계값 (`--buffer_trimming_sec`)
```bash
--buffer_trimming_sec FLOAT
```
- **기본값**: `15.0` (초)
- **설명**: 버퍼가 이 길이를 초과하면 트리밍 실행
- **메모리 vs 정확도 트레이드오프**:
  - 작을수록: 메모리 효율적, 컨텍스트 손실
  - 클수록: 메모리 사용량 증가, 더 나은 컨텍스트

## 3. 음성 활동 감지 (VAD/VAC) 파라미터

### 3.1 VAD 활성화 (`--vad`)
```bash
--vad
```
- **기본값**: `False`
- **설명**: Whisper 내장 Voice Activity Detection 사용
- **장점**: 무음 구간 제거, 처리 효율성 향상

### 3.2 VAC 활성화 (`--vac`)
```bash
--vac
```
- **기본값**: `False`
- **설명**: Voice Activity Controller 사용 (Silero VAD 기반)
- **장점**: 더 정확한 음성 감지, 실시간 처리 최적화
- **요구사항**: `torch`, `torchaudio` 필요

### 3.3 VAC 청크 크기 (`--vac-chunk-size`)
```bash
--vac-chunk-size FLOAT
```
- **기본값**: `0.04` (초)
- **설명**: VAC가 분석할 오디오 청크 크기
- **영향**: 작을수록 더 민감한 음성 감지

## 4. Silero VAD 세부 파라미터 (코드 레벨)

### 4.1 음성 감지 임계값
```python
threshold: float = 0.5
```
- **설명**: 음성으로 판단할 확률 임계값
- **범위**: 0.0 ~ 1.0
- **조정**: 낮을수록 민감, 높을수록 보수적

### 4.2 최소 침묵 지속시간
```python
min_silence_duration_ms: int = 500
```
- **설명**: 음성 종료 판단을 위한 최소 침묵 시간 (밀리초)
- **영향**: 짧을수록 빠른 반응, 길수록 안정적

### 4.3 음성 패딩
```python
speech_pad_ms: int = 100
```
- **설명**: 음성 청크 앞뒤에 추가할 패딩 시간 (밀리초)
- **목적**: 음성 시작/끝 부분 손실 방지

## 5. 디코딩 전략 파라미터

### 5.1 Faster-Whisper 디코딩 파라미터
```python
# whisper_online.py의 FasterWhisperASR.transcribe() 내부
beam_size=5                    # 빔 서치 크기
word_timestamps=True           # 단어별 타임스탬프
condition_on_previous_text=True # 이전 텍스트 조건부 생성
```

#### 빔 서치 크기 조정
- **beam_size=1**: 가장 빠름, 품질 낮음 (그리디 디코딩)
- **beam_size=5**: 기본값, 속도와 품질의 균형
- **beam_size=10**: 느림, 높은 품질

### 5.2 GPU/CPU 컴퓨팅 설정
```python
# GPU 설정 (faster-whisper)
device="cuda"
compute_type="float16"  # float16, int8_float16, int8

# CPU 설정
device="cpu"
compute_type="int8"
```

#### 컴퓨팅 타입 선택
- **float16**: 최고 품질, GPU 메모리 많이 사용
- **int8_float16**: 중간 품질, 메모리 절약
- **int8**: CPU용, 가장 느림

## 6. 모델 캐시 및 로딩 파라미터

### 6.1 모델 캐시 디렉토리 (`--model_cache_dir`)
```bash
--model_cache_dir PATH
```
- **설명**: 다운로드된 모델을 저장할 디렉토리
- **용도**: 네트워크 대역폭 절약, 빠른 로딩

### 6.2 커스텀 모델 디렉토리 (`--model_dir`)
```bash
--model_dir PATH
```
- **설명**: 사용자 정의 모델 경로
- **우선순위**: `--model`과 `--model_cache_dir`보다 우선

## 7. 서버 모드 파라미터

### 7.1 네트워크 설정
```bash
--host HOST          # 기본값: localhost
--port PORT          # 기본값: 43007
```

### 7.2 웜업 파일 (`--warmup-file`)
```bash
--warmup-file PATH
```
- **설명**: 첫 번째 처리 속도 향상을 위한 웜업 오디오 파일
- **효과**: 첫 번째 청크 처리 시간 단축

## 8. 로깅 및 디버깅 파라미터

### 8.1 로그 레벨 (`--log-level`)
```bash
--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
```
- **기본값**: `DEBUG`
- **DEBUG**: 모든 상세 정보
- **INFO**: 일반 정보
- **WARNING**: 경고만
- **ERROR**: 오류만

## 9. 고급 튜닝 파라미터

### 9.1 프롬프트 크기 (코드 레벨)
```python
# whisper_online.py의 prompt() 함수 내부
l < 200  # 200 characters prompt size
```
- **설명**: 이전 텍스트에서 가져올 프롬프트 최대 길이
- **영향**: 길수록 더 나은 컨텍스트, 처리 시간 증가

### 9.2 no_speech_prob 임계값
```python
# FasterWhisperASR.ts_words() 내부
if segment.no_speech_prob > 0.9:
    continue  # 이 세그먼트는 무시
```
- **설명**: 음성이 아닌 것으로 판단할 확률 임계값
- **조정**: 높을수록 더 엄격한 필터링

### 9.3 샘플링 레이트 (고정값)
```python
SAMPLING_RATE = 16000  # Hz
```
- **설명**: 모든 오디오는 16kHz로 리샘플링됨
- **Whisper 요구사항**: 16kHz 고정

## 10. 실제 사용 예시

### 10.1 실시간성 중시 설정
```bash
python whisper_online.py \
    --model tiny \
    --backend faster-whisper \
    --min-chunk-size 0.5 \
    --vac \
    --vac-chunk-size 0.04 \
    --buffer_trimming segment \
    --buffer_trimming_sec 10
```

### 10.2 품질 중시 설정
```bash
python whisper_online.py \
    --model large-v3 \
    --backend faster-whisper \
    --min-chunk-size 2.0 \
    --vad \
    --buffer_trimming sentence \
    --buffer_trimming_sec 30 \
    --lan ko
```

### 10.3 Apple Silicon 최적화 설정
```bash
python whisper_online.py \
    --model large-v3-turbo \
    --backend mlx-whisper \
    --min-chunk-size 1.0 \
    --vac \
    --lan auto
```

### 10.4 서버 모드 설정
```bash
python whisper_online_server.py \
    --host 0.0.0.0 \
    --port 8080 \
    --model medium \
    --warmup-file harvard.wav \
    --vac \
    --log-level INFO
```

## 11. 성능 최적화 가이드라인

### 11.1 지연시간 최소화
1. 작은 모델 사용 (`tiny`, `base`)
2. 작은 청크 크기 (`0.5-1.0`초)
3. VAC 활성화
4. GPU 사용 (faster-whisper)

### 11.2 정확도 최대화
1. 큰 모델 사용 (`large-v3`)
2. 큰 청크 크기 (`2.0-3.0`초)
3. 긴 버퍼 유지 (`30`초)
4. 적절한 언어 설정

### 11.3 메모리 사용량 최소화
1. 작은 모델 사용
2. 짧은 버퍼 트리밍 (`10-15`초)
3. int8 컴퓨팅 타입
4. VAC로 무음 구간 제거

이 가이드를 참조하여 특정 사용 사례에 맞게 파라미터를 조정하시기 바랍니다.
