# Whisper Streaming - Complete Guide to All Configurable Parameters

## 1. Core Model Configuration Parameters

### 1.1 Whisper Model Selection (`--model`)
```bash
--model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo}
```
- **Default**: `large-v2`
- **Description**: Size and version of Whisper model to use
- **Performance vs Speed Trade-off**:
  - `tiny` (39MB): Fastest, low quality
  - `base` (74MB): Fast, basic quality
  - `small` (244MB): Medium speed, good quality
  - `medium` (769MB): Slow, very good quality
  - `large-v3` (1550MB): Slowest, highest quality
  - `large-v3-turbo`: Latest version, 8x faster than large-v3
- **Language-specific models**: `.en` suffix are English-only models (better performance for English)

### 1.2 Backend Selection (`--backend`)
```bash
--backend {faster-whisper,whisper_timestamped,mlx-whisper,openai-api}
```
- **Default**: `faster-whisper`
- **Backend Characteristics**:
  - `faster-whisper`: Fastest (requires GPU), CUDA support
  - `whisper_timestamped`: Medium speed, easy GPU installation
  - `mlx-whisper`: Apple Silicon optimized (M1, M2, etc.)
  - `openai-api`: Cloud-based, no GPU required, costs incurred

### 1.3 Language Setting (`--lan`, `--language`)
```bash
--lan {auto,en,ko,ja,zh,de,fr,es,...}
```
- **Default**: `auto`
- **Supported Languages**: 80+ languages supported
- **Korean**: `ko`
- **Auto Detection**: `auto` (detects language based on first 30 seconds)

### 1.4 Task Type (`--task`)
```bash
--task {transcribe,translate}
```
- **Default**: `transcribe`
- `transcribe`: Transcribe in original language
- `translate`: Translate to English

## 2. Audio Processing and Chunking Parameters

### 2.1 Minimum Chunk Size (`--min-chunk-size`)
```bash
--min-chunk-size FLOAT
```
- **Default**: `1.0` (seconds)
- **Description**: Minimum audio chunk size to process
- **Impact**: 
  - Smaller: Lower latency, higher CPU usage
  - Larger: Higher latency, lower CPU usage, more accurate transcription
- **Recommended Values**: 
  - Real-time priority: `0.5-1.0` seconds
  - Quality priority: `2.0-3.0` seconds

### 2.2 Buffer Trimming Method (`--buffer_trimming`)
```bash
--buffer_trimming {sentence,segment}
```
- **Default**: `segment`
- **segment**: Trim by segments returned by Whisper
- **sentence**: Trim by sentence units based on punctuation (requires sentence splitter)

### 2.3 Buffer Trimming Threshold (`--buffer_trimming_sec`)
```bash
--buffer_trimming_sec FLOAT
```
- **Default**: `15.0` (seconds)
- **Description**: Execute trimming when buffer exceeds this length
- **Memory vs Accuracy Trade-off**:
  - Smaller: Memory efficient, context loss
  - Larger: Increased memory usage, better context

## 3. Voice Activity Detection (VAD/VAC) Parameters

### 3.1 VAD Activation (`--vad`)
```bash
--vad
```
- **Default**: `False`
- **Description**: Use Whisper built-in Voice Activity Detection
- **Benefits**: Remove silent sections, improve processing efficiency

### 3.2 VAC Activation (`--vac`)
```bash
--vac
```
- **Default**: `False`
- **Description**: Use Voice Activity Controller (based on Silero VAD)
- **Benefits**: More accurate voice detection, real-time processing optimization
- **Requirements**: `torch`, `torchaudio` required

### 3.3 VAC Chunk Size (`--vac-chunk-size`)
```bash
--vac-chunk-size FLOAT
```
- **Default**: `0.04` (seconds)
- **Description**: Audio chunk size for VAC analysis
- **Impact**: Smaller values enable more sensitive voice detection

## 4. Detailed Silero VAD Parameters (Code Level)

### 4.1 Voice Detection Threshold
```python
threshold: float = 0.5
```
- **Description**: Probability threshold for voice detection
- **Range**: 0.0 ~ 1.0
- **Adjustment**: Lower = more sensitive, Higher = more conservative

### 4.2 Minimum Silence Duration
```python
min_silence_duration_ms: int = 500
```
- **Description**: Minimum silence time to determine voice end (milliseconds)
- **Impact**: Shorter = faster response, Longer = more stable

### 4.3 Speech Padding
```python
speech_pad_ms: int = 100
```
- **Description**: Padding time to add before and after speech chunks (milliseconds)
- **Purpose**: Prevent loss of speech start/end portions

## 5. Decoding Strategy Parameters

### 5.1 Faster-Whisper Decoding Parameters
```python
# Inside FasterWhisperASR.transcribe() in whisper_online.py
beam_size=5                    # Beam search size
word_timestamps=True           # Word-level timestamps
condition_on_previous_text=True # Conditional generation on previous text
```

#### Beam Search Size Adjustment
- **beam_size=1**: Fastest, low quality (greedy decoding)
- **beam_size=5**: Default, balance of speed and quality
- **beam_size=10**: Slow, high quality

### 5.2 GPU/CPU Computing Settings
```python
# GPU settings (faster-whisper)
device="cuda"
compute_type="float16"  # float16, int8_float16, int8

# CPU settings
device="cpu"
compute_type="int8"
```

#### Computing Type Selection
- **float16**: Highest quality, uses more GPU memory
- **int8_float16**: Medium quality, memory saving
- **int8**: For CPU, slowest

## 6. Model Cache and Loading Parameters

### 6.1 Model Cache Directory (`--model_cache_dir`)
```bash
--model_cache_dir PATH
```
- **Description**: Directory to store downloaded models
- **Purpose**: Save network bandwidth, faster loading

### 6.2 Custom Model Directory (`--model_dir`)
```bash
--model_dir PATH
```
- **Description**: Custom model path
- **Priority**: Takes precedence over `--model` and `--model_cache_dir`

## 7. Server Mode Parameters

### 7.1 Network Settings
```bash
--host HOST          # Default: localhost
--port PORT          # Default: 43007
```

### 7.2 Warmup File (`--warmup-file`)
```bash
--warmup-file PATH
```
- **Description**: Warmup audio file to improve first processing speed
- **Effect**: Reduce first chunk processing time

## 8. Logging and Debugging Parameters

### 8.1 Log Level (`--log-level`)
```bash
--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
```
- **Default**: `DEBUG`
- **DEBUG**: All detailed information
- **INFO**: General information
- **WARNING**: Warnings only
- **ERROR**: Errors only

## 9. Advanced Tuning Parameters

### 9.1 Prompt Size (Code Level)
```python
# Inside prompt() function in whisper_online.py
l < 200  # 200 characters prompt size
```
- **Description**: Maximum length of prompt from previous text
- **Impact**: Longer = better context, increased processing time

### 9.2 no_speech_prob Threshold
```python
# Inside FasterWhisperASR.ts_words()
if segment.no_speech_prob > 0.9:
    continue  # Ignore this segment
```
- **Description**: Probability threshold for non-speech detection
- **Adjustment**: Higher = more strict filtering

### 9.3 Sampling Rate (Fixed Value)
```python
SAMPLING_RATE = 16000  # Hz
```
- **Description**: All audio is resampled to 16kHz
- **Whisper Requirement**: Fixed at 16kHz

## 10. Practical Usage Examples

### 10.1 Real-time Priority Settings
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

### 10.2 Quality Priority Settings
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

### 10.3 Apple Silicon Optimized Settings
```bash
python whisper_online.py \
    --model large-v3-turbo \
    --backend mlx-whisper \
    --min-chunk-size 1.0 \
    --vac \
    --lan auto
```

### 10.4 Server Mode Settings
```bash
python whisper_online_server.py \
    --host 0.0.0.0 \
    --port 8080 \
    --model medium \
    --warmup-file harvard.wav \
    --vac \
    --log-level INFO
```

## 11. Performance Optimization Guidelines

### 11.1 Minimizing Latency
1. Use small models (`tiny`, `base`)
2. Small chunk size (`0.5-1.0` seconds)
3. Enable VAC
4. Use GPU (faster-whisper)

### 11.2 Maximizing Accuracy
1. Use large models (`large-v3`)
2. Large chunk size (`2.0-3.0` seconds)
3. Maintain long buffer (`30` seconds)
4. Proper language settings

### 11.3 Minimizing Memory Usage
1. Use small models
2. Short buffer trimming (`10-15` seconds)
3. int8 computing type
4. Remove silent sections with VAC

Please refer to this guide to adjust parameters for your specific use case.
