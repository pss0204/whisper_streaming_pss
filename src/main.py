import sys
sys.path.append('/home/pss/whisper_streaming')

import soundfile as sf

from whisper_online import *
import librosa
import numpy as np
from run import run_asr

# ASR 설정
asr = CustomFasterWhisperASR("en", "base")

# 오디오 파일 로드
audio_file = "/home/pss/whisper_streaming/harvard.wav"
audio, sr = librosa.load(audio_file, sr=16000, mono=True)

# ASR 실행
results = run_asr(audio, asr)

# 결과 출력
for result in results:
    if result[0] is not None:
        start, end, text = result
        print(f"{start:.2f}-{end:.2f}: {text}")
        
        