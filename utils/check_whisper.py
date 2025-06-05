# check_whisper_setup.py
import sys

def check_dependencies():
    print("=== Whisper Streaming 설치 상태 확인 ===\n")
    
    # 기본 라이브러리 확인
    try:
        import librosa
        import soundfile
        print("✅ 기본 오디오 라이브러리 (librosa, soundfile) - OK")
    except ImportError as e:
        print(f"❌ 기본 라이브러리 오류: {e}")
    
    # Faster-Whisper 확인
    try:
        from faster_whisper import WhisperModel
        print("✅ Faster-Whisper - 설치됨")
        try:
            model = WhisperModel('tiny', device='cuda')
            print("  🚀 GPU 사용 가능")
        except:
            print("  💻 CPU만 사용 가능")
    except ImportError:
        print("❌ Faster-Whisper - 미설치")
    
    # Whisper-Timestamped 확인
    try:
        import whisper_timestamped
        print("✅ Whisper-Timestamped - 설치됨")
    except ImportError:
        print("❌ Whisper-Timestamped - 미설치")
    
    # OpenAI API 확인
    try:
        import openai
        import os
        if os.getenv('OPENAI_API_KEY'):
            print("✅ OpenAI API - 설정됨")
        else:
            print("⚠️  OpenAI API - 키 없음")
    except ImportError:
        print("❌ OpenAI API - 미설치")
    
    # MLX Whisper 확인 (Apple Silicon)
    try:
        import mlx_whisper
        import mlx.core as mx
        print(f"✅ MLX Whisper - 설치됨 (디바이스: {mx.default_device()})")
    except ImportError:
        print("❌ MLX Whisper - 미설치")
    
    # VAC 확인
    try:
        import torch
        import torchaudio
        print("✅ VAC (음성 활동 제어) - 사용 가능")
    except ImportError:
        print("❌ VAC - 미설치 (pip install torch torchaudio)")

if __name__ == "__main__":
    check_dependencies()