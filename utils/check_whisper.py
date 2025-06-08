# check_whisper_setup.py
import sys

def check_dependencies():
    print("=== Whisper Streaming Installation Status Check ===\n")
    
    # Check basic libraries
    try:
        import librosa
        import soundfile
        print("✅ Basic audio libraries (librosa, soundfile) - OK")
    except ImportError as e:
        print(f"❌ Basic library error: {e}")
    
    # Faster-Whisper check
    try:
        from faster_whisper import WhisperModel
        print("✅ Faster-Whisper - Installed")
        try:
            model = WhisperModel('tiny', device='cuda')
            print("  🚀 GPU available")
        except:
            print("  💻 CPU only available")
    except ImportError:
        print("❌ Faster-Whisper - Not installed")
    
    # Whisper-Timestamped check
    try:
        import whisper_timestamped
        print("✅ Whisper-Timestamped - Installed")
    except ImportError:
        print("❌ Whisper-Timestamped - Not installed")
    
    # OpenAI API check
    try:
        import openai
        import os
        if os.getenv('OPENAI_API_KEY'):
            print("✅ OpenAI API - Configured")
        else:
            print("⚠️  OpenAI API - No key")
    except ImportError:
        print("❌ OpenAI API - Not installed")
    
    # MLX Whisper check (Apple Silicon)
    try:
        import mlx_whisper
        import mlx.core as mx
        print(f"✅ MLX Whisper - Installed (device: {mx.default_device()})")
    except ImportError:
        print("❌ MLX Whisper - Not installed")
    
    # VAC check
    try:
        import torch
        import torchaudio
        print("✅ VAC (Voice Activity Control) - Available")
    except ImportError:
        print("❌ VAC - Not installed (pip install torch torchaudio)")

if __name__ == "__main__":
    check_dependencies()