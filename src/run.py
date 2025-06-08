import sys
sys.path.append('/home/pss/whisper_streaming')

import soundfile as sf
from whisper_online import *
import librosa
import numpy as np
from datasets import load_dataset
from jiwer import wer
from src.config import Config

# Global ASR object
asr_model = None

def init_asr(language="en", model_size="large-v2"):
    """Initialize ASR model"""
    global asr_model
    if asr_model is None:
        asr_model = CustomFasterWhisperASR(language, model_size)
        # Check model information - use correct attributes
        print(f"✅ Model loaded: {model_size}")
        try:
            # For faster-whisper, access differently instead of model_size_or_path
            print(f"✅ Device: {asr_model.model.device}")
            print(f"✅ Compute type: {asr_model.model.compute_type}")
            # Remove model_size_or_path or access differently
            print(f"✅ Model size: {model_size}")
        except AttributeError as e:
            print(f"⚠️  Model info access error: {e}")
    return asr_model

def run_asr(audio_array, config=Config):
    """Run ASR on audio array (real-time simulation)"""
    global asr_model
    if asr_model is None:
        raise ValueError("ASR model not initialized. Call init_asr() first.")
    
    processor = OnlineASRProcessor(
        asr=asr_model,
        tokenizer=None,
        buffer_trimming=("sentence", 30)
    )

    asr_model.beam_size = config.beam_size
    
    # Apply whisper_online.py style processing loop
    start_time = time.time()
    results = []
    word_latencies = []
    
    # Use min_chunk_size from config
    min_chunk = config.min_chunk_size
    
    # Settings for simulating audio in memory
    total_duration = len(audio_array) / 16000.0
    beg = 0.0  # Start time (seconds)
    end = 0.0  # End time (seconds)
    
    # Apply the core processing loop from whisper_online.py
    while end < total_duration:
        now = time.time() - start_time
        
        # Wait for min_chunk duration (real-time simulation)
        if now < end + min_chunk:
            time.sleep(min_chunk + end - now)
        
        # Calculate new end time
        end = time.time() - start_time
        if end > total_duration:
            end = total_duration
        
        # Load audio chunk (from beg to end)
        beg_samples = int(beg * 16000)
        end_samples = int(end * 16000)
        
        if end_samples > len(audio_array):
            end_samples = len(audio_array)
            
        chunk = audio_array[beg_samples:end_samples]
        
        if len(chunk) > 0:
            # Insert chunk
            processor.insert_audio_chunk(chunk)
            
            try:
                # ASR processing
                result = processor.process_iter()
                
                # Process results
                current_real_time = time.time() - start_time
                
                if result and result[0] is not None:
                    results.append(result)
                    
                    word_start_time = result[0]
                    word_end_time = result[1]
                    
                    if word_end_time is not None:
                        # Calculate latency
                        latency = current_real_time - word_end_time
                        word_latencies.append(latency)
                        print(f"Word/phrase at {word_start_time:.2f}-{word_end_time:.2f}s, "
                              f"output at real time {current_real_time:.2f}s, latency: {latency:.2f}s")
                        
            except AssertionError as e:
                print(f"Assertion error: {e}")
        
        # Update beg for next iteration
        beg = end
        
        # Prevent infinite loop
        if beg >= total_duration:
            break
    
    # Process final result
    final_result = processor.finish()
    if final_result and final_result[0] is not None:
        results.append(final_result)

    avg_latency = np.mean(word_latencies) if word_latencies else 0.0
    beam_size = asr_model.beam_size

    return results, avg_latency, beam_size, min_chunk

def process_audio_with_asr(sample,config=Config):
    """Function for dataset.map - audio preprocessing and ASR execution"""
    print(f"Processing sample...")  # Add debugging output
    try:
        # Extract audio data
        audio_array = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']
        
        # Resample to 16kHz (whisper requirement)
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        
        # Convert to numpy array
        audio_array = np.array(audio_array, dtype=np.float32)
        
        # Run ASR
        asr_results, avg_latency, beam_size, min_chunk = run_asr(audio_array, config=config)

        # Convert results to text
        pred_text = ""
        if asr_results:
            # Combine all results into one text
            text_parts = []
            for result in asr_results:
                if result and len(result) >= 3 and result[2]:
                    text_parts.append(result[2].strip())
            pred_text = " ".join(text_parts)

        # Extract actual text (ground truth)
        real_text = sample["human_transcript"]
        sample["avg_latency"] = avg_latency
        
        # Add pred to existing sample
        sample['pred'] = pred_text
        sample['beam_size'] = beam_size
        sample['min_chunk'] = min_chunk

        wer_value = wer(real_text, pred_text)
        sample['whisper_wer'] = wer_value
        
        return sample
        
    except Exception as e:
        print(f"Error processing audio sample: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace
        # Return empty prediction text and default WER when error occurs
        sample['pred'] = ""
        sample['whisper_wer'] = 1.0  # Maximum WER value
        return sample

