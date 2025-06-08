#!/usr/bin/env python3
# filepath: /home/pss/whisper_streaming/calculate_wer.py

"""
Improved WER calculation script
"""

import sys
from jiwer import wer
import re

def clean_text(text):
    """Text cleaning function"""
    # Remove special characters and clean whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())  # Convert multiple spaces to single space
    return text.lower().strip()

def calculate_wer_from_files(reference_file, hypothesis_file):
    """Calculate WER from files"""
    
    # Read reference text
    with open(reference_file, 'r', encoding='utf-8') as f:
        reference = f.read()
    
    # Read hypothesis text (extract text only from format mixed with numbers)
    with open(hypothesis_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    hypothesis_parts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove timestamp information and extract text only
        parts = line.split(' ', 3)
        if len(parts) >= 4:
            text = parts[3].strip()
            if text:
                hypothesis_parts.append(text)
    
    hypothesis = ' '.join(hypothesis_parts)
    
    # Clean text
    reference_clean = clean_text(reference)
    hypothesis_clean = clean_text(hypothesis)
    
    print(f"Reference text (first 100 chars): {reference_clean[:100]}...")
    print(f"Hypothesis text (first 100 chars): {hypothesis_clean[:100]}...")
    print()
    
    # Calculate WER
    error_rate = wer(reference_clean, hypothesis_clean)
    
    print(f"Word Error Rate: {error_rate:.4f} ({error_rate*100:.2f}%)")
    
    return error_rate

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_wer_improved.py reference.txt hypothesis.txt")
        sys.exit(1)
    
    reference_file = sys.argv[1]
    hypothesis_file = sys.argv[2]
    
    calculate_wer_from_files(reference_file, hypothesis_file)