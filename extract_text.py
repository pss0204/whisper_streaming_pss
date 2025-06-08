#!/usr/bin/env python3
"""
Script to extract only text from whisper_online.py output
"""
import sys
import re

def extract_text_from_output(input_file, output_file):
    """Extract only text from output file"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    extracted_texts = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Pattern: Extract text following numbers
        # Format: "timestamp start_ms end_ms text"
        parts = line.split(' ', 3)  # Split into maximum 4 parts
        
        if len(parts) >= 4:
            # Fourth part is the actual text
            text = parts[3].strip()
            if text:
                extracted_texts.append(text)
    
    # Combine texts into one
    full_text = ' '.join(extracted_texts)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Text extraction completed: {input_file} -> {output_file}")
    print(f"Extracted text length: {len(full_text)} characters")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_text.py input_file output_file")
        print("Example: python extract_text.py out.txt clean_output.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    extract_text_from_output(input_file, output_file)