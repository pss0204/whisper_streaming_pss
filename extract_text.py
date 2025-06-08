#!/usr/bin/env python3
"""
Script to extract only text from whisper_online.py output and analyze latency
"""
import sys
import re
import numpy as np

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

def analyze_latency(input_file, threshold=5.0):
    """Analyze latency from whisper_online.py output"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    latencies = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Parse: emission_time start_time end_time text
        parts = line.split(' ', 3)
        
        if len(parts) >= 3:
            try:
                emission_time = float(parts[0])  # ms
                start_time = float(parts[1])     # ms ← 이걸 사용해야 함
                end_time = float(parts[2])       # ms
                
                # 올바른 레이턴시 계산: 음성 시작부터 출력까지
                latency = (emission_time - start_time) / 1000
                latencies.append(latency)
                
            except ValueError:
                continue
    
    if not latencies:
        print("No valid latency data found!")
        return
    
    # Convert to numpy array for analysis
    latencies = np.array(latencies)
    
    # Basic statistics
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    std_latency = np.std(latencies)
    
    # Count above threshold
    above_threshold = np.sum(latencies > threshold)
    total_chunks = len(latencies)
    percentage_above = (above_threshold / total_chunks) * 100
    
    print("\n" + "="*50)
    print("LATENCY ANALYSIS")
    print("="*50)
    print(f"Total chunks: {total_chunks}")
    print(f"Average latency: {mean_latency:.3f} seconds")
    print(f"Median latency: {median_latency:.3f} seconds")
    print(f"Standard deviation: {std_latency:.3f} seconds")
    print(f"Chunks above {threshold:.1f}s: {above_threshold}/{total_chunks} ({percentage_above:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_text.py input_file [output_file] [--latency-threshold X.X]")
        print("Examples:")
        print("  python extract_text.py out.txt clean_output.txt")
        print("  python extract_text.py out.txt clean_output.txt --latency-threshold 2.0")
        print("  python extract_text.py out.txt --latency-only")
        print("  python extract_text.py out.txt --latency-only --latency-threshold 5.0")
        sys.exit(1)
    
    input_file = sys.argv[1]
    threshold = 5.0  # 기본 임계값
    latency_only = False
    output_file = None
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--latency-threshold" and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--latency-only":
            latency_only = True
            i += 1
        else:
            if output_file is None and not latency_only:
                output_file = sys.argv[i]
            i += 1
    
    # Execute based on arguments
    if latency_only:
        analyze_latency(input_file, threshold)
    elif output_file:
        extract_text_from_output(input_file, output_file)
        analyze_latency(input_file, threshold)
    else:
        analyze_latency(input_file, threshold)