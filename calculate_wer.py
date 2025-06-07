#!/usr/bin/env python3
# filepath: /home/pss/whisper_streaming/calculate_wer.py

"""
개선된 WER 계산 스크립트
"""

import sys
from jiwer import wer
import re

def clean_text(text):
    """텍스트 정리 함수"""
    # 특수 문자 제거하고 공백 정리
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())  # 다중 공백을 단일 공백으로
    return text.lower().strip()

def calculate_wer_from_files(reference_file, hypothesis_file):
    """파일에서 WER 계산"""
    
    # 참조 텍스트 읽기
    with open(reference_file, 'r', encoding='utf-8') as f:
        reference = f.read()
    
    # 가설 텍스트 읽기 (숫자가 섞인 형식에서 텍스트만 추출)
    with open(hypothesis_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    hypothesis_parts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 타임스탬프 정보 제거하고 텍스트만 추출
        parts = line.split(' ', 3)
        if len(parts) >= 4:
            text = parts[3].strip()
            if text:
                hypothesis_parts.append(text)
    
    hypothesis = ' '.join(hypothesis_parts)
    
    # 텍스트 정리
    reference_clean = clean_text(reference)
    hypothesis_clean = clean_text(hypothesis)
    
    print(f"참조 텍스트 (처음 100자): {reference_clean[:100]}...")
    print(f"가설 텍스트 (처음 100자): {hypothesis_clean[:100]}...")
    print()
    
    # WER 계산
    error_rate = wer(reference_clean, hypothesis_clean)
    
    print(f"Word Error Rate: {error_rate:.4f} ({error_rate*100:.2f}%)")
    
    return error_rate

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python calculate_wer_improved.py reference.txt hypothesis.txt")
        sys.exit(1)
    
    reference_file = sys.argv[1]
    hypothesis_file = sys.argv[2]
    
    calculate_wer_from_files(reference_file, hypothesis_file)