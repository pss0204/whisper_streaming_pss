#!/usr/bin/env python3
"""
whisper_online.py 출력에서 텍스트만 추출하는 스크립트
"""
import sys
import re

def extract_text_from_output(input_file, output_file):
    """출력 파일에서 텍스트만 추출"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    extracted_texts = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 패턴: 숫자들 다음에 오는 텍스트 추출
        # 형식: "timestamp start_ms end_ms text"
        parts = line.split(' ', 3)  # 최대 4개로 분할
        
        if len(parts) >= 4:
            # 네 번째 부분이 실제 텍스트
            text = parts[3].strip()
            if text:
                extracted_texts.append(text)
    
    # 텍스트들을 하나로 합치기
    full_text = ' '.join(extracted_texts)
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"텍스트 추출 완료: {input_file} -> {output_file}")
    print(f"추출된 텍스트 길이: {len(full_text)} 문자")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python extract_text.py input_file output_file")
        print("예시: python extract_text.py out.txt clean_output.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    extract_text_from_output(input_file, output_file)