"""
PDF 구조 분석 스크립트 - 일러스트형 문제의 레이아웃 패턴 파악
정답, 해설, 설명 텍스트의 위치 관계를 분석합니다.
"""

import fitz
import pandas as pd
from typing import List, Dict
import re


def analyze_pdf_structure(pdf_path: str, sample_pages: int = 5) -> None:
    """
    PDF의 구조를 분석하여 텍스트 배치 패턴을 파악합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        sample_pages: 분석할 페이지 수
    """
    
    doc = fitz.open(pdf_path)
    
    print("=== PDF 구조 분석 시작 ===\n")
    
    for page_num in range(min(sample_pages, len(doc))):
        page = doc[page_num]
        
        print(f"📄 Page {page_num + 1}")
        print("=" * 50)
        
        # 1. 모든 텍스트 블록을 Y 좌표 순으로 추출
        text_dict = page.get_text("dict")
        text_blocks = []
        
        for block in text_dict["blocks"]:
            if "lines" in block:
                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span.get("text", "")
                
                if block_text.strip():
                    text_blocks.append({
                        'text': block_text.strip(),
                        'y_position': block["bbox"][1],
                        'bbox': block["bbox"]
                    })
        
        # Y 좌표 순으로 정렬
        text_blocks.sort(key=lambda x: x['y_position'])
        
        # 2. 문항 번호 찾기
        question_numbers = []
        for i, block in enumerate(text_blocks):
            text = block['text']
            if re.search(r'(\d+)\.\s', text):
                match = re.search(r'(\d+)\.\s', text)
                num = int(match.group(1))
                if 781 <= num <= 865:
                    question_numbers.append((i, num, block['y_position']))
        
        print(f"발견된 문항: {[q[1] for q in question_numbers]}")
        
        # 3. 각 문항별 구조 분석
        for q_idx, (block_idx, q_num, q_y) in enumerate(question_numbers):
            print(f"\n🔍 문항 {q_num} 분석:")
            
            # 다음 문항까지의 범위 설정
            if q_idx + 1 < len(question_numbers):
                next_y = question_numbers[q_idx + 1][2]
                end_idx = question_numbers[q_idx + 1][0]
            else:
                next_y = float('inf')
                end_idx = len(text_blocks)
            
            # 해당 문항 영역의 텍스트들 분석
            area_blocks = text_blocks[block_idx:end_idx]
            
            for i, block in enumerate(area_blocks):
                text = block['text']
                y_pos = block['y_position']
                
                # 텍스트 타입 분류
                text_type = classify_text_type(text)
                
                print(f"  [{i:2d}] Y:{y_pos:6.1f} | {text_type:12s} | {text[:80]}")
        
        print("\n" + "="*50 + "\n")
    
    doc.close()


def classify_text_type(text: str) -> str:
    """
    텍스트의 타입을 분류합니다.
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        str: 텍스트 타입
    """
    
    # 문항 번호
    if re.search(r'^\d+\.\s', text):
        return "QUESTION_NUM"
    
    # ■ 마커 (설명 텍스트)
    if text.startswith('■'):
        return "DESCRIPTION"
    
    # 정답 패턴
    if re.search(r'①|②|③|④|⑤', text) or re.search(r'정답[:：]\s*[①②③④⑤]', text):
        return "ANSWER"
    
    # 해설 키워드
    if any(keyword in text for keyword in ['해설', '풀이', '따라서', '그러므로']):
        return "EXPLANATION"
    
    # 법규 조항
    if re.search(r'도로교통법|제\d+조|별표\d+', text):
        return "LAW_CLAUSE"
    
    # 긴 설명문 (100자 이상)
    if len(text) > 100:
        return "LONG_TEXT"
    
    # 선택지
    if re.search(r'^\d+\)', text):
        return "CHOICE"
    
    # 기타
    return "OTHER"


def demo_analyze():
    """
    PDF 구조 분석 데모
    """
    pdf_path = "g:/내 드라이브/4_paper/72_2_license_llm_how/data/4_raw_data_fromkoroad_일러스트.pdf"
    
    print("PDF 구조 분석을 시작합니다...")
    analyze_pdf_structure(pdf_path, sample_pages=3)
    
    print("\n분석 완료! 패턴을 확인하여 필터링 로직을 개선할 수 있습니다.")


if __name__ == "__main__":
    demo_analyze()
