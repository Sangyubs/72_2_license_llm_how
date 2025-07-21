"""
PDF에서 사진 우측 칸의 텍스트를 추출하는 함수들
운전면허 시험 문제의 사진형 문제에서 사진 옆에 있는 설명 텍스트를 추출합니다.
"""

import pandas as pd
import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re
import json


def extract_right_side_text_from_pdf(
    pdf_path: str,
    existing_mapping_path: Optional[str] = None
) -> Dict[str, Dict]:
    """
    PDF에서 사진 우측 칸에 있는 텍스트를 추출하여 문제 번호와 연결합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        existing_mapping_path: 기존 문제-이미지 매핑 JSON 파일 경로 (선택적)
        
    Returns:
        Dict: {
            'question_text_mapping': {question_id: right_side_text},
            'extraction_details': [페이지별 추출 세부정보],
            'extraction_log': [추출 과정 로그]
        }
    """
    
    doc = fitz.open(pdf_path)
    question_text_mapping = {}
    extraction_details = []
    extraction_log = []
    
    # 기존 매핑 정보 로드 (선택적)
    existing_mapping = {}
    if existing_mapping_path and Path(existing_mapping_path).exists():
        with open(existing_mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            existing_mapping = data.get('question_image_mapping', {})
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 1. 페이지에서 텍스트와 위치 정보 추출
        text_dict = page.get_text("dict")
        
        # 2. 페이지에서 이미지 위치 정보 추출
        image_rects = get_image_positions_on_page(page)
        
        # 3. 문항번호 추출
        question_numbers = extract_question_numbers_from_page(page)
        
        # 4. 각 문항에 대해 우측 텍스트 추출
        page_text_mapping = {}
        for question_num in question_numbers:
            question_id = f"question_{question_num}"
            
            # 해당 문항의 이미지 위치 찾기
            image_rect = find_image_rect_for_question(
                question_num, image_rects, page, text_dict
            )
            
            if image_rect:
                # 이미지 우측 영역의 텍스트 추출
                right_side_text = extract_text_right_of_image(
                    page, text_dict, image_rect
                )
                
                if right_side_text.strip():
                    page_text_mapping[question_id] = right_side_text
                    extraction_log.append(
                        f"Page {page_num+1}, Question {question_num}: "
                        f"추출된 텍스트 길이 {len(right_side_text)}"
                    )
        
        question_text_mapping.update(page_text_mapping)
        
        extraction_details.append({
            'page_number': page_num + 1,
            'question_count': len(question_numbers),
            'image_count': len(image_rects),
            'text_extracted_count': len(page_text_mapping),
            'questions': question_numbers,
            'text_mapping': page_text_mapping
        })
    
    doc.close()
    
    return {
        'question_text_mapping': question_text_mapping,
        'extraction_details': extraction_details,
        'extraction_log': extraction_log
    }


def get_image_positions_on_page(page) -> List[fitz.Rect]:
    """
    페이지에서 모든 이미지의 위치 정보를 반환합니다.
    
    Args:
        page: PyMuPDF 페이지 객체
        
    Returns:
        List[fitz.Rect]: 이미지 위치 사각형 리스트
    """
    image_rects = []
    image_list = page.get_images()
    
    for img in image_list:
        img_rects = page.get_image_rects(img)
        if img_rects:
            image_rects.extend(img_rects)
    
    return image_rects


def extract_question_numbers_from_page(page) -> List[str]:
    """
    페이지에서 문항번호를 추출합니다.
    
    Args:
        page: PyMuPDF 페이지 객체
        
    Returns:
        List[str]: 문항번호 리스트
    """
    text = page.get_text()
    
    # 문항번호 패턴들
    patterns = [
        r'(\d+)\.\s',  # "681. " 형태
        r'문제\s*(\d+)',  # "문제 681" 형태
        r'(\d+)번',  # "681번" 형태
    ]
    
    question_numbers = set()
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            question_numbers.add(match.group(1))
    
    return sorted(list(question_numbers))


def find_image_rect_for_question(
    question_num: str, 
    image_rects: List[fitz.Rect], 
    page, 
    text_dict: dict
) -> Optional[fitz.Rect]:
    """
    특정 문항번호에 해당하는 이미지의 위치를 찾습니다.
    
    Args:
        question_num: 문항번호
        image_rects: 페이지의 모든 이미지 위치
        page: PyMuPDF 페이지 객체
        text_dict: 페이지 텍스트 딕셔너리
        
    Returns:
        Optional[fitz.Rect]: 해당 문항의 이미지 위치 (없으면 None)
    """
    
    # 문항번호의 위치 찾기
    question_pattern = f"{question_num}\\."
    question_rect = None
    
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "")
                    if re.search(question_pattern, text):
                        question_rect = fitz.Rect(span["bbox"])
                        break
                if question_rect:
                    break
        if question_rect:
            break
    
    if not question_rect:
        # 문항번호를 찾지 못한 경우, 첫 번째 이미지 반환
        return image_rects[0] if image_rects else None
    
    # 문항번호와 가장 가까운 이미지 찾기
    closest_image = None
    min_distance = float('inf')
    
    for img_rect in image_rects:
        # Y 좌표 기준으로 거리 계산 (같은 줄이나 바로 아래 줄의 이미지)
        distance = abs(img_rect.y0 - question_rect.y0)
        if distance < min_distance:
            min_distance = distance
            closest_image = img_rect
    
    return closest_image


def extract_text_right_of_image(
    page, 
    text_dict: dict, 
    image_rect: fitz.Rect,
    margin: float = 10.0
) -> str:
    """
    이미지 우측 영역의 텍스트를 추출합니다.
    
    Args:
        page: PyMuPDF 페이지 객체
        text_dict: 페이지 텍스트 딕셔너리
        image_rect: 이미지 위치
        margin: 이미지와 텍스트 사이의 여백
        
    Returns:
        str: 추출된 텍스트
    """
    
    # 이미지 우측 영역 정의
    page_rect = page.rect
    right_area = fitz.Rect(
        image_rect.x1 + margin,  # 이미지 오른쪽 끝 + 여백
        image_rect.y0,           # 이미지 상단
        page_rect.x1,            # 페이지 오른쪽 끝
        image_rect.y1            # 이미지 하단
    )
    
    extracted_texts = []
    
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    span_rect = fitz.Rect(span["bbox"])
                    
                    # 스팬이 우측 영역과 겹치는지 확인
                    if span_rect.intersects(right_area):
                        text = span.get("text", "").strip()
                        if text and text not in extracted_texts:
                            extracted_texts.append(text)
    
    return " ".join(extracted_texts)


def save_text_mapping_results(
    text_mapping_results: Dict,
    output_path: str,
    csv_output_path: Optional[str] = None
) -> None:
    """
    텍스트 매핑 결과를 JSON과 CSV 파일로 저장합니다.
    
    Args:
        text_mapping_results: extract_right_side_text_from_pdf() 결과
        output_path: JSON 출력 파일 경로
        csv_output_path: CSV 출력 파일 경로 (선택적)
    """
    
    # JSON 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(text_mapping_results, f, ensure_ascii=False, indent=2)
    
    # CSV 저장 (선택적)
    if csv_output_path:
        question_text_mapping = text_mapping_results['question_text_mapping']
        
        df_data = []
        for question_id, right_text in question_text_mapping.items():
            question_num = question_id.replace('question_', '')
            df_data.append({
                'question_id': question_id,
                'question_number': question_num,
                'right_side_text': right_text,
                'text_length': len(right_text)
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')


def merge_with_existing_mapping(
    text_mapping_path: str,
    image_mapping_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    텍스트 매핑과 기존 이미지 매핑 결과를 통합합니다.
    
    Args:
        text_mapping_path: 텍스트 매핑 JSON 파일 경로
        image_mapping_path: 이미지 매핑 CSV 파일 경로
        output_path: 통합 결과 CSV 출력 경로
        
    Returns:
        pd.DataFrame: 통합된 데이터프레임
    """
    
    # 텍스트 매핑 로드
    with open(text_mapping_path, 'r', encoding='utf-8') as f:
        text_data = json.load(f)
    text_mapping = text_data['question_text_mapping']
    
    # 이미지 매핑 로드
    image_df = pd.read_csv(image_mapping_path)
    
    # 텍스트 정보 추가
    image_df['right_side_text'] = image_df['question_id'].map(text_mapping)
    image_df['has_right_text'] = image_df['right_side_text'].notna()
    image_df['text_length'] = image_df['right_side_text'].str.len().fillna(0)
    
    # 결과 저장
    image_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    return image_df


# 사용 예시 함수
def demo_extract_right_side_text():
    """
    사진 우측 텍스트 추출 데모 함수
    """
    pdf_path = "g:/내 드라이브/4_paper/72_2_license_llm_how/data/3_raw_data_fromkoroad_사진형.pdf"
    existing_mapping_path = "g:/내 드라이브/4_paper/72_2_license_llm_how/data/results/사진형/mapping_results_사진형_corrected.json"
    
    # 텍스트 추출
    results = extract_right_side_text_from_pdf(pdf_path, existing_mapping_path)
    
    # 결과 저장
    output_dir = Path("g:/내 드라이브/4_paper/72_2_license_llm_how/data/results/사진형")
    json_output = output_dir / "text_mapping_results.json"
    csv_output = output_dir / "question_text_mapping.csv"
    
    save_text_mapping_results(results, str(json_output), str(csv_output))
    
    # 기존 매핑과 통합
    image_mapping_csv = output_dir / "question_image_mapping_mapping_results_사진형_corrected.csv"
    integrated_output = output_dir / "integrated_question_image_text_mapping.csv"
    
    integrated_df = merge_with_existing_mapping(
        str(json_output),
        str(image_mapping_csv),
        str(integrated_output)
    )
    
    print(f"텍스트 추출 완료!")
    print(f"- 총 {len(results['question_text_mapping'])}개 문제에서 텍스트 추출")
    print(f"- JSON 결과: {json_output}")
    print(f"- CSV 결과: {csv_output}")
    print(f"- 통합 결과: {integrated_output}")
    
    return results, integrated_df


if __name__ == "__main__":
    demo_extract_right_side_text()
