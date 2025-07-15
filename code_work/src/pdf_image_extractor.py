"""
PDF에서 문항번호와 사진을 매핑하는 함수들
"""

import pandas as pd
import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re
import json


def extract_questions_and_images_from_pdf(
    pdf_path: str, 
    output_dir: str = "extracted_images"
) -> Dict[str, any]:
    """
    PDF에서 문항번호와 해당 이미지를 추출하여 매핑합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        output_dir: 추출된 이미지를 저장할 디렉토리
        
    Returns:
        Dict: {
            'question_image_mapping': {question_id: image_path},
            'page_info': [page별 문항 정보],
            'extraction_log': [추출 과정 로그]
        }
    """
    
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    doc = fitz.open(pdf_path)
    question_image_mapping = {}
    page_info = []
    extraction_log = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 1. 페이지에서 텍스트 추출 (문항번호 찾기용)
        text = page.get_text()
        question_numbers = extract_question_numbers_from_text(text)
        
        # 2. 페이지에서 이미지 추출
        image_list = page.get_images()
        
        # 3. 이미지들의 위치 정보 수집
        images_with_positions = []
        for img_index, img in enumerate(image_list):
            # 이미지 객체 정보
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            
            if pix.n - pix.alpha < 4:  # GRAY나 RGB 이미지만 처리
                # 이미지 위치 정보 추출
                img_rect = page.get_image_rects(img)[0] if page.get_image_rects(img) else None
                
                if img_rect:
                    # 이미지 저장
                    img_filename = f"page_{page_num+1}_img_{img_index+1}.png"
                    img_path = Path(output_dir) / img_filename
                    pix.save(str(img_path))
                    
                    images_with_positions.append({
                        'image_path': str(img_path),
                        'position': img_rect,
                        'y_position': img_rect.y0  # Y 좌표로 정렬용
                    })
            
            pix = None  # 메모리 해제
        
        # 4. 문항번호와 이미지 매핑 (위치 기반)
        page_mapping = map_questions_to_images(
            question_numbers, 
            images_with_positions, 
            page_num + 1,
            text
        )
        
        question_image_mapping.update(page_mapping)
        
        page_info.append({
            'page_number': page_num + 1,
            'question_count': len(question_numbers),
            'image_count': len(images_with_positions),
            'questions': [q[0] if isinstance(q, tuple) else str(q) for q in question_numbers],  # 튜플에서 문항번호만 추출
            'mapping': page_mapping
        })
        
        extraction_log.append(f"Page {page_num+1}: {len(question_numbers)} questions, {len(images_with_positions)} images")
    
    doc.close()
    
    return {
        'question_image_mapping': question_image_mapping,
        'page_info': page_info,
        'extraction_log': extraction_log
    }


def extract_question_numbers_from_text(text: str) -> List[Tuple[str, float]]:
    """
    텍스트에서 문항번호와 해당 위치를 추출합니다.
    
    Args:
        text: 페이지 텍스트
        
    Returns:
        List[Tuple[str, float]]: [(문항번호, y위치), ...]
    """
    
    # 다양한 문항번호 패턴 정의
    patterns = [
        r'(\d+)\.\s',  # "1. ", "2. " 형태
        r'문제\s*(\d+)',  # "문제 1", "문제1" 형태
        r'(\d+)번',  # "1번", "2번" 형태
        r'Q\.?\s*(\d+)',  # "Q.1", "Q 1" 형태
    ]
    
    question_numbers = []
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            question_num = match.group(1)
            # 텍스트에서의 대략적인 위치 추정 (실제 구현시 더 정교한 방법 필요)
            position = match.start() / len(text)  # 0~1 사이 값
            question_numbers.append((question_num, position))
    
    # 중복 제거 및 정렬
    question_numbers = list(set(question_numbers))
    question_numbers.sort(key=lambda x: x[1])  # 위치순 정렬
    
    return question_numbers


def map_questions_to_images(
    question_numbers: List[Tuple[str, float]], 
    images_with_positions: List[Dict], 
    page_num: int,
    page_text: str
) -> Dict[str, str]:
    """
    문항번호와 이미지를 위치 기반으로 매핑합니다.
    
    Args:
        question_numbers: 문항번호와 위치 리스트
        images_with_positions: 이미지와 위치 정보 리스트
        page_num: 페이지 번호
        page_text: 페이지 전체 텍스트
        
    Returns:
        Dict[str, str]: {문항번호: 이미지경로}
    """
    
    mapping = {}
    
    if not question_numbers or not images_with_positions:
        return mapping
    
    # 이미지들을 Y 위치로 정렬
    sorted_images = sorted(images_with_positions, key=lambda x: x['y_position'])
    
    # 각 문항에 대해 가장 가까운 이미지 찾기
    for i, (question_num, q_position) in enumerate(question_numbers):
        
        # 방법 1: 순서 기반 매핑 (가장 단순)
        if i < len(sorted_images):
            mapping[f"question_{question_num}"] = sorted_images[i]['image_path']
        
        # 방법 2: 텍스트 분석 기반 매핑 (더 정교함)
        # 문항 주변 텍스트에서 이미지 관련 키워드 찾기
        question_context = extract_question_context(page_text, question_num)
        if has_image_reference(question_context):
            # 이미지 참조가 있는 경우에만 매핑
            if i < len(sorted_images):
                mapping[f"question_{question_num}"] = sorted_images[i]['image_path']
    
    return mapping


def extract_question_context(text: str, question_num: str, context_size: int = 200) -> str:
    """
    특정 문항 주변의 텍스트 컨텍스트를 추출합니다.
    
    Args:
        text: 전체 텍스트
        question_num: 문항번호
        context_size: 추출할 컨텍스트 크기
        
    Returns:
        str: 문항 주변 텍스트
    """
    
    # 문항번호 패턴으로 위치 찾기
    pattern = rf"{question_num}\.\s"
    match = re.search(pattern, text)
    
    if match:
        start_pos = max(0, match.start() - context_size//2)
        end_pos = min(len(text), match.end() + context_size//2)
        return text[start_pos:end_pos]
    
    return ""


def has_image_reference(text: str) -> bool:
    """
    텍스트에 이미지 참조 키워드가 있는지 확인합니다.
    
    Args:
        text: 확인할 텍스트
        
    Returns:
        bool: 이미지 참조 여부
    """
    
    image_keywords = [
        '그림', '사진', '이미지', '도표', '그래프', 
        '표지', '일러스트', '도식', '보기', '다음',
        '아래', '위', '상단', '하단', '좌측', '우측'
    ]
    
    return any(keyword in text for keyword in image_keywords)


def save_mapping_to_json(mapping_result: Dict, output_path: str) -> None:
    """
    매핑 결과를 JSON 파일로 저장합니다.
    
    Args:
        mapping_result: extract_questions_and_images_from_pdf 결과
        output_path: 저장할 JSON 파일 경로
    """
    
    # 저장용 데이터 구조 변환
    save_data = {
        'question_image_mapping': mapping_result['question_image_mapping'],
        'page_info': mapping_result['page_info'],
        'extraction_log': mapping_result['extraction_log'],
        'extraction_date': pd.Timestamp.now().isoformat()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)


def load_mapping_from_json(json_path: str) -> Dict:
    """
    저장된 매핑 결과를 JSON에서 로드합니다.
    
    Args:
        json_path: JSON 파일 경로
        
    Returns:
        Dict: 매핑 결과
    """
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 사용 예시 함수
def process_license_exam_pdfs(data_dir: str = "data") -> Dict[str, Dict]:
    """
    운전면허 시험 PDF들을 일괄 처리합니다.
    
    Args:
        data_dir: 데이터 디렉토리 경로
        
    Returns:
        Dict: 각 PDF별 매핑 결과
    """
    
    pdf_files = [
        "2_raw_data_fromkoroad_안전표지.pdf",
        "3_raw_data_fromkoroad_사진형.pdf", 
        "4_raw_data_fromkoroad_일러스트.pdf"
    ]
    
    all_results = {}
    
    for pdf_file in pdf_files:
        pdf_path = Path(data_dir) / pdf_file
        if pdf_path.exists():
            print(f"Processing {pdf_file}...")
            
            # 파일명에서 타입 추출
            file_type = pdf_file.split('_')[-1].replace('.pdf', '')
            output_dir = f"extracted_images/{file_type}"
            
            # 매핑 추출
            result = extract_questions_and_images_from_pdf(
                str(pdf_path), 
                output_dir
            )
            
            all_results[file_type] = result
            
            # 결과 저장
            json_path = f"mapping_results_{file_type}.json"
            save_mapping_to_json(result, json_path)
            
            print(f"✓ {file_type}: {len(result['question_image_mapping'])} mappings created")
            for log in result['extraction_log']:
                print(f"  {log}")
        else:
            print(f"❌ File not found: {pdf_file}")
    
    return all_results
