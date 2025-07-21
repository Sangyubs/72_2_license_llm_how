"""
일러스트형 PDF에서 이미지 아래의 설명 텍스트를 추출하는 함수들
운전면허 시험 문제의 일러스트형 문제에서 이미지 아래에 있는 상황 설명 텍스트를 추출합니다.
정답이나 해설은 제외하고 순수한 설명 텍스트만 추출합니다.
"""

import pandas as pd
import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re
import json


def extract_below_image_text_from_pdf(
    pdf_path: str,
    existing_mapping_path: Optional[str] = None
) -> Dict[str, Dict]:
    """
    일러스트형 PDF에서 이미지 아래의 설명 텍스트를 추출하여 문제 번호와 연결합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        existing_mapping_path: 기존 문제-이미지 매핑 JSON 파일 경로 (선택적)
        
    Returns:
        Dict: {
            'question_text_mapping': {question_id: below_image_text},
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
        
        # 4. 각 문항에 대해 이미지 아래 텍스트 추출 (개선된 로직)
        page_text_mapping = {}
        used_images = set()  # 이미 사용된 이미지 추적
        
        print(f"\nPage {page_num+1}: 문항 {len(question_numbers)}개, 이미지 {len(image_rects)}개")
        
        for question_num in question_numbers:
            question_id = f"question_{question_num}"
            
            print(f"\n처리 중: 문항 {question_num}")
            
            # 해당 문항의 이미지 위치 찾기 (사용되지 않은 이미지 중에서)
            available_images = [img for i, img in enumerate(image_rects) if i not in used_images]
            
            if not available_images:
                print(f"Warning: 문항 {question_num}에 사용 가능한 이미지가 없습니다.")
                continue
                
            image_rect = find_image_rect_for_question(
                question_num, available_images, page, text_dict
            )
            
            if image_rect:
                # 사용된 이미지 표시 (인덱스 찾기)
                for i, img in enumerate(image_rects):
                    if (abs(img.x0 - image_rect.x0) < 1 and 
                        abs(img.y0 - image_rect.y0) < 1):
                        used_images.add(i)
                        break
                
                # 이미지 아래 영역의 설명 텍스트 추출
                below_text = extract_text_below_image(
                    page, text_dict, image_rect, question_num
                )
                
                if below_text.strip():
                    page_text_mapping[question_id] = below_text
                    extraction_log.append(
                        f"Page {page_num+1}, Question {question_num}: "
                        f"추출된 텍스트 길이 {len(below_text)}"
                    )
                    print(f"성공: 문항 {question_num} 텍스트 추출 완료 ({len(below_text)}자)")
                else:
                    print(f"Warning: 문항 {question_num}에서 설명 텍스트를 찾을 수 없습니다.")
            else:
                print(f"Error: 문항 {question_num}에 해당하는 이미지를 찾을 수 없습니다.")
        
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
    
    # 일러스트형 문항번호 패턴들 (781~866번 범위)
    patterns = [
        r'(\d+)\.\s',  # "781. " 형태
        r'문제\s*(\d+)',  # "문제 781" 형태
        r'(\d+)번',  # "781번" 형태
    ]
    
    question_numbers = set()
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            num = int(match.group(1))
            # 일러스트형 문제 번호 범위 확인 (781~866)
            if 781 <= num <= 866:
                question_numbers.add(str(num))
    
    return sorted(list(question_numbers))


def find_image_rect_for_question(
    question_num: str, 
    image_rects: List[fitz.Rect], 
    page, 
    text_dict: dict
) -> Optional[fitz.Rect]:
    """
    특정 문항번호에 해당하는 이미지의 위치를 찾습니다.
    개선: 각 문항에 대해 고유한 이미지를 할당하도록 매칭 로직 강화
    
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
        # 문항번호를 찾지 못한 경우 처리
        print(f"Warning: 문항번호 {question_num}을 찾을 수 없습니다.")
        return None
    
    if not image_rects:
        print(f"Warning: 페이지에 이미지가 없습니다.")
        return None
    
    # 개선된 매칭 로직: 문항번호와 이미지의 상대적 위치 고려
    candidate_images = []
    
    for i, img_rect in enumerate(image_rects):
        # 1. 문항번호 아래쪽에 있는 이미지들 찾기
        if img_rect.y0 > question_rect.y0:
            vertical_distance = img_rect.y0 - question_rect.y1  # 문항번호 끝에서 이미지 시작까지
            horizontal_overlap = calculate_horizontal_overlap(question_rect, img_rect)
            
            candidate_images.append({
                'index': i,
                'rect': img_rect,
                'vertical_distance': vertical_distance,
                'horizontal_overlap': horizontal_overlap,
                'total_distance': abs(img_rect.y0 - question_rect.y0)
            })
    
    if not candidate_images:
        # 아래쪽에 이미지가 없으면 가장 가까운 이미지 반환
        print(f"Warning: 문항 {question_num} 아래에 이미지가 없어 가장 가까운 이미지를 사용합니다.")
        closest_image = None
        min_distance = float('inf')
        
        for img_rect in image_rects:
            distance = abs(img_rect.y0 - question_rect.y0)
            if distance < min_distance:
                min_distance = distance
                closest_image = img_rect
        
        return closest_image
    
    # 후보 이미지들을 정렬: 수직거리 우선, 수평겹침 고려
    candidate_images.sort(key=lambda x: (x['vertical_distance'], -x['horizontal_overlap']))
    
    selected_image = candidate_images[0]['rect']
    print(f"문항 {question_num}: 이미지 매칭 성공 (수직거리: {candidate_images[0]['vertical_distance']:.2f})")
    
    return selected_image


def calculate_horizontal_overlap(rect1: fitz.Rect, rect2: fitz.Rect) -> float:
    """
    두 사각형의 수평 겹침 정도를 계산합니다.
    
    Args:
        rect1: 첫 번째 사각형
        rect2: 두 번째 사각형
        
    Returns:
        float: 겹침 비율 (0~1)
    """
    overlap_start = max(rect1.x0, rect2.x0)
    overlap_end = min(rect1.x1, rect2.x1)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_width = overlap_end - overlap_start
    min_width = min(rect1.width, rect2.width)
    
    return overlap_width / min_width if min_width > 0 else 0.0


def extract_text_below_image(
    page, 
    text_dict: dict, 
    image_rect: fitz.Rect,
    question_num: str,
    margin: float = 5.0
) -> str:
    """
    이미지 아래 영역의 설명 텍스트를 추출합니다. (정답/해설 제외)
    개선: 텍스트 추출 범위를 확장하고 더 정교한 필터링 적용
    
    Args:
        page: PyMuPDF 페이지 객체
        text_dict: 페이지 텍스트 딕셔너리
        image_rect: 이미지 위치
        question_num: 문항번호
        margin: 이미지와 텍스트 사이의 여백
        
    Returns:
        str: 추출된 설명 텍스트
    """
    
    # 이미지 아래 영역 정의 (개선된 범위)
    page_rect = page.rect
    
    # 다음 문항번호 찾기 (텍스트 추출 범위 제한용)
    next_question_y = find_next_question_position(text_dict, question_num)
    
    # 추출 범위를 더 넓게 설정
    if next_question_y:
        # 다음 문항이 있으면 그 위치까지
        bottom_limit = next_question_y - 10  # 다음 문항 전 여백 확보
    else:
        # 다음 문항이 없으면 페이지 끝까지 (또는 적절한 거리까지)
        max_distance = 200  # 이미지 아래 최대 200포인트까지
        bottom_limit = min(image_rect.y1 + max_distance, page_rect.y1)
    
    below_area = fitz.Rect(
        page_rect.x0,                # 페이지 왼쪽 끝
        image_rect.y1 + margin,      # 이미지 하단 + 여백
        page_rect.x1,                # 페이지 오른쪽 끝
        bottom_limit                 # 개선된 하단 제한
    )
    
    extracted_texts = []
    all_texts_in_area = []  # 디버깅용
    
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    span_rect = fitz.Rect(span["bbox"])
                    
                    # 스팬이 아래 영역과 겹치는지 확인
                    if span_rect.intersects(below_area):
                        text = span.get("text", "").strip()
                        all_texts_in_area.append(text)  # 디버깅용
                        
                        # 텍스트가 있고, 정답/해설이 아닌 경우
                        if text and not is_answer_or_solution_text(text):
                            # 설명 텍스트 패턴 확장
                            if is_description_text(text):
                                if text not in extracted_texts:
                                    extracted_texts.append(text)
    
    # 디버깅 정보 출력
    print(f"문항 {question_num}: 영역 내 전체 텍스트 {len(all_texts_in_area)}개, 추출 {len(extracted_texts)}개")
    
    # 설명 텍스트들을 정리하여 반환
    return format_description_text(extracted_texts)


def is_description_text(text: str) -> bool:
    """
    설명 텍스트인지 판단합니다. (기존 ■ 기호 외 패턴 확장)
    
    Args:
        text: 확인할 텍스트
        
    Returns:
        bool: 설명 텍스트이면 True
    """
    # 기본 마커들
    markers = ['■', '□', '▲', '●', '○', '▶', '◆']
    
    # 마커로 시작하는 경우
    if any(text.startswith(marker) for marker in markers):
        return True
    
    # 마커가 포함된 경우
    if any(marker in text for marker in markers):
        return True
    
    # 추가 패턴들
    description_patterns = [
        r'^\d+차로',          # "1차로", "2차로" 등
        r'시속\s*\d+',        # "시속 30킬로미터" 등
        r'교차로',            # 교차로 관련
        r'신호',              # 신호 관련
        r'횡단보도',          # 횡단보도 관련
        r'어린이',            # 어린이 관련
        r'보행자',            # 보행자 관련
        r'자전거',            # 자전거 관련
        r'차량.*주행',        # "차량 주행" 등
        r'정차.*중',          # "정차 중" 등
        r'진입.*상',          # "진입 상황" 등
    ]
    
    for pattern in description_patterns:
        if re.search(pattern, text):
            return True
    
    return False


def find_next_question_position(text_dict: dict, current_question_num: str) -> Optional[float]:
    """
    다음 문항번호의 Y 위치를 찾습니다. (텍스트 추출 범위 제한용)
    
    Args:
        text_dict: 페이지 텍스트 딕셔너리
        current_question_num: 현재 문항번호
        
    Returns:
        Optional[float]: 다음 문항의 Y 위치 (없으면 None)
    """
    
    next_num = str(int(current_question_num) + 1)
    next_pattern = f"{next_num}\\."
    
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "")
                    if re.search(next_pattern, text):
                        return span["bbox"][1]  # Y 좌표 반환
    
    return None


def is_answer_or_solution_text(text: str) -> bool:
    """
    정답이나 해설 관련 텍스트인지 판단합니다.
    개선: 필터링을 더 정교하게 조정
    
    Args:
        text: 확인할 텍스트
        
    Returns:
        bool: 정답/해설 텍스트이면 True
    """
    
    # 명확한 정답/해설 키워드들만 필터링
    strict_answer_keywords = [
        '정답', '답:', '해설', '풀이', 
        '①', '②', '③', '④', '⑤',
        '1)', '2)', '3)', '4)', '5)',
        '(1)', '(2)', '(3)', '(4)', '(5)',
        '따라서', '그러므로', '결론',
        '보기', '선택', '문제'
    ]
    
    text_lower = text.lower()
    
    # 엄격한 키워드 체크
    for keyword in strict_answer_keywords:
        if keyword.lower() in text_lower:
            return True
    
    # 숫자만 있는 경우 (정답 번호) - 더 엄격하게
    if text.strip().isdigit() and len(text.strip()) <= 1:
        return True
    
    # 너무 짧은 텍스트 (2글자 이하) 제외
    if len(text.strip()) <= 2:
        return True
        
    return False


def format_description_text(text_list: List[str]) -> str:
    """
    추출된 설명 텍스트들을 정리하여 포맷팅합니다.
    개선: 텍스트 연결 및 정리 로직 강화
    
    Args:
        text_list: 추출된 텍스트 리스트
        
    Returns:
        str: 정리된 설명 텍스트
    """
    
    if not text_list:
        return ""
    
    # 모든 의미있는 텍스트들을 수집
    all_items = []
    
    for text in text_list:
        cleaned_text = text.strip()
        
        if len(cleaned_text) < 3:  # 너무 짧은 텍스트 제외
            continue
            
        # 마커가 있는 텍스트들
        if any(marker in cleaned_text for marker in ['■', '□', '▲', '●', '○', '▶', '◆']):
            all_items.append(cleaned_text)
        # 마커가 없어도 의미있는 설명 텍스트들
        elif is_meaningful_description(cleaned_text):
            # 마커 추가
            if not cleaned_text.startswith('■'):
                cleaned_text = '■ ' + cleaned_text
            all_items.append(cleaned_text)
    
    # 텍스트 연결 및 복원 시도
    merged_items = merge_fragmented_text(all_items)
    
    # 중복 제거
    unique_items = []
    for item in merged_items:
        if item not in unique_items:
            unique_items.append(item)
    
    return '\n'.join(unique_items)


def is_meaningful_description(text: str) -> bool:
    """
    의미있는 설명 텍스트인지 판단합니다.
    
    Args:
        text: 확인할 텍스트
        
    Returns:
        bool: 의미있는 설명이면 True
    """
    meaningful_patterns = [
        r'\d+차로',           # "1차로", "2차로"
        r'시속.*킬로',        # "시속 30킬로미터"
        r'교차로',            # 교차로
        r'신호.*등',          # 신호등
        r'횡단보도',          # 횡단보도
        r'어린이.*버스',      # 어린이통학버스
        r'보행자',            # 보행자
        r'자전거.*운전',      # 자전거 운전자
        r'주행.*중',          # 주행 중
        r'정차.*중',          # 정차 중
        r'대기.*중',          # 대기 중
        r'진입.*상',          # 진입 상황
        r'차량.*통행',        # 차량 통행
        r'도로.*상황',        # 도로 상황
    ]
    
    for pattern in meaningful_patterns:
        if re.search(pattern, text):
            return True
    
    # 일정 길이 이상이고 한글 포함
    if len(text) >= 5 and re.search(r'[가-힣]', text):
        return True
    
    return False


def merge_fragmented_text(text_list: List[str]) -> List[str]:
    """
    잘린 텍스트들을 연결 시도합니다.
    
    Args:
        text_list: 텍스트 리스트
        
    Returns:
        List[str]: 연결된 텍스트 리스트
    """
    if len(text_list) <= 1:
        return text_list
    
    merged = []
    i = 0
    
    while i < len(text_list):
        current = text_list[i]
        
        # 다음 항목과 연결 가능한지 확인
        if i + 1 < len(text_list):
            next_item = text_list[i + 1]
            
            # 연결 조건들
            if should_merge_texts(current, next_item):
                # 마커 제거 후 연결
                current_clean = current.replace('■', '').strip()
                next_clean = next_item.replace('■', '').strip()
                merged_text = f"■ {current_clean} {next_clean}"
                merged.append(merged_text)
                i += 2  # 두 항목 모두 처리됨
                continue
        
        merged.append(current)
        i += 1
    
    return merged


def should_merge_texts(text1: str, text2: str) -> bool:
    """
    두 텍스트를 연결할지 판단합니다.
    
    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트
        
    Returns:
        bool: 연결해야 하면 True
    """
    clean1 = text1.replace('■', '').strip()
    clean2 = text2.replace('■', '').strip()
    
    # 첫 번째가 불완전하게 끝나는 경우
    incomplete_endings = ['·', '(', '차로(', '신호(', '등(']
    
    for ending in incomplete_endings:
        if clean1.endswith(ending):
            return True
    
    # 두 번째가 연결어로 시작하는 경우
    continuation_starts = ['전)', '우회전)', '좌회전)', '직진)', '유턴)']
    
    for start in continuation_starts:
        if clean2.startswith(start):
            return True
    
    return False


def save_illustration_text_mapping_results(
    text_mapping_results: Dict,
    output_path: str,
    csv_output_path: Optional[str] = None
) -> None:
    """
    일러스트 텍스트 매핑 결과를 JSON과 CSV 파일로 저장합니다.
    
    Args:
        text_mapping_results: extract_below_image_text_from_pdf() 결과
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
        for question_id, description_text in question_text_mapping.items():
            question_num = question_id.replace('question_', '')
            df_data.append({
                'question_id': question_id,
                'question_number': question_num,
                'description_text': description_text,
                'text_length': len(description_text),
                'description_count': description_text.count('■')
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')


def merge_with_existing_illustration_mapping(
    text_mapping_path: str,
    image_mapping_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    일러스트 텍스트 매핑과 기존 이미지 매핑 결과를 통합합니다.
    
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
    image_df['description_text'] = image_df['question_id'].map(text_mapping)
    image_df['has_description'] = image_df['description_text'].notna()
    image_df['description_length'] = image_df['description_text'].str.len().fillna(0)
    image_df['description_count'] = image_df['description_text'].str.count('■').fillna(0)
    
    # 결과 저장
    image_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    return image_df


# 사용 예시 함수
def demo_extract_illustration_text():
    """
    일러스트형 이미지 아래 텍스트 추출 데모 함수
    """
    pdf_path = "g:/내 드라이브/4_paper/72_2_license_llm_how/data/4_raw_data_fromkoroad_일러스트.pdf"
    existing_mapping_path = "g:/내 드라이브/4_paper/72_2_license_llm_how/data/results/일러스트/mapping_results_일러스트_corrected_corrected.json"
    
    print("일러스트형 텍스트 추출을 시작합니다...")
    
    # 텍스트 추출
    results = extract_below_image_text_from_pdf(pdf_path, existing_mapping_path)
    
    # 결과 저장
    output_dir = Path("g:/내 드라이브/4_paper/72_2_license_llm_how/data/results/일러스트")
    json_output = output_dir / "illustration_text_mapping_results.json"
    csv_output = output_dir / "question_illustration_text_mapping.csv"
    
    save_illustration_text_mapping_results(results, str(json_output), str(csv_output))
    
    # 기존 매핑과 통합
    image_mapping_csv = output_dir / "question_image_mapping_mapping_results_일러스트_corrected_corrected.csv"
    integrated_output = output_dir / "integrated_question_image_illustration_mapping_v2.csv"
    
    integrated_df = merge_with_existing_illustration_mapping(
        str(json_output),
        str(image_mapping_csv),
        str(integrated_output)
    )
    
    print(f"일러스트 텍스트 추출 완료!")
    print(f"- 총 {len(results['question_text_mapping'])}개 문제에서 텍스트 추출")
    print(f"- JSON 결과: {json_output}")
    print(f"- CSV 결과: {csv_output}")
    print(f"- 통합 결과: {integrated_output}")
    
    # 샘플 결과 출력
    if results['question_text_mapping']:
        sample_question = list(results['question_text_mapping'].keys())[0]
        sample_text = results['question_text_mapping'][sample_question]
        print(f"\n샘플 추출 결과 ({sample_question}):")
        print(f"{sample_text}")
    
    return results, integrated_df


if __name__ == "__main__":
    demo_extract_illustration_text()
