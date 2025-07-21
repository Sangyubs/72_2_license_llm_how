"""
일러스트형 PDF에서 이미지 설명 텍스트를 추출하는 개선된 함수들
기존 로직의 문제점을 해결하여 더 정확한 텍스트 추출을 수행합니다.
"""

import pandas as pd
import fitz  # PyMuPDF
import re
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path


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
    search_nums = [question_num]
    if question_num == "858":
        search_nums.append("585")  # 858번이 585번으로 잘못 표기된 경우
    
    question_rect = None
    
    for search_num in search_nums:
        question_pattern = f"{search_num}\\."
        
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
        
        if question_rect:
            break
    
    if not question_rect:
        return None
    
    # 문항번호 아래쪽에서 가장 가까운 이미지 찾기
    closest_image = None
    min_distance = float('inf')
    
    for img_rect in image_rects:
        # 이미지가 문항번호 아래쪽에 있고
        if img_rect.y0 > question_rect.y1:
            # 수직 거리 계산
            distance = img_rect.y0 - question_rect.y1
            if distance < min_distance:
                min_distance = distance
                closest_image = img_rect
    
    return closest_image


def extract_illustration_descriptions_v2(
    pdf_path: str,
    debug: bool = True
) -> Dict[str, any]:
    """
    일러스트형 PDF에서 문제별 설명 텍스트를 추출합니다.
    
    개선 사항:
    1. 페이지 전체 텍스트를 분석하여 패턴 기반으로 추출
    2. 이미지와 텍스트 매핑 추가
    3. 문항별 영역을 더 정확하게 구분
    
    Args:
        pdf_path: PDF 파일 경로
        debug: 디버깅 정보 출력 여부
        
    Returns:
        Dict: 추출 결과 및 메타데이터
    """
    
    doc = fitz.open(pdf_path)
    question_descriptions = {}
    question_image_mapping = {}
    extraction_log = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        if debug:
            print(f"\n=== Page {page_num + 1} ===")
        
        # 1. 페이지의 모든 텍스트를 위치 정보와 함께 추출
        page_text_blocks = extract_page_text_blocks(page)
        text_dict = page.get_text("dict")
        
        # 2. 이미지 위치 추출
        image_rects = get_image_positions_on_page(page)
        
        # 3. 문항 번호들 찾기
        question_numbers = find_question_numbers_in_blocks(page_text_blocks)
        
        if debug:
            print(f"발견된 문항: {question_numbers}")
            print(f"이미지 개수: {len(image_rects)}")
        
        # 4. 각 문항별 설명 텍스트 및 이미지 매핑
        used_images = set()
        for question_num in question_numbers:
            # 텍스트 추출
            description = extract_question_description(
                page_text_blocks, question_num, debug
            )
            
            # 이미지 매핑
            available_images = [img for i, img in enumerate(image_rects) if i not in used_images]
            image_rect = find_image_rect_for_question(
                question_num, available_images, page, text_dict
            )
            
            if description:
                question_id = f"question_{question_num}"
                question_descriptions[question_id] = description
                
                # 이미지 매핑 정보 저장
                if image_rect:
                    # 사용된 이미지 마킹
                    for i, img in enumerate(image_rects):
                        if (abs(img.x0 - image_rect.x0) < 1 and 
                            abs(img.y0 - image_rect.y0) < 1):
                            used_images.add(i)
                            break
                    
                    question_image_mapping[question_id] = {
                        'page_number': page_num + 1,
                        'image_bbox': [image_rect.x0, image_rect.y0, image_rect.x1, image_rect.y1],
                        'question_number': question_num
                    }
                
                extraction_log.append(f"Page {page_num + 1}, Q{question_num}: {len(description)}자, 이미지: {'○' if image_rect else '✗'}")
                
                if debug:
                    print(f"✓ Q{question_num}: {description[:50]}... (이미지: {'○' if image_rect else '✗'})")
            else:
                if debug:
                    print(f"✗ Q{question_num}: 설명 텍스트 없음")
    
    doc.close()
    
    return {
        'question_descriptions': question_descriptions,
        'question_image_mapping': question_image_mapping,
        'extraction_log': extraction_log,
        'total_questions': len(question_descriptions)
    }


def extract_page_text_blocks(page) -> List[Dict]:
    """
    페이지의 모든 텍스트를 블록 단위로 추출합니다.
    
    Args:
        page: PyMuPDF 페이지 객체
        
    Returns:
        List[Dict]: 텍스트 블록 정보 리스트
    """
    text_dict = page.get_text("dict")
    text_blocks = []
    
    for block in text_dict["blocks"]:
        if "lines" in block:
            block_text = ""
            block_bbox = block["bbox"]
            
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span.get("text", "")
                block_text += line_text + " "
            
            if block_text.strip():
                text_blocks.append({
                    'text': block_text.strip(),
                    'bbox': block_bbox,
                    'y_position': block_bbox[1]  # Y 좌표 (위에서부터)
                })
    
    # Y 좌표 순으로 정렬 (위에서 아래로)
    text_blocks.sort(key=lambda x: x['y_position'])
    
    return text_blocks


def find_question_numbers_in_blocks(text_blocks: List[Dict]) -> List[str]:
    """
    텍스트 블록들에서 문항 번호를 찾습니다.
    
    Args:
        text_blocks: 텍스트 블록 리스트
        
    Returns:
        List[str]: 문항 번호 리스트 (정렬됨)
    """
    question_numbers = set()
    
    # 문항 번호 패턴들
    patterns = [
        r'(\d+)\.\s',           # "781. " 형태
        r'^(\d+)\s',            # 줄 시작의 "781 " 형태
        r'문제\s*(\d+)',        # "문제 781" 형태
        r'(\d+)번',             # "781번" 형태
    ]
    
    for block in text_blocks:
        text = block['text']
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                num = int(match.group(1))
                # 일러스트형 문제 번호 범위 확인 (781~865)
                if 781 <= num <= 865:
                    question_numbers.add(str(num))
                # 858번이 585번으로 잘못 표기된 경우 처리
                elif num == 585:
                    question_numbers.add("858")
    
    return sorted(list(question_numbers))


def extract_question_description(
    text_blocks: List[Dict], 
    question_num: str, 
    debug: bool = False
) -> str:
    """
    특정 문항의 설명 텍스트를 추출합니다.
    
    새로운 접근 방식:
    1. 문항 번호 위치를 찾기
    2. 다음 문항까지의 영역에서 설명 텍스트 패턴 찾기
    3. 정답/해설이 아닌 순수 설명만 필터링
    
    Args:
        text_blocks: 텍스트 블록 리스트
        question_num: 문항 번호
        debug: 디버깅 출력 여부
        
    Returns:
        str: 추출된 설명 텍스트
    """
    
    # 1. 현재 문항과 다음 문항의 위치 찾기
    current_question_y = find_question_position(text_blocks, question_num)
    next_question_y = find_question_position(text_blocks, str(int(question_num) + 1))
    
    if current_question_y is None:
        return ""
    
    # 2. 문항 영역 내의 텍스트 블록들 수집
    question_area_blocks = []
    
    for block in text_blocks:
        block_y = block['y_position']
        
        # 현재 문항 이후, 다음 문항 이전의 블록들
        if block_y > current_question_y:
            if next_question_y is None or block_y < next_question_y:
                question_area_blocks.append(block)
    
    if debug:
        print(f"  Q{question_num} 영역 블록 수: {len(question_area_blocks)}")
    
    # 3. 설명 텍스트 추출 및 필터링
    description_items = []
    
    for block in question_area_blocks:
        text = block['text'].strip()
        
        if is_valid_description(text):
            # 정리된 설명 텍스트로 변환
            cleaned_text = clean_description_text(text)
            if cleaned_text:
                description_items.append(cleaned_text)
                
                if debug:
                    print(f"    추가: {cleaned_text[:30]}...")
    
    # 4. 설명 텍스트들을 정리하여 반환
    return format_final_description(description_items)


def find_question_position(text_blocks: List[Dict], question_num: str) -> Optional[float]:
    """
    특정 문항 번호의 Y 위치를 찾습니다.
    858번이 585번으로 잘못 표기된 경우도 처리합니다.
    
    Args:
        text_blocks: 텍스트 블록 리스트
        question_num: 찾을 문항 번호
        
    Returns:
        Optional[float]: Y 위치 (없으면 None)
    """
    search_nums = [question_num]
    
    # 858번인 경우 585번도 함께 검색
    if question_num == "858":
        search_nums.append("585")
    
    for search_num in search_nums:
        patterns = [
            f"{search_num}\\.",
            f"^{search_num}\\s",
            f"문제\\s*{search_num}",
            f"{search_num}번"
        ]
        
        for block in text_blocks:
            text = block['text']
            
            for pattern in patterns:
                if re.search(pattern, text):
                    return block['y_position']
    
    return None


def is_valid_description(text: str) -> bool:
    """
    유효한 설명 텍스트인지 판단합니다.
    위치와 마커를 기준으로 필터링을 강화했습니다.
    
    Args:
        text: 확인할 텍스트
        
    Returns:
        bool: 유효한 설명이면 True
    """
    
    # 1. 기본 필터링: 너무 짧거나 빈 텍스트
    if len(text.strip()) < 3:
        return False
    
    # 2. ■ 마커로 시작하는 텍스트만 허용
    if not text.strip().startswith('■'):
        return False
    
    # 3. 정답/해설 관련 제외 (위치 기반)
    exclude_keywords = [
        '정답：', '정답:', '답：', '답:',
        '해설：', '해설:', '풀이:', '풀이：'
    ]
    
    for keyword in exclude_keywords:
        if keyword in text:
            return False
    
    # 4. 선택지 번호 제외
    if re.search(r'■\s*[①②③④⑤]', text):
        return False
    
    # 5. 너무 긴 텍스트 제외 (100자 이상은 대부분 해설)
    if len(text) > 100:
        return False
    
    # 6. 법규 조항 키워드 제외
    law_keywords = [
        '도로교통법', '제\d+조', '별표\d+', '시행규칙',
        '따라서', '그러므로', '이하 같다', '에 따라'
    ]
    
    for keyword in law_keywords:
        if re.search(keyword, text):
            return False
    
    # 7. 유효한 설명 텍스트 패턴 확인
    valid_patterns = [
        r'차로',              # "1차로", "2차로"
        r'시속.*킬로',        # "시속 30킬로미터"
        r'교차로',            # 교차로 관련
        r'신호',              # 신호 관련
        r'횡단보도',          # 횡단보도
        r'어린이',            # 어린이 관련
        r'보행자',            # 보행자
        r'자전거',            # 자전거
        r'정차.*중',          # 정차 중
        r'주행.*중',          # 주행 중
        r'진입.*중',          # 진입 중
        r'형.*교차로',        # "+ 형 교차로"
        r'등화.*중',          # "등화 중"
        r'상황',              # 상황 설명
        r'운전.*중',          # 운전 중
    ]
    
    # 하나라도 매치되면 유효한 설명 텍스트
    for pattern in valid_patterns:
        if re.search(pattern, text):
            return True
    
    return False


def clean_description_text(text: str) -> str:
    """
    설명 텍스트를 정리합니다.
    
    Args:
        text: 원본 텍스트
        
    Returns:
        str: 정리된 텍스트
    """
    
    # 1. 기본 정리
    cleaned = text.strip()
    
    # 2. 연속 공백 제거
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 3. ■ 마커가 없으면 추가하지 않음 (이미 is_valid_description에서 필터링됨)
    
    return cleaned


def format_final_description(description_items: List[str]) -> str:
    """
    최종 설명 텍스트를 포맷팅합니다.
    
    Args:
        description_items: 설명 텍스트 리스트
        
    Returns:
        str: 포맷팅된 최종 텍스트
    """
    
    if not description_items:
        return ""
    
    # 1. 중복 제거
    unique_items = []
    for item in description_items:
        if item not in unique_items:
            unique_items.append(item)
    
    # 2. 길이 필터링 (5자 이상 100자 이하)
    filtered_items = [
        item for item in unique_items 
        if 5 <= len(item) <= 100
    ]
    
    # 3. 줄바꿈으로 연결
    return '\n'.join(filtered_items)


def save_results_v2(
    results: Dict,
    output_dir: str
) -> None:
    """
    추출 결과를 저장합니다.
    
    Args:
        results: extract_illustration_descriptions_v2() 결과
        output_dir: 출력 디렉토리
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # JSON 저장 (전체 결과)
    json_file = output_path / "illustration_descriptions_v2.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # CSV 저장 (텍스트 설명)
    csv_file = output_path / "illustration_descriptions_v2.csv"
    csv_data = []
    
    for question_id, description in results['question_descriptions'].items():
        question_num = question_id.replace('question_', '')
        
        # 이미지 정보 가져오기
        image_info = results.get('question_image_mapping', {}).get(question_id, {})
        has_image = 'image_bbox' in image_info
        
        csv_data.append({
            'question_id': question_id,
            'question_number': question_num,
            'description_text': description,
            'description_length': len(description),
            'bullet_count': description.count('■'),
            'line_count': description.count('\n') + 1,
            'has_image_mapping': has_image,
            'page_number': image_info.get('page_number', ''),
            'image_bbox': str(image_info.get('image_bbox', '')) if has_image else ''
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    # 이미지 매핑 CSV 별도 저장
    if 'question_image_mapping' in results:
        mapping_csv_file = output_path / "question_image_mapping_v2.csv"
        mapping_data = []
        
        for question_id, mapping_info in results['question_image_mapping'].items():
            question_num = question_id.replace('question_', '')
            mapping_data.append({
                'question_id': question_id,
                'question_number': question_num,
                'page_number': mapping_info['page_number'],
                'image_x0': mapping_info['image_bbox'][0],
                'image_y0': mapping_info['image_bbox'][1],
                'image_x1': mapping_info['image_bbox'][2],
                'image_y1': mapping_info['image_bbox'][3],
                'image_width': mapping_info['image_bbox'][2] - mapping_info['image_bbox'][0],
                'image_height': mapping_info['image_bbox'][3] - mapping_info['image_bbox'][1]
            })
        
        mapping_df = pd.DataFrame(mapping_data)
        mapping_df.to_csv(mapping_csv_file, index=False, encoding='utf-8-sig')
        
        print(f"이미지 매핑 저장: {mapping_csv_file}")
    
    print(f"결과 저장 완료:")
    print(f"- JSON: {json_file}")
    print(f"- CSV: {csv_file}")
    print(f"- 총 {len(results['question_descriptions'])}개 문제 추출")
    print(f"- 이미지 매핑: {len(results.get('question_image_mapping', {}))}개")


def merge_with_existing_image_mapping(
    v2_results: Dict,
    existing_mapping_csv: str,
    output_dir: str
) -> Dict:
    """
    v2 결과와 기존 이미지 매핑 CSV를 결합하여 완전한 매핑을 생성합니다.
    
    Args:
        v2_results: extract_illustration_descriptions_v2() 결과
        existing_mapping_csv: 기존 이미지 매핑 CSV 파일 경로
        output_dir: 출력 디렉토리
        
    Returns:
        Dict: 결합된 결과
    """
    import pandas as pd
    
    # 기존 이미지 매핑 로드
    existing_df = pd.read_csv(existing_mapping_csv)
    print(f"기존 이미지 매핑 로드: {len(existing_df)}개 문제")
    
    # v2 텍스트 결과
    v2_descriptions = v2_results['question_descriptions']
    v2_image_mapping = v2_results.get('question_image_mapping', {})
    
    # 결합된 결과 생성
    merged_data = []
    merged_image_mapping = {}
    
    for _, row in existing_df.iterrows():
        question_id = row['question_id']
        question_num = str(row['question_number'])
        
        # v2에서 추출된 텍스트 가져오기
        description_text = v2_descriptions.get(question_id, "")
        
        # v2 이미지 매핑 정보 (좌표)
        v2_image_info = v2_image_mapping.get(question_id, {})
        
        # 기존 이미지 파일 경로 정보와 v2 좌표 정보 결합
        merged_image_mapping[question_id] = {
            'question_number': question_num,
            'page_number': row.get('page_number', ''),
            'image_path': row['image_path'],
            'image_filename': row['image_filename'],
            'has_extracted_image': True,  # 기존 매핑에는 모든 이미지가 있음
            'has_description_text': bool(description_text),
            'v2_image_bbox': v2_image_info.get('image_bbox', []),
            'v2_mapping_success': question_id in v2_image_mapping
        }
        
        merged_data.append({
            'question_id': question_id,
            'question_number': question_num,
            'description_text': description_text,
            'description_length': len(description_text) if description_text else 0,
            'bullet_count': description_text.count('■') if description_text else 0,
            'line_count': description_text.count('\n') + 1 if description_text else 0,
            'has_description_text': bool(description_text),
            'page_number': row.get('page_number', ''),
            'image_path': row['image_path'],
            'image_filename': row['image_filename'],
            'has_extracted_image': True,
            'v2_mapping_success': question_id in v2_image_mapping,
            'v2_image_bbox': str(v2_image_info.get('image_bbox', '')) if v2_image_info else ''
        })
    
    # 결과 저장
    output_path = Path(output_dir)
    
    # 통합 CSV 저장
    merged_csv = output_path / "merged_illustration_mapping.csv"
    merged_df = pd.DataFrame(merged_data)
    merged_df.to_csv(merged_csv, index=False, encoding='utf-8-sig')
    
    # 통합 JSON 저장
    merged_json = output_path / "merged_illustration_mapping.json"
    merged_results = {
        'question_descriptions': v2_descriptions,
        'question_image_mapping': merged_image_mapping,
        'extraction_log': v2_results.get('extraction_log', []),
        'total_questions': len(v2_descriptions),
        'total_images': len(existing_df),
        'merge_summary': {
            'questions_with_text': len([d for d in merged_data if d['has_description_text']]),
            'questions_with_images': len([d for d in merged_data if d['has_extracted_image']]),
            'v2_successful_mappings': len([d for d in merged_data if d['v2_mapping_success']]),
            'complete_mappings': len([d for d in merged_data if d['has_description_text'] and d['has_extracted_image']])
        }
    }
    
    with open(merged_json, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)
    
    # 요약 출력
    summary = merged_results['merge_summary']
    print(f"\n=== 매핑 결합 완료 ===")
    print(f"총 문제 수: 85개 (781~865)")
    print(f"설명 텍스트가 있는 문제: {summary['questions_with_text']}개")
    print(f"이미지 파일이 있는 문제: {summary['questions_with_images']}개")
    print(f"v2에서 좌표 매핑 성공: {summary['v2_successful_mappings']}개")
    print(f"완전 매핑 (텍스트+이미지): {summary['complete_mappings']}개")
    print(f"결과 저장: {merged_csv}")
    print(f"JSON 저장: {merged_json}")
    
    return merged_results


def demo_extract_v2():
    """
    개선된 텍스트 추출 데모 함수
    """
    pdf_path = "g:/내 드라이브/4_paper/72_2_license_llm_how/data/4_raw_data_fromkoroad_일러스트.pdf"
    output_dir = "g:/내 드라이브/4_paper/72_2_license_llm_how/data/results/일러스트"
    
    print("=== 개선된 일러스트 텍스트 추출 시작 ===")
    
    # 텍스트 추출
    results = extract_illustration_descriptions_v2(pdf_path, debug=True)
    
    # 결과 저장
    save_results_v2(results, output_dir)
    
    # 요약 출력
    print(f"\n=== 추출 완료 ===")
    print(f"총 추출된 문제 수: {results['total_questions']}")
    print(f"추출 로그: {len(results['extraction_log'])}개")
    
    # 샘플 출력 (781번 포함 확인)
    descriptions = results['question_descriptions']
    if 'question_781' in descriptions:
        print(f"\n781번 추출 성공:")
        print(f"{descriptions['question_781']}")
    else:
        print(f"\n781번 추출 실패 - 다른 샘플:")
        if descriptions:
            sample_key = list(descriptions.keys())[0]
            print(f"{sample_key}: {descriptions[sample_key]}")
    
    return results


def demo_extract_and_merge():
    """
    v2 추출과 기존 매핑 결합 데모
    """
    pdf_path = "g:/내 드라이브/4_paper/72_2_license_llm_how/data/4_raw_data_fromkoroad_일러스트.pdf"
    output_dir = "g:/내 드라이브/4_paper/72_2_license_llm_how/data/results/일러스트"
    existing_mapping_csv = "g:/내 드라이브/4_paper/72_2_license_llm_how/data/results/일러스트/question_image_mapping_mapping_results_일러스트_corrected_corrected.csv"
    
    print("=== v2 텍스트 추출 + 기존 이미지 매핑 결합 ===")
    
    # 1. v2 텍스트 추출
    print("\n1단계: 텍스트 추출 중...")
    v2_results = extract_illustration_descriptions_v2(pdf_path, debug=False)
    
    # 2. 기존 매핑과 결합
    print("\n2단계: 기존 이미지 매핑과 결합 중...")
    merged_results = merge_with_existing_image_mapping(
        v2_results, existing_mapping_csv, output_dir
    )
    
    # 3. v2 단독 결과도 저장
    print("\n3단계: v2 단독 결과 저장 중...")
    save_results_v2(v2_results, output_dir)
    
    return merged_results


if __name__ == "__main__":
    # 기본 v2 추출
    # demo_extract_v2()
    
    # 기존 매핑과 결합한 추출
    demo_extract_and_merge()
