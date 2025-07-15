"""
문항-이미지 매핑 실행 스크립트
운전면허 시험 PDF에서 문항번호와 이미지를 자동으로 연결합니다.
"""

import sys
from pathlib import Path
from typing import Dict

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from pdf_image_extractor import (
    extract_questions_and_images_from_pdf,
    process_license_exam_pdfs,
    save_mapping_to_json,
    load_mapping_from_json
)
from mapping_validator import (
    validate_mapping_results,
    generate_mapping_report,
    main_mapping_workflow,
    export_mapping_to_dataframe
)
import pandas as pd


def setup_environment():
    """
    필요한 라이브러리를 설치합니다.
    """
    required_packages = [
        'PyMuPDF',  # fitz
        'opencv-python',  # cv2
        'pillow',  # PIL
        'pandas',
        'matplotlib',
        'numpy'
    ]
    
    print("필요한 패키지 설치 중...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').split('_')[0])
            print(f"✅ {package} - 이미 설치됨")
        except ImportError:
            print(f"❌ {package} - 설치 필요")
            print(f"   pip install {package}")


def run_single_pdf_processing(pdf_path: str) -> pd.DataFrame:
    """
    단일 PDF 파일을 처리합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        
    Returns:
        pd.DataFrame: 매핑 결과 데이터프레임
    """
    
    print(f"\n{'='*60}")
    print(f"📁 처리 대상: {Path(pdf_path).name}")
    print(f"{'='*60}")
    
    if not Path(pdf_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
        return pd.DataFrame()
    
    try:
        # 전체 워크플로우 실행
        result_df = main_mapping_workflow(pdf_path)
        
        print(f"\n📋 처리 결과:")
        print(f"   총 매핑 수: {len(result_df)}")
        print(f"   이미지가 있는 문항: {result_df['has_image'].sum()}")
        
        return result_df
        
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def run_batch_processing(data_dir: str = "../../data") -> Dict[str, pd.DataFrame]:
    """
    여러 PDF 파일을 일괄 처리합니다.
    
    Args:
        data_dir: 데이터 디렉토리 경로
        
    Returns:
        Dict[str, pd.DataFrame]: 파일별 처리 결과
    """
    
    pdf_files = [
        "2_raw_data_fromkoroad_안전표지.pdf",
        "3_raw_data_fromkoroad_사진형.pdf", 
        "4_raw_data_fromkoroad_일러스트.pdf"
    ]
    
    all_results = {}
    
    print(f"\n🚀 일괄 처리 시작 (총 {len(pdf_files)}개 파일)")
    print(f"데이터 디렉토리: {Path(data_dir).absolute()}")
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = Path(data_dir) / pdf_file
        
        print(f"\n[{i}/{len(pdf_files)}] {pdf_file}")
        
        if pdf_path.exists():
            result_df = run_single_pdf_processing(str(pdf_path))
            if not result_df.empty:
                file_key = pdf_file.replace('.pdf', '').split('_')[-1]
                all_results[file_key] = result_df
        else:
            print(f"⚠️  파일 건너뜀 (존재하지 않음): {pdf_file}")
    
    # 통합 결과 저장
    if all_results:
        print(f"\n📊 전체 결과 요약:")
        total_mappings = 0
        for file_type, df in all_results.items():
            mappings = len(df)
            total_mappings += mappings
            print(f"   {file_type}: {mappings}개 매핑")
        
        print(f"   전체: {total_mappings}개 매핑")
        
        # 통합 데이터프레임 생성
        combined_df = pd.DataFrame()
        for file_type, df in all_results.items():
            df_copy = df.copy()
            df_copy['source_file'] = file_type
            combined_df = pd.concat([combined_df, df_copy], ignore_index=True)
        
        # 통합 결과를 data/results/ 디렉토리에 저장
        base_data_dir = Path(data_dir).absolute()
        results_dir = base_data_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        combined_path = results_dir / "all_question_image_mappings.csv"
        combined_df.to_csv(combined_path, index=False, encoding='utf-8-sig')
        print(f"   통합 파일 저장: {combined_path}")
    
    return all_results


def validate_and_fix_mappings(mapping_json_path: str) -> Dict:
    """
    저장된 매핑 결과를 검증하고 수정합니다.
    
    Args:
        mapping_json_path: 매핑 JSON 파일 경로
        
    Returns:
        Dict: 수정된 매핑 결과
    """
    
    print(f"\n🔍 매핑 검증: {mapping_json_path}")
    
    if not Path(mapping_json_path).exists():
        print(f"❌ 매핑 파일을 찾을 수 없습니다: {mapping_json_path}")
        return {}
    
    # 매핑 로드
    mapping_result = load_mapping_from_json(mapping_json_path)
    
    # 검증
    validation_df = validate_mapping_results(mapping_result)
    print("\n📋 검증 결과:")
    print(validation_df[['page_number', 'question_count', 'mapped_count', 'status']].to_string(index=False))
    
    # 문제가 있는 페이지 확인
    problem_pages = validation_df[validation_df['status'] != 'Good']
    if not problem_pages.empty:
        print(f"\n⚠️  문제가 있는 페이지: {len(problem_pages)}개")
        for _, row in problem_pages.iterrows():
            print(f"   Page {row['page_number']}: {row['status']} (문항:{row['question_count']}, 매핑:{row['mapped_count']})")
    
    return mapping_result


def create_quick_preview(mapping_result: Dict, max_items: int = 10) -> None:
    """
    매핑 결과의 빠른 미리보기를 생성합니다.
    
    Args:
        mapping_result: 매핑 결과
        max_items: 표시할 최대 항목 수
    """
    
    print(f"\n👀 매핑 미리보기 (최대 {max_items}개):")
    print("-" * 80)
    
    mapping = mapping_result['question_image_mapping']
    
    for i, (question_id, image_path) in enumerate(list(mapping.items())[:max_items]):
        image_name = Path(image_path).name
        exists = "✅" if Path(image_path).exists() else "❌"
        print(f"{i+1:2d}. {question_id:15s} → {image_name:30s} {exists}")
    
    total_count = len(mapping)
    if total_count > max_items:
        print(f"    ... 및 {total_count - max_items}개 더")
    
    print("-" * 80)


def main():
    """
    메인 실행 함수
    """
    
    print("🎯 운전면허 시험 문항-이미지 매핑 도구")
    print("=" * 60)
    
    # 환경 설정 확인
    setup_environment()
    
    # 데이터 디렉토리 확인
    data_dir = "../../data"
    if not Path(data_dir).exists():
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {Path(data_dir).absolute()}")
        print("상대 경로를 조정하거나 올바른 경로를 지정해주세요.")
        return
    
    print(f"📂 데이터 디렉토리: {Path(data_dir).absolute()}")
    
    # 사용자 선택
    print("\n실행할 작업을 선택하세요:")
    print("1. 단일 PDF 파일 처리")
    print("2. 전체 PDF 파일 일괄 처리")
    print("3. 기존 매핑 결과 검증")
    print("4. 종료")
    
    choice = input("\n선택 (1-4): ").strip()
    
    if choice == "1":
        # 단일 파일 처리
        print("\n사용 가능한 PDF 파일:")
        pdf_files = list(Path(data_dir).glob("*raw_data_fromkoroad*.pdf"))
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"  {i}. {pdf_file.name}")
        
        if not pdf_files:
            print("❌ PDF 파일을 찾을 수 없습니다.")
            return
        
        file_choice = input(f"\n파일 선택 (1-{len(pdf_files)}): ").strip()
        try:
            file_index = int(file_choice) - 1
            if 0 <= file_index < len(pdf_files):
                run_single_pdf_processing(str(pdf_files[file_index]))
            else:
                print("❌ 잘못된 선택입니다.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
    
    elif choice == "2":
        # 일괄 처리
        run_batch_processing(data_dir)
    
    elif choice == "3":
        # 기존 매핑 검증
        data_dir_path = Path(data_dir).absolute()
        results_dir = data_dir_path / "results"
        
        # results 디렉토리의 모든 하위 폴더에서 JSON 파일 찾기
        json_files = list(results_dir.rglob("mapping_results_*.json")) if results_dir.exists() else []
        
        if not json_files:
            print("❌ 매핑 결과 파일(mapping_results_*.json)을 찾을 수 없습니다.")
            print(f"   검색 경로: {results_dir}")
            return
        
        print("\n사용 가능한 매핑 파일:")
        for i, json_file in enumerate(json_files, 1):
            relative_path = json_file.relative_to(results_dir)
            print(f"  {i}. {relative_path}")
        
        file_choice = input(f"\n파일 선택 (1-{len(json_files)}): ").strip()
        try:
            file_index = int(file_choice) - 1
            if 0 <= file_index < len(json_files):
                mapping_result = validate_and_fix_mappings(str(json_files[file_index]))
                if mapping_result:
                    create_quick_preview(mapping_result)
            else:
                print("❌ 잘못된 선택입니다.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
    
    elif choice == "4":
        print("👋 프로그램을 종료합니다.")
        return
    
    else:
        print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()
