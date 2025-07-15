"""
GUI 보정 도구 간편 실행 스크립트
"""
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from mapping_validator import create_manual_correction_interface
from pdf_image_extractor import load_mapping_from_json, save_mapping_to_json

def main():
    print("🖱️ GUI 매핑 보정 도구")
    print("=" * 50)
    
    # 사용 가능한 매핑 파일 찾기
    results_dir = Path("../../data/results")
    json_files = list(results_dir.rglob("mapping_results_*.json")) if results_dir.exists() else []
    
    if not json_files:
        print("❌ 매핑 결과 파일을 찾을 수 없습니다.")
        print(f"   검색 경로: {results_dir.absolute()}")
        print(f"   먼저 'python run_mapping.py'로 매핑을 실행해주세요.")
        return
    
    print("📂 사용 가능한 매핑 파일:")
    for i, json_file in enumerate(json_files, 1):
        try:
            relative_path = json_file.relative_to(results_dir)
            print(f"  {i}. {relative_path}")
        except ValueError:
            print(f"  {i}. {json_file.name}")
    
    # 파일 선택
    choice = input(f"\n파일 선택 (1-{len(json_files)}): ").strip()
    try:
        file_index = int(choice) - 1
        if 0 <= file_index < len(json_files):
            selected_file = json_files[file_index]
            
            print(f"\n🔄 매핑 파일 로드 중: {selected_file.name}")
            mapping_result = load_mapping_from_json(str(selected_file))
            
            # 매핑 정보 미리보기
            total_mappings = len(mapping_result.get('question_image_mapping', {}))
            print(f"   📊 총 매핑 수: {total_mappings}개")
            
            print("\n🖱️ GUI 보정 도구를 실행합니다...")
            print("   💡 창이 열리면 매핑을 검토하고 수정하세요")
            print("   💡 완료 후 [저장] 버튼을 클릭하면 창이 닫힙니다")
            
            # GUI 실행
            corrected_result = create_manual_correction_interface(mapping_result, str(selected_file))
            
            # 결과 저장
            output_file = selected_file.parent / f"{selected_file.stem}_corrected.json"
            save_mapping_to_json(corrected_result, str(output_file))
            
            # 변경사항 요약
            original_count = len(mapping_result.get('question_image_mapping', {}))
            corrected_count = len(corrected_result.get('question_image_mapping', {}))
            
            print(f"\n✅ 보정 완료!")
            print(f"   📊 보정 전 매핑: {original_count}개")
            print(f"   📊 보정 후 매핑: {corrected_count}개")
            print(f"   📁 보정된 결과: {output_file}")
            
            # CSV도 다시 생성
            from mapping_validator import export_mapping_to_dataframe
            final_df = export_mapping_to_dataframe(corrected_result)
            csv_output = selected_file.parent / f"question_image_mapping_{selected_file.stem}_corrected.csv"
            final_df.to_csv(csv_output, index=False, encoding='utf-8-sig')
            print(f"   📁 보정된 CSV: {csv_output}")
            
        else:
            print("❌ 잘못된 선택입니다.")
    except ValueError:
        print("❌ 숫자를 입력해주세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
