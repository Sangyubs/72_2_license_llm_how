# 문항-이미지 매핑 도구 사용법

## 개요

운전면허 시험 PDF에서 문항번호와 이미지를 자동으로 연결하는 도구입니다.

## 주요 기능

1. **PDF 파싱**: PDF에서 문항번호와 이미지를 추출
2. **자동 매핑**: 위치 기반으로 문항과 이미지 연결
3. **검증 및 보정**: 매핑 결과 검증 및 수동 보정 지원
4. **리포트 생성**: HTML 형태의 상세 리포트 생성

## 설치 방법

### 1. 필수 패키지 설치
```bash
# Windows
install_requirements.bat

# 또는 수동 설치
pip install PyMuPDF opencv-python pillow pandas matplotlib numpy scipy
```

### 2. 디렉토리 구조 확인
```
code_work/src/
├── pdf_image_extractor.py      # 핵심 추출 함수
├── mapping_validator.py        # 검증 및 보정 함수  
├── run_mapping.py             # 실행 스크립트
└── install_requirements.bat    # 패키지 설치 스크립트
```

## 사용법

### 방법 1: 대화형 실행 -> 추천
```bash
cd code_work/src
python run_mapping.py
```

메뉴에서 선택:
- 1: 단일 PDF 파일 처리
- 2: 전체 PDF 파일 일괄 처리  
- 3: 기존 매핑 결과 검증

### 방법 2: 프로그래밍 방식 사용

```python
from pdf_image_extractor import extract_questions_and_images_from_pdf
from mapping_validator import main_mapping_workflow

# 단일 PDF 처리
pdf_path = "../../data/3_raw_data_fromkoroad_사진형.pdf"
result_df = main_mapping_workflow(pdf_path)

# 결과 확인
print(f"총 매핑 수: {len(result_df)}")
print(result_df.head())
```

## 출력 파일

### 📁 저장 위치
모든 결과 파일은 `G:\내 드라이브\4_paper\72_2_license_llm_how\data\results\` 아래에 PDF별로 분류되어 저장됩니다.

```
data/
└── results/
    ├── 사진형/
    │   ├── extracted_images/          # 추출된 이미지들
    │   ├── question_image_mapping_*.csv
    │   ├── mapping_report_*.html
    │   └── mapping_results_사진형.json
    ├── 일러스트/
    │   ├── extracted_images/
    │   ├── question_image_mapping_*.csv
    │   ├── mapping_report_*.html
    │   └── mapping_results_일러스트.json
    ├── 안전표지/
    │   └── ...
    └── all_question_image_mappings.csv  # 통합 결과
```

### 1. 이미지 파일
- `data/results/{파일타입}/extracted_images/` 디렉토리에 페이지별 이미지 저장
- 파일명 형식: `page_{페이지번호}_img_{이미지번호}.png`

### 2. 매핑 결과
- `data/results/{파일타입}/question_image_mapping_{파일명}.csv`: 매핑 결과 데이터프레임
- `data/results/{파일타입}/mapping_results_{파일타입}.json`: 상세 매핑 정보
- `data/results/{파일타입}/mapping_report_{파일명}.html`: 검증 리포트

### 3. 통합 결과
- `data/results/all_question_image_mappings.csv`: 모든 파일의 통합 매핑 결과

## 데이터 구조

### CSV 출력 형식
```
question_id,question_number,image_path,image_filename,page_number,has_image,source_file
question_1,1,data/results/사진형/extracted_images/page_1_img_1.png,page_1_img_1.png,1,True,사진형
question_2,2,data/results/사진형/extracted_images/page_1_img_2.png,page_1_img_2.png,1,True,사진형
```

### JSON 매핑 정보
```json
{
  "question_image_mapping": {
    "question_1": "data/results/사진형/extracted_images/page_1_img_1.png",
    "question_2": "data/results/사진형/extracted_images/page_1_img_2.png"
  },
  "page_info": [
    {
      "page_number": 1,
      "question_count": 2,
      "image_count": 2,
      "questions": ["1", "2"],
      "mapping": {"question_1": "...", "question_2": "..."}
    }
  ],
  "extraction_log": ["Page 1: 2 questions, 2 images"]
}
```

## 매핑 알고리즘

### 1. 문항번호 추출 패턴
- `1. `, `2. ` 형태 (점과 공백)
- `문제 1`, `문제1` 형태  
- `1번`, `2번` 형태
- `Q.1`, `Q 1` 형태

### 2. 매핑 방법
1. **순서 기반**: 페이지 내에서 문항과 이미지의 순서로 매핑
2. **위치 기반**: 텍스트와 이미지의 Y 좌표를 고려한 매핑
3. **컨텍스트 기반**: 문항 주변 텍스트에서 이미지 참조 키워드 확인

### 3. 검증 메트릭
- **매핑률**: 문항 대비 매핑된 비율
- **이미지 활용률**: 이미지 대비 매핑된 비율  
- **상태**: Good(80%+) / Warning(50%+) / Error(<50%)

## 문제 해결

### 자주 발생하는 문제

1. **PyMuPDF 설치 오류**
   ```bash
   pip install --upgrade pip
   pip install PyMuPDF
   ```

2. **이미지 추출 실패**
   - PDF가 보호되어 있는 경우: 보호 해제 후 재시도
   - 이미지가 벡터 형식인 경우: 래스터 이미지로 변환 필요

3. **매핑 정확도 저조**
   - 문항번호 패턴 확인 및 수정
   - 수동 보정 인터페이스 활용
   - 페이지별 검증 리포트 확인

### 로그 확인
```python
# 상세 로그 확인
mapping_result = extract_questions_and_images_from_pdf(pdf_path)
for log in mapping_result['extraction_log']:
    print(log)
```

## 고급 사용법

### 1. 커스텀 매핑 패턴 추가
```python
# pdf_image_extractor.py의 extract_question_numbers_from_text 함수 수정
patterns = [
    r'(\d+)\.\s',  # 기본 패턴
    r'your_custom_pattern',  # 커스텀 패턴 추가
]
```

### 2. 매핑 결과 후처리
```python
# 매핑 결과 로드 및 수정
from pdf_image_extractor import load_mapping_from_json, save_mapping_to_json

# 특정 파일타입의 결과 로드
mapping_result = load_mapping_from_json("../../data/results/사진형/mapping_results_사진형.json")

# 수동 매핑 추가
mapping_result['question_image_mapping']['question_new'] = 'data/results/사진형/extracted_images/page_5_img_3.png'

# 수정된 결과 저장
save_mapping_to_json(mapping_result, "../../data/results/사진형/mapping_results_사진형_fixed.json")
```

### 3. 대량 처리 스크립트
```python
import glob
from pathlib import Path

# 모든 PDF 파일 처리
data_dir = Path("../../data")
pdf_files = list(data_dir.glob("*raw_data*.pdf"))
for pdf_path in pdf_files:
    if "사진" in pdf_path.name or "일러스트" in pdf_path.name:
        main_mapping_workflow(str(pdf_path))
        
# 결과 확인
results_dir = data_dir / "results"
for result_folder in results_dir.iterdir():
    if result_folder.is_dir():
        csv_files = list(result_folder.glob("*.csv"))
        print(f"{result_folder.name}: {len(csv_files)}개 결과 파일")
```

## 참고사항

- 처리 시간: 페이지당 2-5초 소요
- 메모리 사용량: PDF 크기에 비례 (일반적으로 100-500MB)
- 지원 형식: PDF (보호되지 않은 파일)
- 권장 이미지 해상도: 최소 150 DPI
