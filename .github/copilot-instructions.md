# GitHub Copilot 지침: LLM 법규 지식 평가 프로젝트 (v2)

## 1\. 일반 가이드라인 (General Guidelines)

이 프로젝트의 모든 코드 생성 시 다음 원칙을 반드시 준수해야 합니다.

  * **You have to answer in Korean.**
  * **Only modify the minimum code necessary to complete this task. Do not change any other logic, structure, or unrelated code. Only focus on the given task.**
  * **연구 계획 참조**: 코드 작성에 필요한 모든 학술적 배경, 연구 가설, 실험 방법론은 프로젝트 루트의 `research_plan.pdf` 파일을 최우선으로 참고합니다.
  * **순차적 과업 수행**: 각 기능은 독립적인 함수로 구현하며, 아래에 제시된 `[Task]` 순서에 따라 하나씩 개발을 진행합니다. 하나의 함수는 하나의 책임만 갖도록(Single Responsibility Principle) 설계합니다.
  * **라이브러리**: 데이터 처리는 `pandas`, LLM 연동은 `openai`, 통계 분석은 `scipy`, 시각화는 `matplotlib`과 `seaborn` 사용을 원칙으로 합니다.
  * **코드 스타일**: Python 3.9+ 문법과 타입 힌트(type hints)를 적극적으로 사용하고, 모든 함수에는 기능, 파라미터, 반환 값을 설명하는 Docstring을 작성합니다.

-----

## 2\. 핵심 용어 및 데이터 스키마 정의

프로젝트 전반에서 사용될 핵심 용어와 데이터 구조입니다. 변수명과 클래스명은 이를 따릅니다.

### 핵심 용어 (Key Terminology)

  * [cite\_start]**역 포템킨 이해 (Inverse Potemkin Understanding, IPU)**: 정답은 맞혔으나(`ACC=1`), 그 근거가 되는 법규는 모르는(`LRA=0`) 상태를 의미합니다. [cite: 14, 29]
  * [cite\_start]**객관식 지름길 (Multiple-Choice Shortcut)**: 법규 지식 없이 선택지 간의 관계나 통계적 단서만으로 정답을 추론하는 현상을 의미합니다. [cite: 41, 47] [cite\_start]이는 IPU 현상의 주요 원인 중 하나로 간주됩니다. [cite: 50, 51]
  * [cite\_start]**지식 상태 분류 (Knowledge States)**: LLM의 답변을 정답 여부와 법적 근거 제시 여부에 따라 4가지 상태로 분류합니다. [cite: 163, 164]
      * [cite\_start]**A1 (진정한 지식)**: 정답과 법적 근거를 모두 맞힌 상태. [cite: 164, 165]
      * [cite\_start]**A2 (가짜 지식 / IPU)**: 정답은 맞혔지만 법적 근거는 틀린 상태. [cite: 164, 166]
      * [cite\_start]**B1 (부분적 지식 / 포템킨 이해)**: 정답은 틀렸지만 법적 근거는 맞힌 상태. [cite: 164, 167]
      * [cite\_start]**B2 (무지식)**: 정답과 법적 근거를 모두 틀린 상태. [cite: 164, 168]
  * [cite\_start]**ACC (Accuracy, 정답률)**: LLM이 문제의 정답을 맞혔는지의 비율입니다. [cite: 140]
  * [cite\_start]**LRA (Legal Reason Accuracy, 법적 근거 정확도)**: 정답을 맞혔고, 동시에 답변에 포함된 법적 근거가 실제 법률 및 조항과 정확히 일치하는지의 비율입니다. [cite: 141]
  * [cite\_start]**FLR (Fake Legal Rate)**: 정답을 맞힌 경우 중, 법적 근거는 틀리게 제시한 비율입니다. [cite: 142] [cite\_start]`(ACC - LRA) / ACC`로 계산합니다. [cite: 142]

### 데이터 스키마 (Data Schema)

실험 데이터와 결과는 다음 구조를 갖는 `pandas` DataFrame으로 관리합니다. 이는 통계 분석에 최적화된 Tidy Data 형식입니다.

```python
import pandas as pd

# 예상되는 DataFrame 구조
columns = [
    'question_id',                # 문제 고유 ID
    'model_name',                 # 실험에 사용된 LLM 이름
    'round',                      # 반복 실험 회차 (1~30)
    'knowledge_type',             # '법규지식형' or '그 외'
    'question_type',              # '문장형' or '사진/일러스트형'
    'format_type',                # '객관식' or '오픈형' or 'OX형'
    'raw_response',               # LLM이 생성한 원본 답변
    'predicted_answer',           # 파싱된 LLM의 정답
    'predicted_reason',           # 파싱된 LLM의 이유
    'predicted_legal_basis',      # 파싱된 LLM의 법적 근거
    'ground_truth_answer',        # 원본 문제의 정답
    'ground_truth_legal_basis',   # 원본 문제의 법적 근거 (Annotation된 데이터)
    'is_correct',                 # 정답 여부 (ACC 계산용, 1 or 0)
    'is_lra_correct',             # 법적 근거 일치 여부 (LRA 계산용, 1 or 0)
    'knowledge_state'             # A1, A2, B1, B2 중 하나의 값
]
results_df = pd.DataFrame(columns=columns)
```

-----

## 3\. 개발 과업 목록 (Development Tasks)

아래 태스크 순서에 따라 코드를 개발합니다.

### [완료] [Task 1] 데이터 로드 및 전처리 함수

운전면허 시험 문제은행 원본 데이터를 불러와 위에서 정의한 DataFrame 스키마에 맞게 전처리하는 함수를 구현합니다.

  * `load_and_preprocess_data(file_path: str) -> pd.DataFrame:`
      * CSV 또는 JSON 파일을 로드합니다.
      * [cite\_start]연구계획서의 분류 기준에 따라 각 문제에 `knowledge_type`, `question_type`을 할당합니다. [cite: 68]
          * [cite\_start]**`knowledge_type`**: 정답 선택지가 도로교통법에 명확히 근거하면 '법규지식형', 아니면 '그 외'로 분류합니다. [cite: 71, 76]
          * [cite\_start]**`question_type`**: 원본 데이터의 분류(문장형, 사진/일러스트형 등)를 그대로 사용합니다. [cite: 81]
      * [cite\_start]연구계획서의 기준에 따라 `ground_truth_legal_basis`를 Annotation합니다. [cite: 84] [cite\_start](https://www.google.com/search?q=%EC%98%88: "도로교통법 시행규칙 별표 6"). [cite: 85]

### [Task 2] 문제 형식 변환 함수

[cite\_start]객관식 문제를 오픈형 또는 O/X 문제로 변환하는 함수를 구현합니다. [cite: 97]

  * `transform_to_open_ended(question_row: pd.Series) -> str:`
      * 객관식 선택지를 제거하고, 서술형으로 답변할 수 있는 형태의 문제 텍스트를 생성합니다.
  * `transform_to_ox(question_row: pd.Series) -> str:`
      * [cite\_start]가설 2-1 검증을 위해, 4지 1답 문제를 참/거짓을 판단하는 O/X 문제로 변환합니다. [cite: 127, 131]

### [완료] [Task 3] 프롬프트 생성 함수

연구계획서에 명시된 답변 양식을 포함하는 프롬프트를 생성하는 함수를 구현합니다.

  * `create_prompt(question_row: pd.Series) -> str:`
      * 문제 형식(`객관식`, `오픈형`, `OX형`)에 따라 적절한 프롬프트를 동적으로 생성합니다.
      * [cite\_start]아래 `[답변 양식]`이 프롬프트에 반드시 포함되어야 합니다. [cite: 134]
        ```
        1. 정답: (선택지 번호 또는 서술형/OX 답변)
        2. 이유: (서술형 설명)
        3. 법적 근거: (법령명 + 조 또는 별표, 없으면 '없음')
        ```

### [Task 4] LLM 응답 요청 및 파싱 함수

LLM API를 호출하고, 반환된 텍스트에서 `정답`, `이유`, `법적 근거`를 파싱하여 분리하는 함수를 구현합니다.

  * `get_llm_response(prompt: str, model_name: str) -> str:`: LLM API를 호출하고 원본 응답을 문자열로 반환합니다.
  * `parse_response(response_text: str) -> dict:`: 원본 응답을 받아 `{'answer': ..., 'reason': ..., 'legal_basis': ...}` 형태의 딕셔너리를 반환합니다.

### [Task 5] 실험 실행 파이프라인

전체 실험을 실행하는 메인 로직을 구현합니다.

  * `run_experiment(data_df: pd.DataFrame, model_name: str) -> pd.DataFrame:`
      * [cite\_start]연구계획서의 실험 설계에 따라 **30회 반복** 실험을 수행합니다. [cite: 173]
      * [cite\_start]매 회차(round)마다 문제 유형별로 **50개의 문제를 비복원 추출**합니다. [cite: 173]
      * 추출된 각 문제에 대해 프롬프트를 생성하고(`Task 3`), LLM 응답을 받은 뒤(`Task 4`), 결과를 파싱합니다.
      * 응답 결과와 정답을 비교하여 `is_correct`와 `is_lra_correct`를 계산합니다.
          * [cite\_start]`is_lra_correct`는 `is_correct == 1`이면서, `predicted_legal_basis`가 `ground_truth_legal_basis`와 **법률명, 조, 별표까지 정확히 일치**할 때만 `1`이 됩니다. [cite: 123, 141] [cite\_start]'항', '호' 등 세부 항목은 평가에서 제외합니다. [cite: 149]
          * [cite\_start]오픈형 문제의 정답 비교 시에는 `answer matching` 기법을 활용합니다. [cite: 129, 136]
      * [cite\_start]결과를 바탕으로 `knowledge_state`를 할당합니다 (A1, A2, B1, B2). [cite: 164]
      * 모든 결과를 위에서 정의한 `results_df` 스키마에 맞춰 행으로 추가하고 최종 DataFrame을 반환합니다.

-----

## 4\. 분석 및 통계 검정 (Analysis & Statistical Testing)

실험 결과를 분석하고 가설을 통계적으로 검증하는 함수들을 구현합니다.

### [Task 6] 평가지표 계산 함수

`results_df`를 입력받아 각 그룹별 평가지표를 계산합니다.

  * `calculate_metrics(results_df: pd.DataFrame) -> pd.DataFrame:`
      * `model_name`, `knowledge_type`, `question_type`, `format_type`으로 그룹화(group by)합니다.
      * [cite\_start]각 그룹에 대해 `ACC`, `LRA`, `FLR`을 계산하여 요약된 DataFrame을 반환합니다. [cite: 140, 141, 142]

### [Task 7] 가설 검증 통계 함수

`scipy.stats` 라이브러리를 사용하여 연구 가설을 검증합니다.

  * `verify_hypothesis_1(results_df: pd.DataFrame):`
      * [cite\_start]**가설 1 검증**: 법규지식형 문제(`knowledge_type == '법규지식형'`) 전체에서 `ACC`와 `LRA` 간의 차이가 통계적으로 유의미한지 **대응표본 t-검정 (paired t-test)** 을 수행하고 p-value를 출력합니다. [cite: 62, 117]
  * `verify_hypothesis_2(results_df: pd.DataFrame):`
      * [cite\_start]**가설 2-1 검증**: 법규지식형 문제에서, `객관식`일 때와 `OX형`일 때의 `ACC - LRA` 값의 차이가 통계적으로 유의미한지 **독립표본 t-검정 (independent t-test)** 을 수행하고 p-value를 출력합니다. [cite: 63, 127]
      * [cite\_start]**가설 2-2 검증**: 법규지식형 문제에서, `문장형`일 때와 `사진/일러스트형`일 때의 `ACC - LRA` 값의 차이가 통계적으로 유의미한지 **독립표본 t-검정 (independent t-test)** 을 수행하고 p-value를 출력합니다. [cite: 64, 133]