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

  * [cite\_start]**포템킨 이해 (Inverse Potemkin Understanding, IPU)**: 정답은 맞혔으나(`ACC=1`), 그 근거가 되는 법규는 모르는(`LRA=0`) 상태를 의미합니다. [cite: 14, 29]
  * [cite\_start]**객관식 지름길 (Multiple-Choice Shortcut)**: 법규 지식 없이 선택지 간의 관계나 통계적 단서만으로 정답을 추론하는 현상을 의미합니다. [cite: 41, 47] [cite\_start]이는 IPU 현상의 주요 원인 중 하나로 간주됩니다. [cite: 50, 51]
  * [cite\_start]**지식 상태 분류 (Knowledge States)**: LLM의 답변을 정답 여부와 법적 근거 제시 여부에 따라 4가지 상태로 분류합니다. [cite: 163, 164]
      * [cite\_start]**A1 (진정한 지식)**: 정답과 법적 근거를 모두 맞힌 상태. [cite: 164, 165]
      * [cite\_start]**A2 (가짜 지식 / IPU)**: 정답은 맞혔지만 법적 근거는 틀린 상태. [cite: 164, 166]
      * [cite\_start]**B1 (부분적 지식 / 포템킨 이해)**: 정답은 틀렸지만 법적 근거는 맞힌 상태. [cite: 164, 167]
      * [cite\_start]**B2 (무지식)**: 정답과 법적 근거를 모두 틀린 상태. [cite: 164, 168]
  * [cite\_start]**ACC (Accuracy, 정답률)**: LLM이 문제의 정답을 맞혔는지의 비율입니다. [cite: 140]
  * [cite\_start]**LRA (Legal Reason Accuracy, 법적 근거 정확도)**: **전체 문제 중** 정답과 법적 근거를 모두 정확하게 제시한 비율입니다. [cite: 141]
  * [cite\_start]**FLR (Fake Legal Rate)**: **정답을 맞힌 경우 중**, 법적 근거는 틀리게 제시한 비율입니다. [cite: 142] [cite\_start]`(ACC - LRA) / ACC`로 계산할 수 있습니다. [cite: 142]

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

## 3. 개발 과업 목록 (Development Tasks)

아래 태스크 순서에 따라 코드를 개발합니다.

### [완료] [Task 1] 실험 실행 스크립트 (`run_experiment.py`)

`config.yaml` 파일을 기반으로 LLM 평가 실험을 자동화하고 결과를 CSV 파일로 저장하는 스크립트를 구현합니다.

-   **설정 관리**: `config.yaml` 파일을 통해 `실험 유형`, `모델명`, `반복 횟수`, `샘플링 방식` 등 모든 실험 조건을 제어합니다.
-   **실험 유형 분기**:
    -   **`choice` 실험**: 원본 문제 은행(`questionbank.xlsx`)에서 카테고리별로 문제를 샘플링하여 객관식 문제를 풉니다.
    -   **`ox` 실험**: `choice` 실험 결과 파일에 있는 `ox_question`을 기반으로 동일한 문제에 대해 OX 문제를 풉니다. 결과는 원본 파일에 `llm_response_ox` 컬럼으로 병합되어 저장됩니다.
-   **프롬프트 생성**: `prompts.py`에 정의된 템플릿과 `config.yaml`에 명시된 컬럼을 조합하여 동적으로 프롬프트를 생성합니다.
-   **비동기 API 호출**: `asyncio`와 `tqdm_asyncio`를 사용하여 다수의 LLM API 요청을 병렬로 처리하여 실험 시간을 단축합니다.
-   **결과 저장**: 각 `iteration`별 실험 결과를 `llm_response`, `llm_response_ox` 등의 컬럼에 저장하고, 최종 결과를 CSV 파일로 출력합니다.

### [Task 2] 분석 파이프라인 스크립트 (`analysis.py`)

`run_experiment.py`를 통해 생성된 결과 CSV 파일을 입력받아, 통계 분석을 수행하고 최종 결과를 생성하는 스크립트를 구현합니다.

-   **파이프라인 구조**: **파싱 → 채점 → 분석 → 통계 검증**의 4단계로 구성됩니다.
-   **데이터 모델**: 하나의 원본 문제에 대한 `llm_response`(객관식)와 `llm_response_ox`(OX)를 두 개의 독립된 데이터(`format_type`)로 취급하여 처리 후 통합합니다.
-   **파싱**: `safe_json_loads` 함수를 사용하여 LLM의 JSON 응답 문자열을 `predicted_answer`와 `predicted_legal_basis`로 파싱합니다.
-   **채점**:
    -   `is_correct`: 예측 답변과 정답을 비교하여 채점합니다.
    -   `is_lra_correct`: 법적 근거의 일치 여부를 **LLM을 이용해 비동기 방식**으로 채점합니다. 이 채점은 정답 여부와 관계없이 수행됩니다.
-   **분석 및 저장**: `iteration` 및 `format_type`별로 그룹화하여 `ACC`, `LRA`, `FLR`을 계산하고, 채점 상세 결과와 분석 요약 결과를 별도의 CSV 파일로 저장합니다.

-----

## 4. 분석 및 통계 검정 (Analysis & Statistical Testing)

`analysis.py` 스크립트 내에 통계 검증 단계를 구현하여 연구 가설을 검증합니다.

### [Task 3] 가설 검증 통계 함수

`scipy.stats` 라이브러리를 사용하여 `analysis.py`의 최종 단계에서 통계 검증을 수행하고 결과를 출력합니다.

-   `verify_hypotheses(analysis_df: pd.DataFrame, graded_df: pd.DataFrame):`
    -   **가설 1 검증 (객관식 ACC vs LRA)**:
        -   **대상**: `format_type`이 'choice'인 데이터.
        -   **방법**: `iteration`별 `ACC`와 `LRA` 값에 대해 **대응표본 t-검정(paired t-test)**을 수행합니다.
        -   **결과**: p-value를 출력하여 `ACC`와 `LRA` 간의 평균 차이가 통계적으로 유의미한지 확인합니다.
    -   **가설 2a 검증 (객관식 vs OX)**:
        -   **대상**: `format_type`이 **'choice_paired'**인 그룹과 **'ox'**인 그룹. (`choice_paired`는 OX 문제로도 출제된 문제들의 객관식 결과 그룹을 의미)
        -   **방법**: 두 그룹 간 `ACC`, `LRA`, `FLR` 각각의 분포에 대해 **독립표본 t-검정(independent t-test)**을 수행합니다.
        -   **결과**: p-value를 출력하여 문제 형식에 따른 각 지표의 평균 차이가 유의미한지 확인합니다.
    -   **가설 2b 검증 (문제 유형별 비교)**:
        -   **대상**: `cls2` 컬럼을 기준으로 '문장형' 그룹과 '사진_일러스트형' 그룹.
        -   **방법**: 두 그룹 간 `ACC`, `LRA`, `FLR` 각각의 분포에 대해 **독립표본 t-검정(independent t-test)**을 수행합니다.
        -   **결과**: p-value를 출력하여 문제 유형에 따른 각 지표의 평균 차이가 유의미한지 확인합니다.