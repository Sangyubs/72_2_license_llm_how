## 입력된 파일의 컬럼 설명
- `number`: 문제 번호
- `law_answer`: 법적 근거에 대한 정답
- `answer`: 객관식(choice) 문제의 정답
- `ox_answer`: OX 문제의 정답
- `llm_response`: 객관식(choice) 문제에 대한 LLM의 응답 (JSON 형식)
- `iteration`: 실험 반복 번호(그룹 번호)
- `llm_response_ox`: OX 문제에 대한 LLM의 응답 (JSON 형식)
- 이외 컬럼은 무시해도 좋다. 

## 핵심 분석 전략
이 분석의 핵심은 **하나의 원본 문제**에 대해 **'객관식(choice) 질문'**과 **'OX 질문'**의 결과를 각각 독립적으로 분석하고, 최종적으로 결과를 통합하여 비교하는 것입니다.

- **데이터 모델**: 입력 파일의 한 행(row)은 '하나의 문제'를 의미하며, 이 행에는 객관식 응답(`llm_response`)과 OX 응답(`llm_response_ox`)이 모두 포함될 수 있습니다.
- **분석 흐름**:
  1.  **데이터 분리**: 원본 데이터를 두 개의 논리적 스트림으로 나눕니다.
      - **스트림 1 (Choice)**: 모든 행의 `llm_response` 컬럼을 사용합니다.
      - **스트림 2 (OX)**: `llm_response_ox` 컬럼에 유효한 값이 있는 행만 필터링하여 사용합니다.
  2.  **독립적 처리**: 각 스트림에 대해 파싱, 채점, 법적 근거 평가를 독립적으로 수행합니다.
      - Choice 스트림은 `answer`를 정답으로, OX 스트림은 `ox_answer`를 정답으로 사용합니다.
  3.  **결과 통합**: 처리된 두 스트림의 결과를 `format_type` 컬럼('choice', 'ox')으로 명확히 구분하여, 하나의 긴 데이터프레임(long-form DataFrame)으로 수직으로 병합(concatenate)합니다.

이러한 구조를 통해, 사용자는 최종 결과물에서 `format_type`을 기준으로 그룹화하여 각 문제 유형별 정확도와 법적 근거 일치도를 명확하게 확인할 수 있습니다.

## 리팩토링된 분석 파이프라인
`analysis.py` 스크립트는 아래 3단계 파이프라인에 따라 분석을 수행합니다.

### 1단계: 파싱 (Parsing)
- **목표**: 원본 CSV 파일의 `llm_response`와 `llm_response_ox` 컬럼에 있는 JSON 형식의 문자열을 파싱하여 구조화된 데이터로 변환합니다.
- **프로세스**:
  1.  **데이터 분리**: 원본 데이터를 'Choice'와 'OX' 두 스트림으로 논리적으로 분리합니다.
      - **Choice 스트림**: 모든 행의 `llm_response` 컬럼을 사용합니다.
      - **OX 스트림**: `llm_response_ox` 컬럼에 유효한 값이 있는 행만 필터링하여 사용합니다.
  2.  **JSON 파싱**: 각 스트림의 응답 컬럼(`llm_response`, `llm_response_ox`)에 대해 `safe_json_loads` 함수를 적용하여 `predicted_answer`와 `predicted_legal_basis` 컬럼을 생성합니다.
  3.  **통합**: 두 스트림의 파싱 결과를 `format_type` ('choice', 'ox')으로 구분하여 하나의 DataFrame으로 병합합니다.

### 2단계: 채점 (Grading)
- **목표**: 파싱된 데이터를 바탕으로 문제의 정답 여부와 법적 근거의 정확성을 평가합니다.
- **프로세스**:
  1.  **정답 채점 (`is_correct`)**:
      - `format_type`이 'choice'인 경우, `predicted_answer`와 `answer`를 비교합니다.
      - `format_type`이 'ox'인 경우, `predicted_answer`와 `ox_answer`를 비교합니다.
      - 결과를 `is_correct` 컬럼 (1 또는 0)에 저장합니다.
  2.  **법적 근거 채점 (`is_lra_correct`)**:
      - **채점 대상**: LLM이 법적 근거를 제시한 모든 경우(`predicted_legal_basis`가 비어있지 않은 모든 행)에 대해 채점을 수행합니다. **이는 문제의 정답 여부(`is_correct`)와 무관하게 진행됩니다.**
      - 결과를 `is_lra_correct` 컬럼 (1 또는 0)에 저장합니다.

### 3단계: 분석 및 저장 (Analysis & Save)
- **목표**: 채점된 데이터를 바탕으로 최종 평가지표를 계산하고 결과를 파일로 저장합니다.
- **프로세스**:
  1.  **분석 그룹 생성**:
      - **`choice_paired` 그룹 생성**: `llm_response_ox` 컬럼에 응답이 있는, 즉 OX 문제로도 출제된 문제들의 객관식(`format_type` == 'choice') 결과만 필터링하여 `format_type`을 `choice_paired`로 지정합니다.
      - **분석 데이터 통합**: 기존의 `choice`, `ox` 데이터와 새로 생성한 `choice_paired` 데이터를 모두 포함하는 분석용 데이터프레임을 만듭니다.
  2.  **그룹별 평가지표 계산**: `iteration` 및 `format_type`('choice', 'ox', 'choice_paired') 별로 그룹화하여 `ACC`, `LRA`, `FLR`을 계산합니다.
  3.  **결과 저장**: 최종 지표 요약 테이블과 전체 채점 상세 결과를 각각 별도의 CSV 파일로 저장합니다.

### 4단계: 가설 검증 (통계 분석)
- **목표**: 계산된 평가지표를 바탕으로 연구 가설을 통계적으로 검증합니다.
- **프로세스**:
  1.  **가설 1 (객관식 ACC vs LRA)**: `format_type`이 'choice'인 데이터에 대해 `ACC`와 `LRA` 간 **대응표본 t-검정**을 수행합니다.
  2.  **가설 2a (객관식 vs OX)**: **`format_type`이 'choice_paired'인 그룹과 'ox'인 그룹 간**에 `ACC`, `LRA`, `FLR` 지표에 대해 **독립표본 t-검정**을 수행합니다.
  3.  **가설 2b (문제 유형별 비교)**: `cls_2` 컬럼을 기준으로 그룹을 나누어 `ACC`, `LRA`, `FLR` 지표에 대해 **독립표본 t-검정**을 수행합니다.
- [프롬프트 템플릿](code_work/scripts/prompts.py)