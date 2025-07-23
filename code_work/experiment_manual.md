# 실험 실행 가이드 (Experiment Manual)

이 문서는 `run_experiment.py` 스크립트를 사용하여 LLM 평가 실험을 수행하는 방법을 안내합니다. 실험은 `config.yaml` 파일을 통해 제어됩니다.

## 1. 개요

`run_experiment.py`는 다양한 유형의 문제(객관식, OX, 주관식)에 대해 LLM의 답변을 생성하고, 그 결과를 CSV 파일로 저장하는 자동화 스크립트입니다. 모든 실험 설정은 `code_work/configs/` 폴더 내의 `.yaml` 파일을 통해 관리되므로, 코드 수정 없이 다양한 조건의 실험을 반복 수행할 수 있습니다.

## 2. 사전 준비

실험을 실행하기 전에 다음 사항을 확인해야 합니다.

### 가. 필수 라이브러리 설치

프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 필요한 라이브러리를 설치합니다.

```bash
poetry install
```

### 나. API 키 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고, 아래와 같이 Google AI Studio에서 발급받은 API 키를 입력합니다.

```
GOOGLE_API_KEY="여기에_발급받은_API_키를_입력하세요"
```

## 3. 실험 실행 방법

터미널에서 프로젝트 루트 디렉토리 기준으로 다음 명령어를 실행합니다.

### 가. 기본 실행

`configs/config.yaml` 파일을 기본 설정으로 사용하여 실험을 실행합니다.

```bash
python code_work/scripts/run_experiment.py
```

### 나. 특정 설정 파일로 실행

`configs` 폴더에 있는 다른 설정 파일(예: `my_config.yaml`)을 사용하려면 `--config` 인자를 추가합니다.

```bash
python code_work/scripts/run_experiment.py --config my_config.yaml
```

## 4. 설정 파일 (`config.yaml`) 상세 설명

실험의 모든 동작은 `config.yaml` 파일을 통해 제어됩니다. 파일 구조는 크게 **전역 설정**과 **실험 유형별 상세 설정**으로 나뉩니다.

### 가. 전역 설정 (Global Settings)

파일 상단에 위치하며 모든 실험에 공통적으로 적용됩니다.

- `experiment_type`: 실행할 실험의 유형을 지정합니다.
  - `'choice'`: 객관식 문제 실험
  - `'ox'`: `choice` 실험 결과에 기반한 OX 문제 실험
  - `'open'`: 주관식 문제 실험
- `model_name`: 실험에 사용할 Google Gemini 모델의 이름을 지정합니다. (예: `gemini-1.5-flash-latest`)
- `output_dir`: 실험 결과 CSV 파일이 저장될 디렉토리를 지정합니다. (예: `results/`)

### 나. 실험 유형별 상세 설정 (Experiments)

`experiments` 키 아래에 각 실험 유형별로 상세 설정을 정의합니다. `experiment_type`에서 선택된 유형의 설정이 사용됩니다.

#### 1) 객관식 (`choice`) 실험

- `prompt_type`: 프롬프트 유형을 지정합니다. (예: `'zero-shot'`, `'few-shot'`)
- `num_iterations`: 전체 실험을 몇 번 반복할지 지정합니다. 매 반복마다 데이터를 새로 샘플링합니다.
- `num_samples`: 한 번의 반복에서 샘플링할 **총 문제 수**를 지정합니다.
- `input_file`: 문제 데이터가 포함된 Excel 파일의 경로를 지정합니다. (예: `data/questionbank.xlsx`)
- `samples_per_category`: 각 문제 유형(카테고리)별로 몇 개의 문제를 샘플링할지 상세하게 지정합니다.
  - **주의**: 여기에 명시된 모든 샘플 수의 합은 `num_samples`와 정확히 일치해야 합니다.
- `prompt_columns`: `input_file`에서 문제와 선택지를 가져올 때 사용할 컬럼(열)의 이름을 지정합니다.
  - `question`: 문제 내용이 담긴 컬럼명
  - `options`: 선택지 내용이 담긴 컬럼명

#### 2) OX (`ox`) 실험

이 실험은 기존 `choice` 실험의 결과를 바탕으로 특정 카테고리의 문제들을 OX 문제로 변환하여 다시 풀어보는 시나리오입니다.

- `base_result_file`: 기반으로 삼을 `choice` 실험 결과 CSV 파일의 경로를 지정합니다.
- `target_category`: `base_result_file`에서 OX 문제를 풀 대상이 될 문제의 카테고리를 지정합니다.
- `prompt_columns`:
  - `question`: `base_result_file`에서 OX 문제 내용이 담긴 컬럼명을 지정합니다. (예: `ox_question`)

#### 3) 주관식 (`open`) 실험

- `prompt_type`: 프롬프트 유형을 지정합니다.
- `num_iterations`: 실험 반복 횟수입니다.
- `num_samples`: 반복당 샘플링할 총 문제 수입니다.
- `input_file`: 원본 데이터 파일 경로입니다.
- `prompt_columns`:
  - `question`: `input_file`에서 주관식 문제 내용이 담긴 컬럼명을 지정합니다. (예: `open_question`)

## 5. 실험 절차 예시

### 예시: '문장형_4지1답' 문제에 대해 OX 실험 진행하기

1.  **1단계: `choice` 실험 실행**
    - `config.yaml` 파일의 `experiment_type`을 `'choice'`로 설정합니다.
    - `choice` 섹션의 `samples_per_category`에 `'문장형_4지1답'`이 포함되어 있는지 확인하고, 나머지 설정을 원하는 대로 조정합니다.
    - 스크립트를 실행합니다: `python code_work/scripts/run_experiment.py`
    - 실행이 완료되면 `results/` 폴더에 `choice_zero-shot_..._4x100.csv`와 같은 결과 파일이 생성됩니다.

2.  **2단계: `ox` 실험 실행**
    - `config.yaml` 파일의 `experiment_type`을 `'ox'`로 변경합니다.
    - `ox` 섹션의 설정을 다음과 같이 수정합니다.
      - `base_result_file`: 방금 전 생성된 `choice` 실험 결과 파일 경로(예: `results/choice_zero-shot_..._4x100.csv`)를 입력합니다.
      - `target_category`: `'문장형_4지1답'`으로 설정합니다.
    - 스크립트를 다시 실행합니다: `python code_work/scripts/run_experiment.py`
    - 실행이 완료되면 `results/` 폴더에 원본 파일명 뒤에 `_with_ox`가 붙은 새로운 파일(예: `choice_zero-shot_..._4x100_with_ox.csv`)이 생성됩니다. 이 파일에는 기존 `choice` 실험 결과에 `llm_response_ox` 컬럼이 추가되어 있습니다.
