import os
import asyncio
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
import argparse
import yaml

# prompts.py 파일에서 프롬프트 관련 변수들을 가져옵니다.
from prompts import SYSTEM_INSTRUCTION, ANSWER_FORMAT_TEMPLATES, FEW_SHOT_EXAMPLES

# --- 1. 설정 로딩 함수 (argparse + yaml) ---
def load_config():
    """
    커맨드 라인 인자로 YAML 설정 파일 경로를 받아, 해당 파일의 내용을 읽어
    파라미터 딕셔너리로 반환합니다.
    """
    # 스크립트의 상위 폴더에 있는 'configs' 디렉토리를 기본 경로로 설정
    script_dir = Path(__file__).parent
    configs_dir = script_dir.parent / 'configs'

    parser = argparse.ArgumentParser(description="LLM 법규 지식 평가 실험 스크립트")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',  # 기본 설정 '파일 명'을 지정
        help="configs 폴더에 있는 실행할 실험의 설정 YAML 파일명"
    )
    args = parser.parse_args()

    # 기본 경로와 인자로 받은 파일명을 조합하여 최종 경로 생성
    config_file_path = configs_dir / args.config

    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"'{config_file_path}' 설정 파일을 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        raise FileNotFoundError(f"에러: 설정 파일 '{config_file_path}'을(를) 찾을 수 없습니다.")
        
    return config

# --- 2. 메인 비동기 실행 함수 ---
async def run_experiment(config: dict):
    """실험을 비동기적으로 실행하고 결과를 저장하는 메인 함수"""
    # --- 1. 전역 및 실험 유형별 설정 로드 ---
    experiment_type = config['experiment_type']
    model_name = config['model_name']
    output_dir = Path(config['output_dir'])

    # 현재 실험 유형에 맞는 상세 설정을 가져옴
    try:
        exp_config = config['experiments'][experiment_type]
    except KeyError:
        raise ValueError(f"'{experiment_type}'에 대한 설정이 'experiments' 섹션에 없습니다.")

    # 'ox' 실험 유형을 위한 별도 함수 호출
    if experiment_type == 'ox':
        await run_ox_experiment_on_choice_results(config)
        return

    # --- 2. 'choice', 'open' 등 일반 실험 로직 ---
    prompt_type = exp_config['prompt_type']
    num_iterations = exp_config['num_iterations']
    input_file_path = exp_config['input_file']
    prompt_columns = exp_config['prompt_columns']
    
    # --- 3. 유효성 검사 및 샘플링 설정 ---
    samples_per_category = exp_config.get('samples_per_category')
    num_samples = exp_config.get('num_samples')

    if experiment_type == 'choice' and samples_per_category:
        total_category_samples = sum(samples_per_category.values())
        if num_samples is not None and total_category_samples != num_samples:
            raise ValueError(
                f"설정 오류: 'samples_per_category'의 총합({total_category_samples})이 "
                f"'num_samples'({num_samples})와 일치하지 않습니다."
            )
    elif num_samples is None:
        raise ValueError(f"'{experiment_type}' 유형 실험에는 'num_samples' 또는 'samples_per_category' 설정이 필요합니다.")

    # --- 4. 출력 경로 설정 및 기본 정보 출력 ---
    # 파일명 자동 생성 (예: choice_zero-shot_gemini-1.5-flash-latest_4x100.csv)
    filename_parts = [
        experiment_type,
        prompt_type,
        model_name.replace('/', '_'), # 모델 이름에 포함될 수 있는 '/'를 '_'로 변경
        f"{num_iterations}x{num_samples}"
    ]
    output_filename = "_".join(filename_parts) + ".csv"
    output_csv_path = output_dir / output_filename
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n실험 (유형: {experiment_type})을(를) 시작합니다...")
    print(f"입력 파일: {input_file_path}")
    print(f"결과 저장 경로: {output_csv_path}")

    # --- 5. 모델 및 데이터 로딩 ---
    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_INSTRUCTION)
    answer_format = ANSWER_FORMAT_TEMPLATES.get(experiment_type)
    if not answer_format: raise ValueError(f"알 수 없는 실험 유형입니다: {experiment_type}")
    
    try:
        # 단일 Excel 파일의 '0_final' 시트를 읽도록 수정
        df = pd.read_excel(input_file_path, sheet_name='0_final')
    except FileNotFoundError:
        print(f"에러: '{input_file_path}' 파일을 찾을 수 없습니다."); return
    except ValueError as e:
        print(f"에러: '{input_file_path}' 파일에서 '0_final' 시트를 찾을 수 없거나 다른 문제가 발생했습니다. ({e})"); return

    # --- 6. 데이터 샘플링 및 실험 반복 ---
    all_results = []
    for i in range(num_iterations):
        print(f"\n--- Iteration {i + 1}/{num_iterations} 진행 중 ---")
        
        df_sampled_list = []
        if experiment_type == 'choice' and samples_per_category:
            # 카테고리별 샘플링
            for category, n_samples in samples_per_category.items():
                df_category = df[df['category'] == category]
                if len(df_category) < n_samples:
                    raise ValueError(
                        f"데이터 부족: '{category}' 카테고리의 문제 수({len(df_category)}개)가 "
                        f"요청된 샘플 수({n_samples}개)보다 적습니다."
                    )
                df_sampled_list.append(df_category.sample(n=n_samples, random_state=i))
            df_sampled = pd.concat(df_sampled_list).sample(frac=1, random_state=i).reset_index(drop=True)
        else:
            # 기존 방식 (전체에서 샘플링)
            if num_samples is None:
                raise ValueError("샘플링할 문제 수('num_samples')가 지정되지 않았습니다.")
            df_sampled = df.sample(n=num_samples, random_state=i).copy()

        # --- 7. 프롬프트 생성 및 비동기 작업 실행 ---
        few_shot_text = ""
        if prompt_type == 'few-shot':
            # prompts.py의 FEW_SHOT_EXAMPLES를 사용하도록 수정
            few_shot_text = FEW_SHOT_EXAMPLES + "\n\n---\n\n"

        tasks = []
        for _, row in df_sampled.iterrows():
            # config에 정의된 컬럼명을 사용하여 동적으로 프롬프트 구성
            question_col = prompt_columns['question']
            question_text = row[question_col]

            # 'choice' 유형이고 선택지가 유효할 때만 선택지를 포함한 프롬프트 생성
            if experiment_type == 'choice':
                options_col = prompt_columns.get('options')
                if options_col and pd.notna(row[options_col]):
                    options_text = row[options_col]
                    user_prompt_text = f"""{few_shot_text}문제: {question_text}\n선택지:\n{options_text}\n{answer_format}"""
                else: # 선택지가 없는 경우, 기본 프롬프트 사용
                    user_prompt_text = f"""{few_shot_text}문제: {question_text}\n{answer_format}"""
            else: # 'choice'가 아닌 다른 모든 유형 (ox, open-ended 등)
                user_prompt_text = f"""{few_shot_text}문제: {question_text}\n{answer_format}"""


            prompt_parts = [user_prompt_text]
            
            # 미디어 파일 처리 로직 (필요 시 구현)
            media_path = row.get('media_path')
            if pd.notna(media_path) and media_path.strip():
                pass # 미디어 파일 처리 로직 ...

            task = model.generate_content_async(
                contents=prompt_parts,
                generation_config=genai.types.GenerationConfig(temperature=0.2),
            )
            tasks.append(task)
        
        try:
            responses = await tqdm_asyncio.gather(*tasks, desc=f"Iteration {i+1} Progress")
            df_sampled[f'llm_response'] = [res.text for res in responses]
            df_sampled['iteration'] = i + 1
            all_results.append(df_sampled)
        except Exception as e:
            print(f"API 요청 중 에러 발생 (Iteration {i + 1}): {e}"); continue

    if not all_results: print("실험 결과가 없습니다."); return

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n모든 실험 완료! 결과가 '{output_csv_path}' 파일에 저장되었습니다.")


# --- 3. 'ox' 실험 실행 함수 ---
async def run_ox_experiment_on_choice_results(config: dict):
    """
    기존 'choice' 실험 결과 파일을 기반으로 'ox' 문제를 풀어
    결과를 새로운 파일에 저장하는 함수.
    """
    print("\n--- 'OX' 실험 모드를 시작합니다 (기존 결과 파일 기반) ---")
    
    # 설정 로드
    exp_config = config['experiments']['ox']
    base_result_path = Path(exp_config['base_result_file'])
    target_category = exp_config['target_category']
    prompt_columns = exp_config['prompt_columns']
    
    model_name = config['model_name'] # 전역 모델 이름 사용
    answer_format = ANSWER_FORMAT_TEMPLATES['OX']
    
    # 결과 파일 경로 설정 (원본 파일명에 '_with_ox' 추가)
    new_filename = base_result_path.stem + '_with_ox' + base_result_path.suffix
    output_csv_path = base_result_path.parent / new_filename

    print(f"기반 데이터 파일: {base_result_path}")
    print(f"대상 카테고리: {target_category}")
    print(f"결과 저장 경로: {output_csv_path}")

    # 데이터 로드
    try:
        df = pd.read_csv(base_result_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"에러: 기반 결과 파일 '{base_result_path}'을(를) 찾을 수 없습니다.")

    # 대상 문제 필터링
    df_target = df[df['category'] == target_category].copy()
    if df_target.empty:
        print(f"경고: '{target_category}' 카테고리에 해당하는 문제가 없습니다. 실험을 종료합니다.")
        return

    # LLM 설정
    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_INSTRUCTION)

    # 비동기 작업 생성
    tasks = []
    for _, row in df_target.iterrows():
        question_col = prompt_columns['question']
        question_text = row[question_col]
        user_prompt_text = f"문제: {question_text}\n{answer_format}"
        
        task = model.generate_content_async(
            contents=[user_prompt_text],
            generation_config=genai.types.GenerationConfig(temperature=0.2),
        )
        tasks.append(task)

    # LLM 응답 요청 및 결과 저장
    print(f"\n총 {len(tasks)}개의 OX 문제에 대한 LLM 응답을 요청합니다...")
    try:
        responses = await tqdm_asyncio.gather(*tasks, desc="OX 문제 처리 중")
        # llm_response가 텍스트 형식이라고 가정
        df_target['llm_response_ox'] = [res.text for res in responses]
    except Exception as e:
        print(f"API 요청 중 에러 발생: {e}")
        return

    # 원본 데이터프레임에 OX 결과 병합
    # pd.merge를 사용하여 'llm_response_ox' 컬럼을 추가합니다.
    # (df.update는 기존에 있는 컬럼의 값만 변경하고 새 컬럼을 추가하지 않음)
    df = pd.merge(
        df,
        df_target[['number', 'iteration', 'llm_response_ox']],
        on=['number', 'iteration'],
        how='left'  # 원본(choice)의 모든 행을 유지
    )

    # 결과 저장
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n'OX' 실험 완료! 결과가 '{output_csv_path}' 파일에 저장되었습니다.")


# --- 4. 스크립트 실행 ---
if __name__ == "__main__":
    config = load_config()
    
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY: raise ValueError("API 키를 찾을 수 없습니다.")
    genai.configure(api_key=API_KEY)

    asyncio.run(run_experiment(config))