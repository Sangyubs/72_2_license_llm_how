import pandas as pd
import yaml
import os
import json
import sys
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple
import argparse
from pathlib import Path
import asyncio
import numpy as np
from scipy import stats

# 스크립트가 프로젝트 루트에서 실행될 수 있도록 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# llm_client와 prompts는 가상으로 존재한다고 가정합니다.
from code_work.src.llm_client import LLMClient
from code_work.scripts.prompts import LEAGAL_BASIS_MATCHING_PROMPT

def load_config() -> Dict[str, Any]:
    """
    커맨드 라인 인자로 YAML 설정 파일 경로를 받아, 해당 파일의 내용을 읽어
    파라미터 딕셔너리로 반환합니다.
    """
    script_dir = Path(__file__).parent
    configs_dir = script_dir.parent / 'configs'

    parser = argparse.ArgumentParser(description="LLM 법규 지식 평가 분석 스크립트")
    parser.add_argument(
        '--config',
        type=str,
        default='analysis_config.yaml',
        help="configs 폴더에 있는 실행할 분석의 설정 YAML 파일명"
    )
    args = parser.parse_args()
    config_file_path = configs_dir / args.config

    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"'{config_file_path}' 설정 파일을 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        raise FileNotFoundError(f"에러: 설정 파일 '{config_file_path}'을(를) 찾을 수 없습니다.")
        
    return config

def safe_json_loads(s: Optional[str]) -> Dict[str, Any]:
    """JSON 문자열을 안전하게 파싱합니다. 실패 시 빈 딕셔너리를 반환합니다."""
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        s = s.strip().replace('```json', '').replace('```', '')
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}

def parse_llm_responses(df_raw: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    LLM 응답(choice, ox)을 파싱하여 긴 형식의 데이터프레임으로 변환합니다.
    """
    print("\n--- 1단계: LLM 응답 파싱 시작 ---")
    cols = config['column_names']
    
    processed_dfs = []

    # 1. 객관식(Choice) 응답 파싱
    df_choice = df_raw.copy()
    df_choice['format_type'] = 'choice'
    df_choice['response_col'] = df_choice[cols['llm_choice_response']]
    df_choice['gt_answer_col'] = df_choice[cols['gt_choice_answer']]
    processed_dfs.append(df_choice)
    
    # 2. OX 응답 파싱 (유효한 응답만 필터링)
    df_ox = df_raw[df_raw[cols['llm_ox_response']].notna() & (df_raw[cols['llm_ox_response']] != '')].copy()
    if not df_ox.empty:
        df_ox['format_type'] = 'ox'
        df_ox['response_col'] = df_ox[cols['llm_ox_response']]
        df_ox['gt_answer_col'] = df_ox[cols['gt_ox_answer']]
        processed_dfs.append(df_ox)

    # 3. 통합 및 파싱 적용
    df_combined = pd.concat(processed_dfs, ignore_index=True)
    
    print("JSON 형식 응답을 파싱합니다...")
    tqdm.pandas(desc="파싱 진행률")
    parsed_data = df_combined['response_col'].progress_apply(safe_json_loads)
    
    # 파싱된 결과가 비어있을 경우를 대비하여 빈 데이터프레임 생성
    if parsed_data.empty:
        parsed_df = pd.DataFrame(columns=[config['column_names']['predicted_answer'], config['column_names']['predicted_legal_basis']])
    else:
        parsed_df = pd.json_normalize(parsed_data).rename(columns={
            'answer': config['column_names']['predicted_answer'],
            'legal_basis': config['column_names']['predicted_legal_basis']
        })
    
    # 필요한 컬럼이 없을 경우 추가
    for col in [config['column_names']['predicted_answer'], config['column_names']['predicted_legal_basis']]:
        if col not in parsed_df.columns:
            parsed_df[col] = np.nan

    result_df = pd.concat([df_combined.reset_index(drop=True), parsed_df], axis=1)
    print("파싱 완료.")
    return result_df

async def grade_responses(df_parsed: pd.DataFrame, config: Dict[str, Any], client: LLMClient) -> pd.DataFrame:
    """
    파싱된 데이터를 바탕으로 정답 여부와 법적 근거 정확성을 채점합니다.
    """
    print("\n--- 2단계: 응답 채점 시작 ---")
    cols = config['column_names']
    df = df_parsed.copy()

    # 답변 문자열을 정렬하여 순서에 무관하게 비교하는 함수
    def normalize_answer(answer_str: str) -> str:
        if not isinstance(answer_str, str) or answer_str.strip() == '':
            return ''
        # 쉼표로 분리 -> 공백 제거 -> 정렬 -> 다시 쉼표로 합치기
        parts = sorted([part.strip() for part in answer_str.split(',')])
        return ','.join(parts)

    # 1. 정답 채점 (is_correct)
    print("정답 여부(is_correct)를 채점합니다...")
    # apply를 사용하여 각 답변을 정규화
    gt_answers = df['gt_answer_col'].astype(str).str.upper().apply(normalize_answer)
    pred_answers = df[cols['predicted_answer']].astype(str).str.upper().apply(normalize_answer)
    df[cols['is_correct']] = (gt_answers == pred_answers).astype(int)

    # 2. 법적 근거 채점 (is_lra_correct) - 비동기 처리
    df[cols['is_lra_correct']] = 0
    
    grading_mask = df[cols['predicted_legal_basis']].notna() & (df[cols['predicted_legal_basis']] != '') & \
                   df[cols['gt_legal_basis']].notna() & (df[cols['gt_legal_basis']] != '')

    if not grading_mask.any():
        print("법적 근거 채점 대상이 없습니다.")
        return df

    print(f"총 {grading_mask.sum()}개의 법적 근거에 대해 LLM 채점을 시작합니다 (비동기 처리, 동시 50개 제한)...")
    
    semaphore = asyncio.Semaphore(50)

    async def get_lra_score(row: pd.Series) -> Tuple[int, int]:
        prompt = LEAGAL_BASIS_MATCHING_PROMPT.format(
            gt_legal_basis=row[cols['gt_legal_basis']],
            llm_legal_basis=row[cols['predicted_legal_basis']]
        )
        async with semaphore:
            response = await client.generate_async(prompt)
        try:
            # LLM의 JSON 응답을 파싱하여 judgment 값을 점수로 사용
            parsed_json = safe_json_loads(response)
            score = parsed_json.get('judgment', 0)
            return row.name, score
        except Exception:
            return row.name, 0

    tasks = [asyncio.create_task(get_lra_score(row)) for _, row in df[grading_mask].iterrows()]
    results = await tqdm_asyncio.gather(*tasks, desc="법적 근거 채점 진행률")

    for index, score in results:
        df.loc[index, cols['is_lra_correct']] = score
        
    print("채점 완료.")
    return df

def analyze_and_save_results(df_graded: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    채점된 데이터를 분석하여 평가지표를 계산하고 결과를 저장합니다.
    분석 결과와 전체 채점 데이터를 반환합니다.
    """
    print("\n--- 3단계: 분석 및 결과 저장 시작 ---")
    cols = config['column_names']
    
    # 1. 평가지표 계산을 위한 데이터 준비
    # OX 문제로 출제된 문제들의 객관식(choice) 결과만 필터링하여 'choice_paired' 그룹 생성
    ox_response_col = cols['llm_ox_response']
    ox_present_mask = df_graded[ox_response_col].notna() & (df_graded[ox_response_col] != '')
    
    df_choice_paired = df_graded[(df_graded['format_type'] == 'choice') & ox_present_mask].copy()
    df_choice_paired['format_type'] = 'choice_paired' # 새로운 분석 단위명 부여
    
    # 기존 데이터(choice, ox)와 새로 만든 choice_paired 데이터를 합쳐서 분석 수행
    df_for_analysis = pd.concat([df_graded, df_choice_paired], ignore_index=True)

    # 2. 평가지표 계산 (iteration 및 format_type 별)
    def calculate_metrics(df_group: pd.DataFrame) -> pd.Series:
        # ...기존 calculate_metrics 함수 내용과 동일...
        is_correct = df_group[cols['is_correct']]
        is_lra_correct = df_group[cols['is_lra_correct']]
        
        acc = is_correct.mean()
        # LRA: 정답을 맞히고(is_correct=1) 법적 근거도 맞힌(is_lra_correct=1) 경우의 전체 대비 비율
        lra = ((is_correct == 1) & (is_lra_correct == 1)).sum() / len(df_group) if len(df_group) > 0 else 0
        # FLR: 정답을 맞힌 경우 중, 법적 근거는 틀린 비율
        flr = (acc - lra) / acc if acc > 0 else 0
        
        return pd.Series({
            'ACC': acc,
            'LRA': lra,
            'FLR': flr,
            'count': len(df_group)
        })

    print("iteration 및 format_type 별로 평가지표를 계산합니다...")
    analysis_results = df_for_analysis.groupby([cols['iteration'], 'format_type']).apply(calculate_metrics).reset_index()

    # 3. 결과 저장
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graded_output_path = output_dir / config['graded_output_filename']
    analysis_output_path = output_dir / config['analysis_output_filename']
    
    df_graded.to_csv(graded_output_path, index=False, encoding='utf-8-sig')
    analysis_results.to_csv(analysis_output_path, index=False, encoding='utf-8-sig')
    
    print(f"전체 채점 결과가 '{graded_output_path}'에 저장되었습니다.")
    print(f"분석 요약 결과가 '{analysis_output_path}'에 저장되었습니다.")
    print("\n[분석 요약]\n", analysis_results.to_string())
    
    return df_graded, analysis_results

def verify_hypotheses(df_graded: pd.DataFrame, df_analysis: pd.DataFrame, config: Dict[str, Any], output_path: Path):
    """
    분석된 데이터를 바탕으로 통계적 가설 검증을 수행하고 결과를 파일에 저장합니다.
    자유도, 평균, 표준편차, 신뢰구간을 포함한 상세 결과를 출력합니다.
    """
    results_log = []
    def log_and_print(message: str):
        print(message)
        results_log.append(message)

    log_and_print("\n--- 4단계: 가설 검증 (통계 분석) 시작 ---")
    cols = config['column_names']

    # 가설 1: 객관식 문제에서 ACC와 LRA 간의 유의미한 차이 (대응표본 t-검정)
    log_and_print("\n[가설 1] 객관식 ACC vs LRA (대응표본 t-검정)")
    choice_results = df_analysis[df_analysis['format_type'] == 'choice']
    if not choice_results.empty and 'ACC' in choice_results and 'LRA' in choice_results and len(choice_results) > 1:
        acc_series = choice_results['ACC']
        lra_series = choice_results['LRA']
        
        log_and_print(f"  - ACC: Mean={acc_series.mean():.4f}, Std={acc_series.std():.4f}")
        log_and_print(f"  - LRA: Mean={lra_series.mean():.4f}, Std={lra_series.std():.4f}")
        
        result = stats.ttest_rel(acc_series, lra_series)
        ci = result.confidence_interval(confidence_level=0.95)
        
        log_and_print(f"  - 결과: t({result.df:.0f}) = {result.statistic:.4f}, p = {result.pvalue:.4f}")
        log_and_print(f"  - 95% CI of difference: [{ci.low:.4f}, {ci.high:.4f}]")
    else:
        log_and_print("  - 검증에 필요한 데이터가 부족합니다.")

    # 가설 2a: 동일 문제에 대한 객관식(choice_paired)과 OX 문제 간 차이 (독립표본 t-검정)
    log_and_print("\n[가설 2a] 동일 문제에 대한 객관식 vs OX (독립표본 t-검정)")
    choice_paired_metrics = df_analysis[df_analysis['format_type'] == 'choice_paired']
    ox_metrics = df_analysis[df_analysis['format_type'] == 'ox']
    
    if not choice_paired_metrics.empty and not ox_metrics.empty:
        for metric in ['ACC', 'LRA', 'FLR']:
            log_and_print(f"\n  * '{metric}' 지표 비교:")
            group1 = choice_paired_metrics[metric]
            group2 = ox_metrics[metric]
            
            log_and_print(f"    - choice_paired: Mean={group1.mean():.4f}, Std={group1.std():.4f}, n={len(group1)}")
            log_and_print(f"    - ox:            Mean={group2.mean():.4f}, Std={group2.std():.4f}, n={len(group2)}")

            if len(group1) > 1 and len(group2) > 1:
                result = stats.ttest_ind(group1, group2, equal_var=False) # Welch's t-test
                ci = result.confidence_interval(confidence_level=0.95)
                log_and_print(f"    - 결과: t({result.df:.2f}) = {result.statistic:.4f}, p = {result.pvalue:.4f}")
                log_and_print(f"    - 95% CI of difference: [{ci.low:.4f}, {ci.high:.4f}]")
            else:
                log_and_print("    - 검증에 필요한 데이터가 부족합니다.")
    else:
        log_and_print("  - 검증에 필요한 데이터가 부족합니다 (choice_paired 또는 OX 데이터 없음).")

    # 가설 2b: 문제 유형(cls_2) 간 ACC, LRA, FLR 차이 (독립표본 t-검정)
    q_type_col = cols.get('question_type')
    log_and_print(f"\n[가설 2b] 문제 유형별 비교 ('{q_type_col}' 컬럼 기준, 독립표본 t-검정)")
    if q_type_col and q_type_col in df_graded.columns:
        # 문제 유형별로 평가지표 다시 계산
        type_analysis = df_graded.groupby([cols['iteration'], q_type_col]).apply(
            lambda df_group: pd.Series({
                'ACC': df_group[cols['is_correct']].mean(),
                'LRA': ((df_group[cols['is_correct']] == 1) & (df_group[cols['is_lra_correct']] == 1)).sum() / len(df_group) if len(df_group) > 0 else 0,
                'FLR': (lambda acc, lra: (acc - lra) / acc if acc > 0 else 0)(
                    df_group[cols['is_correct']].mean(),
                    ((df_group[cols['is_correct']] == 1) & (df_group[cols['is_lra_correct']] == 1)).sum() / len(df_group) if len(df_group) > 0 else 0
                )
            })
        ).reset_index()
        
        # 비교할 두 그룹을 명시적으로 지정
        type1_name = '문장형'
        type2_name = '사진_일러스트형'
        
        available_types = type_analysis[q_type_col].unique()

        if type1_name in available_types and type2_name in available_types:
            log_and_print(f"  - 비교 그룹: '{type1_name}' vs '{type2_name}'")
            
            for metric in ['ACC', 'LRA', 'FLR']:
                log_and_print(f"\n  * '{metric}' 지표 비교:")
                group1 = type_analysis[type_analysis[q_type_col] == type1_name][metric]
                group2 = type_analysis[type_analysis[q_type_col] == type2_name][metric]

                log_and_print(f"    - {type1_name}: Mean={group1.mean():.4f}, Std={group1.std():.4f}, n={len(group1)}")
                log_and_print(f"    - {type2_name}: Mean={group2.mean():.4f}, Std={group2.std():.4f}, n={len(group2)}")

                if len(group1) > 1 and len(group2) > 1:
                    result = stats.ttest_ind(group1, group2, equal_var=False)
                    ci = result.confidence_interval(confidence_level=0.95)
                    log_and_print(f"    - 결과: t({result.df:.2f}) = {result.statistic:.4f}, p = {result.pvalue:.4f}")
                    log_and_print(f"    - 95% CI of difference: [{ci.low:.4f}, {ci.high:.4f}]")
                else:
                    log_and_print("    - 검증에 필요한 데이터가 부족합니다.")
        else:
            log_and_print(f"  - 검증에 필요한 그룹('{type1_name}', '{type2_name}')이 데이터에 모두 존재하지 않습니다. 발견된 유형: {list(available_types)}")
    else:
        log_and_print(f"  - 검증에 필요한 '{q_type_col}' 컬럼이 데이터에 없습니다.")
    
    log_and_print("\n통계 분석이 완료되었습니다.")

    # 결과 파일로 저장
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results_log))
        print(f"통계 검증 결과가 '{output_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"에러: 통계 검증 결과를 파일에 저장하는 중 오류가 발생했습니다 - {e}")


async def main():
    """메인 실행 함수 (비동기)"""
    config = load_config()
    
    # 1. 데이터 로드
    df_list = []
    for filename in config['input_files']:
        input_path = os.path.join(config['input_dir'], filename)
        print(f"\n'{input_path}' 파일 로드 중...")
        df_list.append(pd.read_csv(input_path))
    df_raw = pd.concat(df_list, ignore_index=True)

    # 2. 파이프라인 실행
    llm_client = LLMClient(model_name=config['judge_model_name'])
    
    df_parsed = parse_llm_responses(df_raw, config)
    df_graded = await grade_responses(df_parsed, config, llm_client)
    df_graded_final, df_analysis = analyze_and_save_results(df_graded, config)
    
    # 통계 분석 결과 저장 경로 설정 및 함수 호출
    output_dir = Path(config['output_dir'])
    # config 파일에 키가 없을 경우를 대비해 기본값('stats_analysis_results.txt') 사용
    stats_filename = config.get('stats_output_filename', 'stats_analysis_results.txt')
    stats_output_path = output_dir / stats_filename
    verify_hypotheses(df_graded_final, df_analysis, config, stats_output_path)


if __name__ == '__main__':
    # Windows에서 asyncio 이벤트 루프 정책 설정
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
