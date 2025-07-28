import os
import sys
import pandas as pd
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import json
from tqdm.asyncio import tqdm as aio_tqdm
import time
import asyncio
import random
from typing import Dict, List, Optional, Any

# --- 경로 설정 ---
# prompts.py를 import하기 위해 프로젝트 루트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from code_work.scripts.prompts import ox_transform_prompt

# --- 설정 ---
# True로 설정하면 처음 10개 행만 테스트합니다.
# False로 설정하면 전체 데이터에 대해 작업을 수행합니다.
IS_TEST_MODE = False
INPUT_FILE_PATH = os.path.join(project_root, "data", "ox_question_processed.csv")
OUTPUT_FILE_PATH = os.path.join(project_root, "data", "ox_question_generated.csv")
# 동시에 처리할 최대 작업 수를 설정합니다 (API 속도 제한에 맞춰 조절)
MAX_CONCURRENT_TASKS = 20

def configure_api() -> Optional[genai.GenerativeModel]:
    """Gemini API 키를 환경 변수에서 로드하고 모델을 설정합니다.
    
    Returns:
        GenerativeModel 인스턴스 또는 None (설정 실패 시)
    """
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        genai.configure(api_key=api_key)
        print("Gemini API 키가 성공적으로 설정되었습니다.")
        return genai.GenerativeModel('gemini-2.5-pro')
    except Exception as e:
        print(f"API 설정 중 오류 발생: {e}")
        return None

async def generate_ox_pair_async(model: genai.GenerativeModel, mcqa_text: str, retries: int = 3, delay: int = 5) -> Dict[str, Any]:
    """
    Gemini API를 비동기적으로 호출하여 O/X 문제와 정답 쌍을 생성합니다.
    API 호출 실패 시 재시도 로직을 포함합니다.
    
    Args:
        model: Gemini 모델 인스턴스
        mcqa_text: 변환할 객관식 문제 텍스트
        retries: 재시도 횟수
        delay: 재시도 간 대기 시간 (초)
        
    Returns:
        생성된 O/X 문제와 답변을 포함한 딕셔너리
    """
    if not mcqa_text or pd.isna(mcqa_text):
        return {"ox_question": "", "ox_answer": "", "error": "Input is empty"}

    input_data_json = json.dumps({"mcqa": mcqa_text}, ensure_ascii=False)
    full_prompt = f"{ox_transform_prompt}\n\n## **입력 데이터**\n\n```json\n{input_data_json}\n```\n\n## **출력**"

    for attempt in range(retries):
        try:
            # 비동기 API 호출
            response = await model.generate_content_async(full_prompt)
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned_text)
            return result
        except ResourceExhausted as e:
            print(f"API 속도 제한 초과 (시도 {attempt + 1}/{retries}). 60초 후 재시도합니다.")
            if attempt < retries - 1:
                await asyncio.sleep(60)
            else:
                return {"ox_question": "", "ox_answer": "", "error": f"API rate limit: {e}"}
        except Exception as e:
            print(f"API 호출 또는 JSON 파싱 오류 (시도 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                return {"ox_question": "", "ox_answer": "", "error": str(e)}
    return {"ox_question": "", "ox_answer": "", "error": "Max retries reached"}

async def generate_ox_pair_with_target_answer(model: genai.GenerativeModel, mcqa_text: str, target_answer: str, retries: int = 3, delay: int = 5) -> Dict[str, Any]:
    """
    특정 답변(O 또는 X)을 목표로 하여 O/X 문제를 생성합니다.
    
    Args:
        model: Gemini 모델 인스턴스
        mcqa_text: 변환할 객관식 문제 텍스트
        target_answer: 목표 답변 ('O' 또는 'X')
        retries: 재시도 횟수
        delay: 재시도 간 대기 시간 (초)
        
    Returns:
        생성된 O/X 문제와 답변을 포함한 딕셔너리
    """
    if not mcqa_text or pd.isna(mcqa_text):
        return {"ox_question": "", "ox_answer": "", "error": "Input is empty"}

    # 프롬프트에 목표 답변을 명시적으로 추가
    input_data_json = json.dumps({
        "mcqa": mcqa_text,
        "target_answer": target_answer
    }, ensure_ascii=False)
    
    # 목표 답변을 지시하는 프롬프트 수정
    modified_prompt = f"""{ox_transform_prompt}

**중요**: 생성할 O/X 문제의 정답은 반드시 '{target_answer}'가 되어야 합니다. 
이를 위해 문제 진술을 적절히 조정하여 답이 '{target_answer}'가 되도록 만드세요.

## **입력 데이터**

```json
{input_data_json}
```

## **출력**"""

    for attempt in range(retries):
        try:
            response = await model.generate_content_async(modified_prompt)
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned_text)
            
            # 생성된 답변이 목표 답변과 일치하는지 확인
            if result.get('ox_answer') == target_answer:
                return result
            else:
                print(f"목표 답변({target_answer})과 불일치: {result.get('ox_answer')}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    continue
                else:
                    return {"ox_question": "", "ox_answer": "", "error": f"Failed to generate {target_answer} answer"}
        except ResourceExhausted as e:
            print(f"API 속도 제한 초과 (시도 {attempt + 1}/{retries}). 60초 후 재시도합니다.")
            if attempt < retries - 1:
                await asyncio.sleep(60)
            else:
                return {"ox_question": "", "ox_answer": "", "error": f"API rate limit: {e}"}
        except Exception as e:
            print(f"API 호출 또는 JSON 파싱 오류 (시도 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                return {"ox_question": "", "ox_answer": "", "error": str(e)}
    
    return {"ox_question": "", "ox_answer": "", "error": "Max retries reached"}

async def generate_balanced_ox_dataset(model: genai.GenerativeModel, target_df: pd.DataFrame, target_ratio: float = 0.5) -> List[Dict[str, Any]]:
    """
    O와 X 답변의 비율을 지정된 비율로 맞춰서 O/X 문제를 생성합니다.
    
    Args:
        model: Gemini 모델 인스턴스
        target_df: 처리할 데이터프레임
        target_ratio: O 답변의 목표 비율 (기본값: 0.5, 즉 50:50)
        
    Returns:
        생성된 결과 리스트
    """
    total_count = len(target_df)
    target_o_count = int(total_count * target_ratio)
    target_x_count = total_count - target_o_count
    
    print(f"목표 비율 - O: {target_o_count}개 ({target_ratio*100:.1f}%), X: {target_x_count}개 ({(1-target_ratio)*100:.1f}%)")
    
    # 세마포어 설정
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    # 각 문제에 대해 목표 답변을 할당
    target_answers = ['O'] * target_o_count + ['X'] * target_x_count
    
    # 랜덤하게 섞어서 순서 무작위화
    random.shuffle(target_answers)
    
    async def run_with_semaphore(mcqa_text: str, target_answer: str) -> Dict[str, Any]:
        async with semaphore:
            return await generate_ox_pair_with_target_answer(model, mcqa_text, target_answer)
    
    # 각 문제와 목표 답변을 매칭하여 태스크 생성
    tasks = []
    for idx, (_, row) in enumerate(target_df.iterrows()):
        if idx < len(target_answers):
            task = run_with_semaphore(row['mcqa'], target_answers[idx])
            tasks.append(task)
    
    print(f"API 요청을 최대 {MAX_CONCURRENT_TASKS}개씩 동시에 처리합니다...")
    results = await aio_tqdm.gather(*tasks)
    print("모든 API 요청 처리가 완료되었습니다.")
    
    # 결과 통계 출력
    o_generated = sum(1 for r in results if r.get('ox_answer') == 'O')
    x_generated = sum(1 for r in results if r.get('ox_answer') == 'X')
    errors = sum(1 for r in results if r.get('error'))
    
    print(f"생성 결과 - O: {o_generated}개, X: {x_generated}개, 오류: {errors}개")
    
    return results

async def main() -> None:
    """메인 실행 함수 (비동기)"""
    print("O/X 문제 생성 스크립트 (균형 조정 모드)를 시작합니다.")
    
    model = configure_api()
    if not model:
        return

    try:
        df = pd.read_csv(INPUT_FILE_PATH)
        print(f"성공적으로 '{INPUT_FILE_PATH}' 파일을 읽었습니다. (총 {len(df)}개 행)")
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{INPUT_FILE_PATH}'을(를) 찾을 수 없습니다.")
        return

    if IS_TEST_MODE:
        print(f"테스트 모드로 실행합니다. 처음 10개 행만 처리합니다.")
        target_df = df.head(10).copy()
    else:
        print(f"전체 모드로 실행합니다. {len(df)}개 행을 모두 처리합니다.")
        target_df = df.copy()

    # 균형 조정된 O/X 문제 생성 (50:50 비율)
    results = await generate_balanced_ox_dataset(model, target_df, target_ratio=0.5)

    # 생성된 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)

    # 원본 DataFrame에 새로운 열 추가 (순서가 보장됨)
    target_df.reset_index(drop=True, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    
    target_df['ox_question'] = results_df['ox_question']
    target_df['ox_answer'] = results_df['ox_answer']
    
    if 'error' in results_df.columns:
        target_df['error'] = results_df['error']

    try:
        target_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
        print(f"\n작업 완료! 균형 조정된 결과가 '{OUTPUT_FILE_PATH}'에 저장되었습니다.")
    except Exception as e:
        print(f"\n오류: 결과를 파일에 저장하는 중 문제가 발생했습니다. {e}")

if __name__ == "__main__":
    # 비동기 main 함수를 실행합니다.
    asyncio.run(main())