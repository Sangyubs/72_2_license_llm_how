import os
import sys
import pandas as pd
import google.generativeai as genai
import json
from tqdm.asyncio import tqdm as aio_tqdm
import time
import asyncio

# --- 경로 설정 ---
# prompts.py를 import하기 위해 프로젝트 루트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from code_work.scripts.prompts import ox_transform_prompt

# --- 설정 ---
# True로 설정하면 처음 5개 행만 테스트합니다.
# False로 설정하면 전체 데이터에 대해 작업을 수행합니다.
IS_TEST_MODE = False
INPUT_FILE_PATH = os.path.join(project_root, "data", "ox_question_processed.csv")
OUTPUT_FILE_PATH = os.path.join(project_root, "data", "ox_question_generated.csv")
# 동시에 처리할 최대 작업 수를 설정합니다 (API 속도 제한에 맞춰 조절)
MAX_CONCURRENT_TASKS = 50

def configure_api():
    """Gemini API 키를 환경 변수에서 로드하고 모델을 설정합니다."""
    try:
        # 환경 변수 이름을 "GEMINI_API_KEY"로 수정했습니다.
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        genai.configure(api_key=api_key)
        print("Gemini API 키가 성공적으로 설정되었습니다.")
        # 모델 이름을 "gemini-1.5-pro-latest"로 수정했습니다.
        return genai.GenerativeModel('gemini-2.5-pro')
    except Exception as e:
        print(f"API 설정 중 오류 발생: {e}")
        return None

async def generate_ox_pair_async(model, mcqa_text: str, retries=3, delay=5):
    """
    Gemini API를 비동기적으로 호출하여 O/X 문제와 정답 쌍을 생성합니다.
    API 호출 실패 시 재시도 로직을 포함합니다.
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
        except Exception as e:
            print(f"API 호출 또는 JSON 파싱 오류 (시도 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay) # 비동기 sleep
            else:
                return {"ox_question": "", "ox_answer": "", "error": str(e)}
    return {"ox_question": "", "ox_answer": "", "error": "Max retries reached"}

async def main():
    """메인 실행 함수 (비동기)"""
    print("O/X 문제 생성 스크립트 (비동기)를 시작합니다.")
    
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

    # 세마포어를 생성하여 동시 실행 작업 수를 제어합니다.
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def run_with_semaphore(mcqa_text):
        async with semaphore:
            return await generate_ox_pair_async(model, mcqa_text)

    # 각 행에 대한 비동기 작업을 생성합니다.
    tasks = [run_with_semaphore(row['mcqa']) for index, row in target_df.iterrows()]
    
    # aio_tqdm.gather를 사용하여 진행 상황을 표시하며 모든 작업을 동시에 실행합니다.
    # asyncio.gather는 입력된 작업의 순서를 보장하여 결과를 반환합니다.
    print(f"API 요청을 최대 {MAX_CONCURRENT_TASKS}개씩 동시에 처리합니다...")
    results = await aio_tqdm.gather(*tasks)
    print("모든 API 요청 처리가 완료되었습니다.")

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
        print(f"\n작업 완료! 결과가 '{OUTPUT_FILE_PATH}'에 저장되었습니다.")
    except Exception as e:
        print(f"\n오류: 결과를 파일에 저장하는 중 문제가 발생했습니다. {e}")

if __name__ == "__main__":
    # 비동기 main 함수를 실행합니다.
    asyncio.run(main())