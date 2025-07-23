import os
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio

# analysis.py가 있는 scripts 폴더에서 두 단계 상위인 프로젝트 루트를 기준으로 .env를 찾습니다.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

class LLMClient:
    """Google Gemini API를 사용하기 위한 클라이언트 클래스"""
    def __init__(self, model_name: str):
        """
        LLMClient를 초기화합니다.
        API 키는 환경 변수 'GOOGLE_API_KEY'에서 로드합니다.

        Args:
            model_name (str): 사용할 LLM 모델의 이름.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        # 일관된 출력을 위한 생성 설정
        self.generation_config = genai.GenerationConfig(
            temperature=0
        )

    def generate(self, prompt: str) -> str:
        """
        주어진 프롬프트를 사용하여 LLM으로부터 텍스트 응답을 생성합니다. (동기 방식)
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            print(f"Google Gemini API 호출 중 오류 발생: {e}")
            return ""

    async def generate_async(self, prompt: str) -> str:
        """
        주어진 프롬프트를 사용하여 LLM으로부터 텍스트 응답을 비동기적으로 생성합니다.

        Args:
            prompt (str): LLM에 전달할 프롬프트.

        Returns:
            str: LLM이 생성한 텍스트 응답. 실패 시 빈 문자열을 반환합니다.
        """
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            print(f"Google Gemini API 비동기 호출 중 오류 발생: {e}")
            return ""