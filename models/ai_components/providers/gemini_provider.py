# models/LLM/providers/gemini_provider.py
import google.generativeai as genai
from typing import Optional
from ..common.app_logger import AppLogger

class GeminiProvider:
    def __init__(self, api_key: str):
        self.logger = AppLogger()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def generate(self, prompt: str) -> Optional[str]:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.log_exception(f"Gemini API Error: {str(e)}")
            return None