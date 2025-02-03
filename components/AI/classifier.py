import json

import ollama


class ProblemClassifier:
    def __init__(self):
        print("문제 유형, 주제, 난이도 등을 분류하는 모델입니다.")
        # 시스템 프롬프트를 사용자 입력으로 설정 (예시 문제를 복사·붙여넣기)
        self.system_prompt = input("Enter the system prompt: (예시 문제를 복사 붙여넣기 해주세요) ")

        def classify(self, text):
            """
            Local AI model 을 호출하여 문제 텍스트 분류
            :param self: problem text
            :param text: problem text
            :return: json
            """
            try:
                response = ollama.chat(
                    model='mistral-small:24b',
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text}
                    ],
                    options={'temperature': 0.2}
                )
                llm_response = response.get('message', {}).get('content', '')
                return self._parse_response(llm_response)
            except Exception as e:
                print(f"Error classifying problem text: {e}")
                return {"type": "unknown", "topics": [], "difficulty": 3}


        def _parse_response(self, response):
            """
            AI 모델의 응답을 파싱하여 문제 유형, 주제, 난이도를 반환합니다.
            """
            # 문제 유형, 주제, 난이도를 분류하는 로직
            try:
                clean_response = response.strip("`").replace("json\n", "").strip()
                return json.loads(clean_response)
            except Exception as e:
                print(f"Error parsing AI response: {e}")
                return {"type": "unknown", "topics": [], "difficulty": 3}

