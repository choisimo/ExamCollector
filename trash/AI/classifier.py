import json

import ollama


class ProblemClassifier:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt or self._default_prompt()

    def _default_prompt(self):
        print("문제 유형 분류 시스템을 초기화합니다.")
        example = '''예시 문제:
        1. [수학] 이차방정식 x²-5x+6=0의 해를 구하시오
        2. [과학] 광합성 과정을 설명하시오
        3. [영어] 다음 문장을 영작하시오'''
        return input(f"시스템 프롬프트 입력 (예시:\n{example}\n): ")

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
        try:
            clean = response.strip("` \n").replace("json", "")
            result = json.loads(clean)
            return {
                "type": result.get("type", "unknown"),
                "topics": result.get("topics", []),
                "difficulty": result.get("difficulty", 3)
            }
        except Exception as e:
            print(f"응답 파싱 오류: {str(e)}")
            return {"type": "unknown", "topics": [], "difficulty": 3}



class EnhancedProblemClassifier(ProblemClassifier):
    def classify(self, text, context=None):
        enhanced_prompt = f"""
        다음 문제를 분석하세요. 이전 문제 유형은 {context['prev_type']}였고,
        현재 챕터는 {context['chapter']}입니다.

        문제: {text}
        """

        try:
            response = ollama.chat(
                model='mistral',
                messages=[
                    {"role": "system", "content": enhanced_prompt},
                    {"role": "user", "content": text}
                ]
            )
            return self._parse_response(response['message']['content'])
        except Exception as e:
            return self._safe_classification(text)

    def _safe_classification(self, text):
        # 기본 분류 규칙
        math_keywords = ['방정식', '함수', '기하']
        science_keywords = ['화학', '물리', '생물']

        return {
            "type": "수학" if any(k in text for k in math_keywords) else "과학",
            "topics": self._detect_topics(text),
            "difficulty": self._estimate_difficulty(text)
        }

    def _detect_topics(self, text):
        # 간단한 키워드 매칭
        topics = []
        if '확률' in text: topics.append('확률')
        if '통계' in text: topics.append('통계')
        return topics if topics else ['일반']

    def _estimate_difficulty(self, text):
        length = len(text)
        if length > 500: return 5
        if length > 300: return 4
        if length > 150: return 3
        return 2