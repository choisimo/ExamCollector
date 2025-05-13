import os
import json
import openai
from typing import Dict, List
from infrastructure.config.settings import AI_AGENT_CONFIG_DIR

class QuestionTaggerService:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    async def tag(self, clusters: Dict[int, List[str]], agent_name: str = "default", retry: bool = False) -> Dict[int, Dict[str, List[str]]]:
        # Load AI agent configuration
        config_file = os.path.join(AI_AGENT_CONFIG_DIR, f"{agent_name}.json")
        if not os.path.exists(config_file):
            config_file = os.path.join(AI_AGENT_CONFIG_DIR, "default.json")
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        model = config.get("model")
        system_prompt = config.get("system_prompt")
        instructions = config.get("instructions")
        temperature = config.get("temperature", 0.3)
        max_tokens = config.get("max_tokens", 500)
        result: Dict[int, Dict[str, List[str]]] = {}
        for cid, questions in clusters.items():
            prompt = f"{instructions}\n{json.dumps(questions, ensure_ascii=False)}"
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            try:
                tags = json.loads(content)
            except json.JSONDecodeError:
                tags = {}
            result[cid] = tags
        return result
