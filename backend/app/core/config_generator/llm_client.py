from typing import Dict, Any
from openai import AsyncClient
import logging
import json

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = AsyncClient(api_key=api_key)
        self.model = model

    async def generate_json_response(self, system_prompt: str, user_prompt: str) -> Dict[Any, Any]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise 