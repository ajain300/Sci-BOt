from openai import AsyncOpenAI
from typing import Dict, Any
import json
import os
import logging
from ..schemas.optimization import OptimizationConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are an expert in scientific optimization and experimental design. 
Your task is to convert natural language descriptions of optimization problems into structured JSON configurations.
The configuration must follow this exact schema:

{
    "parameters": {
        "parameter_name": {
            "min": number,
            "max": number,
            "type": "continuous" | "discrete"
        },
        ...
    },
    "objective": "string describing the objective function",
    "constraints": ["optional array of constraint strings"]
}

Example:
{
    "parameters": {
        "temperature": {
            "min": 20,
            "max": 100,
            "type": "continuous"
        },
        "pressure": {
            "min": 1,
            "max": 10,
            "type": "continuous"
        }
    },
    "objective": "maximize reaction yield",
    "constraints": ["temperature must not exceed pressure * 10"]
}

Format the response as valid JSON matching this schema exactly."""

async def generate_config(prompt: str) -> OptimizationConfig:
    """Generate optimization configuration from natural language prompt using LLM."""
    try:
        logger.info(f"Generating config for prompt: {prompt}")
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        response = await client.chat.completions.create(
            model="gpt-4o-latest",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        logger.info(f"Received response: {content}")
        
        config_dict = json.loads(content)
        return OptimizationConfig(**config_dict)
        
    except Exception as e:
        logger.error(f"Error generating configuration: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to generate configuration: {str(e)}") 