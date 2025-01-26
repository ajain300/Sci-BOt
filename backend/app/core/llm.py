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
            "min": number | "n/a",
            "max": number | "n/a",
            "type": "continuous" | "discrete" | "derived",
            "derived_from": "optional string explaining how this parameter is derived"
        },
        ...
    },
    "objective": "string describing the objective function",
    "objective_variable": "string name of the variable being optimized",
    "constraints": ["optional array of constraint strings"]
}

Important rules:
1. Include ALL variables mentioned in constraints or the problem description in parameters
2. For derived parameters (like those determined by constraints), use:
   - type: "derived"
   - min: "n/a"
   - max: "n/a"
   - derived_from: explanation of how it's derived

Example:
{
    "parameters": {
        "material_a_concentration": {
            "min": 15,
            "max": 35,
            "type": "continuous"
        },
        "material_b_concentration": {
            "min": 15,
            "max": 35,
            "type": "continuous"
        },
        "material_c_concentration": {
            "min": "n/a",
            "max": "n/a",
            "type": "derived",
            "derived_from": "100 - material_a_concentration - material_b_concentration"
        },
        "temperature": {
            "min": 35,
            "max": 400,
            "type": "continuous"
        }
    },
    "objective": "maximize Young's modulus",
    "objective_variable": "Young's modulus",
    "constraints": ["material_a_concentration + material_b_concentration + material_c_concentration = 100"]
}

Format the response as valid JSON matching this schema exactly."""

async def generate_config(prompt: str) -> OptimizationConfig:
    """Generate optimization configuration from natural language prompt using LLM."""
    try:
        logger.info(f"Generating config for prompt: {prompt}")
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        response = await client.chat.completions.create(
            model="gpt-4o",
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