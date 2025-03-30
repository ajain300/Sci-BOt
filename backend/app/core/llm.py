from openai import AsyncOpenAI
from typing import Dict, Any
import json
import os
import logging
from ..schemas.optimization import OptimizationConfig, ObjectiveConfig
from ..schemas.feature_schemas import *
from ..schemas.target_schemas import *
from ..core.config_generator.parser import ConfigurationParser
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are an expert in scientific optimization and experimental design. 
Your task is to convert natural language descriptions of optimization problems into structured JSON configurations.
The configuration must follow this exact schema:

"features": {
    "feature_name": {
        "min": number | "n/a",
        "max": number | "n/a",
        "type": "composition" | "continuous" | "discrete",
        "derived_from": "optional string explaining how this parameter is derived, based on the context of the problem"
    },
    ...
}     "name": "feature_name",
            "type": "continuous",
            "min": number,
            "max": number,
            "scaling": "lin"
        },
        {
            "name": "another_feature",
            "type": "discrete",
            "categories": ["option1", "option2", "option3"]
        },
        {
            "name": "composition_feature",
            "type": "composition",
            "columns": {
                "parts": ["component1", "component2", "component3"],
                "range": {
                    "component1": [min_value, max_value],
                    "component2": [min_value, max_value],
                    "component3": [min_value, max_value]
                }
            },
            "scaling": "lin"
        }
    ],
    "objectives": [
        {
            "name": "string describing the objective function",
            "direction": "maximize" | "minimize" | "target" | "range",
            "weight": number,
            "target_value": number (optional, for target optimization),
            "range_min": number (optional, for range optimization),
            "range_max": number (optional, for range optimization),
            "unit": string (optional),
            "scaling": string (optional)
        }
    ],
    "acquisition_function": "diversity_uncertainty" | "best_score",
    "constraints": ["constraint1", "constraint2"]
}


Important rules:
1. Features must be a LIST of objects, not a dictionary
2. Each feature must have a "name" and "type" field
3. For continuous features, include "min" and "max" values
4. For discrete features, include a "categories" array
5. For composition features, include "columns" with "parts" and "range"
6. All objectives must have a "name" and "direction" field
7. Determine the acquisition function based on the problem, if the goal is exploration, use "diversity_uncertainty", if the goal is to optimize, use "best_score".

Example:
{
    "features": [
        { 
            "name": "material_composition",
            "type": "composition",
            "columns": {
                "parts": [
                    "material_a_concentration",
                    "material_b_concentration",
                    "material_c_concentration"
                ],
                "range": {
                    "material_a_concentration": [15.0, 35.0],
                    "material_b_concentration": [15.0, 35.0],
                    "material_c_concentration": [30.0, 70.0]
                }
            },
            "scaling": "lin"
        },
        {
            "name": "temperature",
            "type": "continuous",
            "min": 20.0,
            "max": 100.0,
            "scaling": "lin"
        },
        {
            "name": "pressure",
            "type": "continuous",
            "min": 1.0,
            "max": 10.0,
            "scaling": "lin"
        }
    ],
    "objectives": [
        {
            "name": "reaction_yield",
            "direction": "maximize",
            "weight": 1.0
        },
        {
            "name": "cost",
            "direction": "minimize",
            "weight": 1.0
        }
    ],
    "acquisition_function": "diversity_uncertainty",
    "constraints": [
        "material_a_concentration + material_b_concentration + material_c_concentration = 100"
    ]
}

Format the response as valid JSON matching this schema exactly."""

async def generate_config(prompt: str) -> OptimizationConfig:
    """Generate optimization configuration from natural language prompt using LLM."""
    try:
        logger.info(f"Generating config for prompt: {prompt}")
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        logger.info(f"Received response: {content}")
        
        raw_config = json.loads(content)
        
        # Use ConfigurationParser to parse features and objectives
        parser = ConfigurationParser()
        features = parser.parse_llm_features(raw_config["features"])
        objectives = parser.parse_llm_objectives(raw_config["objectives"])
        
        # Construct the final config
        config = OptimizationConfig(
            features=features,
            objectives=objectives,
            acquisition_function=raw_config.get("acquisition_function", "diversity_uncertainty"),
            constraints=raw_config.get("constraints", [])
        )
        
        return config
        
    except Exception as e:
        logger.error(f"Error generating configuration: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to generate configuration: {str(e)}")