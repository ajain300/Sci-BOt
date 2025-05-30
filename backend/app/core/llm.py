from openai import AsyncOpenAI
from typing import Dict, Any
import json
import os
import logging
from ..schemas.optimization import OptimizationConfig, ObjectiveConfig
from ..schemas.feature_schemas import *
from ..schemas.target_schemas import *
from ..schemas.optimization import DataPoint
from ..core.config_generator.parser import ConfigurationParser
from ..core.config_generator.prompts import CONFIG_GENERATOR_SYSTEM_PROMPT, ANALYSIS_SYSTEM_PROMPT
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variable to store raw prompt
raw_prompt = None

async def generate_config(prompt: str) -> OptimizationConfig:
    """Generate optimization configuration from natural language prompt using LLM."""
    try:
        logger.info(f"Generating config for prompt: {prompt}")
        
        # Save raw prompt to global variable
        global raw_prompt
        raw_prompt = prompt
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CONFIG_GENERATOR_SYSTEM_PROMPT},
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
    
    
async def evaluate_active_learning_result(config: OptimizationConfig, 
                                          data: List[DataPoint],
                                          suggestions: List[Dict[str, float | str]],
                                          scores: List[float]
                                          ) -> Dict[str, str]:
    """Provide the suggested active learning results to the LLM, and ask it to evaluate the results.
    Go on to suggest any edits to the configuration, or any issues.
    """
    try:
        logger.info(f"Evaluating active learning results for config: {config}")
        
        # Use ConfigurationParser to parse features and objectives
        parser = ConfigurationParser()
        
        # Convert config to JSON string
        config_json = json.dumps(config.model_dump())
        
        # convert  to JSON string
        data_json = json.dumps([data_point.model_dump() for data_point in data])
        
        # combine the suggestions and scores into a list of tuples of dictionaries, having the form:
        # [
        #     {
        #         "suggestion": {"param1": value1, "param2": value2, ...},
        #         "predictions": {"objective1": value1, "objective2": value2, ...}
        #     },
        # ]
        suggestions_and_scores = [
            {
                "suggestion": suggestion,
                "predictions": scores[i]
            }
            for i, suggestion in enumerate(suggestions)
        ]
        
        # convert suggestions to JSON string
        suggestions_json = json.dumps(suggestions_and_scores)
        
        # Create a prompt for the LLM
        prompt = f"""
        Here is the raw prompt describing the optimization problem:
        {raw_prompt}
        
        Here is the configuration:
        {config_json}
        
        Here is the prior data:
        {data_json}
        
        Here are the suggestions from Bayesian Optimization, from which we need to rank the best suggestions:
        {suggestions_json}
        
        All compositions satisfy constraints, so don't check for that.
        """
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        logger.info(f"Received response: {content}")
        
        # parse the response
        analysis_result = json.loads(content)
        
        # sort the suggestions by rank
        analysis_result["suggestions"].sort(key=lambda x: x["rank"])
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error evaluating active learning results: {str(e)}", exc_info=True)