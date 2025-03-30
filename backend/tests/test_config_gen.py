import pytest
from fastapi.testclient import TestClient
import json
from backend.app.core.config_generator.parser import ConfigurationParser
from backend.app.schemas.optimization import OptimizationConfig
sample_llm_reponse = """
{
    "features": [
        {
            "name": "primary_alloy_composition",
            "type": "composition",
            "columns": {
                "parts": [
                    "Aluminum_concentration",
                    "Copper_concentration",
                    "Magnesium_concentration",
                    "Titanium_concentration",
                    "Vanadium_concentration",
                    "Nickel_concentration"
                ],
                "range": {
                    "Aluminum_concentration": [0, 100],
                    "Copper_concentration": [0, 100],
                    "Magnesium_concentration": [0, 100],
                    "Titanium_concentration": [0, 100],
                    "Vanadium_concentration": [0, 100],
                    "Nickel_concentration": [0, 100]
                }
            },
            "scaling": "lin"
        },
        {
            "name": "secondary_additions",
            "type": "composition",
            "columns": {
                "parts": [
                    "Scandium_concentration",
                    "Zirconium_concentration",
                    "Niobium_concentration"
                ],
                "range": {
                    "Scandium_concentration": [0, 10],
                    "Zirconium_concentration": [0, 10],
                    "Niobium_concentration": [0, 10]
                }
            },
            "scaling": "lin"
        }
    ],
    "objectives": [
        {
            "name": "material_stiffness",
            "direction": "maximize",
            "weight": 1.0
        }
    ],
    "acquisition_function": "diversity_uncertainty",
    "constraints": []
}
"""

def test_config_gen():
    
    raw_config = json.loads(sample_llm_reponse)
    
    # Use ConfigurationParser to parse features and objectives
    parser = ConfigurationParser()
    features = parser.parse_llm_features(raw_config["features"])
    objectives = parser.parse_llm_objectives(raw_config["objectives"])
    
    config = OptimizationConfig(
        features=features,
        objectives=objectives,
        acquisition_function=raw_config["acquisition_function"],
        constraints=[]
    )
    
    assert config is not None
    assert config.features is not None
    assert config.objectives is not None
    assert config.constraints is not None
