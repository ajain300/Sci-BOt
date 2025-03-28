from typing import List, Dict
from backend.app.schemas.optimization import OptimizationConfig, DataPoint
from backend.app.schemas.feature_schemas import *

def convert_config_to_dict(config: OptimizationConfig) -> Dict:
    """Convert OptimizationConfig to dictionary format."""
    features_list = []
    
    for feature in config.features:
        if isinstance(feature, CompositionFeatureConfig):
            feature_dict = {
                "name": feature.name,
                "type": "composition",
                "columns": {
                    "parts": feature.columns.parts,
                    "range": feature.columns.range
                },
                "scaling": feature.scaling
            }
            
        elif isinstance(feature, ContinuousFeatureConfig):
            feature_dict = {
                "name": feature.name,
                "type": "continuous",
                "min": feature.min,
                "max": feature.max,
                "scaling": getattr(feature, 'scaling', 'lin')
            }
            
        elif isinstance(feature, DiscreteFeatureConfig):
            feature_dict = {
                "name": feature.name,
                "type": "discrete",
                "categories": feature.categories
            }
            
        features_list.append(feature_dict)
    
    return {
        "features": features_list,
        "objectives": [
            {
                "name": obj.name,
                "direction": obj.direction,
                "weight": obj.weight
            } for obj in config.objectives
        ],
        "acquisition_function": config.acquisition_function,
        "constraints": config.constraints if hasattr(config, 'constraints') else []
    }

def convert_data_points_to_dict(data_points: List[DataPoint]) -> List[dict]:
    """
    Convert DataPoint objects to JSON-serializable dictionaries.
    
    Args:
        data_points: List of DataPoint objects
        
    Returns:
        List[dict]: List of JSON-serializable dictionaries
    """
    return [
        {
            "parameters": {
                k: float(v) for k, v in point.parameters.items()  # Ensure all values are float
            },
            "objective_values": point.objective_values if hasattr(point, 'objective_values') else {
                "reaction_yield": point.objective_value,
                "cost": None
            } if point.objective_value is not None else {
                "reaction_yield": None,
                "cost": None
            }
        }
        for point in data_points
    ]