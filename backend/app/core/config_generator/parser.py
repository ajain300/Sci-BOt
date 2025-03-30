from typing import Dict, Any, List, Union
from ...schemas.feature_schemas import (
    CompositionFeatureConfig, 
    ContinuousFeatureConfig, 
    DiscreteFeatureConfig,
    CompositionColumns,
    FeatureBase
)
from ...schemas.target_schemas import (
    UncertaintyTargetConfig,
    ExtremeTargetConfig,
    ValueTargetConfig,
    RangeTargetConfig,
    TargetBase
)
from ...schemas.optimization import ObjectiveConfig
import logging

logger = logging.getLogger(__name__)

class ConfigurationParser:
    @staticmethod
    def parse_feature(feature_config: Dict[str, Any]):
        """Parse a single feature configuration using existing schemas"""
        feature_type = feature_config["type"]
        
        try:
            if feature_type == "composition":
                return CompositionFeatureConfig(
                    name=feature_config["name"],
                    type=feature_type,
                    columns=CompositionColumns(
                        parts=feature_config["columns"]["parts"],
                        range=feature_config["columns"]["range"]
                    ),
                    scaling=feature_config.get("scaling", "lin")
                )
            elif feature_type == "continuous":
                return ContinuousFeatureConfig(
                    name=feature_config["name"],
                    type=feature_type,
                    min=feature_config["min"],
                    max=feature_config["max"]
                )
            elif feature_type == "discrete":
                return DiscreteFeatureConfig(
                    name=feature_config["name"],
                    type=feature_type,
                    categories=feature_config["categories"]
                )
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
        except Exception as e:
            logger.error(f"Failed to parse feature {feature_config.get('name')}: {str(e)}")
            raise

    @staticmethod
    def parse_target(target_config: Dict[str, Any]):
        """Parse a single target configuration using existing schemas"""
        target_type = target_config["type"]
        base_params = {
            "name": target_config["name"],
            "weight": target_config.get("weight", 1.0),
            "scaling": target_config.get("scaling"),
            "unit": target_config.get("unit")
        }
        
        try:
            if target_type == "uncertainty":
                return UncertaintyTargetConfig(
                    **base_params,
                    direction=target_config["direction"]
                )
            elif target_type == "extreme":
                return ExtremeTargetConfig(
                    **base_params,
                    direction=target_config["direction"]
                )
            elif target_type == "target":
                return ValueTargetConfig(
                    **base_params,
                    target_value=target_config["target_value"]
                )
            elif target_type == "range":
                return RangeTargetConfig(
                    **base_params,
                    range_min=target_config["range_min"],
                    range_max=target_config["range_max"]
                )
            else:
                raise ValueError(f"Unknown target type: {target_type}")
        except Exception as e:
            logger.error(f"Failed to parse target {target_config.get('name')}: {str(e)}")
            raise

    @staticmethod
    def parse_llm_features(features_input: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]) -> List[FeatureBase]:
        """
        Parse features from LLM output to feature objects.
        Can handle both list format and dictionary format.
        """
        features = []
        
        # Check if features_input is a list (new format) or dict (old format)
        if isinstance(features_input, list):
            # List format - process each feature object directly
            for feature_config in features_input:
                try:
                    # Handle continuous feature values that might need conversion
                    if feature_config["type"] == "continuous":
                        if isinstance(feature_config["min"], str) and feature_config["min"] == "n/a":
                            feature_config["min"] = 0
                        if isinstance(feature_config["max"], str) and feature_config["max"] == "n/a":
                            feature_config["max"] = 100
                        
                        # Convert to float if needed
                        if isinstance(feature_config["min"], (int, str)):
                            feature_config["min"] = float(feature_config["min"])
                        if isinstance(feature_config["max"], (int, str)):
                            feature_config["max"] = float(feature_config["max"])
                    
                    feature = ConfigurationParser.parse_feature(feature_config)
                    features.append(feature)
                except Exception as e:
                    logger.error(f"Error parsing feature {feature_config.get('name')}: {str(e)}")
                    # Optionally skip invalid features instead of failing
                    continue
        else:
            # Dictionary format (legacy) - convert to list format first
            for name, feature_data in features_input.items():
                # Create a feature config with name included
                feature_config = {"name": name, **feature_data}
                
                # Handle 'n/a' values in min/max
                if feature_config["type"] == "continuous":
                    if isinstance(feature_config["min"], str) and feature_config["min"] == "n/a":
                        feature_config["min"] = 0
                    if isinstance(feature_config["max"], str) and feature_config["max"] == "n/a":
                        feature_config["max"] = 100
                    
                    # Convert to float if needed
                    if isinstance(feature_config["min"], (int, str)):
                        feature_config["min"] = float(feature_config["min"])
                    if isinstance(feature_config["max"], (int, str)):
                        feature_config["max"] = float(feature_config["max"])
                
                try:
                    feature = ConfigurationParser.parse_feature(feature_config)
                    features.append(feature)
                except Exception as e:
                    logger.error(f"Error parsing feature {name}: {str(e)}")
                    # Optionally skip invalid features instead of failing
                    continue
                
        return features

    @staticmethod
    def parse_llm_objectives(objectives_list: List[Dict[str, Any]]) -> List[ObjectiveConfig]:
        """Parse objectives from LLM output to ObjectiveConfig objects"""
        objectives = []
        
        for objective in objectives_list:
            # Don't modify the direction - it's already in the correct format
            obj_config = ObjectiveConfig(
                name=objective["name"],
                direction=objective["direction"],  # Keep original direction
                weight=objective.get("weight", 1.0),
                type=objective.get("type", "extreme"),
                target_value=objective.get("target_value"),
                range_min=objective.get("range_min"),
                range_max=objective.get("range_max"),
                unit=objective.get("unit"),
                scaling=objective.get("scaling", "lin")  # Add default scaling
            )
            
            objectives.append(obj_config)
                
        return objectives

    def parse_features(self, raw_features: List[Dict[str, Any]]) -> List[FeatureBase]:
        features = []
        for feature_config in raw_features:
            features.append(self.parse_feature(feature_config))
        return features

    def parse_targets(self, raw_targets: List[Dict[str, Any]]) -> List[TargetBase]:
        targets = []
        for target_config in raw_targets:
            targets.append(self.parse_target(target_config))
        return targets 