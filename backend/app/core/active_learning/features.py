from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class FeatureBase(BaseModel):
    name: str
    type: str

class CompositionFeatureConfig(FeatureBase):
    type: str = "composition"
    columns: Dict[str, Union[List[str], Dict[str, List[float]]]] = Field(
        ...,
        description="Dictionary containing parts list and their ranges"
    )
    scaling: str = Field(
        default="lin",
        description="Scaling type for the composition feature"
    )

class ContinuousFeatureConfig(FeatureBase):
    type: str = "continuous"
    min: float
    max: float
    scaling: str = Field(
        default="lin",
        description="Scaling type for the continuous feature"
    )

class DiscreteFeatureConfig(FeatureBase):
    type: str = "discrete"
    categories: List[Union[str, int, float]]

class Feature(ABC):
    def __init__(self, feature_config: Union[CompositionFeatureConfig, ContinuousFeatureConfig, DiscreteFeatureConfig]):
        self.name = feature_config.name
        self.feature_config = feature_config
        self.X_columns: List[str] = []

    @abstractmethod
    def process(self) -> None:
        pass

    def get_columns(self) -> List[str]:
        return self.X_columns

class CompositionFeature(Feature):
    def __init__(self, feature_config: CompositionFeatureConfig):
        super().__init__(feature_config)
        self.parts = feature_config.columns.parts
        self.ranges = feature_config.columns.range
        
    def process(self) -> None:
        """Process composition feature"""
        self.X_columns = self.parts
        # Validate ranges
        for part in self.parts:
            if part not in self.ranges:
                self.ranges[part] = [0, 100]
            elif not isinstance(self.ranges[part], list) or len(self.ranges[part]) != 2:
                raise ValueError(f"Invalid range for part {part}: {self.ranges[part]}")

class ContinuousFeature(Feature):
    def __init__(self, feature_config: ContinuousFeatureConfig):
        super().__init__(feature_config)
        self.min = feature_config.min
        self.max = feature_config.max
        self.scaling = feature_config.scaling
        
        logger.debug(f"Created ContinuousFeature: min={self.min}, max={self.max}, scaling={self.scaling}")
        
    def process(self) -> None:
        """Process continuous feature"""
        self.X_columns = [self.name]

class DiscreteFeature(Feature):
    def __init__(self, feature_config: DiscreteFeatureConfig):
        super().__init__(feature_config)
        self.categories = feature_config.categories
        
    def process(self) -> None:
        """Process discrete feature"""
        self.X_columns = [self.name]
    
    def add_OH_info(self, one_hot_columns: List[str]) -> None:
        self.one_hot_columns = {}
        for i, encoded_col in enumerate(one_hot_columns):
            self.one_hot_columns[encoded_col] = {
                'OH_encoding': [1 if i == j else 0 for j in range(len(one_hot_columns))]
            }

def create_feature(config: Union[Dict, BaseModel]) -> Feature:
    """Factory function to create appropriate feature object"""
    logger.debug(f"Creating feature with config: {config}")
    if isinstance(config, dict):
        if config["type"] == "composition":
            return CompositionFeature(CompositionFeatureConfig(**config))
        elif config["type"] == "continuous":
            return ContinuousFeature(ContinuousFeatureConfig(**config))
        elif config["type"] == "discrete":
            return DiscreteFeature(DiscreteFeatureConfig(**config))
    else:
        if config.type == "composition":
            return CompositionFeature(config)
        elif config.type == "continuous":
            return ContinuousFeature(config)
        elif config.type == "discrete":
            return DiscreteFeature(config)
    
    raise ValueError(f"Unknown feature type: {config['type']}")

def extract_parameter_columns(features: List[FeatureBase]) -> List[str]:
    """
    Extract parameter column names from a list of feature objects.
    
    Args:
        features: List of feature configuration objects
        
    Returns:
        List of parameter column names
        
    Example:
        >>> features = [
        ...     CompositionFeatureConfig(
        ...         name="composition",
        ...         columns={"parts": ["A", "B"], "range": {"A": [0, 100], "B": [0, 100]}}
        ...     ),
        ...     ContinuousFeatureConfig(name="temperature", min=0, max=100),
        ...     DiscreteFeatureConfig(name="catalyst", categories=["X", "Y", "Z"])
        ... ]
        >>> extract_parameter_columns(features)
        ['A', 'B', 'temperature', 'catalyst']
    """
    parameter_cols = []
    
    
    for feature in features:
        if feature.type == 'composition':
            parameter_cols.extend(feature.columns.parts)
        else:
            parameter_cols.append(feature.name)
    
    return parameter_cols