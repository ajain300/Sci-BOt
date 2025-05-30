from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional

class FeatureBase(BaseModel):
    name: str
    type: str

class CompositionColumns(BaseModel):
    parts: List[str] = Field(
        ...,
        description="List of component names in the composition"
    )
    range: Dict[str, List[float]] = Field(
        ...,
        description="Dictionary mapping each component to its allowed range [min, max]"
    )

class CompositionFeatureConfig(FeatureBase):
    type: str = "composition"
    columns: CompositionColumns = Field(
        ...,
        description="Composition feature configuration"
    )
    scaling: Optional[str] = Field(
        default="lin",
        description="Scaling type for the composition values"
    )
class ContinuousFeatureConfig(FeatureBase):
    type: str = "continuous"
    min: float
    max: float
    scaling: Optional[str] = Field(
        default="lin",
        description="Scaling type for the continuous values"
    )

class DiscreteFeatureConfig(FeatureBase):
    type: str = "discrete"
    categories: List[Union[str, int, float]] 