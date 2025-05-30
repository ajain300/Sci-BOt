from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from .feature_schemas import CompositionFeatureConfig, ContinuousFeatureConfig, DiscreteFeatureConfig
## ENUM CLASSES TO DEFINE THE TYPES OF OBJECTIVES AND ACQUISITION FUNCTIONS
class OptimizationDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"
    RANGE = "range"

class AcquisitionFunction(str, Enum):
    EXPECTED_IMPROVEMENT = "expected_improvement"
    DIVERSITY_UNCERTAINTY = "diversity_uncertainty"
    BEST_SCORE = "best_score"
    COMBINED_SINGLE_EI = "combined_single_ei"

## BASE MODELS TO DEFINE THE STRUCTURE OF THE CONFIGURATION
class ConfigGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Natural language prompt describing the optimization problem")
    
class ObjectiveConfig(BaseModel):
    name: str = Field(..., description="Name of the objective")
    weight: float = Field(default=1.0, description="Weight of this objective in multi-objective optimization")
    direction: OptimizationDirection = Field(..., description="Type of optimization (maximize/minimize/target/range)")
    target_value: Optional[float] = Field(
        None, 
        description="Target value when direction is 'target'"
    )
    range_min: Optional[float] = Field(
        None, 
        description="Minimum acceptable value when direction is 'range'"
    )
    range_max: Optional[float] = Field(
        None, 
        description="Maximum acceptable value when direction is 'range'"
    )
    unit: Optional[str] = Field(
        None, 
        description="Unit of the objective"
    )
    scaling: str = Field(
        default="lin",
        description="Scaling type for the objective"
    )


    @field_validator('target_value')
    @classmethod
    def validate_target(cls, v: Optional[float], info) -> Optional[float]:
        if info.data.get('direction') == OptimizationDirection.TARGET and v is None:
            raise ValueError("target_value must be provided when direction is 'target'")
        return v

    @field_validator('range_min', 'range_max')
    @classmethod
    def validate_range(cls, v: Optional[float], info) -> Optional[float]:
        if info.data.get('direction') == OptimizationDirection.RANGE:
            if v is None:
                raise ValueError(f"{info.field_name} must be provided when direction is 'range'")
            if info.field_name == 'range_max' and info.data.get('range_min') is not None:
                if v <= info.data['range_min']:
                    raise ValueError("range_max must be greater than range_min")
        return v

class OptimizationConfig(BaseModel):
    features: List[Union[CompositionFeatureConfig, ContinuousFeatureConfig, DiscreteFeatureConfig]]
    objectives: List[ObjectiveConfig] = Field(
        ..., 
        description="List of objectives to optimize"
    )
    acquisition_function: AcquisitionFunction = Field(..., description="Acquisition function to use")
    constraints: Optional[List[str]] = Field(default=None, description="Optional constraints")
    
class DataPoint(BaseModel):
    parameters: Dict[str, float | str] = Field(..., description="Parameter values")
    objective_values: Dict[str, float] = Field(
        ...,  # Make it required instead of optional
        description="Measured objective values keyed by objective name"
    )
    
class ActiveLearningRequest(BaseModel):
    config: OptimizationConfig
    data: List[DataPoint] = Field(default_factory=list, description="Historical data points")
    n_suggestions: int = Field(default=1, description="Number of parameter suggestions to generate")
    
class Suggestion(BaseModel):
    rank: int = Field(..., description="Rank of this suggestion")
    suggestion: Dict[str, float | str] = Field(..., description="Suggested parameter values to try")
    predictions: List[Dict[str, float | str]] = Field(..., description="Predicted objective values")
    reason: str = Field(..., description="Explanation for this suggestion")

class SuggestionsResponse(BaseModel):
    suggestions: List[Suggestion] = Field(..., description="Ranked and explained suggestions")

class ActiveLearningResponse(BaseModel):
    suggestions: List[Dict[str, float | str]] = Field(..., description="Suggested parameter values to try")
    scores: List[List[Dict[str, float | str]]] = Field(..., description="Expected improvement for each suggestion")
    
class ProcessDataRequest(BaseModel):
    config: OptimizationConfig
    data: List[DataPoint]
    
class ProcessDataResponse(BaseModel):
    statistics: Dict[str, Any] = Field(..., description="Statistical analysis of the data")
    best_point: DataPoint = Field(..., description="Best performing point found")
    parameter_importance: Dict[str, float] = Field(..., description="Relative importance of each parameter")


class ConfigUpdateRequest(BaseModel):
    type: str
    id: str
    property: str
    value: float 