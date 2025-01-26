from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any

class ConfigGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Natural language prompt describing the optimization problem")
    
class OptimizationConfig(BaseModel):
    parameters: Dict[str, Dict[str, Union[float, str, Dict[str, Any]]]] = Field(
        ..., 
        description="Parameter space configuration including derived parameters"
    )
    objective: str = Field(..., description="Objective function to optimize")
    objective_variable: str = Field(..., description="Name of the variable being optimized")
    constraints: Optional[List[str]] = Field(default=None, description="Optional constraints")
    
class DataPoint(BaseModel):
    parameters: Dict[str, float] = Field(..., description="Parameter values")
    objective_value: Optional[float] = Field(default=None, description="Measured objective value")
    
class ActiveLearningRequest(BaseModel):
    config: OptimizationConfig
    data: List[DataPoint] = Field(default_list=[], description="Historical data points")
    n_suggestions: int = Field(default=1, description="Number of parameter suggestions to generate")
    
class ActiveLearningResponse(BaseModel):
    suggestions: List[Dict[str, float]] = Field(..., description="Suggested parameter values to try")
    expected_improvements: List[float] = Field(..., description="Expected improvement for each suggestion")
    
class ProcessDataRequest(BaseModel):
    config: OptimizationConfig
    data: List[DataPoint]
    
class ProcessDataResponse(BaseModel):
    statistics: Dict[str, Any] = Field(..., description="Statistical analysis of the data")
    best_point: DataPoint = Field(..., description="Best performing point found")
    parameter_importance: Dict[str, float] = Field(..., description="Relative importance of each parameter") 