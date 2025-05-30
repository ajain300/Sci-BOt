from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from ...schemas.optimization import OptimizationDirection
from ...schemas.target_schemas import *
from .utils.basic_acquisition import Expected_Improvement

class Target(ABC):
    def __init__(
        self,
        config: Union[UncertaintyTargetConfig, ExtremeTargetConfig, ValueTargetConfig, RangeTargetConfig],
        X_columns: List[str],
        variable_info: Optional[Dict[str, Any]] = None,
        properties: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        initial_columns: Optional[List[str]] = None
    ):
        self.name = config.name
        self.type = config.type
        self.weight = config.weight
        self.scaling = config.scaling
        self.unit = config.unit
        self.direction = config.direction
        
        # Store target-specific values based on type
        if isinstance(config, ValueTargetConfig):
            self.target_value = config.target_value
        elif isinstance(config, RangeTargetConfig):
            self.range_min = config.range_min
            self.range_max = config.range_max
            
        # Store additional information
        self.config = config
        self.data = data
        self.X_columns = X_columns
        self.initial_columns = initial_columns if initial_columns else X_columns
        self.variable_info = variable_info if variable_info else {}
        self.properties = properties if properties else {}
        
        self._target_type_check()
        
    def _target_type_check(self):
        """Validate target configuration based on type."""
        if isinstance(self.config, ValueTargetConfig) and self.target_value is None:
            raise ValueError(f"Target value must be specified for target {self.name} with type 'target'")
            
        if isinstance(self.config, RangeTargetConfig):
            if self.range_min is None or self.range_max is None:
                raise ValueError(f"Range min and max must be specified for target {self.name} with type 'range'")
            if self.range_min >= self.range_max:
                raise ValueError(f"Range min must be less than range max for target {self.name}")

    @abstractmethod
    def process(self):
        pass

    def get_columns(self) -> List[str]:
        return self.X_columns
    
    def score(self, mean, std) -> float:
        if self.type == 'uncertainty':
            return score_uncertainty(std, self.direction)
        elif self.type == 'extreme':
            return score_extreme(mean, self.direction)
        elif self.type == 'target':
            return score_target(mean, self.target_value)
        elif self.type == 'range':
            return score_range(mean, [self.range_min, self.range_max])
        else:
            raise ValueError(f"Invalid target type '{self.type}'")

    def EI(self, mean, std, y_data) -> float:
        if self.type == 'uncertainty' or self.type == 'range':
            return self.score(mean, std)
        elif self.type == 'target':
            y_best = y_data[np.argmin(self.target_value - y_data)]
            return Expected_Improvement(mean, std, y_best, type = 'MIN')
        elif self.type == 'extreme':
            y_best = y_data[np.argmin(y_data)] if self.direction == 'MIN' else y_data[np.argmax(y_data)]
            return Expected_Improvement(mean, std, y_best, type = self.direction)

class UncertaintyTarget(Target):
    def process(self):
        for col in self.X_columns:
            self.variable_info[col] = {'scaling': self.properties['scaling']}

class ExtremeTarget(Target):
    def process(self):
        for col in self.X_columns:
            self.variable_info[col] = {'scaling': self.properties['scaling']}

class ValueTarget(Target):
    def process(self):
        for col in self.X_columns:
            self.variable_info[col] = {'scaling': self.properties['scaling']}

class RangeTarget(Target):
    def process(self):
        for col in self.X_columns:
            self.variable_info[col] = {'scaling': self.properties['scaling']}

def create_target(
    config: Union[Dict, BaseModel],
    X_columns: List[str],
    variable_info: Optional[Dict[str, Any]] = None,
    properties: Optional[Dict[str, Any]] = None
) -> Target:
    """Factory function to create appropriate target object"""
    if isinstance(config, dict):
        if config["type"] == "uncertainty":
            return UncertaintyTarget(UncertaintyTargetConfig(**config), X_columns, variable_info, properties)
        elif config["type"] == "extreme":
            return ExtremeTarget(ExtremeTargetConfig(**config), X_columns, variable_info, properties)
        elif config["type"] == "target":
            return ValueTarget(ValueTargetConfig(**config), X_columns, variable_info, properties)
        elif config["type"] == "range":
            return RangeTarget(RangeTargetConfig(**config), X_columns, variable_info, properties)
    else:
        if config.type == "uncertainty":
            return UncertaintyTarget(config, X_columns, variable_info, properties)
        elif config.type == "extreme":
            return ExtremeTarget(config, X_columns, variable_info, properties)
        elif config.type == "target":
            return ValueTarget(config, X_columns, variable_info, properties)
        elif config.type == "range":
            return RangeTarget(config, X_columns, variable_info, properties)
    
    raise ValueError(f"Unknown target type: {config['type']}")
            
# Helper functions for scoring

def score_uncertainty(values: float, goal: str) -> float:
    if goal == 'MIN':
        return (max(values) - values) / (max(values) - min(values))
    elif goal == 'MAX':
        return values / max(values)

def score_extreme(values: float, goal: str) -> float:
    if goal == 'MIN':
        return (max(values) - values) / (max(values) - min(values) + 0.0001)
    elif goal == 'MAX':
        return values / max(values)

def score_target(value: float, goal: float, accepted_error_range = 0.1) -> float:
    std_dev = accepted_error_range * value / 3
    return np.exp(-(value - goal)**2 / (2 * std_dev**2))

def score_range(value: List[float], choice: float) -> float:
    return 1 if choice >= value[0] and choice <= value[1] else 0

