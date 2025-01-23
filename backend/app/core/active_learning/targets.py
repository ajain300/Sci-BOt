from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict

# Dictate the types: goals
TARGET_TYPES_DICT = {
    'Uncertainty': ['MIN', 'MAX'],
    'Extreme' : ['MIN', 'MAX'],
    'Target' : float,
    'Range' : [float, float]
}

class Target:
    def __init__(self, target_config, 
                target_type, target_goal,
                 data = None, 
                 X_columns = None, 
                 variable_info = None, 
                 properties = None, 
                 initial_columns = None):
        self.name = target_config['name']
        self.type = target_type
        self.goal = target_goal if not isinstance(target_goal, int) else float(target_goal)
        self.target_config = target_config
        self.data = data
        self.X_columns = X_columns
        self.initial_columns = initial_columns if initial_columns else X_columns
        self.variable_info = variable_info if variable_info else {}
        self.properties = properties if properties else {}
        
        self._target_type_check()

    @abstractmethod
    def process(self):
        pass

    def get_columns(self) -> List[str]:
        return self.X_columns
    
    def _target_type_check(self):
        if self.type in TARGET_TYPES_DICT:
            valid_goals = TARGET_TYPES_DICT[self.type]
            if isinstance(valid_goals, list):
                if isinstance(valid_goals[0], type):
                    if not isinstance(self.goal, list):
                        raise ValueError(f"Invalid goal format '{type(self.goal).__name__}' for target type '{self.type}'. Valid goal type is: {valid_goals[0].__name__}")
                    else:
                        for i, goal in enumerate(self.goal):
                            if not isinstance(goal, valid_goals[i]):
                                raise ValueError(f"Invalid goal type '{type(goal).__name__}' for target type '{self.type}'. Valid goal type is: {valid_goals[0].__name__}")
                else:
                    if self.goal not in valid_goals:
                        raise ValueError(f"Invalid goal '{self.goal}' for target type '{self.type}'. Valid goals are: {valid_goals}")
            elif isinstance(valid_goals, type):
                if not isinstance(self.goal, valid_goals):
                    raise ValueError(f"Invalid goal type '{type(self.goal).__name__}' for target type '{self.type}'. Valid goal type is: {valid_goals.__name__}")
        else:
            raise ValueError(f"Invalid target type '{self.type}'. Valid target types are: {list(TARGET_TYPES_DICT.keys())}")

    def score(self, mean, std) -> float:
        if self.type == 'Uncertainty':
            return score_uncertainty(std, self.goal)
        elif self.type == 'Extreme':
            return score_extreme(mean, self.goal)
        elif self.type == 'Target':
            return score_target(mean, self.goal)
        elif self.type == 'Range':
            return score_range(mean, self.goal)
        else:
            raise ValueError(f"Invalid target type '{self.type}'")
    
class TargetFeature(Target):
    def process(self):
        for col in self.X_columns:
            self.variable_info[col] = {'scaling': self.properties['scaling']}
            
# Helper functions for scoring

def score_uncertainty(values: float, goal: str) -> float:
    # print("scoring uncertainty")
    if goal == 'MIN':
        return (max(values) - values) / (max(values) - min(values))
    elif goal == 'MAX':
        return values / max(values)

def score_extreme(values: float, goal: str) -> float:
    # print("scoring extreme")
    if goal == 'MIN':
        return (max(values) - values) / (max(values) - min(values) + 0.0001)
    elif goal == 'MAX':
        return values / max(values)

def score_target(value: float, goal: float, accepted_error_range = 0.1) -> float:
    # print("scoring target")
    std_dev = accepted_error_range * value / 3  # Calculate standard deviation as 10% of the value
    return np.exp(-(value - goal)**2 / (2 * std_dev**2))

def score_range(value: List[float], choice: float) -> float:
    return 1 if choice >= value[0] and choice <= value[1] else 0

