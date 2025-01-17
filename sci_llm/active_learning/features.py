from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict

class Feature(ABC):
    def __init__(self, feature_config, data = None, X_columns = None, variable_info = None, properties = None, initial_columns = None):
        self.name = feature_config['name']
        self.feature_config = feature_config
        self.data = data
        self.X_columns = X_columns
        self.OH_columns = []
        self.initial_columns = initial_columns if initial_columns else X_columns
        self.variable_info = variable_info if variable_info else {}
        self.properties = properties if properties else {}

    @abstractmethod
    def process(self):
        pass

    def get_columns(self) -> List[str]:
        return self.X_columns
    
    # def set_columns(self, columns: List[str]):
    #     self.X_columns = columns

class CompositionFeature(Feature):
    def process(self):
        for col in self.X_columns:
            self.variable_info[col] = {'scaling': self.feature_config.get('scaling', 'lin')}
        
class CorrelatedFeature(Feature):
    def process(self):
        pass

class DiscreteFeature(Feature):
    def process(self):
        if 'requirements' not in self.feature_config:
            self.feature_config['requirements'] = []
        self.feature_config['requirements'].append({'allowed_values': self.feature_config['range']})

class GeneralFeature(Feature):
    def process(self):
        columns = self.feature_config['columns']
        if self.X_columns is not None:    
            self.X_columns.extend(columns)
        else:
            self.X_columns = [columns]

# Factory method to get the appropriate feature class
def get_feature_class(feature_type: str):
    feature_classes = {
        'composition': CompositionFeature,
        'correlated': CorrelatedFeature,
        'fidelity': FidelityFeature,
        'general': GeneralFeature
    }
    return feature_classes.get(feature_type, GeneralFeature)