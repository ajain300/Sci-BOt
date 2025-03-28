import itertools
import numpy as np
import pandas as pd
from .dataset import AL_Dataset
from .acquisition import *
from .models import GaussianProcessModel
import logging
import os
from typing import List, Tuple, Dict
from ...schemas.optimization import DataPoint

# logger = logging.getLogger(__name__)

class Active_Learning_Loop:
    rec_to_acq_func = {'diversity_uncertainty': Diversity_Batch,
                       'best_score': NaiveMOFunction,
                       }
    model_type_to_obj = {'GPR': GaussianProcessModel}
    
    def __init__(self, data : AL_Dataset, 
                 rec_type = 'best_score', 
                 model_type = 'GPR', 
                 name = "Active_Learning_Loop", 
                 directory = "active_learning",
                 **kwargs):
        f"""
        The function initializes Active_Learning_Loop.

        :param data: The dataset to be used.
        :param rec_type: The type of recommendation to be used. One of {self.rec_to_acq_func.keys()}.
        """
        # Name and path vars
        self.name = name
        self.path = os.path.join(directory, name)
        os.makedirs(self.path, exist_ok = True)
        
        # Create logger with the path as the log file
        logging.basicConfig(filename=os.path.join(self.path, f'{self.name}.log'), level=logging.INFO)
        self.logger = logging.getLogger()
        
        self.dataset = data

        if rec_type not in self.rec_to_acq_func:
            raise Exception(f"Recommendation type must be one of {self.rec_to_acq_func.keys()}, got {rec_type}")
        self.rec_type = rec_type
        self.model = self.model_type_to_obj[model_type](data, logger = self.logger)
        
    def run_Loop(self, **kwargs):
        """
        We initialize the model, compute the hypervolume, and then optimize the acquisition function to get
        new observations. 
        
        The new observations are then returned.
        :return: The predictions of the model.
        """
        logger.info("Running Active_Learning_Loop.run_Loop")
        # Train the model
        self.model.train()
        # self.model.validate_training(os.path.join(self.path, 'training_validation.png'))
        
        # Initialize the acquisition function
        acq_func : AcquisitionFunction = self.rec_to_acq_func[self.rec_type](self.model, 
                                                       self.dataset, 
                                                       targets = self.dataset.targets,
                                                       **kwargs)

        recs, sample_mean, sample_std = acq_func.recommend(**kwargs)
        logger.info(f"recs: {recs}")
        recs = self.dataset.make_rec_readable(recs)

        for prop, preds in sample_mean.items():
            recs[prop + '_mean'] = preds
            recs[prop + '_std'] = sample_std[prop]

        return recs

    async def get_suggestions(self, data: List[DataPoint], n_suggestions: int) -> Tuple[List[Dict[str, float]], List[float]]:
        """Convert input data to AL_Dataset format and get suggestions using run_Loop."""
        # Convert input data to format needed by AL_Dataset
        param_data = {}
        objective_data = {}
        
        for point in data:
            for param_name, value in point.parameters.items():
                if param_name not in param_data:
                    param_data[param_name] = []
                param_data[param_name].append(value)
            
            objective_data['objective'] = point.objective_values
        
        # Create DataFrame from the data
        param_df = pd.DataFrame(param_data)
        objective_df = pd.DataFrame(objective_data)
        
        # Update dataset with new data
        self.dataset.update_data(param_df, objective_df)
        
        # Get recommendations using run_Loop
        recommendations_df = self.run_Loop(n_suggestions=n_suggestions)
        
        # Convert recommendations DataFrame to required format
        suggestions = []
        expected_improvements = []
        
        # Get parameter columns (excluding mean and std columns)
        param_cols = [col for col in recommendations_df.columns 
                     if not (col.endswith('_mean') or col.endswith('_std'))]
        
        # Convert each row to a suggestion dictionary
        for _, row in recommendations_df.iterrows():
            suggestion = {param: float(row[param]) for param in param_cols}
            suggestions.append(suggestion)
            
            # Get the expected improvement (using mean prediction)
            if 'objective_mean' in recommendations_df.columns:
                expected_improvements.append(float(row['objective_mean']))
            else:
                expected_improvements.append(0.0)
                
        return suggestions[:n_suggestions], expected_improvements[:n_suggestions]
