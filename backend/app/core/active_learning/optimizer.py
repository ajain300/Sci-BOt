import numpy as np
from typing import List, Dict, Tuple
from ...schemas.optimization import OptimizationConfig, DataPoint, ActiveLearningResponse
from ...core.active_learning.al_loop import Active_Learning_Loop
from .dataset import AL_Dataset
import logging
import traceback

logger = logging.getLogger("app.core.active_learning")

class ActiveLearningOptimizer:
    def __init__(self, config: OptimizationConfig, data: List[DataPoint]):
        self.config = config
        print("in ActiveLearningOptimizer, config type:", type(config))
        # Create AL_Dataset with the configuration
        self.dataset = AL_Dataset(
            data_config=config,
            data=data
        )
        
        # Initialize ActiveLearningLoop with the dataset
        self.active_learning_loop = Active_Learning_Loop(
            data=self.dataset,
            rec_type=config.acquisition_function,  # or 'diversity_uncertainty' based on your needs
            model_type='GPR'
        )
    
    async def get_suggestions(self, n_suggestions: int = 1) -> ActiveLearningResponse:
        """Get next points to evaluate based on active learning strategy."""
        try:
            logger.debug("Processing data")
            self.dataset.process_data()
        
            logger.debug("Getting AL suggestions using run_Loop")
            
            # Get recommendations using run_Loop directly
            recommendations_df = self.active_learning_loop.run_Loop(n_suggestions=n_suggestions)
            
            # Convert recommendations DataFrame to required format
            suggestions = []
            expected_improvements = []
            
            # Get parameter columns (excluding mean and std columns)
            param_cols = [col for col in recommendations_df.columns 
                         if not (col.endswith('_mean') or col.endswith('_std'))]
            
            # Convert each row to a suggestion dictionary
            logger.info(f"recommendations_df: {recommendations_df.to_string()}")
            logger.debug(f"param_cols: {param_cols}")
            for _, row in recommendations_df.iterrows():
                logger.debug(f"row: {row}")
                suggestion = {param: float(row[param]) for param in param_cols}
                suggestions.append(suggestion)
                
                # Get the expected improvement (using mean prediction)
                if 'objective_mean' in recommendations_df.columns:
                    expected_improvements.append(float(row['objective_mean']))
                else:
                    expected_improvements.append(0.0)
            
            return ActiveLearningResponse(
                suggestions=suggestions,
                expected_improvements=expected_improvements
            )
            
        except Exception as e:
            logger.error(f"Error in get_suggestions: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise