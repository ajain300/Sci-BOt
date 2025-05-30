import numpy as np
from typing import List, Dict, Tuple
from ...schemas.optimization import OptimizationConfig, DataPoint, ActiveLearningResponse
from ...core.active_learning.al_loop import Active_Learning_Loop
from ...core.llm import evaluate_active_learning_result
from .dataset import AL_Dataset
import logging
import traceback

logger = logging.getLogger("app.core.active_learning")

class ActiveLearningOptimizer:
    def __init__(self, config: OptimizationConfig, data: List[DataPoint]):
        self.config = config
        self.data = data
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
            recommendations_df, objective_names = self.active_learning_loop.run_Loop(n_suggestions=n_suggestions)
            
            # Convert recommendations DataFrame to required format
            suggestions = []
            scores = []
            
            # Get parameter columns (excluding mean and std columns)
            param_cols = [col for col in recommendations_df.columns 
                         if not (col.endswith('_mean') or col.endswith('_std'))]
            
 
            # Convert each row to a suggestion dictionary
            logger.info(f"recommendations_df: {recommendations_df.to_string()}")
            logger.debug(f"param_cols: {param_cols}")
            for _, row in recommendations_df.iterrows():
                logger.debug(f"row: {row}")
                def try_float(value):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return value
                suggestion = {
                    param: try_float(row[param]) 
                    for param in param_cols}
                suggestions.append(suggestion)
                
                # Get the objective mean and std
                objective_values = []
                for objective in objective_names:
                    objective_values.append({
                        objective: float(row[objective + '_mean']),
                        objective + '_std': float(row[objective + '_std'])
                    })
                scores.append(objective_values)
                
            self.suggestions = suggestions
            self.scores = scores
            
            return ActiveLearningResponse(
                suggestions=suggestions,
                scores=scores,
            )
            
        except Exception as e:
            logger.error(f"Error in get_suggestions: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
            
    async def evaluate_suggestions(self) -> Dict[str, str]:
        """Evaluate the suggestions using the LLM."""
        try:
            return await evaluate_active_learning_result(self.config, self.data, self.suggestions, self.scores)
        except Exception as e:
            logger.error(f"Error in evaluate_suggestions: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise