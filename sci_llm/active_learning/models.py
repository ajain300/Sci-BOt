import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Kernel
from sklearn.model_selection import GridSearchCV
import pandas as pd
from .dataset import AL_Dataset
import warnings
from sklearn.exceptions import ConvergenceWarning
import logging
import matplotlib.pyplot as plt

# Suppress only ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

class BaseModel:
    def train(self, X, y):
        # Placeholder for training logic
        pass

    def inference(self, X):
        # Placeholder for testing logic
        pass

    def validate_training(self, filename : str):
        # Placeholder for validation logic
        pass

class GaussianProcessModel(BaseModel):
    def __init__(self, dataset : AL_Dataset, **kwargs):
        # Check if provided kwargs are the right names
        valid_kwargs = ['kernel_function', 'hyperparams', 'logger']
        invalid_kwargs = [kwarg for kwarg in kwargs.keys() if kwarg not in valid_kwargs]
        if invalid_kwargs:
            raise ValueError(f"Invalid kwargs provided: {', '.join(invalid_kwargs)}")
        
        # Define the kernel
        self.kernel = kwargs.get('kernel_function', RBF() + WhiteKernel(noise_level_bounds=(1e-6, 1e-1)))
        # Check if provided kernel is a valid kernel function
        if not isinstance(self.kernel, Kernel):
            raise ValueError("Invalid kernel_function provided.")
        
        # Define the parameter grid for hyperparameter optimization
        self.param_grid = kwargs.get('hyperparams', self._get_default_param_grid())

        # Set the dataset for this model
        self.dataset = dataset

        # Set logger if given
        self.logger = kwargs.get('logger', logging.getLogger(__name__))


    def train(self, **kwargs):
        X_train, Y, Y_std = self.dataset.get_training_format()
        return self._train(X_train, Y, y_var = Y_std, **kwargs)
    
    def _train(self, X, y, y_var = 1e-5, **kwargs):
        self.logger.info("Training the model.")
        y = y.reshape(-1,1)

        opt_params = self._perform_grid_search(X, y)
        kernel = update_kernel_params(self.kernel, opt_params)

        # Perform hyperparameter optimization using GridSearchCV
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=y_var)  # Initial dummy alpha
        
        gpr.fit(X, y)

        self.model = gpr

    def _perform_grid_search(self, X, y):
        self.logger.info("Performing hyperparam grid search.")
        gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-4)  # Initial dummy alpha
        gpr_optimized = GridSearchCV(gpr, self.param_grid, cv=5)
        gpr_optimized.fit(X, y)
        return gpr_optimized.best_params_

    def inference(self, predict_data : pd.DataFrame, 
                  output_X_pred = False,
                  as_dataframe = False,
                  **kwargs):
        """
        If single variable inference, then property should be given.
        If multiple variable inference then iterate through each property and return the ouput
        such that the output is a dataframe with the input columns and the predicted columns
        """
        # Transform the input with dataset
        if kwargs.get("property", None) is not None:
            X, X_cols = self.dataset.get_inference_format(predict_data, target_property = kwargs.get("property"))
        
            if output_X_pred:
                mean, std = self.model.predict(X, return_std=True)
                return mean, std, X
            
            if as_dataframe:
                mean, std = self.model.predict(X, return_std=True)
                df_out = pd.DataFrame(X, columns = X_cols)
                # for i, target in enumerate(targets):
                df_out[self.dataset.target.name] = mean
                df_out[self.dataset.target.name + '_std'] = std
                return df_out
            
            return self.model.predict(X, return_std=True)
            
        else:
            # If no property is given, then return the mean and std of all the properties
            X, X_cols = self.dataset.get_inference_format(predict_data)
            mean, std = self.model.predict(X, return_std=True)
            if as_dataframe:
                df_out = pd.DataFrame(X, columns = X_cols)
                df_out['mean'] = mean
                df_out['std'] = std
                return df_out
            return mean, std
        
        # return self.model.predict(X, return_std=True)

    def _get_default_param_grid(self):
        return {
        'kernel__k1__length_scale': np.logspace(-1, 1, 10),
        'kernel__k2__noise_level': np.logspace(-3, 0, 10)}
    
    def validate_training(self, filename):
        # Import matplotlib for plotting

        # Retrieve the training data
        _, y_train, y_std = self.dataset.get_training_format()
        X_train = self.dataset.input_data[self.dataset.X_columns].copy()
        
        # Convert X_train to a DataFrame if it's not already
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        # Get predictions on the training data
        mean_pred, std_pred = self.inference(X_train)

        # Flatten the arrays to ensure they are 1D
        y_actual = y_train.flatten()
        y_pred = mean_pred.flatten()

        # Create parity plot
        plt.figure(figsize=(8, 6))
        
        # plt.scatter(y_actual, y_pred, alpha=0.7, label=)
        plt.errorbar(y_actual, y_pred, yerr=std_pred.flatten(), fmt='o', alpha=0.7)
        plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()],
                 'k--', color='red', label='Ideal')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Parity Plot on Training Data')
        plt.legend()
        plt.savefig(filename)

def update_kernel_params(kernel, best_params):
    for param, value in best_params.items():
        if param.startswith('kernel__'):
            parts = param.split('__')
            attr = kernel
            for part in parts[1:-1]:
                attr = getattr(attr, part)
            setattr(attr, parts[-1], value)
    return kernel