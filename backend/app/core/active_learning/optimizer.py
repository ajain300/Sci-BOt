import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm
from typing import List, Dict, Tuple
from ...schemas.optimization import OptimizationConfig, DataPoint, ActiveLearningResponse

class ActiveLearningOptimizer:
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.gp = GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF(),
            n_restarts_optimizer=10,
            random_state=42
        )
        
    def _convert_to_array(self, data_points: List[DataPoint]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert data points to numpy arrays for GP."""
        X = []
        y = []
        param_names = list(self.config.parameters.keys())
        
        for point in data_points:
            X.append([point.parameters[name] for name in param_names])
            y.append(point.objective_value)
            
        return np.array(X), np.array(y)
    
    def _convert_to_dict(self, X: np.ndarray) -> List[Dict[str, float]]:
        """Convert numpy array back to dictionary format."""
        param_names = list(self.config.parameters.keys())
        return [{name: float(x[i]) for i, name in enumerate(param_names)} for x in X]
    
    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Calculate expected improvement at points X."""
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        mu_sample = self.gp.predict(self.X_train)
        mu_sample_opt = np.min(mu_sample)
        
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        
        return ei
    
    async def get_suggestions(self, data: List[DataPoint], n_suggestions: int = 1) -> ActiveLearningResponse:
        """Get next points to evaluate based on expected improvement."""
        if not data:
            # If no data, return random suggestions
            suggestions = []
            for _ in range(n_suggestions):
                point = {}
                for name, param in self.config.parameters.items():
                    point[name] = np.random.uniform(param["min"], param["max"])
                suggestions.append(point)
            return ActiveLearningResponse(
                suggestions=suggestions,
                expected_improvements=[0.0] * n_suggestions
            )
        
        # Convert data to arrays and fit GP
        self.X_train, self.y_train = self._convert_to_array(data)
        self.gp.fit(self.X_train, self.y_train)
        
        # Generate random candidates and select best by EI
        n_random = 1000
        X_random = []
        for _ in range(n_random):
            point = []
            for name, param in self.config.parameters.items():
                point.append(np.random.uniform(param["min"], param["max"]))
            X_random.append(point)
        X_random = np.array(X_random)
        
        ei = self._expected_improvement(X_random)
        indices = np.argsort(ei.ravel())[-n_suggestions:]
        
        return ActiveLearningResponse(
            suggestions=self._convert_to_dict(X_random[indices]),
            expected_improvements=ei[indices].tolist()
        ) 