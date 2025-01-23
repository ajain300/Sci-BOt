import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Any
from ..schemas.optimization import OptimizationConfig, DataPoint, ProcessDataResponse

async def analyze_data(config: OptimizationConfig, data: List[DataPoint]) -> ProcessDataResponse:
    """Analyze optimization data to extract insights."""
    if not data:
        raise ValueError("No data points provided for analysis")
    
    # Convert data to arrays
    param_names = list(config.parameters.keys())
    X = []
    y = []
    for point in data:
        X.append([point.parameters[name] for name in param_names])
        y.append(point.objective_value)
    X = np.array(X)
    y = np.array(y)
    
    # Find best point
    best_idx = np.argmin(y)
    best_point = data[best_idx]
    
    # Calculate parameter importance using Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importance = dict(zip(param_names, rf.feature_importances_))
    
    # Calculate statistics
    statistics = {
        "num_points": len(data),
        "best_objective": float(y[best_idx]),
        "mean_objective": float(np.mean(y)),
        "std_objective": float(np.std(y)),
        "parameter_ranges": {
            name: {
                "min": float(np.min(X[:, i])),
                "max": float(np.max(X[:, i])),
                "mean": float(np.mean(X[:, i])),
                "std": float(np.std(X[:, i]))
            }
            for i, name in enumerate(param_names)
        }
    }
    
    return ProcessDataResponse(
        statistics=statistics,
        best_point=best_point,
        parameter_importance=importance
    ) 