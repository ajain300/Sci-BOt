import pytest
from httpx import AsyncClient
from backend.app.main import app
from backend.app.schemas.optimization import OptimizationConfig, DataPoint, ObjectiveConfig, OptimizationDirection

TEST_CONFIG_JSON = {
    "features": {
        "material_a_concentration": {
            "name": "material_a_concentration",
            "min": 15.0,
            "max": 35.0,
            "type": "float"
        },
        "material_b_concentration": {
            "name": "material_b_concentration",
            "min": 15.0,
            "max": 35.0,
            "type": "float"
        },
        "material_c_concentration": {
            "name": "material_c_concentration",
            "min": "n/a",
            "max": "n/a",
            "type": "derived",
            "derived_from": "100 - material_a_concentration - material_b_concentration"
        },
        "temperature": {
            "name": "temperature",
            "min": 20.0,
            "max": 100.0,
            "type": "float"
        },
        "pressure": {
            "name": "pressure",
            "min": 1.0,
            "max": 10.0,
            "type": "float"
        }
    },
    "objectives": [
        {
            "name": "reaction_yield",
            "direction": "maximize",
            "weight": 1.0
        },
        {
            "name": "cost",
            "direction": "minimize",
            "weight": 1.0
        }
    ],
    "acquisition_function": "diversity_uncertainty",
    "constraints": [
        "material_a_concentration + material_b_concentration + material_c_concentration = 100"
    ]
}

# You can then create the OptimizationConfig from the JSON:
TEST_CONFIG = OptimizationConfig.model_validate(TEST_CONFIG_JSON)


TEST_DATA_POINTS = [
    DataPoint(
        parameters={"temperature": 50, "pressure": 5},
        objective_value=0.75
    ),
    DataPoint(
        parameters={"temperature": 75, "pressure": 7},
        objective_value=0.85
    ),
    DataPoint(
        parameters={"temperature": 60, "pressure": 6},
        objective_value=0.80
    )
]

@pytest.mark.asyncio
async def test_analyze_data():
    async with AsyncClient(base_url="http://test") as client:
        response = await client.post(
            app.url_path_for("analyze_data"),
            json={
                "config": TEST_CONFIG.model_dump(),
                "data": [point.model_dump() for point in TEST_DATA_POINTS]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check statistics
        assert "statistics" in data
        stats = data["statistics"]
        assert stats["num_points"] == 3
        assert stats["best_objective"] == 0.85
        assert "mean_objective" in stats
        assert "std_objective" in stats
        
        # Check parameter ranges
        assert "parameter_ranges" in stats
        ranges = stats["parameter_ranges"]
        assert "temperature" in ranges
        assert "pressure" in ranges
        
        # Check best point
        assert "best_point" in data
        best = data["best_point"]
        assert best["objective_value"] == 0.85
        assert best["parameters"]["temperature"] == 75
        assert best["parameters"]["pressure"] == 7
        
        # Check parameter importance
        assert "parameter_importance" in data
        importance = data["parameter_importance"]
        assert "temperature" in importance
        assert "pressure" in importance
        assert 0 <= importance["temperature"] <= 1
        assert 0 <= importance["pressure"] <= 1

@pytest.mark.asyncio
async def test_analyze_data_no_data():
    async with AsyncClient(base_url="http://test") as client:
        response = await client.post(
            app.url_path_for("analyze_data"),
            json={
                "config": TEST_CONFIG.model_dump(),
                "data": []
            }
        )
        assert response.status_code == 400

@pytest.mark.asyncio
async def test_analyze_data_invalid_config():
    async with AsyncClient(base_url="http://test") as client:
        response = await client.post(
            app.url_path_for("analyze_data"),
            json={
                "config": {
                    "parameters": {},  # Empty parameters
                    "objective": "test"
                },
                "data": [point.model_dump() for point in TEST_DATA_POINTS]
            }
        )
        assert response.status_code == 400 