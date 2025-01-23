import pytest
from httpx import AsyncClient
from ..app.main import app
from ..app.schemas.optimization import OptimizationConfig, DataPoint

TEST_CONFIG = OptimizationConfig(
    parameters={
        "temperature": {"min": 20, "max": 100, "type": "float"},
        "pressure": {"min": 1, "max": 10, "type": "float"}
    },
    objective="Maximize reaction yield"
)

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
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/optimization/analyze", json={
            "config": TEST_CONFIG.dict(),
            "data": [point.dict() for point in TEST_DATA_POINTS]
        })
        
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
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/optimization/analyze", json={
            "config": TEST_CONFIG.dict(),
            "data": []
        })
        assert response.status_code == 400

@pytest.mark.asyncio
async def test_analyze_data_invalid_config():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/optimization/analyze", json={
            "config": {
                "parameters": {},  # Empty parameters
                "objective": "test"
            },
            "data": [point.dict() for point in TEST_DATA_POINTS]
        })
        assert response.status_code == 400 