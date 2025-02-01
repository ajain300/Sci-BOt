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
    )
]

@pytest.mark.asyncio
async def test_get_suggestions_no_data():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/optimization/suggest", json={
            "config": TEST_CONFIG.dict(),
            "data": [],
            "n_suggestions": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "suggestions" in data
        assert "expected_improvements" in data
        assert len(data["suggestions"]) == 3
        
        # Check suggestion bounds
        for suggestion in data["suggestions"]:
            assert 20 <= suggestion["temperature"] <= 100
            assert 1 <= suggestion["pressure"] <= 10

@pytest.mark.asyncio
async def test_get_suggestions_with_data():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/optimization/suggest", json={
            "config": TEST_CONFIG.dict(),
            "data": [point.dict() for point in TEST_DATA_POINTS],
            "n_suggestions": 2
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["suggestions"]) == 2
        assert len(data["expected_improvements"]) == 2
        
        # Check that suggestions are different from existing points
        for suggestion in data["suggestions"]:
            assert suggestion not in [p.parameters for p in TEST_DATA_POINTS]

@pytest.mark.asyncio
async def test_get_suggestions_invalid_config():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/optimization/suggest", json={
            "config": {
                "parameters": {},  # Empty parameters
                "objective": "test"
            },
            "data": [],
            "n_suggestions": 1
        })
        assert response.status_code == 400 