import pytest
from httpx import AsyncClient
from backend.app.main import app
from backend.app.schemas.optimization import OptimizationConfig

@pytest.mark.asyncio
async def test_generate_config():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/optimization/config", json={
            "prompt": "Optimize a chemical reaction with temperature between 20-100°C and pressure between 1-10 atm to maximize yield"
        })
        
        assert response.status_code == 200
        config = OptimizationConfig(**response.json())
        
        # Check if config has required fields
        assert "temperature" in config.parameters
        assert "pressure" in config.parameters
        
        # Check parameter ranges
        temp_param = config.parameters["temperature"]
        assert temp_param["min"] >= 20
        assert temp_param["max"] <= 100
        
        pressure_param = config.parameters["pressure"]
        assert pressure_param["min"] >= 1
        assert pressure_param["max"] <= 10
        
        # Check objective
        assert "yield" in config.objective.lower()

@pytest.mark.asyncio
async def test_generate_config_invalid_prompt():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/optimization/config", json={
            "prompt": ""  # Empty prompt
        })
        assert response.status_code == 400 