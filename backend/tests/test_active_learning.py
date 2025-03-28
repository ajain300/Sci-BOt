import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.schemas.optimization import OptimizationConfig, DataPoint
from backend.app.utils.serialization import convert_config_to_dict, convert_data_points_to_dict

TEST_CONFIG_JSON = TEST_CONFIG_JSON = {
    "features": [
        # Composition feature for materials
        {
            "name": "material_composition",
            "type": "composition",
            "columns": {
                "parts": [
                    "material_a_concentration",
                    "material_b_concentration",
                    "material_c_concentration"
                ],
                "range": {
                    "material_a_concentration": [15.0, 35.0],
                    "material_b_concentration": [15.0, 35.0],
                    "material_c_concentration": [30.0, 70.0]
                }
            },
            "scaling": "lin"
        },
        # Continuous features
        {
            "name": "temperature",
            "type": "continuous",
            "min": 20.0,
            "max": 100.0,
            "scaling": "lin"
        },
        {
            "name": "pressure",
            "type": "continuous",
            "min": 1.0,
            "max": 10.0,
            "scaling": "lin"
        }
    ],
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
        parameters={
            "material_a_concentration": 25.0,
            "material_b_concentration": 25.0,
            "material_c_concentration": 50.0,
            "temperature": 50.0,
            "pressure": 5.0
        },
        objective_values={
            "reaction_yield": 0.75,
            "cost": 100.0
        }
    ),
    DataPoint(
        parameters={
            "material_a_concentration": 30.0,
            "material_b_concentration": 20.0,
            "material_c_concentration": 50.0,
            "temperature": 75.0,
            "pressure": 7.0
        },
        objective_values={
            "reaction_yield": 0.85,
            "cost": 150.0
        }
    ),
    DataPoint(
        parameters={
            "material_a_concentration": 30.0,
            "material_b_concentration": 30.0,
            "material_c_concentration": 40.0,
            "temperature": 40.0,
            "pressure": 10.0
        },
        objective_values={
            "reaction_yield": 0.35,
            "cost": 50.0
        }
    ),
    DataPoint(
    parameters={
        "material_a_concentration": 20.0,
        "material_b_concentration": 30.0,
        "material_c_concentration": 50.0,
        "temperature": 60.0,
        "pressure": 6.0
    },
    objective_values={
        "reaction_yield": 0.65,
        "cost": 90.0
    }
    ),
    DataPoint(
        parameters={
            "material_a_concentration": 35.0,
            "material_b_concentration": 25.0,
            "material_c_concentration": 40.0,
            "temperature": 80.0,
            "pressure": 8.0
        },
        objective_values={
            "reaction_yield": 0.92,
            "cost": 160.0
        }
    )
    ]
# @pytest.mark.asyncio
# async def test_get_suggestions_no_data():
#     async with AsyncClient(base_url="http://test") as client:
#         response = await client.post(
#             app.url_path_for("get_suggestions"),
#             json={
#                 "config": TEST_CONFIG.model_dump(),
#                 "data": [],
#                 "n_suggestions": 3
#             }
#         )
        
#         assert response.status_code == 200
#         data = response.json()
        
#         # Check response structure
#         assert "suggestions" in data
#         assert "expected_improvements" in data
#         assert len(data["suggestions"]) == 3
        
#         # Check suggestion bounds
#         for suggestion in data["suggestions"]:
#             assert 20 <= suggestion["temperature"] <= 100
#             assert 1 <= suggestion["pressure"] <= 10

@pytest.mark.asyncio
async def test_get_suggestions_with_data():
    client = TestClient(app)
    url = "/optimization/suggest"
    
    # Convert the config to a dict with proper string handling for n/a values
    config_dict = convert_config_to_dict(TEST_CONFIG)
    data_points_dict = convert_data_points_to_dict(TEST_DATA_POINTS)
    
    print("config_dict type:", type(config_dict))
    
    request_data = {
        "config": config_dict,
        "data": data_points_dict,
        "n_suggestions": 2
    }
    
    print("\nRequest URL:", url)
    print("Request Data:", request_data)
    
    response = client.post(
        url,
        json=request_data
    )
    
    print(f"Response Status: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print(f"Response Body: {response.text}")
    
    assert response.status_code == 200
# @pytest.mark.asyncio
# async def test_get_suggestions_invalid_config():
#     async with AsyncClient(base_url="http://test") as client:
#         response = await client.post(
#             app.url_path_for("get_suggestions"),
#             json={
#                 "config": {
#                     "parameters": {},  # Empty parameters
#                 "objective": "test"
#             },
#             "data": [],
#             "n_suggestions": 1
#         })
#         assert response.status_code == 400 