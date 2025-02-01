from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ...schemas.optimization import (
    ConfigGenerationRequest,
    OptimizationConfig,
    ActiveLearningRequest,
    ActiveLearningResponse,
    ProcessDataRequest,
    ProcessDataResponse
)
from ...core.llm import generate_config
from ...core.active_learning import ActiveLearningOptimizer
from ...core.data_processing import analyze_data
import io
import csv

router = APIRouter(prefix="/optimization", tags=["optimization"])

@router.post("/config", response_model=OptimizationConfig)
async def create_optimization_config(request: ConfigGenerationRequest):
    """Generate optimization configuration from natural language prompt."""
    try:
        config = await generate_config(request.prompt)
        return config
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/template")
async def get_csv_template(config: OptimizationConfig):
    """Generate a CSV template with headers based on the configuration."""
    try:
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Get all parameter names and add objective variable
        headers = list(config.parameters.keys()) + [config.objective_variable]
        writer.writerow(headers)
        
        # Create response with CSV file
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=optimization_template.csv"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/suggest", response_model=ActiveLearningResponse)
async def get_suggestions(request: ActiveLearningRequest):
    """Get next points to evaluate using active learning."""
    try:
        optimizer = ActiveLearningOptimizer(request.config)
        response = await optimizer.get_suggestions(request.data, request.n_suggestions)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analyze", response_model=ProcessDataResponse)
async def process_data(request: ProcessDataRequest):
    """Analyze optimization data and extract insights."""
    try:
        response = await analyze_data(request.config, request.data)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 