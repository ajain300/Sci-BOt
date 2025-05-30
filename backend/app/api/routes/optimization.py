from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ...schemas.optimization import (
    ConfigGenerationRequest,
    OptimizationConfig,
    ActiveLearningRequest,
    SuggestionsResponse,
    ProcessDataRequest,
    ProcessDataResponse,
    ConfigUpdateRequest
)
from ...core.llm import generate_config
from ...core.active_learning import ActiveLearningOptimizer
from ...core.data_processing import analyze_data
from ...core.logging_config import setup_logging
from pprint import pformat
import io
import logging
import csv
import traceback

logger = logging.getLogger(__name__)


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
        # Get feature columns
        feature_columns = []
        for feature in config.features:
            if feature.type != "composition":
                feature_columns.append(feature.name)
            else:
                feature_columns.extend(feature.columns.parts)
        # Get all parameter names and add objective variable
        headers = feature_columns + [objective.name for objective in config.objectives]
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

@router.post("/suggest", response_model=SuggestionsResponse)
async def get_suggestions(request: ActiveLearningRequest) -> SuggestionsResponse:
    try:
        optimizer = ActiveLearningOptimizer(request.config, request.data)
        response = await optimizer.get_suggestions(request.n_suggestions)
        analysis_result = await optimizer.evaluate_suggestions()
        return SuggestionsResponse(**analysis_result)
        
    except Exception as e:
        logger.error(f"Error in get_suggestions: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

@router.post("/analyze", response_model=ProcessDataResponse)
async def process_data(request: ProcessDataRequest):
    """Analyze optimization data and extract insights."""
    try:
        response = await analyze_data(request.config, request.data)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 



@router.post("/config/update")
async def update_config(request: ConfigUpdateRequest):
    try:
        # Update the configuration in your backend storage
        # This will depend on how you're storing the configuration
        await update_optimization_config(
            type=request.type,
            id=request.id,
            property=request.property,
            value=request.value
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update configuration: {str(e)}"
        )