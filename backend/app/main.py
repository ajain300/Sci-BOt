from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict
from .core.logging_config import setup_logging

from .api.routes import optimization

# Setup logging before creating the app
setup_logging()

app = FastAPI(
    title="Sci-Opt API",
    description="Scientific Optimization API with Active Learning and LLM Integration",
    version="1.0.0"
)

origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(optimization.router)

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"} 