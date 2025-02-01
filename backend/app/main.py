from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict

from .api.routes import optimization

app = FastAPI(
    title="Sci-Opt API",
    description="Scientific Optimization API with Active Learning and LLM Integration",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(optimization.router)

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"} 