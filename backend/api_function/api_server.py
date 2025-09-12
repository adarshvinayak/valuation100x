#!/usr/bin/env python3
"""
FastAPI Server for DeepResearch - Lightweight API Function
Handles requests and delegates analysis to SQS worker
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import uvicorn
import boto3
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Try to import Redis with fallback
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQS Configuration
QUEUE_URL = os.environ.get("QUEUE_URL")
sqs = boto3.client("sqs") if QUEUE_URL else None

# Global variables for storage
redis_client = None
analysis_states: Dict[str, Dict[str, Any]] = {}

# Pydantic models
class AnalysisRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: Optional[str] = Field(None, description="Company name (optional)")
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Ticker cannot be empty')
        return v.upper().strip()

class AnalysisResponse(BaseModel):
    analysis_id: str
    polling_url: str
    status: str = "queued"

class AnalysisStatus(BaseModel):
    analysis_id: str
    status: str
    progress: int = 0
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    current_step: Optional[str] = None
    current_step_name: Optional[str] = None
    current_step_description: Optional[str] = None
    user_message: Optional[str] = None
    total_steps: Optional[int] = None
    completed_steps: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

# Create FastAPI app
app = FastAPI(
    title="DeepResearch API",
    description="Lightweight API for comprehensive stock analysis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage functions
async def store_analysis_state(analysis_id: str, state: Dict[str, Any]):
    """Store analysis state in Redis or memory"""
    if redis_client:
        try:
            await redis_client.setex(f"analysis:{analysis_id}", 3600, json.dumps(state, default=str))
            return
        except Exception as e:
            logger.warning(f"Redis storage failed: {e}")
    
    # Fallback to memory
    analysis_states[analysis_id] = state

async def get_analysis_state(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve analysis state from Redis or memory"""
    if redis_client:
        try:
            data = await redis_client.get(f"analysis:{analysis_id}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis retrieval failed: {e}")
    
    # Fallback to memory
    return analysis_states.get(analysis_id)

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "deepresearch-api",
        "version": "2.0.0",
        "architecture": "sqs-worker"
    }

@app.get("/api/validate/{ticker}")
async def validate_ticker(ticker: str):
    """Validate ticker symbol using FMP API"""
    try:
        # Import FMP tools from common layer
        import sys
        sys.path.append('/opt/python')  # Lambda layer path
        from tools.fmp import validate_ticker_symbol
        
        result = await validate_ticker_symbol(ticker.upper())
        return result
        
    except Exception as e:
        logger.error(f"Ticker validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/api/analysis/comprehensive/start")
async def start_comprehensive_analysis(request: AnalysisRequest) -> AnalysisResponse:
    """Start comprehensive analysis by sending job to SQS queue"""
    
    if not QUEUE_URL or not sqs:
        raise HTTPException(status_code=503, detail="SQS queue not configured")
    
    analysis_id = str(uuid.uuid4())
    ticker = request.ticker.upper()
    
    try:
        # Store initial state
        initial_state = {
            "analysis_id": analysis_id,
            "ticker": ticker,
            "company_name": request.company_name,
            "status": "queued",
            "progress": 0,
            "current_step": "queued",
            "current_step_name": "Queued",
            "current_step_description": "Analysis job queued for processing",
            "user_message": "Your analysis has been queued and will start shortly...",
            "started_at": datetime.utcnow(),
            "total_steps": 8,
            "completed_steps": 0
        }
        
        await store_analysis_state(analysis_id, initial_state)
        
        # Send message to SQS queue
        message_body = json.dumps({
            "analysis_id": analysis_id,
            "ticker": ticker,
            "company_name": request.company_name
        })
        
        response = sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=message_body
        )
        
        logger.info(f"✅ Analysis job queued: {analysis_id} for ticker {ticker}")
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            polling_url=f"/api/analysis/{analysis_id}/status",
            status="queued"
        )
        
    except Exception as e:
        logger.error(f"Failed to queue analysis job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@app.get("/api/analysis/{analysis_id}/status")
async def get_analysis_status(analysis_id: str) -> AnalysisStatus:
    """Get analysis status via polling"""
    
    try:
        state = await get_analysis_state(analysis_id)
        
        if not state:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return AnalysisStatus(
            analysis_id=analysis_id,
            status=state.get("status", "unknown"),
            progress=state.get("progress", 0),
            ticker=state.get("ticker"),
            company_name=state.get("company_name"),
            current_step=state.get("current_step"),
            current_step_name=state.get("current_step_name"),
            current_step_description=state.get("current_step_description"),
            user_message=state.get("user_message"),
            total_steps=state.get("total_steps"),
            completed_steps=state.get("completed_steps"),
            started_at=state.get("started_at"),
            completed_at=state.get("completed_at"),
            error=state.get("error"),
            results=state.get("results")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Initialize Redis connection (optional)
@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection if available"""
    global redis_client
    
    logger.info("🚀 Starting DeepResearch API (Lightweight)")
    
    if REDIS_AVAILABLE and redis:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            redis_client = await redis.from_url(redis_url)
            await redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using memory storage.")
            redis_client = None
    else:
        logger.info("📦 Using memory storage (Redis not available)")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
    logger.info("👋 API server shutdown complete")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


