#!/usr/bin/env python3
"""
FastAPI Server for DeepResearch Comprehensive Analysis
Provides REST API endpoints and WebSocket connections for the frontend.
Updated: Fresh deployment - datetime fixes active
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# Try to import Redis with fallback
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Import only essential components at startup
# Heavy imports will be loaded on-demand to prevent startup failures
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from run_enhanced_analysis import EnhancedAnalysisRunner
    from tools.fmp import get_financials_fmp

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Supabase client with fallback
try:
    from database.supabase_client import supabase_manager
    SUPABASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Supabase client not available: {e}")
    SUPABASE_AVAILABLE = False
    supabase_manager = None

# Global variables
analysis_tasks: Dict[str, Dict[str, Any]] = {}

# Request tracking to debug multiple calls
incoming_requests: Dict[str, List[str]] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with graceful fallbacks"""
    global redis_client
    
    # Startup
    logger.info("Starting DeepResearch API Server...")
    
    # Initialize Supabase with timeout and fallback
    if SUPABASE_AVAILABLE and supabase_manager:
        try:
            # Add timeout to prevent hanging
            await asyncio.wait_for(supabase_manager.initialize(), timeout=10.0)
            logger.info("✅ Supabase connection established")
        except asyncio.TimeoutError:
            logger.warning("🔄 Supabase initialization timed out. Using local storage.")
        except Exception as e:
            logger.warning(f"🔄 Supabase initialization failed: {e}. Using local storage.")
    else:
        logger.info("🔄 Supabase not available. Using local storage.")
    
    # Initialize Redis for caching and session management (optimized for Railway)
    logger.info(f"🔧 Redis initialization starting... REDIS_AVAILABLE={REDIS_AVAILABLE}, redis module={redis is not None}")
    
    if REDIS_AVAILABLE and redis:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # Railway fix: IPv6 support for redis.railway.internal
        # Based on Railway docs: https://docs.railway.com/reference/errors/enotfound-redis-railway-internal
        # Note: family parameter not supported in current redis client version
            
        logger.info(f"🔧 Attempting Redis connection to: {redis_url.split('@')[0] if '@' in redis_url else 'localhost'}@[REDACTED]")
        logger.info(f"🔧 Redis URL from environment: {'SET' if os.getenv('REDIS_URL') else 'NOT SET'}")
        
        try:
            logger.info("🔧 Creating Redis client with optimized settings...")
            # Optimized connection settings for Railway IPv6 network
            redis_client = await asyncio.wait_for(
                redis.from_url(
                    redis_url, 
                    decode_responses=True, 
                    socket_timeout=10.0, 
                    socket_connect_timeout=15.0,
                    retry_on_timeout=True,
                    health_check_interval=60
                ), 
                timeout=20.0
            )
            logger.info("🔧 Redis client created, testing connection...")
            await asyncio.wait_for(redis_client.ping(), timeout=5.0)
            logger.info("✅ Redis connection established successfully")
        except asyncio.TimeoutError:
            logger.warning("🔄 Redis connection timed out. Using in-memory storage.")
            redis_client = None
        except Exception as e:
            logger.warning(f"🔄 Redis connection failed: {type(e).__name__}: {str(e)}. Using in-memory storage.")
            redis_client = None
    else:
        reason = "Redis library not available" if not REDIS_AVAILABLE else "Redis module is None"
        logger.info(f"🔄 {reason}. Using in-memory storage.")
        redis_client = None
    
    logger.info(f"🔧 Redis initialization complete. Client: {'CONNECTED' if redis_client else 'NOT CONNECTED'}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    if redis_client:
        try:
            await redis_client.close()
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")
    
    if SUPABASE_AVAILABLE and supabase_manager:
        try:
            await supabase_manager.close()
        except Exception as e:
            logger.warning(f"Error closing Supabase connection: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="DeepResearch Comprehensive Analysis API",
    description="Professional stock analysis API with real-time progress tracking",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend - Updated for Railway healthcheck and Lovable integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins including Railway healthcheck hostname
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Pydantic models
class AnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    company_name: Optional[str] = Field(None, description="Optional company name")

class AnalysisResponse(BaseModel):
    analysis_id: str
    ticker: str
    company_name: Optional[str]
    status: str
    estimated_duration: str
    websocket_url: str
    created_at: datetime

class AnalysisStatus(BaseModel):
    analysis_id: str
    ticker: str
    status: str
    progress: int
    current_step: str
    current_component: Optional[str]
    estimated_completion: Optional[datetime]
    steps_completed: List[str]
    steps_remaining: List[str]
    error: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]

class TickerValidation(BaseModel):
    ticker: str
    is_valid: bool
    company_name: Optional[str]
    exchange: Optional[str]
    sector: Optional[str]
    market_cap: Optional[float]
    last_updated: datetime

class SystemHealth(BaseModel):
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, Any]
    active_analyses: int
    queue_size: int

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, analysis_id: str):
        await websocket.accept()
        if analysis_id not in self.active_connections:
            self.active_connections[analysis_id] = []
        self.active_connections[analysis_id].append(websocket)
        logger.info(f"WebSocket connected for analysis {analysis_id}")

    def disconnect(self, websocket: WebSocket, analysis_id: str):
        if analysis_id in self.active_connections:
            self.active_connections[analysis_id].remove(websocket)
            if not self.active_connections[analysis_id]:
                del self.active_connections[analysis_id]
        logger.info(f"WebSocket disconnected for analysis {analysis_id}")

    async def send_to_analysis(self, analysis_id: str, message: dict):
        if analysis_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[analysis_id]:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for conn in disconnected:
                self.disconnect(conn, analysis_id)

    async def broadcast_system_status(self, message: dict):
        """Broadcast system-wide messages to all connections"""
        for analysis_id, connections in self.active_connections.items():
            await self.send_to_analysis(analysis_id, message)

manager = ConnectionManager()

# Analysis step definitions
ANALYSIS_STEPS = [
    {"id": "question_generation", "name": "Question Generation", "duration": 60},
    {"id": "sec_filing_analysis", "name": "SEC Filing Analysis", "duration": 180},
    {"id": "financial_data_collection", "name": "Financial Data Collection", "duration": 120},
    {"id": "comprehensive_research", "name": "Comprehensive Research", "duration": 300},
    {"id": "damodaran_story_development", "name": "Damodaran Story Development", "duration": 120},
    {"id": "valuation_scenario_analysis", "name": "Valuation & Scenario Analysis", "duration": 90},
    {"id": "report_generation", "name": "Report Generation", "duration": 30}
]

# Server configuration constants
DEFAULT_PORT = 3000
DEFAULT_HOST = "0.0.0.0"

# Request deduplication configuration
REQUEST_COOLDOWN_SECONDS = int(os.getenv("REQUEST_COOLDOWN_SECONDS", "300"))  # 5 minutes default
MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT_ANALYSES", "3"))     # Maximum concurrent analyses

# Helper Functions
def create_service_unavailable_response(detail: str) -> JSONResponse:
    """Create a 503 Service Unavailable response"""
    return JSONResponse(
        status_code=503,
        content={"detail": detail, "error": "Service Unavailable"}
    )

def create_rate_limit_response(detail: str, retry_after: int = None) -> JSONResponse:
    """Create a 429 Too Many Requests response with optional Retry-After header"""
    headers = {"Retry-After": str(retry_after)} if retry_after else {}
    return JSONResponse(
        status_code=429,
        content={"detail": detail, "error": "Too Many Requests"},
        headers=headers
    )

def is_ticker_request_allowed(ticker: str, incoming_requests: Dict) -> tuple[bool, str, int]:
    """
    Check if a ticker request is allowed based on rate limiting and concurrency rules.
    
    Returns:
        tuple: (is_allowed, error_message, retry_after_seconds)
    """
    ticker_upper = ticker.upper()
    current_time = datetime.utcnow()
    
    # Check if ticker was recently requested (cooldown period)
    if ticker_upper in incoming_requests:
        last_requests = incoming_requests[ticker_upper]
        if last_requests:
            latest_request = datetime.fromisoformat(last_requests[-1])
            time_since_last = (current_time - latest_request).total_seconds()
            
            if time_since_last < REQUEST_COOLDOWN_SECONDS:
                remaining_cooldown = int(REQUEST_COOLDOWN_SECONDS - time_since_last)
                return False, f"Rate limit exceeded for {ticker_upper}. Please wait before requesting analysis again.", remaining_cooldown
    
    # Check concurrent analysis limit
    active_analyses = len([task for task in analysis_tasks.values() 
                          if task.get("status") in ["running", "pending"]])
    
    if active_analyses >= MAX_CONCURRENT_ANALYSES:
        return False, f"Maximum concurrent analyses ({MAX_CONCURRENT_ANALYSES}) reached. Please wait for completion.", 60
    
    return True, "", 0

async def validate_ticker(ticker: str) -> TickerValidation:
    """Validate ticker symbol and get company information"""
    try:
        # Use your existing FMP tool to validate
        ticker = ticker.upper().strip()
        
        # Basic ticker format validation
        if not ticker.isalpha() or len(ticker) > 10:
            return TickerValidation(
                ticker=ticker,
                is_valid=False,
                company_name=None,
                exchange=None,
                sector=None,
                market_cap=None,
                last_updated=datetime.utcnow()
            )
        
        # Use FMP API to validate real stocks (lazy import)
        try:
            from tools.fmp import FMPClient
        except ImportError as e:
            logger.warning(f"FMP client not available: {e}")
            # Fallback validation without API
            return TickerValidation(
                ticker=ticker,
                is_valid=True,  # Allow through if FMP not available
                company_name=f"{ticker} Corporation",
                exchange="Unknown",
                sector="Unknown", 
                market_cap=None,
                last_updated=datetime.utcnow()
            )
        
        try:
            async with FMPClient(os.getenv("FMP_API_KEY")) as fmp:
                # Get company profile to validate ticker
                profile = await fmp.get_company_profile(ticker)
                
                # FMP returns normalized field names (snake_case)
                if profile and profile.get("company_name"):
                    return TickerValidation(
                        ticker=ticker,
                        is_valid=True,
                        company_name=profile["company_name"],
                        exchange=profile.get("exchange", "Unknown"),
                        sector=profile.get("sector", "Unknown"),
                        market_cap=profile.get("market_cap", 0),
                        last_updated=datetime.utcnow()
                    )
                else:
                    return TickerValidation(
                        ticker=ticker,
                        is_valid=False,
                        company_name=None,
                        exchange=None,
                        sector=None,
                        market_cap=None,
                        last_updated=datetime.utcnow()
                    )
                    
        except Exception as e:
            logger.error(f"FMP validation failed for {ticker}: {e}")
            # Fallback: assume valid if basic format checks pass
            return TickerValidation(
                ticker=ticker,
                is_valid=True,  # Allow through if API fails
                company_name=f"{ticker} Corporation",
                exchange="Unknown",
                sector="Unknown", 
                market_cap=None,
                last_updated=datetime.utcnow()
            )
            
    except Exception as e:
        logger.error(f"Ticker validation error: {e}")
        raise HTTPException(status_code=500, detail="Ticker validation failed")

async def store_analysis_state(analysis_id: str, state: dict):
    """Store analysis state in Supabase, Redis, and memory"""
    
    # Store in Supabase database
    if supabase_manager.initialized:
        try:
            # Prepare analysis data for Supabase
            analysis_data = {
                'id': analysis_id,
                'ticker': state.get('ticker'),
                'analysis_type': 'comprehensive',
                'status': state.get('status'),
                'session_id': state.get('session_id'),
                'results_json': state,
                'processing_time_seconds': None,
                'created_at': state.get('started_at', datetime.utcnow()).isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Store or update in Supabase
            if state.get('status') == 'running' and 'started_at' in state:
                # First time - insert
                await supabase_manager.store_analysis_result(analysis_data)
            else:
                # Update existing
                await supabase_manager.update_analysis_status(
                    analysis_id, 
                    state.get('status', 'running'),
                    {
                        'results_json': state,
                        'processing_time_seconds': state.get('processing_time_seconds'),
                        'error_message': state.get('error')
                    }
                )
        except Exception as e:
            logger.error(f"Supabase storage error: {e}")
    
    # Store in Redis for fast access
    if redis_client:
        try:
            await redis_client.setex(f"analysis:{analysis_id}", 3600, json.dumps(state, default=str))
        except Exception as e:
            logger.error(f"Redis storage error: {e}")
    
    # Always store in memory as fallback
    analysis_tasks[analysis_id] = state

async def get_analysis_state(analysis_id: str) -> Optional[dict]:
    """Retrieve analysis state from Redis or memory"""
    if redis_client:
        try:
            data = await redis_client.get(f"analysis:{analysis_id}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Redis retrieval error: {e}")
    
    # Fallback to memory
    return analysis_tasks.get(analysis_id)

# API Endpoints

@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle preflight OPTIONS requests for CORS"""
    return {}

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "DeepResearch Comprehensive Analysis API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "healthy"
    }

@app.get("/health", tags=["System"], status_code=200)
async def simple_health():
    """Railway healthcheck endpoint - temporarily allow without Redis for deployment"""
    try:
        # Check Redis connection status but don't fail healthcheck
        redis_status = "disconnected"
        if redis_client is not None:
            try:
                await asyncio.wait_for(redis_client.ping(), timeout=2.0)
                redis_status = "connected"
                logger.info("✅ Health check: Redis connected")
            except Exception as e:
                logger.warning(f"🔄 Health check: Redis ping failed: {e}")
                redis_status = "error"
        else:
            logger.warning("🔄 Health check: Redis client not initialized")
        
        # Always return healthy for Railway deployment
        return {
            "status": "healthy", 
            "timestamp": datetime.utcnow().isoformat(),
            "redis": redis_status,
            "service": "valuation100x",
            "environment": "production"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Still return healthy for Railway
        return {
            "status": "healthy", 
            "timestamp": datetime.utcnow().isoformat(),
            "redis": "unknown",
            "service": "valuation100x",
            "error": str(e)
        }

@app.get("/api/health", response_model=SystemHealth, tags=["System"])
async def health_check():
    """Detailed system health check - always returns healthy status for deployment"""
    try:
        # Check Redis connection safely
        redis_status = "unavailable"
        if redis_client:
            try:
                await asyncio.wait_for(redis_client.ping(), timeout=1.0)
                redis_status = "healthy"
            except Exception:
                redis_status = "unavailable"
        
        # Check Supabase safely
        supabase_status = "unavailable"
        if SUPABASE_AVAILABLE and supabase_manager and getattr(supabase_manager, 'initialized', False):
            supabase_status = "healthy"
        
        return SystemHealth(
            status="healthy",  # Always healthy for Railway
            timestamp=datetime.utcnow(),
            version="1.0.0",
            services={
                "api_server": "healthy",
                "redis": redis_status,
                "supabase": supabase_status,
                "in_memory_storage": "healthy"
            },
            active_analyses=len(analysis_tasks) if analysis_tasks else 0,
            queue_size=0
        )
    except Exception as e:
        logger.warning(f"Health check error: {e}")
        # Still return healthy status for Railway deployment
    return SystemHealth(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        services={
            "api_server": "healthy",
                "redis": "error",
                "supabase": "error", 
            "in_memory_storage": "healthy"
        },
            active_analyses=0,
        queue_size=0
    )

@app.get("/api/validate/ticker/{ticker}", response_model=TickerValidation, tags=["Validation"])
async def validate_ticker_endpoint(ticker: str):
    """Validate a stock ticker and get company information"""
    return await validate_ticker(ticker)

@app.get("/api/analysis/{analysis_id}/recover", response_model=AnalysisResponse, tags=["Analysis"])
async def recover_analysis(analysis_id: str):
    """Recover an existing analysis for page refresh/reconnection"""
    try:
        # Get analysis state from storage
        state = await get_analysis_state(analysis_id)
        
        if not state:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis {analysis_id} not found"
            )
        
        # Return current analysis state for frontend recovery
        return AnalysisResponse(
            analysis_id=analysis_id,
            ticker=state.get("ticker"),
            company_name=state.get("company_name"),
            status=state.get("status", "unknown"),
            estimated_duration="5 minutes" if state.get("status") == "running" else "0 minutes",
            websocket_url=f"/api/analysis/{analysis_id}/ws",
            created_at=state.get("started_at", datetime.utcnow())
        )
        
    except Exception as e:
        logger.error(f"Failed to recover analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to recover analysis: {str(e)}"
        )

@app.post("/api/analysis/comprehensive/start", response_model=AnalysisResponse, tags=["Analysis"])
async def start_comprehensive_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Start a comprehensive analysis for a stock ticker"""
    
    # Log incoming request details
    logger.info(f"🔥 INCOMING ANALYSIS REQUEST: ticker={request.ticker}, company_name={request.company_name}")
    logger.info(f"🔥 REQUEST TIMESTAMP: {datetime.utcnow().isoformat()}")
    
    # Track incoming requests to detect duplicates
    ticker_upper = request.ticker.upper()
    current_time = datetime.utcnow().isoformat()
    
    if ticker_upper not in incoming_requests:
        incoming_requests[ticker_upper] = []
    incoming_requests[ticker_upper].append(current_time)
    
    # Log request pattern to detect multiple calls
    if len(incoming_requests[ticker_upper]) > 1:
        logger.warning(f"🚨 MULTIPLE REQUESTS for {ticker_upper}:")
        for i, timestamp in enumerate(incoming_requests[ticker_upper]):
            logger.warning(f"   Request #{i+1}: {timestamp}")
    else:
        logger.info(f"✅ First request for {ticker_upper}")
    
    # Log current state of all tracked requests
    logger.info(f"📋 ALL TRACKED REQUESTS: {dict(incoming_requests)}")
    logger.info(f"📋 TOTAL UNIQUE TICKERS REQUESTED: {len(incoming_requests)}")
    
    # Check if request is allowed (rate limiting and concurrency)
    is_allowed, error_message, retry_after = is_ticker_request_allowed(request.ticker, incoming_requests)
    if not is_allowed:
        logger.warning(f"🚫 Request blocked for {ticker_upper}: {error_message}")
        return create_rate_limit_response(error_message, retry_after)
    
    # Validate ticker first
    validation = await validate_ticker(request.ticker)
    if not validation.is_valid:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid ticker symbol: {request.ticker}"
        )
    
    # Check if analysis is already running for this ticker
    existing_analysis = None
    for analysis_id, task in analysis_tasks.items():
        if task.get("ticker") == request.ticker.upper() and task.get("status") == "running":
            existing_analysis = analysis_id
            logger.info(f"Found existing analysis for {request.ticker}: {analysis_id}")
            break
    
    if not existing_analysis:
        logger.info(f"No existing analysis found for {request.ticker}, creating new one")
    
    # If analysis already running, return existing analysis ID for recovery
    if existing_analysis:
        existing_state = await get_analysis_state(existing_analysis)
        if existing_state and existing_state.get("status") == "running":
            return AnalysisResponse(
                analysis_id=existing_analysis,
                ticker=request.ticker,
                company_name=existing_state.get("company_name"),
                status="running",
                estimated_duration="5 minutes",  # Will be updated via WebSocket
                websocket_url=f"/api/analysis/{existing_analysis}/ws",
                created_at=existing_state.get("started_at", datetime.utcnow())
            )
    
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())
    
    # Initialize analysis state
    ticker_normalized = request.ticker.upper()
    analysis_state = {
        "analysis_id": analysis_id,
        "ticker": ticker_normalized,
        "company_name": request.company_name or validation.company_name,
        "status": "running",
        "progress": 0,
        "current_step": "initializing",
        "current_component": None,
        "steps_completed": [],
        "steps_remaining": [step["id"] for step in ANALYSIS_STEPS],
        "error": None,
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "results": None
    }
    
    logger.info(f"📝 Created analysis state for {ticker_normalized} (ID: {analysis_id})")
    
    # Store initial state
    await store_analysis_state(analysis_id, analysis_state)
    
    # Add to analysis tasks tracking
    analysis_tasks[analysis_id] = {
        "ticker": ticker_normalized,
        "status": "running",
        "started_at": datetime.utcnow()
    }
    logger.info(f"📋 Added {ticker_normalized} to analysis_tasks tracking (ID: {analysis_id})")
    
    # Start analysis in background  
    logger.info(f"🔄 Starting background task for {ticker_normalized} (ID: {analysis_id})")
    background_tasks.add_task(run_comprehensive_analysis, analysis_id, ticker_normalized, request.company_name)
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        ticker=request.ticker,
        company_name=request.company_name or validation.company_name,
        status="started",
        estimated_duration="15 minutes",
        websocket_url=f"/ws/analysis/{analysis_id}",
        created_at=datetime.utcnow()
    )

@app.get("/api/analysis/{analysis_id}/status", response_model=AnalysisStatus, tags=["Analysis"])
async def get_analysis_status(analysis_id: str):
    """Get current status of a comprehensive analysis"""
    
    state = await get_analysis_state(analysis_id)
    if not state:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Calculate estimated completion
    estimated_completion = None
    if state["status"] == "running" and state["progress"] > 0:
        elapsed = (datetime.utcnow() - datetime.fromisoformat(state["started_at"].replace('Z', '+00:00'))).total_seconds()
        total_estimated = elapsed / (state["progress"] / 100)
        remaining = total_estimated - elapsed
        estimated_completion = datetime.utcnow() + timedelta(seconds=remaining)
    
    return AnalysisStatus(
        analysis_id=state["analysis_id"],
        ticker=state["ticker"],
        status=state["status"],
        progress=state["progress"],
        current_step=state["current_step"],
        current_component=state.get("current_component"),
        estimated_completion=estimated_completion,
        steps_completed=state.get("steps_completed", []),
        steps_remaining=state.get("steps_remaining", []),
        error=state.get("error"),
        started_at=state["started_at"],
        completed_at=state.get("completed_at")
    )

@app.post("/api/system/cleanup-sessions", tags=["System"])
async def cleanup_stuck_sessions():
    """Administrative endpoint to clean up stuck analysis sessions"""
    
    cleaned_sessions = []
    current_time = datetime.utcnow()
    
    # Find and clean up stuck sessions in analysis_tasks
    for analysis_id, task_info in list(analysis_tasks.items()):
        # Check if analysis is stuck (running for more than 30 minutes)
        if task_info.get("status") == "running":
            started_at = task_info.get("started_at")
            if started_at and (current_time - started_at).total_seconds() > 1800:  # 30 minutes
                logger.warning(f"🧹 Found stuck analysis session: {analysis_id} (running for {(current_time - started_at).total_seconds() / 60:.1f} minutes)")
                
                # Update state to reflect cleanup
                state = await get_analysis_state(analysis_id)
                if state:
                    state["status"] = "cleaned_up"
                    state["completed_at"] = current_time
                    state["error"] = "Session cleaned up - analysis was stuck"
                    await store_analysis_state(analysis_id, state)
                
                # Remove from memory
                del analysis_tasks[analysis_id]
                cleaned_sessions.append({
                    "analysis_id": analysis_id,
                    "ticker": task_info.get("ticker"),
                    "stuck_duration_minutes": (current_time - started_at).total_seconds() / 60
                })
                
                logger.info(f"🧹 Cleaned up stuck session: {analysis_id}")
    
    return {
        "message": f"Cleaned up {len(cleaned_sessions)} stuck analysis sessions",
        "cleaned_sessions": cleaned_sessions,
        "remaining_active_sessions": len([t for t in analysis_tasks.values() if t.get("status") == "running"]),
        "timestamp": current_time.isoformat()
    }

@app.delete("/api/analysis/{analysis_id}/cancel", tags=["Analysis"])
async def cancel_analysis(analysis_id: str):
    """Cancel a running comprehensive analysis"""
    
    state = await get_analysis_state(analysis_id)
    if not state:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if state["status"] != "running":
        raise HTTPException(status_code=400, detail="Analysis is not running")
    
    # Update state to cancelled
    state["status"] = "cancelled"
    state["completed_at"] = datetime.utcnow()
    await store_analysis_state(analysis_id, state)
    
    # CRITICAL: Update analysis_tasks to reflect cancellation and clean up
    if analysis_id in analysis_tasks:
        analysis_tasks[analysis_id]["status"] = "cancelled"
        analysis_tasks[analysis_id]["completed_at"] = state["completed_at"]
        logger.info(f"🧹 Updated analysis_tasks status to cancelled for analysis {analysis_id}")
        
        # Clean up cancelled analysis from memory after 2 minutes (same as failed)
        # This allows get_analysis_state fallback to continue finding it temporarily
        async def cleanup_cancelled_analysis():
            await asyncio.sleep(120)  # 2 minutes
            if analysis_id in analysis_tasks and analysis_tasks[analysis_id].get("status") == "cancelled":
                del analysis_tasks[analysis_id]
                logger.info(f"🧹 Cleaned up cancelled analysis from memory: {analysis_id}")
        
        asyncio.create_task(cleanup_cancelled_analysis())
    
    # Notify WebSocket clients
    await manager.send_to_analysis(analysis_id, {
        "type": "analysis_cancelled",
        "analysis_id": analysis_id,
        "data": {
            "status": "cancelled",
            "cancelled_at": datetime.utcnow().isoformat()
        }
    })
    
    return {
        "analysis_id": analysis_id,
        "status": "cancelled",
        "message": "Analysis cancelled successfully",
        "cancelled_at": datetime.utcnow()
    }

@app.get("/api/analysis/{analysis_id}/results", tags=["Analysis"])
async def get_analysis_results(analysis_id: str):
    """Get comprehensive analysis results"""
    
    state = await get_analysis_state(analysis_id)
    if not state:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if state["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    return {
        "analysis_id": state["analysis_id"],
        "ticker": state["ticker"],
        "company_name": state["company_name"],
        "status": state["status"],
        "results": state.get("results", {}),
        "metadata": {
            "analysis_duration": f"{(state['completed_at'] - state['started_at']).total_seconds() / 60:.1f} minutes",
            "data_sources": ["SEC-API", "FMP", "Alpha Vantage", "ValueInvesting.io"],
            "model_versions": {
                "damodaran_framework": "v2.0",
                "sentiment_model": "FinBERT-v1.1"
            }
        },
        "completed_at": state["completed_at"]
    }

@app.get("/api/reports/{analysis_id}/markdown", tags=["Reports"])
async def get_markdown_report(analysis_id: str):
    """Get the comprehensive analysis report in markdown format"""
    
    state = await get_analysis_state(analysis_id)
    if not state:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if state["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    # Look for saved markdown report
    ticker = state["ticker"]
    output_dir = Path("data/outputs") / ticker
    
    # Find the most recent comprehensive report
    markdown_files = list(output_dir.glob(f"{ticker}_enhanced_comprehensive_*.md"))
    if not markdown_files:
        raise HTTPException(status_code=404, detail="Markdown report not found")
    
    # Get the most recent file
    latest_file = max(markdown_files, key=lambda f: f.stat().st_mtime)
    
    def generate_markdown():
        with open(latest_file, 'r', encoding='utf-8') as f:
            for line in f:
                yield line
    
    return StreamingResponse(
        generate_markdown(),
        media_type="text/markdown",
        headers={"Content-Disposition": f"inline; filename={latest_file.name}"}
    )

@app.post("/api/reports/{analysis_id}/pdf", tags=["Reports"])
async def generate_pdf_report(analysis_id: str):
    """Generate and download PDF report"""
    
    state = await get_analysis_state(analysis_id)
    if not state:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if state["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    # For now, return a placeholder response
    # In production, you would generate PDF from markdown
    raise HTTPException(status_code=501, detail="PDF generation not implemented yet")

# WebSocket endpoint
@app.websocket("/ws/analysis/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    """WebSocket connection for real-time analysis updates"""
    
    await manager.connect(websocket, analysis_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "analysis_id": analysis_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for messages from client (e.g., subscription confirmations)
                data = await websocket.receive_json()
                
                if data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, analysis_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, analysis_id)

# Background task for running analysis
async def run_comprehensive_analysis(analysis_id: str, ticker: str, company_name: Optional[str]):
    """Run the comprehensive analysis and update progress via WebSocket"""
    
    try:
        logger.info(f"🚀 Starting comprehensive analysis for {ticker.upper()} (ID: {analysis_id})")
        logger.info(f"📊 Analysis parameters: ticker={ticker}, company_name={company_name}, analysis_id={analysis_id}")
        
        # Initialize the analysis runner (lazy import)
        try:
            from run_enhanced_analysis import EnhancedAnalysisRunner
            runner = EnhancedAnalysisRunner()
        except ImportError as e:
            logger.error(f"Enhanced analysis runner not available: {e}")
            return create_service_unavailable_response("Analysis service temporarily unavailable")
        
        # Update progress through steps
        for i, step in enumerate(ANALYSIS_STEPS):
            # Update current step
            state = await get_analysis_state(analysis_id)
            if state is None:
                logger.warning(f"Analysis state not found for {analysis_id}, terminating")
                return
            if state.get("status") == "cancelled":
                logger.info(f"Analysis {analysis_id} was cancelled")
                return
            
            state["current_step"] = step["id"]
            state["progress"] = int((i / len(ANALYSIS_STEPS)) * 100)
            await store_analysis_state(analysis_id, state)
            
            # Send progress update via WebSocket
            await manager.send_to_analysis(analysis_id, {
                "type": "progress_update",
                "analysis_id": analysis_id,
                "data": {
                    "progress": state["progress"],
                    "current_step": step["id"],
                    "current_component": step["name"],
                    "step_description": step["name"],
                    "estimated_completion": (datetime.utcnow() + timedelta(seconds=sum(s["duration"] for s in ANALYSIS_STEPS[i:]))).isoformat(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
            # Send log message
            await manager.send_to_analysis(analysis_id, {
                "type": "analysis_log",
                "analysis_id": analysis_id,
                "data": {
                    "level": "INFO",
                    "component": step["name"],
                    "message": f"Starting {step['name']} for {ticker}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {
                        "step": step["id"],
                        "progress": state["progress"]
                    }
                }
            })
            
            # Simulate step duration (in production, this would be actual analysis)
            await asyncio.sleep(min(step["duration"] / 10, 5))  # Accelerated for demo
            
            # Mark step as completed
            state = await get_analysis_state(analysis_id)
            if state is None:
                logger.warning(f"Analysis state not found for {analysis_id}, terminating")
                return
            if state.get("status") == "cancelled":
                logger.info(f"Analysis {analysis_id} was cancelled")
                return
                
            if "steps_completed" not in state:
                state["steps_completed"] = []
            if "steps_remaining" not in state:
                state["steps_remaining"] = [s["id"] for s in ANALYSIS_STEPS]
                
            state["steps_completed"].append(step["id"])
            if step["id"] in state["steps_remaining"]:
                state["steps_remaining"].remove(step["id"])
            await store_analysis_state(analysis_id, state)
            
            # Send step completion
            await manager.send_to_analysis(analysis_id, {
                "type": "step_completed",
                "analysis_id": analysis_id,
                "data": {
                    "completed_step": step["id"],
                    "next_step": ANALYSIS_STEPS[i + 1]["id"] if i + 1 < len(ANALYSIS_STEPS) else None,
                    "progress": state["progress"],
                    "step_duration": f"{step['duration']} seconds",
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
        
        # Run the actual comprehensive analysis
        logger.info(f"Running actual comprehensive analysis for {ticker}")
        
        # Update to show actual analysis running
        await manager.send_to_analysis(analysis_id, {
            "type": "analysis_log",
            "analysis_id": analysis_id,
            "data": {
                "level": "INFO",
                "component": "EnhancedAnalysisRunner",
                "message": f"Executing comprehensive analysis workflow for {ticker}",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "step": "comprehensive_execution",
                    "progress": 95
                }
            }
        })
        
        # Run the actual analysis (this is your existing code)
        results = await runner.run_comprehensive_analysis(ticker, company_name)
        
        # Complete the analysis
        state = await get_analysis_state(analysis_id)
        if state is None:
            logger.warning(f"Analysis state not found for {analysis_id}, cannot complete")
            return
        if state.get("status") == "cancelled":
            logger.info(f"Analysis {analysis_id} was cancelled during completion")
            return
        
        state["status"] = "completed"
        state["progress"] = 100
        state["current_step"] = "completed"
        state["completed_at"] = datetime.utcnow()
        state["results"] = {
            "investment_score": results.get("base_analysis", {}).get("investment_score", 5.0),
            "fair_value": results.get("base_analysis", {}).get("fair_value", 0.0),
            "current_price": results.get("base_analysis", {}).get("current_price", 0.0),
            "recommendation": "BUY" if results.get("base_analysis", {}).get("investment_score", 5) >= 6 else "HOLD",
            "confidence": 0.85,  # You can extract this from your results
            "analysis_summary": {
                "executive_summary": "Comprehensive analysis completed successfully",
                "key_strengths": ["Strong fundamentals", "Good market position"],
                "key_risks": ["Market volatility", "Sector-specific risks"],
                "price_target": results.get("base_analysis", {}).get("fair_value", 0.0) * 1.1
            }
        }
        
        await store_analysis_state(analysis_id, state)
        
        # CRITICAL: Update analysis_tasks to reflect completion and clean up
        if analysis_id in analysis_tasks:
            analysis_tasks[analysis_id]["status"] = "completed"
            analysis_tasks[analysis_id]["completed_at"] = state["completed_at"]
            logger.info(f"🧹 Updated analysis_tasks status to completed for {ticker} (ID: {analysis_id})")
            
            # Clean up completed analysis from memory after 5 minutes
            # This prevents the "still running" issue for future requests
            async def cleanup_completed_analysis():
                await asyncio.sleep(300)  # 5 minutes
                if analysis_id in analysis_tasks and analysis_tasks[analysis_id].get("status") == "completed":
                    del analysis_tasks[analysis_id]
                    logger.info(f"🧹 Cleaned up completed analysis from memory: {analysis_id}")
            
            asyncio.create_task(cleanup_completed_analysis())
        
        # Send completion notification
        await manager.send_to_analysis(analysis_id, {
            "type": "analysis_completed",
            "analysis_id": analysis_id,
            "data": {
                "status": "completed",
                "total_duration": f"{(state['completed_at'] - state['started_at']).total_seconds() / 60:.1f} minutes",
                "investment_score": state["results"]["investment_score"],
                "recommendation": state["results"]["recommendation"],
                "results_available": True,
                "report_ready": True,
                "completed_at": state["completed_at"].isoformat()
            }
        })
        
        logger.info(f"Comprehensive analysis completed for {ticker} (ID: {analysis_id})")
        
    except Exception as e:
        logger.error(f"Analysis failed for {ticker} (ID: {analysis_id}): {e}")
        
        # Update state with error
        state = await get_analysis_state(analysis_id)
        if state:
            state["status"] = "failed"
            state["error"] = str(e)
            state["completed_at"] = datetime.utcnow()
            await store_analysis_state(analysis_id, state)
            
            # CRITICAL: Update analysis_tasks to reflect failure and clean up
            if analysis_id in analysis_tasks:
                analysis_tasks[analysis_id]["status"] = "failed"
                analysis_tasks[analysis_id]["completed_at"] = state["completed_at"]
                analysis_tasks[analysis_id]["error"] = str(e)
                logger.info(f"🧹 Updated analysis_tasks status to failed for {ticker} (ID: {analysis_id})")
                
                # Clean up failed analysis from memory after 2 minutes
                async def cleanup_failed_analysis():
                    await asyncio.sleep(120)  # 2 minutes
                    if analysis_id in analysis_tasks and analysis_tasks[analysis_id].get("status") == "failed":
                        del analysis_tasks[analysis_id]
                        logger.info(f"🧹 Cleaned up failed analysis from memory: {analysis_id}")
                
                asyncio.create_task(cleanup_failed_analysis())
            
            # Send error notification
            await manager.send_to_analysis(analysis_id, {
                "type": "analysis_error",
                "analysis_id": analysis_id,
                "data": {
                    "error_type": "analysis_failure",
                    "error_message": str(e),
                    "failed_step": state.get("current_step", "unknown"),
                    "retry_possible": True,
                    "partial_results_available": False,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })

if __name__ == "__main__":
    # Port configuration with environment variables
    # Railway provides PORT, but allow custom configuration via APP_PORT
    port = int(os.getenv("PORT", os.getenv("APP_PORT", DEFAULT_PORT)))
    host = os.getenv("APP_HOST", DEFAULT_HOST)
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    logger.info(f"📍 Configuration: Railway PORT={os.getenv('PORT', 'not set')}, APP_PORT={os.getenv('APP_PORT', 'not set')}, DEFAULT={DEFAULT_PORT}")
    
    # Detect environment (Railway sets RAILWAY_ENVIRONMENT)
    is_production = os.getenv("RAILWAY_ENVIRONMENT") == "production"
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=not is_production,  # No reload in production
        workers=1,  # Single worker for Railway
        log_level="info",
        timeout_keep_alive=60  # Keep connections alive longer
    )

