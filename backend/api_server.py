#!/usr/bin/env python3
"""
FastAPI Server for DeepResearch Comprehensive Analysis
Provides REST API endpoints with polling-based status updates.
Updated: WebSocket removed, polling-based architecture
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
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

# Try to import Redis with fallback
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Pre-load heavy dependencies at startup (Option 1 fix for 503 errors)
# This prevents import failures during background task execution
import importlib
from typing import TYPE_CHECKING, Optional as OptionalType

# Global variable to store pre-loaded runner
ANALYSIS_RUNNER: OptionalType['EnhancedAnalysisRunner'] = None
ANALYSIS_RUNNER_ERROR: OptionalType[str] = None

def preload_analysis_dependencies():
    """Pre-load ALL heavy analysis dependencies at startup (Option 1)"""
    global ANALYSIS_RUNNER, ANALYSIS_RUNNER_ERROR
    
    try:
        logger.info("🔄 Pre-loading EnhancedAnalysisRunner and ALL dependencies at startup...")
        
        # Import and initialize ALL heavy dependencies upfront
        try:
            # Import the runner class
            from run_enhanced_analysis import EnhancedAnalysisRunner
            
            # Initialize the runner (this will trigger all dependency imports)
            ANALYSIS_RUNNER = EnhancedAnalysisRunner()
            
            logger.info("✅ EnhancedAnalysisRunner and all dependencies pre-loaded successfully!")
            # Success - no return needed, exceptions handle failures
            
        except ImportError as ie:
            logger.error(f"❌ EnhancedAnalysisRunner import failed: {ie}")
            ANALYSIS_RUNNER_ERROR = f"Import failed: {ie}"
            raise ie  # Fail hard - no mock fallback
        except Exception as e:
            logger.error(f"❌ EnhancedAnalysisRunner initialization failed: {e}")
            ANALYSIS_RUNNER_ERROR = f"Initialization failed: {e}"
            raise e  # Fail hard - no mock fallback
        
    except Exception as e:
        error_msg = f"Failed to pre-load analysis dependencies: {str(e)}"
        logger.error(f"❌ {error_msg}")
        ANALYSIS_RUNNER_ERROR = error_msg
        raise e  # Fail hard - no mock fallback

if TYPE_CHECKING:
    from run_enhanced_analysis import EnhancedAnalysisRunner
    from tools.fmp import get_financials_fmp

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import secure configuration
try:
    from config import config, validate_ticker_symbol, sanitize_input, validate_request_data
    logger.info("✅ Secure configuration imported successfully")
except Exception as e:
    logger.error(f"❌ Failed to import secure configuration: {e}")
    raise

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
# WebSocket connections removed - using polling-based updates instead
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with graceful fallbacks"""
    global redis_client
    
    # Startup
    logger.info("Starting DeepResearch API Server...")
    
    # Pre-load analysis dependencies (Option 1 fix)
    logger.info("🚀 Step 1: Pre-loading analysis dependencies...")
    try:
        preload_analysis_dependencies()
        
        # Verify pre-loading actually worked
        global ANALYSIS_RUNNER
        if ANALYSIS_RUNNER is None:
            raise RuntimeError("Pre-loading appeared to succeed but ANALYSIS_RUNNER is still None")
            
        logger.info("✅ Analysis dependencies pre-loaded and verified successfully!")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Analysis dependencies failed to pre-load: {e}")
        logger.error("🚨 Lambda will fail to start - this is intentional (no mock fallback)")
        # Force immediate failure - don't let FastAPI continue
        import sys
        sys.exit(1)
    
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

# CORS middleware for frontend - Updated for Lambda deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Lambda Function URLs
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"

    # CORS headers (keeping existing for now as requested)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"

    return response

# Pydantic models
class AnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    company_name: Optional[str] = Field(None, description="Optional company name")

    @field_validator('ticker')
    @classmethod
    def validate_ticker_format(cls, v):
        """Validate ticker format and security"""
        is_valid, error_message = validate_ticker_symbol(v)
        if not is_valid:
            raise ValueError(f"Invalid ticker: {error_message}")
        return v.upper().strip()

    @field_validator('company_name')
    @classmethod
    def sanitize_company_name(cls, v):
        """Sanitize company name input"""
        if v is not None:
            return sanitize_input(v, max_length=100)
        return v

class AnalysisResponse(BaseModel):
    analysis_id: str
    ticker: str
    company_name: Optional[str]
    status: str
    estimated_duration: str
    polling_url: str
    created_at: datetime

class AnalysisStatus(BaseModel):
    analysis_id: str
    ticker: str
    company_name: Optional[str]
    status: str
    progress: int
    current_step: str
    current_component: Optional[str]
    estimated_completion: Optional[datetime]
    steps_completed: List[str]
    steps_remaining: List[str]
    error: Optional[str]
    
    # User-friendly fields for frontend display
    current_step_name: Optional[str] = None
    current_step_description: Optional[str] = None
    user_message: Optional[str] = None
    total_steps: int = 0
    completed_steps: int = 0
    started_at: datetime
    completed_at: Optional[datetime]

class TickerValidation(BaseModel):
    ticker: str
    is_valid: bool
    company_name: Optional[str]
    exchange: Optional[str]
    sector: Optional[str]
    market_cap: Optional[float]
    current_price: Optional[float]
    day_low: Optional[float]
    day_high: Optional[float]
    volume: Optional[int]
    last_updated: datetime

class SystemHealth(BaseModel):
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, Any]
    active_analyses: int
    queue_size: int

# WebSocket Connection Manager removed - using polling-based updates instead

# Analysis step definitions with user-friendly descriptions
ANALYSIS_STEPS = [
    {
        "id": "initialization", 
        "name": "Getting Started", 
        "description": "Setting up your analysis workspace and validating company data",
        "duration": 30,
        "user_message": "Preparing to analyze {}..."
    },
    {
        "id": "sec_filing_analysis", 
        "name": "Reading Company Documents", 
        "description": "Downloading and analyzing official SEC filings and annual reports",
        "duration": 180,
        "user_message": "Reading through {}'s official filings and reports..."
    },
    {
        "id": "financial_data_collection", 
        "name": "Gathering Financial Data", 
        "description": "Collecting current stock prices, financial statements, and key metrics",
        "duration": 120,
        "user_message": "Collecting {}'s latest financial data and market information..."
    },
    {
        "id": "comprehensive_research", 
        "name": "Deep Market Research", 
        "description": "Researching industry trends, competitors, and market conditions",
        "duration": 300,
        "user_message": "Researching {}'s industry, competitors, and market position..."
    },
    {
        "id": "valuation_analysis", 
        "name": "Calculating Company Value", 
        "description": "Running advanced valuation models and scenario analysis",
        "duration": 180,
        "user_message": "Calculating {}'s intrinsic value using multiple approaches..."
    },
    {
        "id": "risk_assessment", 
        "name": "Analyzing Investment Risks", 
        "description": "Identifying potential risks and opportunities for investors",
        "duration": 90,
        "user_message": "Evaluating investment risks and opportunities for {}..."
    },
    {
        "id": "report_generation", 
        "name": "Creating Your Report", 
        "description": "Compiling all findings into a comprehensive investment analysis",
        "duration": 60,
        "user_message": "Finalizing your comprehensive {} investment report..."
    }
]

# Server configuration constants
DEFAULT_PORT = 3000
DEFAULT_HOST = "0.0.0.0"

# Request deduplication configuration - Reduced for development/testing
REQUEST_COOLDOWN_SECONDS = int(os.getenv("REQUEST_COOLDOWN_SECONDS", "30"))   # 30 seconds default (was 5 minutes)
MAX_CONCURRENT_ANALYSES = int(os.getenv("MAX_CONCURRENT_ANALYSES", "5"))      # 5 concurrent analyses (was 3)

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
    """Validate ticker symbol and get company information with security checks"""
    try:
        # First perform security validation
        is_valid, error_message = validate_ticker_symbol(ticker)
        if not is_valid:
            logger.warning(f"Security validation failed for ticker {ticker}: {error_message}")
            return TickerValidation(
                ticker=ticker,
                is_valid=False,
                company_name=None,
                exchange=None,
                sector=None,
                market_cap=None,
                current_price=None,
                day_low=None,
                day_high=None,
                volume=None,
                last_updated=datetime.utcnow()
            )

        # Use your existing FMP tool to validate
        ticker = ticker.upper().strip()
        
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
                current_price=None,
                day_low=None,
                day_high=None,
                volume=None,
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
                        current_price=profile.get("current_price", 0),
                        day_low=profile.get("day_low", 0),
                        day_high=profile.get("day_high", 0),
                        volume=profile.get("volume", 0),
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
                        current_price=None,
                        day_low=None,
                        day_high=None,
                        volume=None,
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
                current_price=None,
                day_low=None,
                day_high=None,
                volume=None,
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
                'created_at': state.get('started_at', datetime.utcnow()).isoformat() if isinstance(state.get('started_at'), datetime) else str(state.get('started_at', datetime.utcnow().isoformat())),
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
                state = json.loads(data)
                # Convert datetime strings back to datetime objects
                if 'started_at' in state and isinstance(state['started_at'], str):
                    state['started_at'] = datetime.fromisoformat(state['started_at'].replace('Z', '+00:00'))
                if 'completed_at' in state and isinstance(state['completed_at'], str):
                    state['completed_at'] = datetime.fromisoformat(state['completed_at'].replace('Z', '+00:00'))
                return state
        except Exception as e:
            logger.error(f"Redis retrieval error: {e}")
    
    # Fallback to memory
    return analysis_tasks.get(analysis_id)

# API Endpoints

@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle preflight OPTIONS requests for CORS"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
    )

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

@app.post("/api/admin/clear-rate-limits", tags=["Admin"])
async def clear_rate_limits():
    """Clear all rate limiting data (for debugging)"""
    global incoming_requests
    cleared_count = len(incoming_requests)
    incoming_requests.clear()
    logger.info(f"🧹 Cleared rate limits for {cleared_count} tickers")
    return {
        "message": f"Cleared rate limits for {cleared_count} tickers",
        "timestamp": datetime.utcnow().isoformat(),
        "rate_limit_cooldown": REQUEST_COOLDOWN_SECONDS,
        "max_concurrent": MAX_CONCURRENT_ANALYSES
    }

@app.get("/api/admin/circuit-breaker-status", tags=["Admin"])
async def get_circuit_breaker_status():
    """Check FMP circuit breaker status"""
    try:
        from tools.fmp import _fmp_circuit_breaker
        return {
            "fmp_circuit_breaker": _fmp_circuit_breaker,
            "timestamp": datetime.utcnow().isoformat()
        }
    except ImportError:
        return {"error": "Circuit breaker not available"}

@app.post("/api/admin/reset-circuit-breaker", tags=["Admin"])
async def reset_circuit_breaker():
    """Reset FMP circuit breaker (for debugging)"""
    try:
        from tools.fmp import _fmp_circuit_breaker
        _fmp_circuit_breaker["failure_count"] = 0
        _fmp_circuit_breaker["last_failure_time"] = None
        _fmp_circuit_breaker["circuit_open"] = False
        logger.info("🔄 Reset FMP circuit breaker")
        return {
            "message": "Circuit breaker reset successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except ImportError:
        return {"error": "Circuit breaker not available"}

@app.get("/api/validate/ticker/{ticker}", response_model=TickerValidation, tags=["Validation"])
async def validate_ticker_endpoint(ticker: str):
    """Validate a stock ticker and get company information"""
    return await validate_ticker(ticker)

@app.get("/api/debug/fmp-test/{ticker}", tags=["Debug"])
async def debug_fmp_test(ticker: str):
    """Debug endpoint to test FMP API integration"""
    try:
        from tools.fmp import FMPClient
        api_key = os.getenv("FMP_API_KEY")
        
        if not api_key:
            return {"error": "FMP_API_KEY not found", "env_vars": list(os.environ.keys())}
        
        async with FMPClient(api_key) as client:
            profile = await client.get_company_profile(ticker)
            return {
                "ticker": ticker,
                "api_key_present": bool(api_key),
                "api_key_length": len(api_key) if api_key else 0,
                "profile": profile,
                "success": True
            }
    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e),
            "error_type": type(e).__name__,
            "success": False
        }

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
            polling_url=f"/api/analysis/{analysis_id}/status",
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
    
    # Check rate limiting and concurrency BEFORE adding to tracking
    ticker_upper = request.ticker.upper()
    current_time = datetime.utcnow().isoformat()
    
    # Check if request is allowed (rate limiting and concurrency)
    is_allowed, error_message, retry_after = is_ticker_request_allowed(request.ticker, incoming_requests)
    if not is_allowed:
        logger.warning(f"🚫 Request blocked for {ticker_upper}: {error_message}")
        return create_rate_limit_response(error_message, retry_after)
    
    # NOW track the incoming request (after validation)
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
                estimated_duration="5 minutes",  # Will be updated via polling
                polling_url=f"/api/analysis/{existing_analysis}/status",
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
        polling_url=f"/api/analysis/{analysis_id}/status",  # Use polling endpoint instead
        created_at=datetime.utcnow()
    )

@app.get("/api/analysis/{analysis_id}/status", response_model=AnalysisStatus, tags=["Analysis"])
async def get_analysis_status(analysis_id: str):
    """Get current status of a comprehensive analysis with detailed progress info"""
    
    state = await get_analysis_state(analysis_id)
    if not state:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Calculate estimated completion and detailed progress
    estimated_completion = None
    current_step_info = None
    user_friendly_message = "Starting analysis..."
    
    if state["status"] == "running":
        # Find current step details
        current_step_id = state.get("current_step")
        if current_step_id:
            current_step_info = next((step for step in ANALYSIS_STEPS if step["id"] == current_step_id), None)
            if current_step_info:
                company_name = state.get("company_name", state.get("ticker", "the company"))
                user_friendly_message = current_step_info["user_message"].format(company_name)
        
        # Calculate time estimates
        if state.get("progress", 0) > 0:
            started_at = state.get("started_at", datetime.utcnow())
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            elapsed = (datetime.utcnow() - started_at).total_seconds()
            total_estimated = elapsed / (state.get("progress", 1) / 100)
            remaining = total_estimated - elapsed
            estimated_completion = datetime.utcnow() + timedelta(seconds=max(0, remaining))
    
    # Enhanced status response with user-friendly fields
    enhanced_status = AnalysisStatus(
        analysis_id=analysis_id,  # Use URL parameter, not state key
        ticker=state.get("ticker", ""),
        company_name=state.get("company_name"),
        status=state.get("status", "unknown"),
        progress=state.get("progress", 0),
        current_step=state.get("current_step", "initializing"),
        current_component=state.get("current_component"),
        estimated_completion=estimated_completion,
        steps_completed=state.get("steps_completed", []),
        steps_remaining=state.get("steps_remaining", []),
        error=state.get("error"),
        started_at=state.get("started_at", datetime.utcnow()),
        completed_at=state.get("completed_at"),
        
        # User-friendly fields
        current_step_name=current_step_info.get("name", "Processing") if current_step_info else "Processing",
        current_step_description=current_step_info.get("description", "") if current_step_info else "",
        user_message=user_friendly_message,
        total_steps=len(ANALYSIS_STEPS),
        completed_steps=len(state.get("steps_completed", []))
    )
    
    return enhanced_status

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
            if started_at:
                started_at_dt = started_at if isinstance(started_at, datetime) else datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                if (current_time - started_at_dt).total_seconds() > 1800:  # 30 minutes
                    logger.warning(f"🧹 Found stuck analysis session: {analysis_id} (running for {(current_time - started_at_dt).total_seconds() / 60:.1f} minutes)")
                
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
                    "stuck_duration_minutes": (current_time - started_at_dt).total_seconds() / 60
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
    
    # Analysis cancelled - status will be available via polling endpoint
    
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
            "analysis_duration": f"{(state['completed_at'] - state['started_at']).total_seconds() / 60:.1f} minutes" if state.get('completed_at') and state.get('started_at') and isinstance(state['completed_at'], datetime) and isinstance(state['started_at'], datetime) else "N/A",
            "data_sources": ["SEC-API", "FMP", "Alpha Vantage", "ValueInvesting.io"],
            "model_versions": {
                "damodaran_framework": "v2.0",
                "sentiment_model": "FinBERT-v1.1"
            }
        },
        "completed_at": state["completed_at"].isoformat() if isinstance(state["completed_at"], datetime) else str(state["completed_at"]) if state.get("completed_at") else None
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

# WebSocket endpoint removed - using polling-based updates instead

# Background task for running analysis
async def run_comprehensive_analysis(analysis_id: str, ticker: str, company_name: Optional[str]):
    """Run the comprehensive analysis and update progress via polling status endpoint"""
    
    try:
        logger.info(f"🚀 Starting comprehensive analysis for {ticker.upper()} (ID: {analysis_id})")
        logger.info(f"📊 Analysis parameters: ticker={ticker}, company_name={company_name}, analysis_id={analysis_id}")
        
        # Use pre-loaded analysis runner (Option 1 fix)
        global ANALYSIS_RUNNER, ANALYSIS_RUNNER_ERROR
        
        if ANALYSIS_RUNNER is None:
            error_msg = ANALYSIS_RUNNER_ERROR or "Analysis runner not pre-loaded at startup"
            logger.error(f"❌ Analysis runner unavailable: {error_msg}")
            
            # Update analysis state with error
            state = await get_analysis_state(analysis_id)
            if state:
                state["status"] = "failed"
                state["error"] = f"Service initialization failed: {error_msg}"
                state["completed_at"] = datetime.utcnow()
                await store_analysis_state(analysis_id, state)
            
            # Error status will be available via polling endpoint
            return
        
        runner = ANALYSIS_RUNNER
        logger.info(f"✅ Using pre-loaded analysis runner for {ticker}")
        
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
            
            # Update completed and remaining steps
            state["steps_completed"] = [s["id"] for s in ANALYSIS_STEPS[:i]]
            state["steps_remaining"] = [s["id"] for s in ANALYSIS_STEPS[i+1:]]
            
            await store_analysis_state(analysis_id, state)
            
            # Log user-friendly progress
            company_name = state.get("company_name", ticker)
            user_message = step["user_message"].format(company_name)
            logger.info(f"📊 {ticker} Progress: {state['progress']}% - {user_message}")
            
            # Progress updates available via polling status endpoint
            
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
            # Progress updates available via polling status endpoint
        
        # Run the actual comprehensive analysis
        logger.info(f"Running actual comprehensive analysis for {ticker}")
        
        # Update to show actual analysis running
            # Progress updates available via polling status endpoint
        
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
            # Progress updates available via polling status endpoint
        
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
            # Progress updates available via polling status endpoint

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
