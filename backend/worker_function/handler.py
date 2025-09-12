#!/usr/bin/env python3
"""
DeepResearch Worker Function - SQS Handler
Executes heavy financial analysis jobs from SQS queue
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add common layer to Python path
sys.path.append('/opt/python')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import analysis runner from common layer
from run_enhanced_analysis import EnhancedAnalysisRunner

# Try to import Redis for state management
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# ✅ Pre-load the heavy runner in the global scope for performance
# This will be reused across warm Lambda invocations
logger.info("🔄 Pre-loading EnhancedAnalysisRunner...")
try:
    ANALYSIS_RUNNER = EnhancedAnalysisRunner()
    logger.info("✅ EnhancedAnalysisRunner pre-loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to pre-load EnhancedAnalysisRunner: {e}")
    ANALYSIS_RUNNER = None

# Redis client for state management
redis_client = None
if REDIS_AVAILABLE and redis:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        redis_client = redis.from_url(redis_url)
        redis_client.ping()
        logger.info("✅ Redis connected for state management")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None

# Analysis steps for progress tracking
ANALYSIS_STEPS = [
    {
        "id": "initialization",
        "name": "Initialization",
        "description": "Setting up analysis environment",
        "user_message": "Preparing analysis environment...",
        "progress_weight": 5
    },
    {
        "id": "sec_documents",
        "name": "SEC Document Processing",
        "description": "Downloading and processing SEC filings",
        "user_message": "Downloading fresh SEC documents and creating vector index...",
        "progress_weight": 20
    },
    {
        "id": "base_analysis",
        "name": "Base Financial Analysis",
        "description": "Running core Damodaran analysis",
        "user_message": "Analyzing financial statements and calculating intrinsic value...",
        "progress_weight": 25
    },
    {
        "id": "comprehensive_report",
        "name": "Comprehensive Report",
        "description": "Generating detailed analysis report",
        "user_message": "Creating comprehensive investment analysis report...",
        "progress_weight": 20
    },
    {
        "id": "markdown_formatting",
        "name": "Report Formatting",
        "description": "Formatting final report",
        "user_message": "Formatting final report with charts and visualizations...",
        "progress_weight": 10
    },
    {
        "id": "results_saving",
        "name": "Saving Results",
        "description": "Saving analysis results",
        "user_message": "Saving analysis results and generating outputs...",
        "progress_weight": 10
    },
    {
        "id": "quality_checks",
        "name": "Quality Validation",
        "description": "Running quality checks",
        "user_message": "Validating analysis quality and completeness...",
        "progress_weight": 5
    },
    {
        "id": "completion",
        "name": "Completion",
        "description": "Analysis completed successfully",
        "user_message": "Analysis completed! Results are ready for review.",
        "progress_weight": 5
    }
]

async def update_analysis_state(analysis_id: str, updates: Dict[str, Any]):
    """Update analysis state in Redis or log if not available"""
    try:
        if redis_client:
            # Get current state
            current_data = redis_client.get(f"analysis:{analysis_id}")
            if current_data:
                current_state = json.loads(current_data)
            else:
                current_state = {}
            
            # Update with new data
            current_state.update(updates)
            current_state["updated_at"] = datetime.utcnow().isoformat()
            
            # Store back to Redis
            redis_client.setex(f"analysis:{analysis_id}", 3600, json.dumps(current_state, default=str))
            logger.info(f"📊 Updated analysis state for {analysis_id}: {updates}")
        else:
            logger.warning(f"📊 State update (Redis unavailable): {analysis_id} - {updates}")
    except Exception as e:
        logger.error(f"❌ Failed to update analysis state: {e}")

def handler(event, context):
    """
    SQS handler for processing analysis jobs
    This handler is triggered by messages on the SQS queue
    """
    logger.info("🚀 Worker function invoked")
    
    if not ANALYSIS_RUNNER:
        logger.error("❌ Analysis runner not available - cannot process jobs")
        return {"statusCode": 500, "body": "Analysis runner not loaded"}
    
    # Process each SQS record
    for record in event['Records']:
        try:
            message_body = json.loads(record['body'])
            analysis_id = message_body['analysis_id']
            ticker = message_body['ticker']
            company_name = message_body.get('company_name')
            
            logger.info(f"📊 Starting analysis for job_id: {analysis_id} on ticker: {ticker}")
            
            # Update status to processing
            await update_analysis_state(analysis_id, {
                "status": "processing",
                "current_step": "initialization",
                "current_step_name": "Initialization",
                "current_step_description": "Setting up analysis environment",
                "user_message": "Analysis is starting...",
                "progress": 5
            })
            
            # Execute the comprehensive analysis
            try:
                # Create a custom progress callback
                async def progress_callback(step_id: str, progress: int, message: str = None):
                    """Update progress during analysis"""
                    step_info = next((s for s in ANALYSIS_STEPS if s["id"] == step_id), None)
                    if step_info:
                        await update_analysis_state(analysis_id, {
                            "current_step": step_id,
                            "current_step_name": step_info["name"],
                            "current_step_description": step_info["description"],
                            "user_message": message or step_info["user_message"],
                            "progress": progress
                        })
                
                # Run the analysis with progress tracking
                logger.info(f"🔬 Executing comprehensive analysis for {ticker}")
                
                # Update to processing state
                await progress_callback("sec_documents", 10, "Downloading SEC documents...")
                
                # Run the main analysis
                results = await ANALYSIS_RUNNER.run_comprehensive_analysis(
                    ticker=ticker,
                    company_name=company_name,
                    verbose=True
                )
                
                # Update progress through completion
                await progress_callback("completion", 100, "Analysis completed successfully!")
                
                # Mark as completed with results
                await update_analysis_state(analysis_id, {
                    "status": "completed",
                    "progress": 100,
                    "completed_at": datetime.utcnow().isoformat(),
                    "results": results,
                    "current_step": "completed",
                    "current_step_name": "Completed",
                    "user_message": "Analysis completed successfully! Results are ready for review."
                })
                
                logger.info(f"✅ Successfully completed analysis for job_id: {analysis_id}")
                
            except Exception as analysis_error:
                error_message = f"Analysis execution failed: {str(analysis_error)}"
                logger.error(f"❌ Analysis error for {analysis_id}: {error_message}")
                
                # Mark as failed
                await update_analysis_state(analysis_id, {
                    "status": "failed",
                    "error": error_message,
                    "completed_at": datetime.utcnow().isoformat(),
                    "current_step": "failed",
                    "current_step_name": "Failed",
                    "user_message": f"Analysis failed: {error_message}"
                })
                
                # Re-raise to trigger SQS retry/DLQ handling
                raise analysis_error
                
        except Exception as e:
            error_message = f"ERROR processing SQS message: {str(e)}"
            logger.error(error_message)
            
            # Try to update state if we have analysis_id
            try:
                if 'analysis_id' in locals():
                    await update_analysis_state(analysis_id, {
                        "status": "failed",
                        "error": error_message,
                        "completed_at": datetime.utcnow().isoformat()
                    })
            except:
                pass
            
            # Re-raise the exception to trigger SQS retry/DLQ handling
            raise e
    
    return {"statusCode": 200, "body": "Processing completed"}

# For testing locally
if __name__ == "__main__":
    # Test event
    test_event = {
        'Records': [{
            'body': json.dumps({
                'analysis_id': 'test-123',
                'ticker': 'AAPL',
                'company_name': 'Apple Inc.'
            })
        }]
    }
    
    result = handler(test_event, None)
    print(f"Test result: {result}")


