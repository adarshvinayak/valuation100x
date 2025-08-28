"""
AWS Lambda handler for FastAPI application using Mangum
"""
import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set environment variables for Lambda
os.environ.setdefault("AWS_LAMBDA_RUNTIME_API", "")

# Import the FastAPI app
from api_server import app

# Import Mangum for ASGI to Lambda adapter
try:
    from mangum import Mangum
    
    # Create the Lambda handler
    handler = Mangum(app, lifespan="off")
    
except ImportError:
    # Fallback if Mangum is not available
    def handler(event, context):
        return {
            "statusCode": 500,
            "body": "Mangum not installed. Please install with: pip install mangum"
        }


