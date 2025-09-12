# tests/test_worker.py
"""
Unit tests for the Worker Function
Tests the SQS handler that processes analysis jobs
"""

import json
import pytest
import sys
import os

# Add the worker_function directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'worker_function'))

# Import the worker handler
from handler import handler as worker_handler

@pytest.fixture
def mock_sqs_event():
    """Creates a mock SQS event for a ticker analysis job."""
    return {
        "Records": [
            {
                "messageId": "19dd0b57-b21e-4ac1-bd88-01bbb068cb78",
                "receiptHandle": "MessageReceiptHandle",
                "body": json.dumps({
                    "analysis_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                    "ticker": "AAPL",
                    "company_name": "Apple Inc."
                }),
                "attributes": {
                    "ApproximateReceiveCount": "1",
                    "SentTimestamp": "1523232000000",
                    "SenderId": "123456789012",
                    "ApproximateFirstReceiveTimestamp": "1523232000001"
                },
                "messageAttributes": {},
                "md5OfBody": "7b270e59b47ff90a553787216d55d91d",
                "eventSource": "aws:sqs",
                "eventSourceARN": "arn:aws:sqs:us-east-1:123456789012:MyQueue",
                "awsRegion": "us-east-1"
            }
        ]
    }

@pytest.fixture
def mock_context():
    """Creates a mock Lambda context object."""
    class MockContext:
        def __init__(self):
            self.function_name = "test-worker-function"
            self.function_version = "1"
            self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-worker-function"
            self.memory_limit_in_mb = "3008"
            self.remaining_time_in_millis = lambda: 900000
            self.log_group_name = "/aws/lambda/test-worker-function"
            self.log_stream_name = "2023/01/01/[$LATEST]test123"
            self.aws_request_id = "test-request-id-123"
    
    return MockContext()

def test_worker_handler_success(mock_sqs_event, mock_context, mocker):
    """
    Tests that the worker handler can successfully process a mock SQS event.
    """
    # Mock the heavy analysis runner to prevent it from actually running
    mock_runner = mocker.patch('handler.ANALYSIS_RUNNER')
    mock_runner.run_comprehensive_analysis.return_value = {
        "status": "completed",
        "results": {
            "investment_score": 8.5,
            "fair_value": 150.0,
            "current_price": 145.0
        }
    }
    
    # Mock the state update function to prevent Redis calls
    mocker.patch('handler.update_analysis_state')
    
    # Call the handler with the mock event
    result = worker_handler(mock_sqs_event, mock_context)
    
    # Verify the handler returned success
    assert result["statusCode"] == 200
    assert result["body"] == "Processing completed"
    
    # Verify the analysis runner was called with correct parameters
    mock_runner.run_comprehensive_analysis.assert_called_once_with(
        ticker="AAPL", 
        company_name="Apple Inc.",
        verbose=True
    )
    
    print("✅ Worker handler test passed - SQS event processed successfully")

def test_worker_handler_analysis_failure(mock_sqs_event, mock_context, mocker):
    """
    Tests that the worker handler properly handles analysis failures.
    """
    # Mock the analysis runner to raise an exception
    mock_runner = mocker.patch('handler.ANALYSIS_RUNNER')
    mock_runner.run_comprehensive_analysis.side_effect = Exception("Analysis failed due to API timeout")
    
    # Mock the state update function
    mock_update_state = mocker.patch('handler.update_analysis_state')
    
    # The handler should re-raise the exception for SQS retry handling
    with pytest.raises(Exception, match="Analysis failed due to API timeout"):
        worker_handler(mock_sqs_event, mock_context)
    
    # Verify the failure was logged in state
    mock_update_state.assert_called()
    
    print("✅ Worker handler failure test passed - Exception properly handled")

def test_worker_handler_invalid_message(mock_context, mocker):
    """
    Tests that the worker handler handles invalid SQS messages gracefully.
    """
    invalid_event = {
        "Records": [
            {
                "messageId": "invalid-message",
                "body": "invalid-json-content"
            }
        ]
    }
    
    # Mock the state update function
    mocker.patch('handler.update_analysis_state')
    
    # The handler should raise an exception for invalid JSON
    with pytest.raises(json.JSONDecodeError):
        worker_handler(invalid_event, mock_context)
    
    print("✅ Invalid message test passed - JSON decode error properly handled")

def test_worker_handler_missing_analysis_runner(mock_sqs_event, mock_context, mocker):
    """
    Tests that the worker handler fails gracefully when ANALYSIS_RUNNER is not available.
    """
    # Mock ANALYSIS_RUNNER as None (not loaded)
    mocker.patch('handler.ANALYSIS_RUNNER', None)
    
    # Call the handler
    result = worker_handler(mock_sqs_event, mock_context)
    
    # Verify it returns an error status
    assert result["statusCode"] == 500
    assert "Analysis runner not loaded" in result["body"]
    
    print("✅ Missing runner test passed - Proper error handling when runner unavailable")

if __name__ == "__main__":
    # Run tests directly if executed as script
    pytest.main([__file__, "-v"])







