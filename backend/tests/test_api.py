# tests/test_api.py
"""
Unit tests for the API Function
Tests the FastAPI endpoints for the lightweight API
"""

import pytest
import sys
import os
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

# Add the api_function directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api_function'))

# Import the FastAPI app
from api_server import app

# Create test client
client = TestClient(app)

def test_health_check():
    """
    Tests the health check endpoint of the API.
    """
    response = client.get("/health")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "deepresearch-api"
    assert data["version"] == "2.0.0"
    assert data["architecture"] == "sqs-worker"
    
    print("✅ Health check test passed")

def test_validate_ticker_endpoint():
    """
    Tests the ticker validation endpoint.
    Note: This test mocks the FMP API call to avoid external dependencies.
    """
    with patch('api_server.validate_ticker_symbol') as mock_validate:
        # Mock the validation response
        mock_validate.return_value = {
            "is_valid": True,
            "company_name": "Apple Inc.",
            "ticker": "AAPL",
            "exchange": "NASDAQ"
        }
        
        response = client.get("/api/validate/AAPL")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["is_valid"] is True
        assert data["company_name"] == "Apple Inc."
        assert data["ticker"] == "AAPL"
        
        # Verify the mock was called with uppercase ticker
        mock_validate.assert_called_once_with("AAPL")
        
        print("✅ Ticker validation test passed")

@patch('api_server.sqs')
@patch('api_server.store_analysis_state')
def test_start_analysis_endpoint(mock_store_state, mock_sqs):
    """
    Tests the analysis start endpoint.
    Mocks SQS to avoid actual message sending.
    """
    # Mock SQS send_message response
    mock_sqs.send_message.return_value = {
        'MessageId': 'test-message-id-123',
        'MD5OfBody': 'test-md5-hash'
    }
    
    # Mock state storage
    mock_store_state.return_value = None
    
    # Test data
    test_request = {
        "ticker": "TSLA",
        "company_name": "Tesla Inc."
    }
    
    response = client.post(
        "/api/analysis/comprehensive/start",
        json=test_request
    )
    
    assert response.status_code == 200
    
    data = response.json()
    assert "analysis_id" in data
    assert data["status"] == "queued"
    assert data["polling_url"].startswith("/api/analysis/")
    assert data["polling_url"].endswith("/status")
    
    # Verify SQS message was sent
    mock_sqs.send_message.assert_called_once()
    call_args = mock_sqs.send_message.call_args
    
    # Check the message body contains correct ticker
    import json
    message_body = json.loads(call_args[1]['MessageBody'])
    assert message_body["ticker"] == "TSLA"
    assert message_body["company_name"] == "Tesla Inc."
    
    print("✅ Start analysis test passed")

@patch('api_server.get_analysis_state')
def test_get_analysis_status_endpoint(mock_get_state):
    """
    Tests the analysis status endpoint.
    """
    # Mock analysis state
    mock_state = {
        "analysis_id": "test-analysis-123",
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "status": "processing",
        "progress": 45,
        "current_step": "base_analysis",
        "current_step_name": "Base Financial Analysis",
        "current_step_description": "Running core Damodaran analysis",
        "user_message": "Analyzing financial statements...",
        "total_steps": 8,
        "completed_steps": 3,
        "started_at": "2023-01-01T10:00:00"
    }
    
    mock_get_state.return_value = mock_state
    
    response = client.get("/api/analysis/test-analysis-123/status")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["analysis_id"] == "test-analysis-123"
    assert data["status"] == "processing"
    assert data["progress"] == 45
    assert data["ticker"] == "AAPL"
    assert data["current_step_name"] == "Base Financial Analysis"
    
    # Verify the mock was called with correct analysis_id
    mock_get_state.assert_called_once_with("test-analysis-123")
    
    print("✅ Analysis status test passed")

def test_get_analysis_status_not_found():
    """
    Tests the analysis status endpoint when analysis is not found.
    """
    with patch('api_server.get_analysis_state') as mock_get_state:
        # Mock returning None (analysis not found)
        mock_get_state.return_value = None
        
        response = client.get("/api/analysis/nonexistent-id/status")
        
        assert response.status_code == 404
        assert "Analysis not found" in response.json()["detail"]
        
        print("✅ Analysis not found test passed")

def test_start_analysis_invalid_ticker():
    """
    Tests the analysis start endpoint with invalid ticker.
    """
    test_request = {
        "ticker": "",  # Empty ticker should fail validation
        "company_name": "Test Company"
    }
    
    response = client.post(
        "/api/analysis/comprehensive/start",
        json=test_request
    )
    
    assert response.status_code == 422  # Validation error
    
    print("✅ Invalid ticker test passed")

@patch.dict(os.environ, {}, clear=True)  # Clear QUEUE_URL
def test_start_analysis_no_sqs_config():
    """
    Tests the analysis start endpoint when SQS is not configured.
    """
    test_request = {
        "ticker": "AAPL",
        "company_name": "Apple Inc."
    }
    
    response = client.post(
        "/api/analysis/comprehensive/start",
        json=test_request
    )
    
    assert response.status_code == 503
    assert "SQS queue not configured" in response.json()["detail"]
    
    print("✅ No SQS config test passed")

if __name__ == "__main__":
    # Run tests directly if executed as script
    pytest.main([__file__, "-v"])







