"""
Test configuration and shared fixtures
"""
import os
import sys
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ['TESTING'] = 'true'
os.environ['ENV_FILE_PATH'] = r'C:\Users\rnuser\Documents\deepresearch\.env'

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    env_vars = {
        'OPENAI_API_KEY': 'test-openai-key',
        'TAVILY_API_KEY': 'test-tavily-key',
        'FMP_API_KEY': 'test-fmp-key',
        'ALPHAVANTAGE_API_KEY': 'test-alpha-key',
        'POLYGON_API_KEY': 'test-polygon-key',
        'SEC_API_KEY': 'test-sec-key'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def sample_financial_data():
    """Sample financial data for testing"""
    return {
        "ratios_ttm": {
            "roe": 0.18,
            "roa": 0.12,
            "debt_to_equity": 0.25,
            "current_ratio": 1.8,
            "gross_margin": 0.38,
            "net_margin": 0.21,
            "interest_coverage": 15.2
        },
        "market_cap": 2800000000000,
        "pe_ttm": 25.4,
        "ev_ebitda_ttm": 18.2,
        "revenue_ttm": 394000000000,
        "ebitda_ttm": 120000000000
    }

@pytest.fixture
def sample_price_data():
    """Sample price data for testing"""
    return [
        {"date": "2024-01-01", "close": 150.0, "volume": 1000000},
        {"date": "2024-01-02", "close": 152.0, "volume": 1100000},
        {"date": "2024-01-03", "close": 148.0, "volume": 1200000},
        {"date": "2024-01-04", "close": 155.0, "volume": 900000},
        {"date": "2024-01-05", "close": 153.0, "volume": 1050000}
    ]

@pytest.fixture
def sample_sentiment_texts():
    """Sample texts for sentiment analysis testing"""
    return {
        "positive": [
            "Strong revenue growth exceeded analyst expectations",
            "Management expressed confidence in future prospects",
            "Excellent quarter with outstanding performance"
        ],
        "negative": [
            "Declining margins and increased competition concerns",
            "Disappointing results with weak guidance",
            "Significant headwinds affecting profitability"
        ],
        "neutral": [
            "Results were in line with expectations",
            "The company reported quarterly earnings",
            "Management provided standard outlook"
        ]
    }

@pytest.fixture
def mock_api_responses():
    """Mock API responses for external services"""
    return {
        "fmp_financials": {
            "symbol": "AAPL",
            "price": 150.0,
            "marketCap": 2800000000000,
            "peRatio": 25.4,
            "eps": 5.89
        },
        "alpha_vantage_prices": {
            "Time Series (Daily)": {
                "2024-01-01": {"4. close": "150.0"},
                "2024-01-02": {"4. close": "152.0"}
            }
        }
    }
