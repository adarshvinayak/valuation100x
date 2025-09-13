"""
Polygon.io API Integration
Provides fallback financial data when FMP fails
"""
import asyncio
import os
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from .cache import get_cache

logger = logging.getLogger(__name__)

class PolygonClient:
    """Polygon.io API client for financial data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = None
        self.cache = get_cache()
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    @retry(
        stop=stop_after_attempt(2),  # Reduced from 3 to 2 attempts
        wait=wait_exponential(multiplier=2, min=10, max=60)  # Longer waits to respect rate limits
    )
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with retries"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}/{endpoint}"
        request_params = {"apikey": self.api_key}
        if params:
            request_params.update(params)
        
        # Check cache first
        cache_key = f"polygon:{endpoint}:{str(sorted(request_params.items()))}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for Polygon request: {endpoint}")
            return cached_result
        
        logger.info(f"Making Polygon API request: {url}")
        
        async with self.session.get(url, params=request_params) as response:
            logger.info(f"Polygon API response status: {response.status}")
            
            if response.status != 200:
                response_text = await response.text()
                logger.error(f"Polygon API error - Status: {response.status}, Response: {response_text}")
                response.raise_for_status()
            
            data = await response.json()
            
            # Check for API error messages
            if isinstance(data, dict) and data.get("status") == "ERROR":
                error_msg = data.get('error', 'Unknown error')
                logger.error(f"Polygon API returned error: {error_msg}")
                
                # Don't retry on rate limit errors - fail fast
                if "exceeded the maximum requests" in error_msg.lower() or "rate limit" in error_msg.lower():
                    logger.warning("Polygon rate limit reached - failing fast to avoid wasted compute")
                    raise Exception(f"Polygon Rate Limit: {error_msg}")
                
                raise Exception(f"Polygon API Error: {error_msg}")
            
            # Cache the result
            self.cache.set(cache_key, data, ttl_hours=24)
            
            return data
    
    async def get_ticker_details(self, ticker: str) -> Dict[str, Any]:
        """Get basic ticker details from Polygon"""
        try:
            endpoint = f"v3/reference/tickers/{ticker}"
            data = await self._make_request(endpoint)
            
            result = data.get("results", {})
            if not result:
                return {}
            
            # Normalize to FMP-like format
            return {
                "ticker": ticker,
                "company_name": result.get("name", f"{ticker} Corporation"),
                "description": result.get("description", ""),
                "sector": result.get("sic_description", "Unknown"),
                "industry": result.get("sic_description", "Unknown"),
                "exchange": result.get("primary_exchange", "Unknown"),
                "market_cap": result.get("market_cap", 0),
                "phone": result.get("phone_number", ""),
                "website": result.get("homepage_url", ""),
                "logo": result.get("branding", {}).get("logo_url", ""),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get ticker details for {ticker}: {e}")
            return {}
    
    async def get_previous_close(self, ticker: str) -> Dict[str, Any]:
        """Get previous close price data"""
        try:
            endpoint = f"v2/aggs/ticker/{ticker}/prev"
            data = await self._make_request(endpoint)
            
            results = data.get("results", [])
            if not results:
                return {}
            
            result = results[0]
            return {
                "ticker": ticker,
                "current_price": result.get("c", 0),  # Close price
                "day_low": result.get("l", 0),        # Low
                "day_high": result.get("h", 0),       # High  
                "volume": result.get("v", 0),         # Volume
                "open": result.get("o", 0),           # Open
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get previous close for {ticker}: {e}")
            return {}

async def get_polygon_fallback_data(ticker: str) -> Dict[str, Any]:
    """
    Get basic company data from Polygon.io as fallback
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.warning("POLYGON_API_KEY not found - fallback unavailable")
        return {}
    
    logger.info(f"Using Polygon.io fallback for {ticker}")
    
    try:
        async with PolygonClient(api_key) as client:
            # Get basic ticker details and price data
            details_task = client.get_ticker_details(ticker)
            price_task = client.get_previous_close(ticker)
            
            details, price_data = await asyncio.gather(
                details_task, price_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(details, Exception):
                logger.error(f"Polygon ticker details failed: {details}")
                details = {}
            if isinstance(price_data, Exception):
                logger.error(f"Polygon price data failed: {price_data}")
                price_data = {}
            
            # Combine the data
            combined_data = {**details, **price_data}
            
            # Ensure we have minimum required fields
            if not combined_data.get("company_name"):
                combined_data["company_name"] = f"{ticker} Corporation"
            if not combined_data.get("sector"):
                combined_data["sector"] = "Unknown"
            if not combined_data.get("exchange"):
                combined_data["exchange"] = "Unknown"
            
            logger.info(f"Polygon fallback successful for {ticker}: {combined_data.get('company_name')}")
            return combined_data
            
    except Exception as e:
        logger.error(f"Polygon fallback failed for {ticker}: {e}")
        return {}
