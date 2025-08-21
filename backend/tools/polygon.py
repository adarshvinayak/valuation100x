"""
Polygon.io API Integration

Fetches stock price data and market information.
"""
import asyncio
import os
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from .cache import get_cache

logger = logging.getLogger(__name__)


class PolygonClient:
    """Polygon.io API client"""
    
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
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with retries"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}/{endpoint}"
        request_params = {"apiKey": self.api_key}
        if params:
            request_params.update(params)
        
        # Check cache first
        cache_key = f"polygon:{endpoint}:{str(sorted(request_params.items()))}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for Polygon request: {endpoint}")
            return cached_result
        
        async with self.session.get(url, params=request_params) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Check for API errors
            if data.get("status") == "ERROR":
                raise ValueError(f"Polygon API error: {data.get('error', 'Unknown error')}")
            
            # Cache the result
            self.cache.set(cache_key, data, ttl_hours=24)
            
            return data
    
    async def get_aggregates(self, 
                           ticker: str, 
                           multiplier: int = 1,
                           timespan: str = "day",
                           from_date: Optional[str] = None,
                           to_date: Optional[str] = None,
                           limit: int = 5000) -> List[Dict[str, Any]]:
        """
        Get aggregate bars for a ticker
        
        Args:
            ticker: Stock symbol
            multiplier: Size of timespan multiplier
            timespan: Size of time window (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Maximum number of results
        """
        try:
            # Set default date range if not provided
            if not to_date:
                to_date = datetime.now().strftime("%Y-%m-%d")
            if not from_date:
                # Default to 400 days ago (approximately 1.1 years of trading days)
                start_date = datetime.now() - timedelta(days=400)
                from_date = start_date.strftime("%Y-%m-%d")
            
            endpoint = f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
            params = {
                "adjusted": "true",
                "sort": "desc",  # Most recent first
                "limit": limit
            }
            
            data = await self._make_request(endpoint, params)
            
            if "results" not in data:
                logger.warning(f"No results found for {ticker}")
                return []
            
            # Convert to normalized format
            prices = []
            for bar in data["results"]:
                try:
                    # Convert timestamp to date
                    timestamp = bar.get("t", 0) / 1000  # Convert from milliseconds
                    date_obj = datetime.fromtimestamp(timestamp)
                    
                    price_record = {
                        "ticker": ticker,
                        "date": date_obj.strftime("%Y-%m-%d"),
                        "timestamp": int(bar.get("t", 0)),
                        "open": float(bar.get("o", 0)),
                        "high": float(bar.get("h", 0)),
                        "low": float(bar.get("l", 0)),
                        "close": float(bar.get("c", 0)),
                        "volume": int(bar.get("v", 0)),
                        "volume_weighted_average": float(bar.get("vw", 0)),
                        "number_of_transactions": int(bar.get("n", 0)),
                        "last_updated": datetime.now().isoformat()
                    }
                    prices.append(price_record)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse price data for {ticker}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(prices)} price records for {ticker}")
            return prices
            
        except Exception as e:
            logger.error(f"Failed to get aggregates for {ticker}: {e}")
            return []
    
    async def get_ticker_details(self, ticker: str) -> Dict[str, Any]:
        """Get ticker details including company information"""
        try:
            endpoint = f"v3/reference/tickers/{ticker}"
            data = await self._make_request(endpoint)
            
            if "results" not in data:
                return {}
            
            result = data["results"]
            
            # Normalize the response
            return {
                "ticker": ticker,
                "name": result.get("name", ""),
                "description": result.get("description", ""),
                "market": result.get("market", ""),
                "type": result.get("type", ""),
                "primary_exchange": result.get("primary_exchange", ""),
                "currency_name": result.get("currency_name", "USD"),
                "cik": result.get("cik", ""),
                "composite_figi": result.get("composite_figi", ""),
                "share_class_figi": result.get("share_class_figi", ""),
                "market_cap": result.get("market_cap", 0),
                "phone_number": result.get("phone_number", ""),
                "address": result.get("address", {}),
                "homepage_url": result.get("homepage_url", ""),
                "total_employees": result.get("total_employees", 0),
                "list_date": result.get("list_date", ""),
                "branding": result.get("branding", {}),
                "share_class_shares_outstanding": result.get("share_class_shares_outstanding", 0),
                "weighted_shares_outstanding": result.get("weighted_shares_outstanding", 0),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get ticker details for {ticker}: {e}")
            return {}
    
    async def get_previous_close(self, ticker: str) -> Dict[str, Any]:
        """Get previous trading day's close"""
        try:
            endpoint = f"v2/aggs/ticker/{ticker}/prev"
            data = await self._make_request(endpoint)
            
            if "results" not in data or not data["results"]:
                return {}
            
            result = data["results"][0]
            
            # Convert timestamp to date
            timestamp = result.get("T", 0) / 1000
            date_obj = datetime.fromtimestamp(timestamp)
            
            return {
                "ticker": ticker,
                "date": date_obj.strftime("%Y-%m-%d"),
                "open": float(result.get("o", 0)),
                "high": float(result.get("h", 0)),
                "low": float(result.get("l", 0)),
                "close": float(result.get("c", 0)),
                "volume": int(result.get("v", 0)),
                "volume_weighted_average": float(result.get("vw", 0)),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get previous close for {ticker}: {e}")
            return {}


async def get_prices_polygon(ticker: str, days: int = 400) -> List[Dict[str, Any]]:
    """
    Main function to get price data from Polygon
    
    Args:
        ticker: Stock symbol
        days: Number of days of historical data to fetch
    
    Returns:
        List of price records sorted by date (most recent first)
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.error("POLYGON_API_KEY not found in environment")
        return []
    
    try:
        async with PolygonClient(api_key) as client:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            prices = await client.get_aggregates(
                ticker,
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d")
            )
            
            return prices
            
    except Exception as e:
        logger.error(f"Failed to get prices for {ticker} from Polygon: {e}")
        return []


async def get_ticker_info_polygon(ticker: str) -> Dict[str, Any]:
    """
    Get ticker information from Polygon
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.error("POLYGON_API_KEY not found in environment")
        return {}
    
    try:
        async with PolygonClient(api_key) as client:
            details = await client.get_ticker_details(ticker)
            return details
            
    except Exception as e:
        logger.error(f"Failed to get ticker info for {ticker} from Polygon: {e}")
        return {}


async def main():
    """CLI entry point for testing Polygon integration"""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test Polygon API integration")
    parser.add_argument("--ticker", required=True, help="Stock ticker")
    parser.add_argument("--prices", action="store_true", help="Get price data")
    parser.add_argument("--info", action="store_true", help="Get ticker information")
    parser.add_argument("--days", type=int, default=30, help="Number of days for price data")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.prices:
        prices = await get_prices_polygon(args.ticker, args.days)
        
        print(f"\nPrice data for {args.ticker} ({len(prices)} records):")
        print("=" * 60)
        
        for i, price in enumerate(prices[:10]):  # Show first 10 records
            print(f"{price['date']}: Close ${price['close']:.2f}, Volume {price['volume']:,}")
            if i == 9 and len(prices) > 10:
                print(f"... and {len(prices) - 10} more records")
    
    if args.info:
        info = await get_ticker_info_polygon(args.ticker)
        
        print(f"\nTicker information for {args.ticker}:")
        print("=" * 50)
        
        if info:
            print(f"Name: {info.get('name')}")
            print(f"Market: {info.get('market')}")
            print(f"Primary Exchange: {info.get('primary_exchange')}")
            print(f"Market Cap: ${info.get('market_cap', 0):,.0f}")
            print(f"Employees: {info.get('total_employees', 0):,}")
        else:
            print("No ticker information available")


if __name__ == "__main__":
    asyncio.run(main())
