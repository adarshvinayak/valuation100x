"""
Alpha Vantage API Integration

Fetches stock price data and market information.
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


class AlphaVantageClient:
    """Alpha Vantage API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
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
        wait=wait_exponential(multiplier=1, min=12, max=20)  # Alpha Vantage has strict rate limits
    )
    async def _make_request(self, params: Dict[str, str]) -> Dict:
        """Make API request with retries"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        request_params = {"apikey": self.api_key}
        request_params.update(params)
        
        # Check cache first
        cache_key = f"alpha:{str(sorted(request_params.items()))}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for Alpha Vantage request: {params.get('function', 'unknown')}")
            return cached_result
        
        async with self.session.get(self.base_url, params=request_params) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
            
            if "Note" in data:
                # Rate limit hit
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                raise Exception("Rate limit exceeded")
            
            # Cache the result (24 hours for price data)
            self.cache.set(cache_key, data, ttl_hours=24)
            
            return data
    
    async def get_daily_prices(self, ticker: str, outputsize: str = "compact") -> List[Dict[str, Any]]:
        """
        Get daily price data
        
        Args:
            ticker: Stock symbol
            outputsize: "compact" (100 days) or "full" (20+ years)
        """
        try:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "outputsize": outputsize
            }
            
            data = await self._make_request(params)
            
            time_series_key = "Time Series (Daily)"
            if time_series_key not in data:
                logger.error(f"No time series data found for {ticker}")
                return []
            
            time_series = data[time_series_key]
            
            # Convert to normalized format
            prices = []
            for date_str, price_data in time_series.items():
                try:
                    price_record = {
                        "ticker": ticker,
                        "date": date_str,
                        "open": float(price_data.get("1. open", 0)),
                        "high": float(price_data.get("2. high", 0)),
                        "low": float(price_data.get("3. low", 0)),
                        "close": float(price_data.get("4. close", 0)),
                        "adjusted_close": float(price_data.get("5. adjusted close", 0)),
                        "volume": int(float(price_data.get("6. volume", 0))),
                        "dividend_amount": float(price_data.get("7. dividend amount", 0)),
                        "split_coefficient": float(price_data.get("8. split coefficient", 1.0)),
                        "last_updated": datetime.now().isoformat()
                    }
                    prices.append(price_record)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse price data for {ticker} on {date_str}: {e}")
                    continue
            
            # Sort by date (most recent first)
            prices.sort(key=lambda x: x["date"], reverse=True)
            
            logger.info(f"Retrieved {len(prices)} price records for {ticker}")
            return prices
            
        except Exception as e:
            logger.error(f"Failed to get daily prices for {ticker}: {e}")
            return []
    
    async def get_company_overview(self, ticker: str) -> Dict[str, Any]:
        """Get company overview and fundamental data"""
        try:
            params = {
                "function": "OVERVIEW",
                "symbol": ticker
            }
            
            data = await self._make_request(params)
            
            if not data or "Symbol" not in data:
                return {}
            
            # Normalize the response
            return {
                "ticker": ticker,
                "name": data.get("Name", ""),
                "description": data.get("Description", ""),
                "sector": data.get("Sector", ""),
                "industry": data.get("Industry", ""),
                "country": data.get("Country", ""),
                "currency": data.get("Currency", "USD"),
                "exchange": data.get("Exchange", ""),
                
                # Market data
                "market_cap": self._safe_float(data.get("MarketCapitalization", 0)),
                "pe_ratio": self._safe_float(data.get("PERatio", 0)),
                "peg_ratio": self._safe_float(data.get("PEGRatio", 0)),
                "price_to_book": self._safe_float(data.get("PriceToBookRatio", 0)),
                "dividend_yield": self._safe_float(data.get("DividendYield", 0)),
                "eps": self._safe_float(data.get("EPS", 0)),
                "beta": self._safe_float(data.get("Beta", 0)),
                
                # Financial metrics
                "revenue_ttm": self._safe_float(data.get("RevenueTTM", 0)),
                "gross_profit_ttm": self._safe_float(data.get("GrossProfitTTM", 0)),
                "ebitda": self._safe_float(data.get("EBITDA", 0)),
                
                # Ratios
                "profit_margin": self._safe_float(data.get("ProfitMargin", 0)),
                "operating_margin": self._safe_float(data.get("OperatingMarginTTM", 0)),
                "return_on_assets": self._safe_float(data.get("ReturnOnAssetsTTM", 0)),
                "return_on_equity": self._safe_float(data.get("ReturnOnEquityTTM", 0)),
                
                # Trading data
                "52_week_high": self._safe_float(data.get("52WeekHigh", 0)),
                "52_week_low": self._safe_float(data.get("52WeekLow", 0)),
                "50_day_ma": self._safe_float(data.get("50DayMovingAverage", 0)),
                "200_day_ma": self._safe_float(data.get("200DayMovingAverage", 0)),
                "shares_outstanding": self._safe_float(data.get("SharesOutstanding", 0)),
                
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get company overview for {ticker}: {e}")
            return {}
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        if value is None or value == "None" or value == "":
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0


async def get_prices_alpha(ticker: str, outputsize: str = "compact") -> List[Dict[str, Any]]:
    """
    Main function to get price data from Alpha Vantage
    
    Args:
        ticker: Stock symbol
        outputsize: "compact" (100 days) or "full" (20+ years)
    
    Returns:
        List of price records sorted by date (most recent first)
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        logger.error("ALPHAVANTAGE_API_KEY not found in environment")
        return []
    
    try:
        async with AlphaVantageClient(api_key) as client:
            prices = await client.get_daily_prices(ticker, outputsize)
            return prices
            
    except Exception as e:
        logger.error(f"Failed to get prices for {ticker} from Alpha Vantage: {e}")
        return []


async def get_overview_alpha(ticker: str) -> Dict[str, Any]:
    """
    Get company overview from Alpha Vantage
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        logger.error("ALPHAVANTAGE_API_KEY not found in environment")
        return {}
    
    try:
        async with AlphaVantageClient(api_key) as client:
            overview = await client.get_company_overview(ticker)
            return overview
            
    except Exception as e:
        logger.error(f"Failed to get overview for {ticker} from Alpha Vantage: {e}")
        return {}


async def main():
    """CLI entry point for testing Alpha Vantage integration"""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test Alpha Vantage API integration")
    parser.add_argument("--ticker", required=True, help="Stock ticker")
    parser.add_argument("--prices", action="store_true", help="Get price data")
    parser.add_argument("--overview", action="store_true", help="Get company overview")
    parser.add_argument("--full", action="store_true", help="Get full price history")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.prices:
        outputsize = "full" if args.full else "compact"
        prices = await get_prices_alpha(args.ticker, outputsize)
        
        print(f"\nPrice data for {args.ticker} ({len(prices)} records):")
        print("=" * 60)
        
        for i, price in enumerate(prices[:10]):  # Show first 10 records
            print(f"{price['date']}: Close ${price['close']:.2f}, Volume {price['volume']:,}")
            if i == 9 and len(prices) > 10:
                print(f"... and {len(prices) - 10} more records")
    
    if args.overview:
        overview = await get_overview_alpha(args.ticker)
        
        print(f"\nCompany overview for {args.ticker}:")
        print("=" * 50)
        
        if overview:
            print(f"Name: {overview.get('name')}")
            print(f"Sector: {overview.get('sector')}")
            print(f"Market Cap: ${overview.get('market_cap', 0):,.0f}")
            print(f"P/E Ratio: {overview.get('pe_ratio', 0):.2f}")
            print(f"Beta: {overview.get('beta', 0):.2f}")
        else:
            print("No overview data available")


if __name__ == "__main__":
    asyncio.run(main())
