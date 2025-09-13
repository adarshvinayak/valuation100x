"""
News and Research Summary Tool

Fetches news articles and research about companies using Tavily search.
"""
import asyncio
import os
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from tavily import AsyncTavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from .cache import get_cache

logger = logging.getLogger(__name__)


class NewsSearcher:
    """News and research searcher using Tavily"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = AsyncTavilyClient(api_key=api_key)
        self.cache = get_cache()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search_company_news(self, 
                                ticker: str, 
                                company_name: str,
                                max_results: int = 8,
                                days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Search for recent news about a company
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            max_results: Maximum number of results
            days_back: How many days back to search
        
        Returns:
            List of news articles with title, url, date, snippet
        """
        # Check cache first
        cache_key = f"news:{ticker}:{company_name}:{max_results}:{days_back}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for news search: {ticker}")
            return cached_result
        
        try:
            # Construct search query
            search_query = f"{company_name} {ticker} stock earnings financial results"
            
            # Add date filter for recent news
            since_date = datetime.now() - timedelta(days=days_back)
            
            # Perform search
            response = await self.client.search(
                query=search_query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False
            )
            
            # Process and normalize results
            news_articles = []
            
            if "results" in response:
                for result in response["results"]:
                    article = {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("content", ""),
                        "date": self._extract_date(result),
                        "source": self._extract_source(result.get("url", "")),
                        "source_type": "news",
                        "ticker": ticker,
                        "relevance_score": result.get("score", 0.0)
                    }
                    
                    # Filter out articles that are too old or irrelevant
                    if self._is_relevant_article(article, ticker, company_name):
                        news_articles.append(article)
            
            # Sort by relevance and date
            news_articles.sort(key=lambda x: (x["relevance_score"], x["date"]), reverse=True)
            
            # Cache the results
            self.cache.set(cache_key, news_articles, ttl_hours=4)  # Shorter TTL for news
            
            logger.info(f"Found {len(news_articles)} relevant news articles for {ticker}")
            return news_articles
            
        except Exception as e:
            logger.error(f"Failed to search news for {ticker}: {e}")
            return []
    
    async def search_earnings_news(self,
                                 ticker: str,
                                 company_name: str,
                                 max_results: int = 5) -> List[Dict[str, Any]]:
        """Search specifically for earnings-related news"""
        try:
            search_query = f"{company_name} {ticker} earnings results financial quarter"
            
            response = await self.client.search(
                query=search_query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False
            )
            
            earnings_articles = []
            
            if "results" in response:
                for result in response["results"]:
                    article = {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("content", ""),
                        "date": self._extract_date(result),
                        "source": self._extract_source(result.get("url", "")),
                        "source_type": "earnings_news",
                        "ticker": ticker,
                        "relevance_score": result.get("score", 0.0)
                    }
                    
                    # Filter for earnings relevance
                    if self._is_earnings_relevant(article):
                        earnings_articles.append(article)
            
            logger.info(f"Found {len(earnings_articles)} earnings articles for {ticker}")
            return earnings_articles
            
        except Exception as e:
            logger.error(f"Failed to search earnings news for {ticker}: {e}")
            return []
    
    async def search_analyst_research(self,
                                    ticker: str,
                                    company_name: str,
                                    max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for analyst research and price targets"""
        try:
            search_query = f"{company_name} {ticker} analyst research price target rating upgrade downgrade"
            
            response = await self.client.search(
                query=search_query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False
            )
            
            research_articles = []
            
            if "results" in response:
                for result in response["results"]:
                    article = {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("content", ""),
                        "date": self._extract_date(result),
                        "source": self._extract_source(result.get("url", "")),
                        "source_type": "analyst_research",
                        "ticker": ticker,
                        "relevance_score": result.get("score", 0.0)
                    }
                    
                    if self._is_research_relevant(article):
                        research_articles.append(article)
            
            logger.info(f"Found {len(research_articles)} research articles for {ticker}")
            return research_articles
            
        except Exception as e:
            logger.error(f"Failed to search analyst research for {ticker}: {e}")
            return []
    
    def _extract_date(self, result: Dict[str, Any]) -> str:
        """Extract or estimate publication date"""
        # Try to extract date from result metadata
        if "published_date" in result:
            return result["published_date"]
        
        # Default to current date if not available
        return datetime.now().strftime("%Y-%m-%d")
    
    def _extract_source(self, url: str) -> str:
        """Extract source name from URL"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            
            # Clean up common domain patterns
            domain = domain.replace("www.", "")
            
            # Map to known sources
            source_mapping = {
                "reuters.com": "Reuters",
                "bloomberg.com": "Bloomberg",
                "wsj.com": "Wall Street Journal",
                "ft.com": "Financial Times",
                "cnbc.com": "CNBC",
                "marketwatch.com": "MarketWatch",
                "fool.com": "The Motley Fool",
                "seekingalpha.com": "Seeking Alpha",
                "yahoo.com": "Yahoo Finance",
                "finance.yahoo.com": "Yahoo Finance"
            }
            
            return source_mapping.get(domain, domain)
            
        except Exception:
            return "Unknown"
    
    def _is_relevant_article(self, article: Dict[str, Any], ticker: str, company_name: str) -> bool:
        """Check if article is relevant to the company"""
        title_lower = article["title"].lower()
        snippet_lower = article["snippet"].lower()
        ticker_lower = ticker.lower()
        company_lower = company_name.lower()
        
        # Must contain ticker or company name
        has_ticker = ticker_lower in title_lower or ticker_lower in snippet_lower
        has_company = any(word in title_lower or word in snippet_lower 
                         for word in company_lower.split() if len(word) > 3)
        
        return has_ticker or has_company
    
    def _is_earnings_relevant(self, article: Dict[str, Any]) -> bool:
        """Check if article is earnings-related"""
        text = (article["title"] + " " + article["snippet"]).lower()
        earnings_keywords = ["earnings", "results", "quarter", "revenue", "profit", "eps"]
        
        return any(keyword in text for keyword in earnings_keywords)
    
    def _is_research_relevant(self, article: Dict[str, Any]) -> bool:
        """Check if article is analyst research-related"""
        text = (article["title"] + " " + article["snippet"]).lower()
        research_keywords = ["analyst", "research", "price target", "rating", "upgrade", 
                           "downgrade", "buy", "sell", "hold", "recommendation"]
        
        return any(keyword in text for keyword in research_keywords)


async def news_summary(ticker: str, 
                      company: str, 
                      n: int = 8,
                      include_research: bool = True) -> List[Dict[str, Any]]:
    """
    Main function to get news summary for a company
    
    Args:
        ticker: Stock ticker symbol
        company: Company name
        n: Number of articles to return
        include_research: Whether to include analyst research
    
    Returns:
        List of news articles and research
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY not found in environment")
        return []
    
    try:
        searcher = NewsSearcher(api_key)
        
        # Search for different types of content
        tasks = [
            searcher.search_company_news(ticker, company, max_results=n//2),
            searcher.search_earnings_news(ticker, company, max_results=n//4)
        ]
        
        if include_research:
            tasks.append(searcher.search_analyst_research(ticker, company, max_results=n//4))
        
        # Execute searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_articles = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Search task failed: {result}")
            elif isinstance(result, list):
                all_articles.extend(result)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                unique_articles.append(article)
        
        # Sort by relevance and date, then limit to n results
        unique_articles.sort(key=lambda x: (x["relevance_score"], x["date"]), reverse=True)
        
        return unique_articles[:n]
        
    except Exception as e:
        logger.error(f"Failed to get news summary for {ticker}: {e}")
        return []


async def main():
    """CLI entry point for testing news search"""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test news search functionality")
    parser.add_argument("--ticker", required=True, help="Stock ticker")
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument("--count", type=int, default=5, help="Number of articles")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Test news search
    articles = await news_summary(args.ticker, args.company, args.count)
    
    print(f"\nNews summary for {args.company} ({args.ticker}):")
    print("=" * 60)
    
    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Source: {article['source']} ({article['source_type']})")
        print(f"   Date: {article['date']}")
        print(f"   URL: {article['url']}")
        print(f"   Snippet: {article['snippet'][:150]}...")
        print(f"   Relevance: {article['relevance_score']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
