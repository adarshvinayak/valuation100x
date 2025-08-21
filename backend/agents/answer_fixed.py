#!/usr/bin/env python3
"""
Answer Agent - Fixed

Comprehensive research agent for stock analysis questions.
Uses the correct LlamaIndex FunctionAgent pattern from the reference implementation.
"""
import asyncio
import json
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Import all tools
from tools.retrieval import retrieve_docs
from tools.fmp import get_financials_fmp
from tools.alpha import get_prices_alpha, get_overview_alpha
from tools.polygon import get_prices_polygon, get_ticker_info_polygon
from tools.bloomberg import get_bloomberg
from tools.technicals import compute_indicators
from tools.sentiment import finbert_sentiment, uncertainty_index, analyze_transcript_sentiment
from tools.news import news_summary

logger = logging.getLogger(__name__)


def create_answer_agent(openai_api_key: str) -> FunctionAgent:
    """Create answer agent with research tools following reference pattern"""
    
    llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.1)
    
    # Create tool functions that the agent can use
    tools = [
        _create_retrieval_tool(),
        _create_financials_tool(),
        _create_prices_tool(),
        _create_technicals_tool(),
        _create_sentiment_tool(),
        _create_news_tool(),
        _create_bloomberg_tool(),
    ]
    
    system_prompt = """You are an expert financial analyst conducting comprehensive stock research.

Your role is to thoroughly research and answer specific questions about publicly traded companies using the available tools and data sources.

**Available Tools:**
1. `retrieve_docs` - Search SEC filings and earnings transcripts for specific information
2. `get_financials` - Get comprehensive financial data (income, balance sheet, cash flow, ratios)  
3. `get_prices` - Get historical price data and market information
4. `compute_technicals` - Calculate technical indicators from price data
5. `analyze_sentiment` - Analyze sentiment from text (earnings calls, news)
6. `get_news` - Search for recent news and analyst research
7. `get_bloomberg` - Access Bloomberg data (currently stub)

**Research Process:**
1. Understand the question and identify what data/analysis is needed
2. Use multiple tools to gather comprehensive information
3. Cross-reference findings across different data sources
4. Quantify findings where possible with specific metrics
5. Assess the reliability and recency of your sources

**Response Requirements:**
You MUST respond with a valid JSON object following this exact schema:

```json
{
  "question": "The original research question",
  "findings": [
    {
      "statement": "A specific factual statement",
      "evidence": [
        {
          "title": "Description of evidence",
          "url": "Source URL or identifier", 
          "date": "2024-01-15",
          "source_type": "SEC filing | news | financial_data | analyst_research"
        }
      ]
    }
  ],
  "metrics": {
    "key_metric_1": "value",
    "key_metric_2": "value"
  },
  "summary": "Concise 2-3 sentence summary answering the question",
  "confidence": 0.85
}
```

**Quality Standards:**
- Every finding must be supported by specific evidence with sources
- Include quantitative metrics whenever possible
- Be objective and fact-based, avoid speculation
- Cite your sources properly with URLs when available
- Assess confidence based on data quality and completeness
- If data is insufficient, state this clearly and explain limitations

**Important Notes:**
- Always format your response as valid JSON - no other text before or after
- Use actual URLs from your tool results in evidence
- Include dates for time-sensitive information  
- The confidence score should reflect data quality and completeness (0.0-1.0)
- If Bloomberg data is not available, rely on other sources and note this limitation"""

    return FunctionAgent(
        tools=tools,
        llm=llm,
        verbose=False,
        system_prompt=system_prompt
    )


def _create_retrieval_tool() -> FunctionTool:
    """Create document retrieval tool"""
    async def retrieve_documents(ticker: str, query: str, k: int = 8) -> str:
        """Search SEC filings and transcripts for relevant information."""
        try:
            results = await retrieve_docs(ticker, query, k)
            
            if not results:
                return f"No relevant documents found for {ticker} with query: {query}"
            
            # Format results for the agent
            formatted_results = []
            for i, result in enumerate(results[:k], 1):
                formatted_results.append(f"""
Document {i}:
Source: {result['form_type']} filed {result['filed_date']}
Relevance Score: {result['score']:.3f}
Content: {result['text'][:500]}...
Source Path: {result['source_path']}
""")
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Retrieval tool error: {e}")
            return f"Error retrieving documents: {str(e)}"
    
    return FunctionTool.from_defaults(fn=retrieve_documents)


def _create_financials_tool() -> FunctionTool:
    """Create financial data tool"""
    async def get_financials(ticker: str) -> str:
        """Get comprehensive financial data including ratios, income statement, balance sheet, and cash flow."""
        try:
            data = await get_financials_fmp(ticker)
            
            if not data:
                return f"No financial data available for {ticker}"
            
            # Return full data as JSON for agent to parse
            return json.dumps(data, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Financials tool error: {e}")
            return f"Error getting financial data: {str(e)}"
    
    return FunctionTool.from_defaults(fn=get_financials)


def _create_prices_tool() -> FunctionTool:
    """Create price data tool"""
    async def get_prices(ticker: str, source: str = "polygon", days: int = 252) -> str:
        """Get historical price data and market information."""
        try:
            if source.lower() == "alpha":
                prices = await get_prices_alpha(ticker, "compact")
                overview = await get_overview_alpha(ticker)
            else:  # Default to Polygon
                prices = await get_prices_polygon(ticker, days)
                overview = await get_ticker_info_polygon(ticker)
            
            if not prices:
                return f"No price data available for {ticker}"
            
            # Return price data for further analysis
            result = {
                "latest_price": prices[0] if prices else {},
                "overview": overview,
                "price_history": prices[:50],  # Recent 50 days
                "total_records": len(prices)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Prices tool error: {e}")
            return f"Error getting price data: {str(e)}"
    
    return FunctionTool.from_defaults(fn=get_prices)


def _create_technicals_tool() -> FunctionTool:
    """Create technical analysis tool"""
    async def compute_technicals(ticker: str, price_data: str = None) -> str:
        """Compute technical indicators from price data."""
        try:
            # If price_data is provided as JSON string, use it
            prices = []
            if price_data:
                try:
                    data = json.loads(price_data)
                    prices = data.get("price_history", [])
                except:
                    pass
            
            # If no price data provided, fetch it
            if not prices:
                prices = await get_prices_polygon(ticker, 400)
            
            if not prices:
                return f"No price data available for technical analysis of {ticker}"
            
            # Compute indicators
            indicators = compute_indicators(prices)
            
            if not indicators:
                return f"Unable to compute technical indicators for {ticker}"
            
            return json.dumps(indicators, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Technicals tool error: {e}")
            return f"Error computing technical indicators: {str(e)}"
    
    return FunctionTool.from_defaults(fn=compute_technicals)


def _create_sentiment_tool() -> FunctionTool:
    """Create sentiment analysis tool"""
    async def analyze_sentiment(ticker: str, texts: str = None) -> str:
        """Analyze sentiment from earnings transcripts or provided text."""
        try:
            result = {"ticker": ticker}
            
            # If specific texts provided, analyze them
            if texts:
                try:
                    text_list = json.loads(texts) if isinstance(texts, str) and texts.startswith('[') else [texts]
                except:
                    text_list = [texts]
                
                sentiments = finbert_sentiment(text_list)
                uncertainty = uncertainty_index(text_list)
                
                result["sentiment_scores"] = sentiments
                result["average_sentiment"] = sum(sentiments) / len(sentiments) if sentiments else 0.0
                result["uncertainty_index"] = uncertainty
            
            # Try to get earnings transcripts from SEC filings
            try:
                transcript_docs = await retrieve_docs(ticker, "earnings call transcript", k=3)
                
                if transcript_docs:
                    transcript_texts = [doc["text"] for doc in transcript_docs]
                    full_transcript = " ".join(transcript_texts)
                    
                    transcript_sentiment = analyze_transcript_sentiment(full_transcript)
                    result["transcript_analysis"] = transcript_sentiment
            except Exception as e:
                logger.warning(f"Could not analyze transcripts for {ticker}: {e}")
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Sentiment tool error: {e}")
            return f"Error analyzing sentiment: {str(e)}"
    
    return FunctionTool.from_defaults(fn=analyze_sentiment)


def _create_news_tool() -> FunctionTool:
    """Create news and research tool"""
    async def get_news(ticker: str, company_name: str = "", max_articles: int = 8) -> str:
        """Get recent news and analyst research about the company."""
        try:
            # Use company name if provided, otherwise use ticker
            company = company_name or ticker
            
            articles = await news_summary(ticker, company, max_articles)
            
            if not articles:
                return f"No recent news found for {ticker}"
            
            result = {
                "articles": articles,
                "total_count": len(articles)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"News tool error: {e}")
            return f"Error getting news: {str(e)}"
    
    return FunctionTool.from_defaults(fn=get_news)


def _create_bloomberg_tool() -> FunctionTool:
    """Create Bloomberg data tool (stub)"""
    async def get_bloomberg_data(ticker: str, data_type: str = "overview") -> str:
        """Get Bloomberg data (currently not implemented - returns stub)."""
        try:
            result = await get_bloomberg(ticker, data_type)
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Bloomberg tool error: {e}")
            return f"Bloomberg integration not available: {str(e)}"
    
    return FunctionTool.from_defaults(fn=get_bloomberg_data)


async def answer_question_with_agent(agent: FunctionAgent, ticker: str, question: str, company_name: str = "") -> Dict[str, Any]:
    """Answer a research question using the answer agent"""
    try:
        logger.info(f"🔬 Researching: {question[:60]}... (This may take 1-2 minutes)")
        
        # Enhance the question with context
        enhanced_question = f"""
Research Question: {question}

Company: {company_name or ticker}
Ticker: {ticker}

Please use the available tools to thoroughly research this question and provide a comprehensive answer with supporting evidence and sources.

Remember to respond ONLY with valid JSON following the required schema.
"""
        
        # Get response from agent (using correct API)
        result = await agent.run(user_msg=enhanced_question)
        
        # Parse and validate the response
        answer_data = _parse_and_validate_response(str(result), question)
        
        logger.info(f"✅ Research complete for: {question[:50]}...")
        return answer_data
        
    except Exception as e:
        logger.error(f"Failed to answer question for {ticker}: {e}")
        return _create_error_response(question, str(e))


def _parse_and_validate_response(response: str, original_question: str) -> Dict[str, Any]:
    """Parse and validate agent response"""
    try:
        # Clean the response
        response = response.strip()
        
        # Remove assistant prefix if present
        if response.startswith("assistant: "):
            response = response[11:]
        
        # Remove any markdown code blocks
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        if response.startswith("```"):
            response = response[3:]
        
        # Parse JSON
        data = json.loads(response)
        
        # Basic validation
        required_fields = ["question", "findings", "summary", "confidence"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Ensure findings is a list
        if not isinstance(data["findings"], list):
            data["findings"] = [data["findings"]]
        
        # Ensure confidence is a number between 0 and 1
        if not isinstance(data["confidence"], (int, float)):
            data["confidence"] = 0.5
        data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
        
        # Ensure metrics exists
        if "metrics" not in data:
            data["metrics"] = {}
        
        return data
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Response validation failed: {e}")
        logger.error(f"Raw response: {response[:500]}...")
        
        # Return minimal valid response
        return {
            "question": original_question,
            "findings": [{
                "statement": "Unable to parse research response properly",
                "evidence": [{
                    "title": "System Error",
                    "url": "internal://json-error",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source_type": "system"
                }]
            }],
            "metrics": {},
            "summary": "Error processing research data",
            "confidence": 0.0
        }


def _create_error_response(question: str, error_msg: str) -> Dict[str, Any]:
    """Create error response in valid format"""
    return {
        "question": question,
        "findings": [{
            "statement": f"Research failed due to system error: {error_msg}",
            "evidence": [{
                "title": "System Error",
                "url": "internal://error",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source_type": "system"
            }]
        }],
        "metrics": {},
        "summary": f"Unable to complete research due to: {error_msg}",
        "confidence": 0.0
    }


# Global agent instance
_answer_agent = None

def get_answer_agent(openai_api_key: str = None) -> FunctionAgent:
    """Get or create the global answer agent instance"""
    global _answer_agent
    
    if _answer_agent is None:
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found")
        
        _answer_agent = create_answer_agent(openai_api_key)
    
    return _answer_agent


async def answer_question(ticker: str, question: str, company_name: str = "") -> Dict[str, Any]:
    """
    Convenience function for answering research questions
    
    This is the main function that will be used by the workflow.
    """
    agent = get_answer_agent()
    return await answer_question_with_agent(agent, ticker, question, company_name)
