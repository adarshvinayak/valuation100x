#!/usr/bin/env python3
"""
Question Generation Agent - Fixed

Generates comprehensive, sector-aware research questions for stock analysis.
Uses the correct LlamaIndex FunctionAgent pattern from the reference implementation.
"""
import os
import logging
from typing import List

from llama_index.core.agent import FunctionAgent
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)

# Sector-specific question templates for fallback
SECTOR_QUESTION_TEMPLATES = {
    "technology": [
        "What are the company's key revenue growth drivers and how sustainable are they?",
        "How does the company's R&D spending and innovation pipeline compare to competitors?",
        "What are the competitive moats and switching costs in the company's core business?",
        "How is the company positioned for cloud transformation and AI adoption trends?",
        "What are the key regulatory and antitrust risks facing the company?"
    ],
    "healthcare": [
        "What is the strength and diversity of the company's drug pipeline?",
        "How does the company's patent cliff exposure affect future revenue?",
        "What are the regulatory approval risks for key products in development?",
        "How sustainable are the company's pricing power and margins?",
        "What are the competitive threats from biosimilars and generics?"
    ],
    "financials": [
        "How is the company's credit quality and loan portfolio performing?",
        "What is the impact of interest rate changes on net interest margin?",
        "How well capitalized is the company relative to regulatory requirements?",
        "What are the key risks to fee income and trading revenues?",
        "How is the company positioned for potential economic downturns?"
    ],
    "energy": [
        "What are the company's production costs and breakeven oil prices?",
        "How is the company managing ESG transition and renewable energy investments?",
        "What is the quality and remaining life of the company's reserves?",
        "How exposed is the company to commodity price volatility?",
        "What are the regulatory and environmental compliance risks?"
    ],
    "consumer_staples": [
        "How strong are the company's brand portfolio and pricing power?",
        "What is the impact of input cost inflation on margins?",
        "How is the company adapting to changing consumer preferences?",
        "What are the competitive dynamics in key product categories?",
        "How resilient are the company's distribution channels?"
    ],
    "industrials": [
        "How cyclical is the company's business and what is the current cycle position?",
        "What is the company's operational leverage and margin sensitivity?",
        "How is the company positioned for infrastructure spending trends?",
        "What are the supply chain risks and mitigation strategies?",
        "How is the company addressing green transition opportunities?"
    ],
    "consumer_discretionary": [
        "How sensitive is the company's business to economic cycles?",
        "What is the company's e-commerce strategy and digital transformation progress?",
        "How differentiated are the company's brands and products?",
        "What are the supply chain and labor cost pressures?",
        "How is the company adapting to changing consumer behavior post-pandemic?"
    ]
}

# Generic fallback questions
GENERIC_QUESTIONS = [
    "What are the company's key revenue growth drivers and how sustainable are they?",
    "How does the company's R&D spending and innovation pipeline compare to competitors?",
    "What are the competitive moats and switching costs in the company's core business?",
    "How is the company positioned for cloud transformation and AI adoption trends?",
    "What are the key regulatory and antitrust risks facing the company?",
    "How has the company's financial performance trended over the past 3-5 years?",
    "What are the key valuation metrics and how do they compare to historical averages?",
    "How is the company performing relative to industry and market benchmarks?",
    "What are the key catalysts that could drive the stock price higher or lower?"
]


def create_question_agent(openai_api_key: str) -> FunctionAgent:
    """Create question generation agent following reference pattern"""
    
    llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    
    system_prompt = """You are a sophisticated equity research analyst specializing in generating comprehensive research questions for stock analysis.

Your task is to generate 6-10 specific, actionable research questions for a given stock ticker and company. The questions should be:

1. **Sector-aware**: Tailored to the specific industry and business model
2. **Non-overlapping**: Each question should cover a distinct aspect of analysis
3. **Comprehensive**: Cover fundamentals, valuation, sentiment, technicals, and risks
4. **Actionable**: Questions that can be researched and answered with available data

**Coverage Areas (ensure all are addressed):**
- Financial fundamentals and performance trends
- Valuation analysis (DCF inputs, multiples, peer comparison)
- Qualitative factors (management, competitive position, moats)
- Sentiment analysis (earnings calls, news, analyst opinions)
- Technical analysis (price trends, momentum, support/resistance)
- Risk analysis (company-specific and macro risks)

**Output Requirements:**
- Provide exactly one question per line
- No numbering, bullets, or markdown formatting
- No preamble or explanation text
- Each question should be a complete, well-formed sentence
- Focus on specific rather than generic questions

For each company, tailor questions to their specific business model, sector dynamics, and current market context.

Generate questions that an experienced analyst would ask when building a comprehensive investment thesis."""

    return FunctionAgent(
        tools=[],
        llm=llm,
        verbose=False,
        system_prompt=system_prompt
    )


def _determine_sector(ticker: str, company_name: str = "") -> str:
    """Determine company sector from ticker and name"""
    # Simple mapping for demo tickers
    sector_mapping = {
        "AAPL": "technology",
        "MSFT": "technology", 
        "NVDA": "technology",
        "AMZN": "consumer_discretionary",
        "GOOGL": "technology",
        "META": "technology",
        "TSLA": "consumer_discretionary",
        "KO": "consumer_staples",
        "PG": "consumer_staples",
        "JNJ": "healthcare",
        "UNH": "healthcare",
        "JPM": "financials",
        "GS": "financials",
        "BAC": "financials",
        "XOM": "energy",
        "CVX": "energy",
        "CAT": "industrials",
        "BA": "industrials",
        "WMT": "consumer_staples",
        "DIS": "consumer_discretionary"
    }
    
    return sector_mapping.get(ticker.upper(), "unknown")


def _get_sector_context(sector: str) -> str:
    """Get sector-specific context for question generation"""
    sector_contexts = {
        "technology": "Focus on innovation cycles, platform dynamics, R&D efficiency, competitive moats from network effects, regulatory risks from antitrust, and scalability of business models.",
        "healthcare": "Emphasize drug pipeline strength, patent cliff risks, regulatory approval timelines, pricing pressures, competitive threats from biosimilars, and demographic tailwinds.",
        "financials": "Concentrate on credit quality, interest rate sensitivity, regulatory capital requirements, loan growth prospects, and fintech disruption risks.",
        "energy": "Highlight commodity price sensitivity, production costs, reserve quality, environmental regulations, capital discipline, and energy transition impacts.",
        "consumer_staples": "Focus on brand strength, pricing power, market share stability, input cost inflation, distribution channels, and demographic shifts.",
        "industrials": "Emphasize cyclical exposure, operational leverage, infrastructure spending, supply chain dynamics, and green transition opportunities.",
        "consumer_discretionary": "Focus on economic sensitivity, e-commerce adaptation, brand differentiation, supply chain costs, and changing consumer behavior."
    }
    
    return sector_contexts.get(sector, "Focus on standard financial analysis covering profitability, growth, valuation, competitive position, and risk factors.")


def _parse_questions(response: str) -> List[str]:
    """Parse the agent response into individual questions"""
    # Clean up the response first
    response = response.strip()
    
    # Remove assistant prefix if present
    if response.startswith("assistant: "):
        response = response[11:]
    
    # Split by newlines and clean up
    lines = response.split('\n')
    questions = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Remove numbering if present
        import re
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = re.sub(r'^[-•]\s*', '', line)
        
        # Ensure it's a question
        if line and not line.endswith('?'):
            line += '?'
        
        if line and len(line) > 20:  # Reasonable minimum length
            questions.append(line)
    
    # Ensure we have 6-10 questions
    if len(questions) < 6:
        logger.warning(f"Only generated {len(questions)} questions, expected 6-10")
    elif len(questions) > 10:
        logger.warning(f"Generated {len(questions)} questions, truncating to 10")
        questions = questions[:10]
    
    return questions


def _get_fallback_questions(ticker: str, sector: str) -> List[str]:
    """Get fallback questions if agent generation fails"""
    logger.warning(f"Using fallback questions for {ticker}")
    
    # Use sector-specific questions if available
    if sector in SECTOR_QUESTION_TEMPLATES:
        return SECTOR_QUESTION_TEMPLATES[sector]
    
    # Use generic questions as last resort
    return GENERIC_QUESTIONS[:8]


async def generate_questions_with_agent(agent: FunctionAgent, ticker: str, company_name: str = "") -> List[str]:
    """Generate questions using the question agent"""
    try:
        # Determine sector and context
        sector = _determine_sector(ticker, company_name)
        sector_context = _get_sector_context(sector)
        
        # Create enhanced prompt
        prompt = f"""Generate research questions for stock analysis:

Ticker: {ticker}
Company: {company_name or ticker}
Sector: {sector}

Sector Context: {sector_context}

Generate 6-10 specific research questions that cover:
1. Financial fundamentals and performance
2. Valuation analysis (DCF inputs, multiples)
3. Qualitative factors (management, competitive position)
4. Sentiment and market perception
5. Technical price analysis
6. Risk factors and potential catalysts

Ensure questions are specific to this company and sector, not generic templates.
Output one question per line with no formatting."""

        # Generate questions using the agent (using correct API)
        result = await agent.run(user_msg=prompt)
        
        # Parse the response into individual questions
        questions = _parse_questions(str(result))
        
        logger.info(f"Generated {len(questions)} questions for {ticker}")
        return questions
        
    except Exception as e:
        logger.error(f"Failed to generate questions for {ticker}: {e}")
        # Fallback to generic questions
        return _get_fallback_questions(ticker, _determine_sector(ticker, company_name))


# Global agent instance
_question_agent = None

def get_question_agent(openai_api_key: str = None) -> FunctionAgent:
    """Get or create the global question agent instance"""
    global _question_agent
    
    if _question_agent is None:
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found")
        
        _question_agent = create_question_agent(openai_api_key)
    
    return _question_agent


async def generate_questions(ticker: str, company_name: str = "") -> List[str]:
    """
    Convenience function for question generation
    
    This is the main function that will be used by the workflow.
    """
    agent = get_question_agent()
    return await generate_questions_with_agent(agent, ticker, company_name)
