#!/usr/bin/env python3
"""
Enhanced Answer Agent with Damodaran Industry Analysis Tools

Adds industry analysis and story validation tools to the existing answer agent.
"""
import os
import logging
from typing import List, Dict, Any
import json
from datetime import datetime

from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Import existing answer agent
from agents.answer_fixed import get_answer_agent, create_research_tools

# Import Damodaran tools
from tools.damodaran_industry_analysis import (
    get_industry_analysis,
    get_value_chain_analysis, 
    get_sector_risk_register
)

logger = logging.getLogger(__name__)


def create_enhanced_research_tools() -> List[FunctionTool]:
    """Create enhanced research tools including Damodaran industry analysis"""
    
    # Get existing research tools
    existing_tools = create_research_tools()
    
    # Add Damodaran industry analysis tools
    damodaran_tools = [
        FunctionTool.from_defaults(
            fn=get_industry_analysis,
            name="get_industry_analysis",
            description="""Get comprehensive industry analysis using Damodaran's framework.
            
            This tool provides:
            - Value chain mapping and company positioning
            - Comprehensive sector risk register
            - Accounting quirks and adjustments
            - Competitive landscape analysis
            
            Use when you need deep industry context for valuation or strategy questions.
            
            Args:
                sector: Company sector (e.g., 'technology', 'healthcare', 'financials')
                company_data_json: JSON string containing company financial and operational data
            
            Returns:
                JSON string with comprehensive industry analysis
            """
        ),
        
        FunctionTool.from_defaults(
            fn=get_value_chain_analysis,
            name="get_value_chain_analysis", 
            description="""Analyze company's position in the industry value chain.
            
            This tool provides:
            - Complete value chain mapping from raw materials to end customer
            - Company's value capture points and competitive positioning  
            - Bargaining power analysis across the chain
            - Integration opportunities and threats
            
            Use when analyzing competitive moats, pricing power, or strategic positioning.
            
            Args:
                sector: Company sector
                company_data_json: JSON string of company data
            
            Returns:
                JSON string with value chain analysis
            """
        ),
        
        FunctionTool.from_defaults(
            fn=get_sector_risk_register,
            name="get_sector_risk_register",
            description="""Get comprehensive sector-specific risk analysis.
            
            This tool identifies and quantifies:
            - Demand drivers and cyclicality
            - Pricing power and cost structure risks
            - Regulatory and technology disruption threats
            - ESG factors and competitive dynamics
            
            Use when assessing investment risks or stress testing scenarios.
            
            Args:
                sector: Company sector
                financial_data_json: JSON string of financial statements
            
            Returns:
                JSON string with detailed risk assessment
            """
        )
    ]
    
    # Combine all tools
    return existing_tools + damodaran_tools


def create_enhanced_answer_agent(openai_api_key: str) -> FunctionAgent:
    """Create enhanced answer agent with Damodaran industry analysis capabilities"""
    
    llm = OpenAI(model="gpt-5-mini", api_key=openai_api_key)
    tools = create_enhanced_research_tools()
    
    system_prompt = """You are an expert equity research analyst with deep expertise in Aswath Damodaran's story-driven valuation methodology and comprehensive industry analysis.

Your enhanced capabilities include:

**CORE RESEARCH TOOLS:**
- SEC filings analysis with semantic search
- Financial data analysis and ratio calculations  
- Technical analysis and price data
- Sentiment analysis of earnings calls and news
- News search and analyst research

**DAMODARAN INDUSTRY ANALYSIS TOOLS:**
- Comprehensive industry analysis using Damodaran's framework
- Value chain mapping and competitive positioning
- Sector-specific risk registers with quantified impacts
- Accounting quirks and sector-specific adjustments

**RESEARCH APPROACH:**
1. **Story-First**: Always consider the business story and how it affects analysis
2. **Sector-Aware**: Use industry-specific tools for deeper context
3. **Risk-Focused**: Identify and quantify key risks using sector analysis
4. **Valuation-Oriented**: Connect all analysis back to investment implications

**WHEN TO USE DAMODARAN TOOLS:**
- **Industry Analysis**: For questions about competitive landscape, market dynamics, or sector trends
- **Value Chain Analysis**: For competitive moat, pricing power, or strategic positioning questions
- **Risk Analysis**: For risk assessment, stress testing, or downside scenario questions

**ENHANCED ANALYSIS FRAMEWORK:**
1. Start with the company's business story and sector context
2. Use industry analysis tools to understand sector dynamics
3. Apply traditional research tools for specific data and metrics
4. Synthesize findings with story validation and risk assessment
5. Provide investment-oriented conclusions with clear reasoning

**QUALITY STANDARDS:**
- Always cite specific data sources and provide context
- Connect sector analysis to company-specific implications
- Quantify risks and opportunities where possible
- Maintain objectivity while building coherent investment thesis
- Flag any story-data inconsistencies or red flags

Remember: Every analysis should contribute to building or validating the investment story while maintaining rigorous analytical standards."""

    return FunctionAgent(
        tools=tools,
        llm=llm,
        verbose=False,
        system_prompt=system_prompt
    )


async def answer_question_enhanced(
    agent: FunctionAgent,
    question: str,
    ticker: str,
    company_name: str,
    story_context: Dict[str, Any] = None,
    sector_analysis: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Answer research question with enhanced story and sector context
    
    Args:
        agent: Enhanced answer agent
        question: Research question
        ticker: Stock ticker
        company_name: Company name
        story_context: Damodaran story context (optional)
        sector_analysis: Sector analysis results (optional)
        
    Returns:
        Enhanced answer with story validation
    """
    try:
        # Build enhanced context
        context_parts = [f"Research question for {company_name} ({ticker}): {question}"]
        
        if story_context:
            context_parts.append(f"\nCompany Story Context:\n{json.dumps(story_context, indent=2)}")
        
        if sector_analysis:
            context_parts.append(f"\nSector Analysis:\n{json.dumps(sector_analysis, indent=2)}")
        
        context_parts.append("""
        \nPlease provide a comprehensive answer that:
        1. Uses appropriate research tools to gather relevant data
        2. Applies Damodaran industry analysis where relevant
        3. Validates findings against the company story (if provided)
        4. Identifies key risks and opportunities
        5. Provides investment-oriented conclusions
        
        Focus on actionable insights that support or challenge the investment thesis.
        """)
        
        enhanced_prompt = "\n".join(context_parts)
        
        # Get enhanced response
        response = agent.chat(enhanced_prompt)
        
        # Structure the response
        return {
            "question": question,
            "answer": str(response),
            "enhanced_analysis": True,
            "story_context_used": story_context is not None,
            "sector_analysis_used": sector_analysis is not None,
            "confidence": 0.8,  # Could be enhanced with actual confidence scoring
            "sources": ["Enhanced research tools", "Damodaran industry analysis"],
            "analysis_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Enhanced question answering failed: {e}")
        return {
            "question": question,
            "answer": f"Analysis failed: {str(e)}",
            "enhanced_analysis": False,
            "error": str(e)
        }


def get_enhanced_answer_agent() -> FunctionAgent:
    """Get the enhanced answer agent"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")
    
    return create_enhanced_answer_agent(openai_api_key)


if __name__ == "__main__":
    # Test the enhanced answer agent
    import asyncio
    
    async def test_enhanced_agent():
        agent = get_enhanced_answer_agent()
        
        test_story = {
            "classification": {"life_cycle_stage": "growth", "business_model_type": "asset_light"},
            "core_business": "Cloud computing and software services",
            "competitive_advantage": "Platform network effects and ecosystem lock-in"
        }
        
        result = await answer_question_enhanced(
            agent,
            "What are the key competitive moats and how sustainable are they?",
            "TEST",
            "Test Company",
            story_context=test_story
        )
        
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_enhanced_agent())
