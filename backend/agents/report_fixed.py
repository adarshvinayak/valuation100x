#!/usr/bin/env python3
"""
Report Agent - Fixed

Generates comprehensive investment reports by synthesizing research findings.
Uses the correct LlamaIndex FunctionAgent pattern from the reference implementation.
"""
import os
import logging
from typing import Dict, Any, List
from datetime import datetime

from llama_index.core.agent import FunctionAgent
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)


def create_report_agent(openai_api_key: str) -> FunctionAgent:
    """Create report generation agent following reference pattern"""
    
    llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.2)
    
    system_prompt = """You are a senior equity research analyst responsible for synthesizing comprehensive investment reports.

Your role is to create concise, actionable investment reports based on research findings, valuation analysis, and scoring results.

**Report Structure:**
1. **Executive Summary** - Key investment thesis in 2-3 sentences
2. **Investment Score** - Overall rating with rationale
3. **Valuation Analysis** - Fair value estimate and methodology
4. **Key Strengths** - Primary positive factors
5. **Key Risks** - Primary concerns and downside factors
6. **Investment Recommendation** - Clear buy/hold/sell guidance
7. **Price Target** - Specific target with timeframe
8. **Research Summary** - Key findings from analysis

**Guidelines:**
- Be concise but comprehensive (aim for 800-1200 words)
- Use professional, objective language
- Include specific metrics and data points
- Highlight both positives and negatives
- Provide actionable insights for investment decisions
- Use bullet points for clarity where appropriate
- Include confidence levels for key assumptions
- Cite key research findings and sources

**Response Format:**
Provide a well-structured markdown report that synthesizes all the provided information into a coherent investment analysis."""

    return FunctionAgent(
        tools=[],
        llm=llm,
        verbose=False,
        system_prompt=system_prompt
    )


async def generate_report_with_agent(
    agent: FunctionAgent,
    ticker: str,
    company_name: str,
    research_answers: List[Dict[str, Any]],
    valuation_data: Dict[str, Any],
    scoring_data: Dict[str, Any]
) -> str:
    """Generate final investment report using the report agent"""
    try:
        # Extract key data for the report
        investment_score = scoring_data.get("score", 5.0)
        confidence = scoring_data.get("confidence", 0.5)
        fair_value = valuation_data.get("median_fv", 0)
        current_price = valuation_data.get("current_price", fair_value)
        p_undervalued = valuation_data.get("p_underv", 0.5)
        
        # Determine recommendation
        if investment_score >= 7:
            recommendation = "STRONG BUY"
        elif investment_score >= 6:
            recommendation = "BUY"
        elif investment_score >= 5:
            recommendation = "HOLD"
        elif investment_score >= 4:
            recommendation = "WEAK HOLD"
        else:
            recommendation = "SELL"
        
        # Create comprehensive prompt with all context
        prompt = f"""Generate a comprehensive investment report for {company_name} ({ticker}).

**Analysis Results:**
- Investment Score: {investment_score:.1f}/10
- Confidence Level: {confidence:.1%}
- Current Price: ${current_price:.2f}
- Fair Value Estimate: ${fair_value:.2f}
- Probability Undervalued: {p_undervalued:.1%}
- Preliminary Recommendation: {recommendation}

**Component Scores:**
"""
        
        # Add component breakdown if available
        if "components" in scoring_data:
            for component, details in scoring_data["components"].items():
                score = details.get("score", 5.0)
                weight = details.get("weight", 0.25)
                prompt += f"- {component.title()}: {score:.1f}/10 (weight: {weight:.0%})\n"
        
        prompt += f"""

**Valuation Context:**
- Current valuation: ${current_price:.2f}
- Fair value estimate: ${fair_value:.2f}
- Valuation gap: {((fair_value/current_price - 1) * 100) if current_price > 0 else 0:.1f}%
- Implied CAGR: {valuation_data.get('implied_cagr', 0.12)*100:.1f}%

**Research Findings Summary:**
"""
        
        # Add key findings from research
        key_findings = []
        for i, answer in enumerate(research_answers, 1):
            question = answer.get("question", "")
            summary = answer.get("summary", "")
            confidence_level = answer.get("confidence", 0.5)
            
            if summary:
                key_findings.append(f"**Q{i}:** {question}")
                key_findings.append(f"**Answer:** {summary}")
                key_findings.append(f"**Confidence:** {confidence_level:.1%}")
                
                # Add key metrics if available
                metrics = answer.get("metrics", {})
                if metrics:
                    key_metrics = []
                    for key, value in list(metrics.items())[:3]:  # Top 3 metrics
                        key_metrics.append(f"{key}: {value}")
                    if key_metrics:
                        key_findings.append(f"**Key Metrics:** {', '.join(key_metrics)}")
                key_findings.append("")  # Blank line
        
        prompt += "\n".join(key_findings)
        
        prompt += f"""

**Analysis Guidelines:**
- Synthesize all research findings into a coherent investment thesis
- Provide specific rationale for the investment score and recommendation
- Highlight the most critical factors for investment decision-making
- Include both bullish and bearish scenarios
- Address key risks and catalysts
- Provide actionable insights for investors

Generate a professional investment report that combines all this analysis into clear, actionable investment guidance.
"""

        # Generate report using the agent (using correct API)
        result = await agent.run(user_msg=prompt)
        report_text = str(result)
        
        # Clean up response format
        if report_text.startswith("assistant: "):
            report_text = report_text[11:]
        
        # Add header and footer
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        final_report = f"""# {company_name} ({ticker}) - Investment Research Report

**Generated:** {timestamp}  
**Analyst Confidence:** {confidence:.1%}  
**Overall Score:** {investment_score:.1f}/10  
**Recommendation:** {recommendation}

---

{report_text}

---

## Disclaimer

*This report is generated by an AI-powered research system for informational purposes only. The analysis is based on publicly available data and should not be considered as personalized investment advice. Please conduct your own due diligence and consult with qualified financial advisors before making investment decisions. Past performance does not guarantee future results.*

**Report ID:** {ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}
"""
        
        logger.info(f"Generated investment report for {ticker}")
        return final_report.strip()
        
    except Exception as e:
        logger.error(f"Failed to generate report for {ticker}: {e}")
        
        # Return a basic fallback report
        return f"""# {company_name} ({ticker}) - Investment Research Report

**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

## Executive Summary

Research analysis completed for {company_name} with an investment score of {scoring_data.get('score', 5.0):.1f}/10.

## Analysis Error

Unable to generate complete report due to system limitations: {str(e)}

## Basic Metrics

- Current Price: ${valuation_data.get('current_price', 0):.2f}
- Fair Value Estimate: ${valuation_data.get('median_fv', 0):.2f}
- Investment Score: {scoring_data.get('score', 5.0):.1f}/10
- Confidence: {scoring_data.get('confidence', 0.5):.1%}

## Recommendation

Please review individual research components for detailed analysis.

---

*Report generation encountered technical difficulties. Manual review recommended.*
"""


# Global agent instance  
_report_agent = None

def get_report_agent(openai_api_key: str = None) -> FunctionAgent:
    """Get or create the global report agent instance"""
    global _report_agent
    
    if _report_agent is None:
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found")
        
        _report_agent = create_report_agent(openai_api_key)
    
    return _report_agent


async def generate_report(
    ticker: str,
    company_name: str,
    research_answers: List[Dict[str, Any]],
    valuation_data: Dict[str, Any],
    scoring_data: Dict[str, Any]
) -> str:
    """
    Convenience function for generating investment reports
    
    This is the main function that will be used by the workflow.
    """
    agent = get_report_agent()
    return await generate_report_with_agent(agent, ticker, company_name, research_answers, valuation_data, scoring_data)
