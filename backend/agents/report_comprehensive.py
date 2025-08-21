#!/usr/bin/env python3
"""
Comprehensive Report Agent

Generates detailed investment reports with full justification, citations, and analysis breakdown.
"""
import os
import logging
from typing import Dict, Any, List
from datetime import datetime
import json

from llama_index.core.agent import FunctionAgent
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)


def create_comprehensive_report_agent(openai_api_key: str) -> FunctionAgent:
    """Create comprehensive report generation agent"""
    
    llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.1)
    
    system_prompt = """You are a senior equity research analyst creating comprehensive investment reports with full transparency and justification.

Your reports must include EVERY detail with complete traceability:

**COMPREHENSIVE REPORT STRUCTURE:**

# 1. EXECUTIVE SUMMARY
- Investment thesis (2-3 sentences)
- Key recommendation with score
- Primary catalysts and risks

# 2. RESEARCH METHODOLOGY
- Questions generated and rationale
- Data sources used
- Analysis timeframe
- Methodology confidence level

# 3. DETAILED QUESTION ANALYSIS
For EACH research question:
- The question itself
- Research methodology used
- Key findings with specific data
- Sources and URLs cited
- Confidence level in findings
- Impact on investment thesis

# 4. FINANCIAL METRICS ANALYSIS
- All financial ratios calculated
- Comparison to industry benchmarks
- Trend analysis over time
- Data sources for each metric
- Calculation methodology
- Interpretation and implications

# 5. VALUATION ANALYSIS
- DCF assumptions and justification
- Multiples analysis methodology
- Fair value calculation steps
- Sensitivity analysis
- Key risk factors to valuation
- Data sources for all inputs

# 6. TECHNICAL ANALYSIS
- All technical indicators calculated
- Trend analysis and patterns
- Support/resistance levels
- Volume analysis
- Technical score justification
- Chart interpretation

# 7. SENTIMENT ANALYSIS
- News articles analyzed (with URLs)
- Sentiment scores and methodology
- Earnings call transcript analysis
- Social media sentiment (if available)
- Uncertainty index calculation
- Overall sentiment interpretation

# 8. SCORING BREAKDOWN
For EACH component (Valuation, Quality, Sentiment, Technical):
- Individual score (X/10)
- Weight in final calculation
- Detailed justification
- Supporting metrics
- Risk factors
- Confidence level

# 9. RISK ANALYSIS
- Regulatory risks
- Market risks
- Company-specific risks
- Macroeconomic factors
- Scenario analysis
- Risk mitigation strategies

# 10. FINAL RECOMMENDATION
- Investment rating (1-10 scale explanation)
- Target price with 12-month timeframe
- Position size recommendation
- Entry/exit strategy
- Key catalysts to monitor
- Stop-loss considerations

# 11. CITATIONS AND SOURCES
- Complete list of all URLs used
- API sources (FMP, Polygon, Tavily, etc.)
- Data collection timestamps
- Source reliability assessment
- Data quality notes

**FORMATTING REQUIREMENTS:**
- Use clear markdown formatting
- Include tables for metrics
- Use bullet points for clarity
- Highlight key findings with **bold**
- Use > blockquotes for important insights
- Include emoji indicators: 📈 📉 ⚠️ ✅ ❌
- Show confidence levels: 🔴 Low | 🟡 Medium | 🟢 High

**TRANSPARENCY REQUIREMENTS:**
- Show ALL calculations
- Explain reasoning for each conclusion
- Include uncertainty where present
- Acknowledge data limitations
- Provide alternative scenarios
- Reference specific data points with sources

Generate a professional, comprehensive report that an institutional investor could use for investment decisions."""

    return FunctionAgent.from_tools(
        tools=[],  # No additional tools needed for report generation
        llm=llm,
        verbose=True,
        system_prompt=system_prompt
    )


async def generate_comprehensive_report(
    report_agent: FunctionAgent,
    ticker: str,
    company_name: str,
    research_answers: List[Dict[str, Any]],
    valuation_data: Dict[str, Any],
    scoring_data: Dict[str, Any],
    financial_data: Dict[str, Any] = None,
    technical_data: Dict[str, Any] = None,
    sentiment_data: Dict[str, Any] = None
) -> str:
    """Generate comprehensive investment report with full justification"""
    
    try:
        # Prepare comprehensive data summary
        analysis_summary = {
            "ticker": ticker,
            "company_name": company_name,
            "analysis_date": datetime.now().isoformat(),
            "research_questions_count": len(research_answers),
            "investment_score": scoring_data.get("final_score", 0),
            "fair_value": valuation_data.get("median_fv", 0),
            "current_price": financial_data.get("current_price", 0) if financial_data else 0
        }
        
        # Extract all citations and URLs
        all_citations = []
        for answer in research_answers:
            findings = answer.get("findings", [])
            for finding in findings:
                evidence = finding.get("evidence", [])
                for cite in evidence:
                    if cite.get("url") and cite["url"] != "internal://json-error":
                        all_citations.append({
                            "title": cite.get("title", ""),
                            "url": cite.get("url", ""),
                            "source_type": cite.get("source_type", ""),
                            "date": cite.get("date", ""),
                            "question": answer.get("question", "")
                        })
        
        # Create detailed research breakdown
        research_breakdown = []
        for i, answer in enumerate(research_answers, 1):
            question_analysis = {
                "number": i,
                "question": answer.get("question", ""),
                "summary": answer.get("summary", ""),
                "confidence": answer.get("confidence", 0),
                "findings_count": len(answer.get("findings", [])),
                "evidence_count": sum(len(f.get("evidence", [])) for f in answer.get("findings", [])),
                "metrics": answer.get("metrics", {}),
                "key_findings": [f.get("statement", "") for f in answer.get("findings", [])[:3]]  # Top 3 findings
            }
            research_breakdown.append(question_analysis)
        
        # Prepare comprehensive input for the agent
        comprehensive_input = f"""
**GENERATE COMPREHENSIVE INVESTMENT REPORT**

**Company:** {company_name} ({ticker})
**Analysis Date:** {datetime.now().strftime("%B %d, %Y")}
**Analysis Summary:** {json.dumps(analysis_summary, indent=2)}

**RESEARCH QUESTIONS & DETAILED ANALYSIS:**
{json.dumps(research_breakdown, indent=2)}

**COMPLETE RESEARCH ANSWERS:**
{json.dumps(research_answers, indent=2)}

**VALUATION ANALYSIS DATA:**
{json.dumps(valuation_data, indent=2)}

**SCORING BREAKDOWN:**
{json.dumps(scoring_data, indent=2)}

**FINANCIAL METRICS:**
{json.dumps(financial_data, indent=2) if financial_data else "Limited financial data available"}

**TECHNICAL ANALYSIS:**
{json.dumps(technical_data, indent=2) if technical_data else "Limited technical data available"}

**SENTIMENT ANALYSIS:**
{json.dumps(sentiment_data, indent=2) if sentiment_data else "Limited sentiment data available"}

**ALL CITATIONS AND SOURCES:**
{json.dumps(all_citations, indent=2)}

**INSTRUCTIONS:**
Create a comprehensive investment report following the detailed structure provided in your system prompt. Include every calculation, justify every conclusion, and cite all sources with URLs. This report will be used by institutional investors for investment decisions.

Focus on:
1. Complete transparency in methodology
2. Detailed justification for each score component
3. Full citation of all data sources
4. Risk analysis with specific scenarios
5. Actionable investment recommendations
6. Professional formatting with clear sections

Generate the report in markdown format with all requested sections and details.
"""
        
        # Generate the comprehensive report
        response = await report_agent.achat(comprehensive_input)
        report_content = str(response.content)
        
        # Add metadata footer
        metadata_footer = f"""

---

## 📊 **Report Metadata**

| **Metric** | **Value** |
|------------|-----------|
| **Ticker** | {ticker} |
| **Company** | {company_name} |
| **Analysis Date** | {datetime.now().strftime("%B %d, %Y at %H:%M UTC")} |
| **Questions Analyzed** | {len(research_answers)} |
| **Sources Cited** | {len(all_citations)} |
| **Investment Score** | {scoring_data.get("final_score", 0):.1f}/10 |
| **Fair Value** | ${valuation_data.get("median_fv", 0):.2f} |
| **Confidence Level** | {scoring_data.get("confidence", 0.5):.1%} |

**Generated by:** Deep Stock Research AI System  
**Model:** GPT-4o-mini  
**Report Version:** Comprehensive v2.0  

---

*This report is generated by an AI system for research purposes. Please conduct your own due diligence before making investment decisions.*
"""
        
        full_report = report_content + metadata_footer
        
        logger.info(f"Generated comprehensive investment report for {ticker}")
        return full_report
        
    except Exception as e:
        logger.error(f"Failed to generate comprehensive report for {ticker}: {e}")
        return f"# Error Generating Report\n\nFailed to generate comprehensive report for {ticker}: {str(e)}"


def get_comprehensive_report_agent() -> FunctionAgent:
    """Get comprehensive report agent instance"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    return create_comprehensive_report_agent(openai_api_key)


if __name__ == "__main__":
    # Test the comprehensive report agent
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def test_comprehensive_report():
        """Test comprehensive report generation"""
        
        # Sample data for testing
        sample_research = [{
            "question": "What are the key revenue trends?",
            "summary": "Strong revenue growth driven by AI demand",
            "confidence": 0.85,
            "findings": [{
                "statement": "Revenue grew 265% YoY in AI segment",
                "evidence": [{
                    "title": "NVIDIA Q3 2024 Earnings",
                    "url": "https://investor.nvidia.com/earnings",
                    "source_type": "earnings",
                    "date": "2024-11-20"
                }]
            }],
            "metrics": {"revenue_growth": 0.265}
        }]
        
        sample_valuation = {
            "median_fv": 120.50,
            "p_underv": 0.65,
            "gap": 0.15
        }
        
        sample_scoring = {
            "final_score": 7.2,
            "confidence": 0.78,
            "components": {
                "valuation": {"score": 8.1, "weight": 0.4},
                "quality": {"score": 7.5, "weight": 0.25},
                "sentiment": {"score": 6.8, "weight": 0.2},
                "technical": {"score": 7.0, "weight": 0.15}
            }
        }
        
        agent = get_comprehensive_report_agent()
        report = await generate_comprehensive_report(
            agent, "NVDA", "NVIDIA Corporation",
            sample_research, sample_valuation, sample_scoring
        )
        
        print("=== COMPREHENSIVE REPORT TEST ===")
        print(report[:1000] + "..." if len(report) > 1000 else report)
    
    asyncio.run(test_comprehensive_report())
