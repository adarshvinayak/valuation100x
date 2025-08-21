#!/usr/bin/env python3
"""
Comprehensive Markdown Report Formatter

Converts analysis results into beautifully formatted markdown reports.
"""
import json
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def format_comprehensive_markdown_report(analysis_data: Dict[str, Any]) -> str:
    """
    Generate a comprehensive markdown report from analysis data
    
    Args:
        analysis_data: Complete analysis results dictionary
        
    Returns:
        Formatted markdown report string
    """
    
    ticker = analysis_data.get("ticker", "UNKNOWN")
    company_name = analysis_data.get("company_name", ticker)
    
    # Extract key metrics
    investment_score = analysis_data.get("investment_score", 0)
    fair_value = analysis_data.get("fair_value", 0)
    current_price = analysis_data.get("current_price", 0)
    probability_undervalued = analysis_data.get("probability_undervalued", 0)
    
    # Component scores
    component_scores = analysis_data.get("component_scores", {})
    scoring_data = analysis_data.get("scoring_data", {})
    
    # Research data
    research_answers = analysis_data.get("research_answers", [])
    research_summary = analysis_data.get("research_summary", {})
    
    # Financial, technical, sentiment data
    financial_data = analysis_data.get("financial_data", {})
    technical_data = analysis_data.get("technical_data", {})
    sentiment_data = analysis_data.get("sentiment_data", {})
    valuation_data = analysis_data.get("valuation_data", {})
    
    # Analysis metadata
    analysis_metadata = analysis_data.get("analysis_metadata", {})
    
    # Build the comprehensive markdown report
    report = f"""# 📊 Comprehensive Investment Analysis Report

## {company_name} ({ticker})

---

### 📈 **Executive Summary**

| **Metric** | **Value** |
|------------|-----------|
| 🎯 **Investment Score** | **{investment_score:.1f}/10** |
| 💰 **Fair Value** | **${fair_value:.2f}** |
| 📊 **Current Price** | **${current_price:.2f}** |
| 📉 **Price Gap** | **{((current_price - fair_value) / fair_value * 100) if fair_value > 0 else 0:.1f}%** |
| 🎲 **Probability Undervalued** | **{probability_undervalued:.1%}** |
| ⏱️ **Analysis Duration** | **{analysis_metadata.get('analysis_duration', 0):.1f} seconds** |
| 📅 **Analysis Date** | **{datetime.now().strftime('%B %d, %Y at %H:%M UTC')}** |

---

## 🎯 **Investment Recommendation**

"""

    # Add recommendation based on score
    if investment_score >= 7:
        recommendation = "🟢 **STRONG BUY**"
        rationale = "High investment score indicates strong fundamentals and attractive valuation."
    elif investment_score >= 6:
        recommendation = "🟢 **BUY**"
        rationale = "Good investment score suggests favorable risk-reward profile."
    elif investment_score >= 5:
        recommendation = "🟡 **HOLD**"
        rationale = "Neutral investment score indicates fair valuation with limited upside."
    elif investment_score >= 4:
        recommendation = "🟠 **WEAK HOLD**" 
        rationale = "Below-average score suggests caution and limited upside potential."
    else:
        recommendation = "🔴 **SELL/AVOID**"
        rationale = "Low investment score indicates significant concerns or overvaluation."

    report += f"""
### {recommendation}

**Rationale:** {rationale}

---

## 📊 **Detailed Scoring Breakdown**

"""

    # Add component scoring breakdown
    if component_scores:
        report += "| **Component** | **Score** | **Weight** | **Weighted Score** | **Performance** |\n"
        report += "|---------------|-----------|------------|--------------------|-----------------|\n"
        
        for component, score in component_scores.items():
            # Get weight from scoring data
            weights = scoring_data.get("component_weights", {})
            weight = weights.get(component, 0)
            weighted = score * weight
            
            # Performance indicator
            if score >= 7:
                performance = "🟢 Strong"
            elif score >= 5:
                performance = "🟡 Average" 
            else:
                performance = "🔴 Weak"
                
            report += f"| **{component.title()}** | {score:.1f}/10 | {weight:.0%} | {weighted:.2f} | {performance} |\n"

    # Add confidence and data quality
    confidence = scoring_data.get("confidence", 0)
    data_quality = research_summary.get("data_quality_score", 0)
    
    report += f"""

### 📈 **Analysis Quality Metrics**

- **Overall Confidence:** {confidence:.1%} 🎯
- **Data Quality Score:** {data_quality:.1%} 📊
- **Questions Answered:** {research_summary.get('questions_answered', 0)}/10 ❓
- **Average Confidence:** {research_summary.get('average_confidence', 0):.1%} 📈
- **Total Citations:** {research_summary.get('total_citations', 0)} 📰

---

## 🔍 **Research Questions & Analysis**

"""

    # Add detailed research questions and answers
    for i, qa in enumerate(research_answers, 1):
        question = qa.get("question", "Unknown question")
        summary = qa.get("summary", "No summary available")
        confidence = qa.get("confidence", 0)
        findings = qa.get("findings", [])
        
        report += f"""
### {i}. {question}

**Summary:** {summary}  
**Confidence:** {confidence:.1%} 📊

"""
        
        # Add key findings
        if findings:
            report += "**Key Findings:**\n"
            for finding in findings[:3]:  # Top 3 findings
                statement = finding.get("statement", "")
                if statement:
                    report += f"- {statement}\n"
        
        # Add evidence sources
        evidence_count = sum(len(f.get("evidence", [])) for f in findings)
        if evidence_count > 0:
            report += f"\n**Sources:** {evidence_count} citations 📚\n"
        
        report += "\n---\n"

    # Add financial analysis
    report += """
## 💰 **Financial Analysis**

"""

    if financial_data:
        ratios = financial_data.get("ratios_ttm", {})
        if ratios:
            report += """
### 📈 **Key Financial Ratios**

| **Metric** | **Value** | **Assessment** |
|------------|-----------|----------------|
"""
            
            # Key ratios with assessments
            financial_metrics = [
                ("ROE", ratios.get("roe", 0), "Return on Equity"),
                ("ROA", ratios.get("roa", 0), "Return on Assets"), 
                ("Debt/Equity", ratios.get("debt_to_equity", 0), "Leverage Ratio"),
                ("Current Ratio", ratios.get("current_ratio", 0), "Liquidity"),
                ("Gross Margin", ratios.get("gross_margin", 0), "Profitability"),
                ("Net Margin", ratios.get("net_margin", 0), "Efficiency")
            ]
            
            for name, value, description in financial_metrics:
                if isinstance(value, (int, float)) and value != 0:
                    if "margin" in name.lower() or "roe" in name.lower() or "roa" in name.lower():
                        formatted_value = f"{value:.1%}"
                        assessment = "🟢 Strong" if value > 0.15 else "🟡 Average" if value > 0.05 else "🔴 Weak"
                    elif "ratio" in name.lower():
                        formatted_value = f"{value:.2f}"
                        if "debt" in name.lower():
                            assessment = "🟢 Strong" if value < 0.3 else "🟡 Average" if value < 0.6 else "🔴 Weak"
                        else:
                            assessment = "🟢 Strong" if value > 1.5 else "🟡 Average" if value > 1.0 else "🔴 Weak"
                    else:
                        formatted_value = f"{value:.2f}"
                        assessment = "🟡 N/A"
                    
                    report += f"| **{name}** | {formatted_value} | {assessment} |\n"

    # Add technical analysis
    report += """

## 📈 **Technical Analysis**

"""

    if technical_data:
        indicators = technical_data.get("indicators", {}) if isinstance(technical_data, dict) else {}
        
        if indicators:
            report += """
### 🎯 **Technical Indicators**

| **Indicator** | **Value** | **Signal** |
|---------------|-----------|------------|
"""
            
            # Technical indicators
            tech_metrics = [
                ("Current Price", indicators.get("current_price"), "$"),
                ("SMA 50", indicators.get("sma_50"), "$"),
                ("Price vs SMA 50", indicators.get("price_vs_sma_50"), "%"),
                ("RSI (14)", indicators.get("rsi_14"), ""),
                ("1M Return", indicators.get("return_1m"), "%"),
                ("Volatility (30d)", indicators.get("volatility_30d"), "%"),
                ("Max Drawdown", indicators.get("max_drawdown"), "%")
            ]
            
            for name, value, unit in tech_metrics:
                if value is not None and value != 0:
                    if unit == "$":
                        formatted_value = f"${value:.2f}"
                    elif unit == "%":
                        formatted_value = f"{value:.1f}%"
                    else:
                        formatted_value = f"{value:.2f}"
                    
                    # Signal assessment
                    if "rsi" in name.lower():
                        signal = "🟢 Bullish" if 30 <= value <= 70 else "🟡 Neutral"
                    elif "return" in name.lower():
                        signal = "🟢 Positive" if value > 0 else "🔴 Negative"
                    elif "price_vs" in name.lower():
                        signal = "🟢 Above MA" if value > 0 else "🔴 Below MA"
                    else:
                        signal = "🟡 Neutral"
                    
                    report += f"| **{name}** | {formatted_value} | {signal} |\n"

    # Add sentiment analysis
    report += """

## 😊 **Sentiment Analysis**

"""

    if sentiment_data:
        avg_sentiment = sentiment_data.get("average_sentiment", 0)
        transcript_analysis = sentiment_data.get("transcript_analysis", {})
        
        report += f"""
### 📰 **Market Sentiment Overview**

- **Overall Sentiment:** {avg_sentiment:.2f} {'🟢 Positive' if avg_sentiment > 0.1 else '🔴 Negative' if avg_sentiment < -0.1 else '🟡 Neutral'}
- **Transcript Sentiment:** {transcript_analysis.get('overall_sentiment', 0):.2f}
- **Uncertainty Index:** {transcript_analysis.get('uncertainty_index', 0):.2f}

"""

    # Add valuation analysis
    report += """

## 💎 **Valuation Analysis**

"""

    if valuation_data:
        report += f"""
### 🎯 **Valuation Summary**

| **Method** | **Value** | **Details** |
|------------|-----------|-------------|
| **Fair Value** | ${valuation_data.get('median_fv', 0):.2f} | DCF-based estimate |
| **Current Price** | ${current_price:.2f} | Market price |
| **Upside/Downside** | {((fair_value - current_price) / current_price * 100) if current_price > 0 else 0:.1f}% | Price vs Fair Value |
| **P(Undervalued)** | {valuation_data.get('p_underv', 0):.1%} | Probability estimate |
| **Implied CAGR** | {valuation_data.get('implied_cagr', 0):.1%} | Reverse DCF |

"""

    # Add risk analysis
    report += """

## ⚠️ **Risk Analysis**

### 🚨 **Key Risk Factors**

"""

    # Extract risks from research answers
    risk_factors = []
    for qa in research_answers:
        question = qa.get("question", "").lower()
        if any(risk_word in question for risk_word in ["risk", "challenge", "regulatory", "competition"]):
            summary = qa.get("summary", "")
            if summary:
                risk_factors.append(summary)

    if risk_factors:
        for i, risk in enumerate(risk_factors[:5], 1):  # Top 5 risks
            report += f"{i}. **{risk}**\n"
    else:
        report += "- Regulatory and compliance risks\n"
        report += "- Market volatility and competition\n" 
        report += "- Operational and financial risks\n"

    # Add data sources and citations
    report += """

---

## 📚 **Data Sources & Citations**

"""

    # Extract all unique citations
    all_citations = []
    for qa in research_answers:
        findings = qa.get("findings", [])
        for finding in findings:
            evidence = finding.get("evidence", [])
            for cite in evidence:
                if cite.get("url") and cite["url"] != "internal://json-error":
                    all_citations.append(cite)

    # Remove duplicates
    unique_urls = set()
    unique_citations = []
    for cite in all_citations:
        url = cite.get("url", "")
        if url not in unique_urls:
            unique_urls.add(url)
            unique_citations.append(cite)

    if unique_citations:
        report += f"""
### 🔗 **Sources Used ({len(unique_citations)} total)**

"""
        source_types = {}
        for cite in unique_citations:
            source_type = cite.get("source_type", "other")
            if source_type not in source_types:
                source_types[source_type] = []
            source_types[source_type].append(cite)

        for source_type, citations in source_types.items():
            report += f"\n#### {source_type.title()} Sources ({len(citations)})\n\n"
            for cite in citations[:10]:  # Limit to 10 per type
                title = cite.get("title", "Untitled")
                url = cite.get("url", "#")
                date = cite.get("date", "")
                report += f"- [{title}]({url})"
                if date:
                    report += f" _{date}_"
                report += "\n"

    # Add API sources
    data_sources = analysis_metadata.get("data_sources_used", [])
    if data_sources:
        report += f"""

### 🔌 **API Data Sources**

"""
        source_descriptions = {
            "news": "📰 Tavily News API - Real-time news and market sentiment",
            "financial_data": "💰 Financial Modeling Prep - Financial statements and ratios", 
            "analyst_research": "🔬 Research Reports - Professional analyst coverage",
            "earnings_news": "📊 Earnings Data - Quarterly results and guidance",
            "technical": "📈 Polygon.io - Market data and technical indicators"
        }
        
        for source in data_sources:
            description = source_descriptions.get(source, f"📊 {source.title()} - Market data")
            report += f"- {description}\n"

    # Add footer
    report += f"""

---

## 📋 **Report Metadata**

| **Field** | **Value** |
|-----------|-----------|
| **Report ID** | {ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')} |
| **Generated** | {datetime.now().strftime('%B %d, %Y at %H:%M UTC')} |
| **Workflow Version** | {analysis_metadata.get('workflow_version', '2.0')} |
| **Analysis Model** | GPT-4o-mini (OpenAI) |
| **System** | Deep Stock Research AI |

---

## ⚖️ **Disclaimer**

*This comprehensive investment analysis report is generated by an AI-powered research system for informational and educational purposes only. The analysis is based on publicly available data and should not be considered as personalized investment advice.*

**Important Notes:**
- Past performance does not guarantee future results
- All investments carry risk of loss
- Market conditions can change rapidly
- Please conduct your own due diligence
- Consult with qualified financial advisors before making investment decisions

**Data Accuracy:** While we strive for accuracy, this analysis is based on publicly available information and automated processing. Users should verify critical information independently.

**Report Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M:%S UTC')}  
**Powered by:** Deep Stock Research AI System v2.0

---

*© 2025 Deep Stock Research AI. This report contains proprietary analysis and should not be redistributed without permission.*
"""

    return report


def generate_summary_table(analysis_results: List[Dict[str, Any]]) -> str:
    """Generate a summary table for multiple analyses"""
    
    if not analysis_results:
        return "No analysis results to display."
    
    report = f"""# 📊 Investment Analysis Summary

**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}  
**Total Companies Analyzed:** {len(analysis_results)}

## 🎯 **Executive Summary Table**

| **Ticker** | **Score** | **Recommendation** | **Fair Value** | **Current Price** | **Upside** | **Confidence** |
|------------|-----------|-------------------|----------------|-------------------|------------|----------------|
"""
    
    for result in analysis_results:
        ticker = result.get("ticker", "N/A")
        score = result.get("investment_score", 0)
        fair_value = result.get("fair_value", 0)
        current_price = result.get("current_price", 0)
        
        # Calculate upside
        upside = ((fair_value - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Get confidence
        scoring_data = result.get("scoring_data", {})
        confidence = scoring_data.get("confidence", 0)
        
        # Recommendation based on score
        if score >= 7:
            recommendation = "🟢 BUY"
        elif score >= 5:
            recommendation = "🟡 HOLD"
        else:
            recommendation = "🔴 SELL"
        
        report += f"| **{ticker}** | {score:.1f}/10 | {recommendation} | ${fair_value:.2f} | ${current_price:.2f} | {upside:+.1f}% | {confidence:.0%} |\n"
    
    report += f"""

---

*Generated by Deep Stock Research AI System*
"""
    
    return report


if __name__ == "__main__":
    # Test the markdown formatter
    sample_data = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "investment_score": 7.2,
        "fair_value": 150.0,
        "current_price": 140.0,
        "component_scores": {"valuation": 8.0, "quality": 7.5, "sentiment": 6.8, "technical": 7.0},
        "research_answers": [{"question": "Test question?", "summary": "Test summary", "confidence": 0.85}]
    }
    
    report = format_comprehensive_markdown_report(sample_data)
    print("=== MARKDOWN FORMATTER TEST ===")
    print(report[:500] + "...")
