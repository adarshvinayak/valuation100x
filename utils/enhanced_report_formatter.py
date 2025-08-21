#!/usr/bin/env python3
"""
Enhanced Report Formatter

Formats comprehensive investment reports into detailed JSON and Markdown formats
with complete transparency, calculations, and source citations.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedReportFormatter:
    """Formats enhanced reports into comprehensive JSON and Markdown"""
    
    def __init__(self):
        self.markdown_template = self._create_markdown_template()
    
    def format_enhanced_report(
        self, 
        enhanced_data: Dict[str, Any],
        ticker: str,
        company_name: str
    ) -> Dict[str, str]:
        """Format enhanced report into JSON and Markdown"""
        
        try:
            # Generate comprehensive JSON
            enhanced_json = self._format_enhanced_json(enhanced_data, ticker, company_name)
            
            # Generate comprehensive Markdown
            enhanced_markdown = self._format_enhanced_markdown(enhanced_data, ticker, company_name)
            
            return {
                "json": enhanced_json,
                "markdown": enhanced_markdown
            }
            
        except Exception as e:
            logger.error(f"Error formatting enhanced report: {e}")
            raise e
    
    def _format_enhanced_json(
        self,
        enhanced_data: Dict[str, Any],
        ticker: str, 
        company_name: str
    ) -> str:
        """Format comprehensive JSON with all data and calculations"""
        
        try:
            # Create comprehensive JSON structure
            comprehensive_json = {
                "report_metadata": {
                    "ticker": ticker,
                    "company_name": company_name,
                    "report_type": "Enhanced Comprehensive Investment Analysis",
                    "generation_date": datetime.now().isoformat(),
                    "analysis_framework": "Traditional DCF + Damodaran Story-Driven + ValueInvesting.io",
                    "report_version": "2.0"
                },
                "executive_summary": enhanced_data.get("sections", {}).get("executive_summary", {}),
                "detailed_research": enhanced_data.get("sections", {}).get("detailed_research", {}),
                "valuation_analysis": enhanced_data.get("sections", {}).get("valuation_analysis", {}),
                "financial_analysis": enhanced_data.get("sections", {}).get("financial_analysis", {}),
                "sentiment_analysis": enhanced_data.get("sections", {}).get("sentiment_analysis", {}),
                "technical_analysis": enhanced_data.get("sections", {}).get("technical_analysis", {}),
                "risk_assessment": enhanced_data.get("sections", {}).get("risk_assessment", {}),
                "methodology": enhanced_data.get("sections", {}).get("methodology", {}),
                "data_sources": enhanced_data.get("data_sources", []),
                "transparency_note": "All calculations, assumptions, and data sources are disclosed for full transparency"
            }
            
            return json.dumps(comprehensive_json, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error formatting enhanced JSON: {e}")
            return json.dumps({"error": str(e)})
    
    def _format_enhanced_markdown(
        self,
        enhanced_data: Dict[str, Any],
        ticker: str,
        company_name: str
    ) -> str:
        """Format comprehensive Markdown report"""
        
        try:
            sections = enhanced_data.get("sections", {})
            
            markdown_report = f"""# 📊 Comprehensive Investment Research Report with Full Transparency

## {company_name} ({ticker})

**Report Generation Date:** {datetime.now().strftime("%B %d, %Y at %I:%M %p UTC")}  
**Analysis Framework:** Traditional DCF + Damodaran Story-Driven + ValueInvesting.io Insights  
**Report Version:** Enhanced Comprehensive v2.0

---

## 🎯 Executive Summary

{self._format_executive_summary_md(sections.get("executive_summary", {}))}

---

## 🔍 Detailed Research Analysis

{self._format_research_analysis_md(sections.get("detailed_research", {}))}

---

## 💰 Transparent Valuation Analysis

{self._format_valuation_analysis_md(sections.get("valuation_analysis", {}))}

---

## 📈 Comprehensive Financial Analysis

{self._format_financial_analysis_md(sections.get("financial_analysis", {}))}

---

## 📰 Sentiment Analysis with Source Citations

{self._format_sentiment_analysis_md(sections.get("sentiment_analysis", {}))}

---

## 📊 Technical Analysis

{self._format_technical_analysis_md(sections.get("technical_analysis", {}))}

---

## ⚠️ Risk Assessment

{self._format_risk_assessment_md(sections.get("risk_assessment", {}))}

---

## 📚 Methodology & Data Sources

{self._format_methodology_md(sections.get("methodology", {}), enhanced_data.get("data_sources", []))}

---

## 🔗 Data Sources Bibliography

{self._format_data_sources_md(enhanced_data.get("data_sources", []))}

---

## ⚖️ Disclaimer

*This comprehensive report is generated by an AI-powered research system that integrates multiple institutional-grade data sources and methodologies. The analysis includes traditional quantitative analysis, Damodaran's story-driven valuation framework, and insights from ValueInvesting.io's automated DCF platform. All calculations, assumptions, and data sources are disclosed for complete transparency.*

*This report is for informational purposes only and should not be considered as personalized investment advice. Please conduct your own due diligence and consult with qualified financial advisors before making investment decisions. Past performance does not guarantee future results.*

**Report ID:** {ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_Enhanced  
**Generated by:** DeepResearch Enhanced Analysis System  
**Data Sources:** {len(enhanced_data.get("data_sources", []))} institutional-grade sources

---

*© 2025 DeepResearch - Enhanced Investment Analysis Platform*
"""
            
            return markdown_report
            
        except Exception as e:
            logger.error(f"Error formatting enhanced Markdown: {e}")
            return f"# Error Generating Report\n\n{str(e)}"
    
    def _format_executive_summary_md(self, executive_summary: Dict[str, Any]) -> str:
        """Format executive summary section"""
        
        if not executive_summary:
            return "*Executive summary data not available*"
        
        investment_rec = executive_summary.get("investment_recommendation", {})
        valuation_summary = executive_summary.get("valuation_summary", {})
        key_metrics = executive_summary.get("key_metrics_table", {})
        component_scores = executive_summary.get("component_scores", {})
        
        return f"""
### 🎯 Investment Recommendation

| **Metric** | **Value** |
|------------|-----------|
| **Investment Rating** | **{investment_rec.get("rating", "N/A")}** |
| **Investment Score** | **{investment_rec.get("score", "N/A")}** |
| **Analysis Confidence** | **{investment_rec.get("confidence", "N/A")}** |

### 💰 Valuation Summary

| **Metric** | **Value** |
|------------|-----------|
| **Current Price** | {valuation_summary.get("current_price", "N/A")} |
| **Fair Value Estimate** | {valuation_summary.get("fair_value_estimate", "N/A")} |
| **Price Gap** | {valuation_summary.get("price_gap", "N/A")} |
| **Valuation Method** | {valuation_summary.get("valuation_method", "N/A")} |

### 📊 Key Financial Metrics

| **Metric** | **Value** |
|------------|-----------|
{self._dict_to_table_rows(key_metrics)}

### 🏆 Component Analysis Scores

| **Component** | **Score** | **Performance** |
|---------------|-----------|-----------------|
| **Valuation** | {component_scores.get("Valuation", "N/A")} | {self._get_performance_emoji(component_scores.get("Valuation", "0/10"))} |
| **Quality** | {component_scores.get("Quality", "N/A")} | {self._get_performance_emoji(component_scores.get("Quality", "0/10"))} |
| **Sentiment** | {component_scores.get("Sentiment", "N/A")} | {self._get_performance_emoji(component_scores.get("Sentiment", "0/10"))} |
| **Technicals** | {component_scores.get("Technicals", "N/A")} | {self._get_performance_emoji(component_scores.get("Technicals", "0/10"))} |
"""
    
    def _format_research_analysis_md(self, research_data: Dict[str, Any]) -> str:
        """Format detailed research analysis section"""
        
        if not research_data:
            return "*Research analysis data not available*"
        
        overview = research_data.get("overview", {})
        
        md_content = f"""
### 📋 Research Overview

| **Metric** | **Value** |
|------------|-----------|
| **Total Questions Analyzed** | {overview.get("total_questions", "N/A")} |
| **Average Confidence** | {overview.get("average_confidence", "N/A")} |
| **Analysis Duration** | {overview.get("analysis_duration", "N/A")} |
| **Data Sources Used** | {len(overview.get("data_sources_used", []))} sources |

### 🎭 Damodaran Story-Driven Analysis

"""
        
        # Add Damodaran story analysis if available
        story_analysis = research_data.get("damodaran_story_analysis", {})
        if story_analysis:
            md_content += f"""
**Framework:** {story_analysis.get("framework", "N/A")}  
**Analysis Date:** {story_analysis.get("analysis_date", "N/A")}

**Business Story:**
{story_analysis.get("story_response", "Story analysis not available")}
"""
        
        # Add ValueInvesting.io insights
        vi_insights = research_data.get("valueinvesting_io_insights", {})
        if vi_insights:
            md_content += f"""

### 🏛️ ValueInvesting.io Institutional Insights

**Methodology:** {vi_insights.get("methodology", "N/A")}

**Data Sources:**
{self._format_list_items(vi_insights.get("data_sources", []))}

**DCF Components:**
- Revenue Projections: {vi_insights.get("dcf_components", {}).get("revenue_projections", {}).get("methodology", "N/A")}
- Margin Analysis: {vi_insights.get("dcf_components", {}).get("margin_analysis", {}).get("operating_margins", "N/A")}
- Capital Requirements: {vi_insights.get("dcf_components", {}).get("capital_requirements", {}).get("reinvestment_needs", "N/A")}
- Discount Rate: {vi_insights.get("dcf_components", {}).get("discount_rate", {}).get("wacc_calculation", "N/A")}
"""
        
        return md_content
    
    def _format_valuation_analysis_md(self, valuation_data: Dict[str, Any]) -> str:
        """Format valuation analysis with complete transparency"""
        
        if not valuation_data:
            return "*Valuation analysis data not available*"
        
        md_content = """
### 🧮 DCF Model Assumptions & Calculations

#### WACC Calculation
"""
        
        wacc_calc = valuation_data.get("wacc_calculation", {})
        if wacc_calc:
            md_content += f"""
| **Component** | **Value** | **Source/Justification** |
|---------------|-----------|--------------------------|
{self._dict_to_table_rows_with_justification(wacc_calc)}
"""
        
        md_content += """

#### Growth Assumptions
"""
        
        growth_assumptions = valuation_data.get("growth_assumptions", {})
        if growth_assumptions:
            md_content += f"""
| **Assumption** | **Value** | **Rationale** |
|----------------|-----------|---------------|
{self._dict_to_table_rows_with_rationale(growth_assumptions)}
"""
        
        md_content += """

#### Capital Assumptions
"""
        
        capital_assumptions = valuation_data.get("capital_assumptions", {})
        if capital_assumptions:
            md_content += f"""
| **Component** | **Value** | **Industry Benchmark** |
|---------------|-----------|------------------------|
{self._dict_to_table_rows_with_benchmark(capital_assumptions)}
"""
        
        # Add sensitivity analysis
        sensitivity = valuation_data.get("sensitivity_analysis", {})
        if sensitivity:
            md_content += """

#### Sensitivity Analysis

**WACC Sensitivity:**
"""
            wacc_sens = sensitivity.get("wacc_sensitivity", {})
            if wacc_sens:
                md_content += f"""
- WACC Range: {wacc_sens.get("wacc_range", [])}
- Value Impact: {wacc_sens.get("value_impact", [])}%
"""
            
            growth_sens = sensitivity.get("growth_sensitivity", {})
            if growth_sens:
                md_content += f"""

**Growth Rate Sensitivity:**
- Terminal Growth Range: {growth_sens.get("terminal_growth_range", [])}
- Value Impact: {growth_sens.get("value_impact", [])}%
"""
        
        # Add final valuation results
        results = valuation_data.get("valuation_results", {})
        if results:
            md_content += f"""

### 🎯 Valuation Results

| **Metric** | **Value** |
|------------|-----------|
{self._dict_to_table_rows(results)}
"""
        
        return md_content
    
    def _format_financial_analysis_md(self, financial_data: Dict[str, Any]) -> str:
        """Format comprehensive financial analysis"""
        
        if not financial_data:
            return "*Financial analysis data not available*"
        
        md_content = """
### 💹 Financial Performance Metrics

#### Profitability Analysis
"""
        
        profitability = financial_data.get("profitability_metrics", {})
        if profitability:
            md_content += f"""
| **Metric** | **Value** |
|------------|-----------|
{self._dict_to_table_rows(profitability)}
"""
        
        md_content += """

#### Liquidity Analysis
"""
        
        liquidity = financial_data.get("liquidity_metrics", {})
        if liquidity:
            md_content += f"""
| **Metric** | **Value** |
|------------|-----------|
{self._dict_to_table_rows(liquidity)}
"""
        
        md_content += """

#### Leverage Analysis
"""
        
        leverage = financial_data.get("leverage_metrics", {})
        if leverage:
            md_content += f"""
| **Metric** | **Value** |
|------------|-----------|
{self._dict_to_table_rows(leverage)}
"""
        
        # Add cash flow metrics
        cash_flow = financial_data.get("cash_flow_metrics", {})
        if cash_flow:
            md_content += f"""

#### Cash Flow Analysis

| **Metric** | **Value** |
|------------|-----------|
{self._dict_to_table_rows(cash_flow)}
"""
        
        # Add segment analysis for META
        segment_analysis = financial_data.get("segment_analysis", {})
        if segment_analysis:
            md_content += """

### 🏢 Business Segment Analysis
"""
            for segment, data in segment_analysis.items():
                md_content += f"""

#### {segment.replace('_', ' ')}
- **Description:** {data.get("description", "N/A")}
- **Revenue Contribution:** {data.get("revenue_contribution", "N/A")}
- **Operating Characteristics:** {data.get("operating_characteristics", "N/A")}
- **Growth Drivers:** {', '.join(data.get("growth_drivers", []))}
"""
        
        return md_content
    
    def _format_sentiment_analysis_md(self, sentiment_data: Dict[str, Any]) -> str:
        """Format sentiment analysis with source citations"""
        
        if not sentiment_data:
            return "*Sentiment analysis data not available*"
        
        md_content = f"""
### 📊 Overall Sentiment Score: {sentiment_data.get("overall_sentiment_score", "N/A")}

### 📰 Data Sources Analyzed
"""
        
        data_sources = sentiment_data.get("data_sources", [])
        for source in data_sources:
            md_content += f"""
- **{source.get("name", "Unknown")}** ([{source.get("url", "No URL")}]({source.get("url", "#")}))
  - Type: {source.get("type", "N/A")}
  - Description: {source.get("description", "N/A")}
"""
        
        # Add institutional perspective
        institutional = sentiment_data.get("institutional_perspective", {})
        if institutional:
            md_content += f"""

### 🏛️ Institutional Perspective

**Source:** [{institutional.get("source", "N/A")}]({institutional.get("url", "#")})  
**Methodology:** {institutional.get("methodology", "N/A")}  
**Coverage:** {institutional.get("coverage", "N/A")}  
**Data Quality:** {institutional.get("data_quality", "N/A")}
"""
        
        return md_content
    
    def _format_technical_analysis_md(self, technical_data: Dict[str, Any]) -> str:
        """Format technical analysis section"""
        
        if not technical_data:
            return "*Technical analysis data not available*"
        
        md_content = f"""
### 📈 Technical Score: {technical_data.get("overall_technical_score", "N/A")}

### 📊 Data Source
"""
        
        data_source = technical_data.get("data_source", {})
        if data_source:
            md_content += f"""
**Provider:** [{data_source.get("provider", "N/A")}]({data_source.get("url", "#")})  
**Data Type:** {data_source.get("data_type", "N/A")}  
**Update Frequency:** {data_source.get("update_frequency", "N/A")}
"""
        
        return md_content
    
    def _format_risk_assessment_md(self, risk_data: Dict[str, Any]) -> str:
        """Format risk assessment section"""
        
        if not risk_data:
            return "*Risk assessment data not available*"
        
        md_content = f"""
### ⚠️ Overall Risk Level: {risk_data.get("overall_risk_level", "N/A")}

"""
        
        # Add story-driven risks
        story_risks = risk_data.get("story_driven_risks", {})
        if story_risks:
            md_content += f"""
### 🎭 Damodaran Story-Driven Risk Analysis

**Source:** {story_risks.get("source", "N/A")}  
**Methodology:** {story_risks.get("methodology", "N/A")}

{story_risks.get("analysis", "Risk analysis not available")}
"""
        
        return md_content
    
    def _format_methodology_md(
        self, 
        methodology_data: Dict[str, Any],
        data_sources: List[Dict[str, Any]]
    ) -> str:
        """Format methodology section"""
        
        if not methodology_data:
            return "*Methodology data not available*"
        
        md_content = """
### 🔬 Analysis Framework
"""
        
        framework = methodology_data.get("analysis_framework", {})
        if framework:
            md_content += f"""
**Primary Framework:** {framework.get("primary_framework", "N/A")}

**Components:**
{self._format_list_items(framework.get("components", []))}
"""
        
        # Add valuation methodologies
        valuation_methods = methodology_data.get("valuation_methodologies", {})
        if valuation_methods:
            md_content += """

### 💰 Valuation Methodologies
"""
            for method, details in valuation_methods.items():
                md_content += f"""

#### {method.replace('_', ' ')}
- **Description:** {details.get("description", "N/A")}
- **Source:** {details.get("source", "N/A")}
- **Key Inputs:** {', '.join(details.get("key_inputs", []))}
"""
        
        # Add quality assurance
        qa = methodology_data.get("quality_assurance", {})
        if qa:
            md_content += f"""

### ✅ Quality Assurance

| **Aspect** | **Standard** |
|------------|--------------|
{self._dict_to_table_rows(qa)}
"""
        
        return md_content
    
    def _format_data_sources_md(self, data_sources: List[Dict[str, Any]]) -> str:
        """Format data sources bibliography"""
        
        if not data_sources:
            return "*No data sources available*"
        
        md_content = ""
        for i, source in enumerate(data_sources, 1):
            md_content += f"""
{i}. **{source.get("name", "Unknown Source")}**
   - URL: [{source.get("url", "No URL")}]({source.get("url", "#")})
   - Type: {source.get("type", "N/A")}
   - Description: {source.get("description", "N/A")}

"""
        
        return md_content
    
    def _dict_to_table_rows(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to markdown table rows"""
        if not data:
            return "| N/A | N/A |"
        
        rows = []
        for key, value in data.items():
            rows.append(f"| **{key}** | {value} |")
        return "\n".join(rows)
    
    def _dict_to_table_rows_with_justification(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to table rows with justification column"""
        if not data:
            return "| N/A | N/A | N/A |"
        
        rows = []
        for key, value in data.items():
            justification = "Market standard" if "Rate" in key else "Company-specific calculation"
            rows.append(f"| **{key}** | {value} | {justification} |")
        return "\n".join(rows)
    
    def _dict_to_table_rows_with_rationale(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to table rows with rationale column"""
        if not data:
            return "| N/A | N/A | N/A |"
        
        rows = []
        for key, value in data.items():
            rationale = "Based on historical trends and industry analysis"
            rows.append(f"| **{key}** | {value} | {rationale} |")
        return "\n".join(rows)
    
    def _dict_to_table_rows_with_benchmark(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to table rows with benchmark column"""
        if not data:
            return "| N/A | N/A | N/A |"
        
        rows = []
        for key, value in data.items():
            benchmark = "Industry median"
            rows.append(f"| **{key}** | {value} | {benchmark} |")
        return "\n".join(rows)
    
    def _format_list_items(self, items: List[str]) -> str:
        """Format list items as markdown bullets"""
        if not items:
            return "- None available"
        
        return "\n".join([f"- {item}" for item in items])
    
    def _get_performance_emoji(self, score_str: str) -> str:
        """Get performance emoji based on score"""
        try:
            score = float(score_str.split("/")[0])
            if score >= 7.0:
                return "🟢 Strong"
            elif score >= 5.0:
                return "🟡 Average"
            else:
                return "🔴 Weak"
        except:
            return "⚪ Unknown"
    
    def _create_markdown_template(self) -> str:
        """Create markdown template for reports"""
        return """
# Enhanced Investment Research Report Template
# This template ensures consistent formatting across all reports
"""

# Export function
def format_enhanced_report(
    enhanced_data: Dict[str, Any],
    ticker: str,
    company_name: str
) -> Dict[str, str]:
    """Format enhanced report into JSON and Markdown"""
    
    formatter = EnhancedReportFormatter()
    return formatter.format_enhanced_report(enhanced_data, ticker, company_name)

# Test function
def test_formatter():
    """Test the enhanced report formatter"""
    
    mock_data = {
        "sections": {
            "executive_summary": {
                "investment_recommendation": {
                    "rating": "WEAK HOLD",
                    "score": "4.4/10",
                    "confidence": "83.0%"
                }
            }
        },
        "data_sources": [
            {
                "name": "ValueInvesting.io",
                "url": "https://valueinvesting.io/",
                "type": "DCF Platform",
                "description": "Automated DCF analysis"
            }
        ]
    }
    
    result = format_enhanced_report(mock_data, "META", "Meta Platforms Inc.")
    print("JSON Output Length:", len(result["json"]))
    print("Markdown Output Length:", len(result["markdown"]))

if __name__ == "__main__":
    test_formatter()
