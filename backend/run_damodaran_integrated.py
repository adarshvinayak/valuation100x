#!/usr/bin/env python3
"""
Damodaran-Enhanced Stock Research System

This script integrates Aswath Damodaran's story-driven valuation methodology 
into the existing working deepresearch workflow.
"""
import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Setup logging with UTF-8 encoding - Lambda aware
lambda_mode = os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None

handlers = [logging.StreamHandler(sys.stdout)]

# Only add file handler if not in Lambda (file system is read-only in Lambda)
if not lambda_mode:
    try:
        # Create logs directory if it doesn't exist (only outside Lambda)
        os.makedirs('logs', exist_ok=True)
        handlers.append(logging.FileHandler('logs/damodaran_integrated.log', encoding='utf-8'))
    except Exception as e:
        print(f"Warning: Could not create log file: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

# Import the existing working workflow
from workflow import run_stock_analysis

# Import Damodaran components
from agents.damodaran_story_agent import get_damodaran_story_agent, develop_company_story
from tools.damodaran_industry_analysis import DamodaranIndustryAnalyzer
from synthesis.damodaran_valuation import damodaran_ensemble_valuation


async def run_enhanced_analysis(ticker: str, company_name: str = None) -> Dict[str, Any]:
    """
    Run enhanced analysis that combines the working workflow with Damodaran methodology
    
    Args:
        ticker: Stock ticker
        company_name: Company name (optional)
        
    Returns:
        Enhanced analysis results with Damodaran components
    """
    logger.info(f"Starting Damodaran-enhanced analysis for {ticker}")
    
    try:
        # Step 1: Run the existing working analysis first
        logger.info("Running base comprehensive analysis...")
        base_results = await run_stock_analysis(ticker, company_name)
        
        if not base_results or "error" in base_results:
            logger.error(f"Base analysis failed for {ticker}")
            return base_results
        
        # Step 2: Extract financial data from base results for Damodaran analysis
        financial_data = base_results.get("financial_data", {})
        
        # If no financial data in base results, fetch it directly
        if not financial_data:
            logger.info("Fetching financial data directly for Damodaran analysis")
            try:
                from tools.fmp import get_financials_fmp
                financial_data = await get_financials_fmp(ticker)
                if not financial_data:
                    logger.warning("Could not fetch financial data for Damodaran analysis")
                    # Continue with base results but note the limitation
                    enhanced_results = dict(base_results)
                    enhanced_results["damodaran_analysis"] = {
                        "error": "Financial data not available for Damodaran analysis",
                        "limitation": "Enhanced analysis requires financial data"
                    }
                    return enhanced_results
            except Exception as e:
                logger.warning(f"Failed to fetch financial data: {e}")
                enhanced_results = dict(base_results)
                enhanced_results["damodaran_analysis"] = {
                    "error": f"Financial data fetch failed: {str(e)}",
                    "limitation": "Enhanced analysis requires financial data"
                }
                return enhanced_results
        
        # Step 3: Develop Damodaran story
        logger.info("Developing Damodaran business story...")
        story_context = {}
        try:
            story_agent = get_damodaran_story_agent()
            story_context = await develop_company_story(
                story_agent, ticker, company_name or ticker, financial_data, "technology"
            )
        except Exception as e:
            logger.warning(f"Story development failed: {e}")
            story_context = {"error": str(e)}
        
        # Step 4: Conduct industry analysis
        logger.info("Conducting Damodaran industry analysis...")
        sector_analysis = {}
        try:
            analyzer = DamodaranIndustryAnalyzer()
            company_data = {"financial_data": financial_data, "story_context": story_context}
            sector_analysis = analyzer.comprehensive_sector_analysis("technology", company_data)
        except Exception as e:
            logger.warning(f"Industry analysis failed: {e}")
            sector_analysis = {"error": str(e)}
        
        # Step 5: Perform Damodaran valuation
        logger.info("Performing Damodaran valuation...")
        damodaran_valuation = {}
        try:
            current_price = base_results.get("current_price", 100.0)
            damodaran_valuation = damodaran_ensemble_valuation(
                story_context, financial_data, ticker, current_price, "technology"
            )
        except Exception as e:
            logger.warning(f"Damodaran valuation failed: {e}")
            damodaran_valuation = {"error": str(e)}
        
        # Step 6: Enhance the base results with Damodaran components
        enhanced_results = dict(base_results)
        enhanced_results.update({
            "damodaran_analysis": {
                "story_context": story_context,
                "sector_analysis": sector_analysis,
                "damodaran_valuation": damodaran_valuation,
                "methodology": "integrated_damodaran_enhancement",
                "analysis_date": datetime.now().isoformat()
            },
            "enhanced_methodology": True,
            "framework": "damodaran_integrated"
        })
        
        # Step 7: Generate enhanced report
        enhanced_report = generate_enhanced_report(enhanced_results)
        enhanced_results["enhanced_report"] = enhanced_report
        
        logger.info(f"Enhanced analysis complete for {ticker}")
        return enhanced_results
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "error": str(e),
            "analysis_date": datetime.now().isoformat()
        }


def generate_enhanced_report(results: Dict[str, Any]) -> str:
    """Generate an enhanced report that includes Damodaran analysis"""
    
    ticker = results.get("ticker", "UNKNOWN")
    company_name = results.get("company_name", ticker)
    
    # Base analysis components
    investment_score = results.get("investment_score", 5.0)
    fair_value = results.get("fair_value", 0.0)
    current_price = results.get("current_price", 0.0)
    
    # Damodaran components
    damodaran_analysis = results.get("damodaran_analysis", {})
    story_context = damodaran_analysis.get("story_context", {})
    sector_analysis = damodaran_analysis.get("sector_analysis", {})
    damodaran_valuation = damodaran_analysis.get("damodaran_valuation", {})
    
    report = f"""
# 📊 Enhanced Investment Analysis Report with Damodaran Framework

## {company_name} ({ticker})

---

### 🎯 **Executive Summary**

| **Metric** | **Value** |
|------------|-----------|
| 🎯 **Investment Score** | **{investment_score:.1f}/10** |
| 💰 **Fair Value** | **${fair_value:.2f}** |
| 📊 **Current Price** | **${current_price:.2f}** |
| 📈 **Framework** | **Damodaran-Enhanced** |

---

## 🎭 **Damodaran Business Story Analysis**

"""
    
    # Add story analysis if available
    if story_context and "error" not in story_context:
        classification = story_context.get("classification", {})
        report += f"""
### **Company Classification**
- **Life Cycle Stage**: {classification.get('life_cycle_stage', 'Unknown')}
- **Business Model**: {classification.get('business_model_type', 'Unknown')}
- **Market Position**: {classification.get('market_position', 'Unknown')}

### **Business Story Elements**
- **Core Business**: {story_context.get('core_business', 'Not specified')}
- **Competitive Advantage**: {story_context.get('competitive_advantage', 'Not specified')}
- **Growth Drivers**: {', '.join(story_context.get('growth_drivers', ['Not specified']))}
- **Key Risks**: {', '.join(story_context.get('key_risks', ['Not specified']))}
"""
    else:
        report += "\n*Story analysis unavailable due to technical issues*\n"
    
    # Add industry analysis if available
    if sector_analysis and "error" not in sector_analysis:
        report += f"""
---

## 🏭 **Industry Analysis**

### **Value Chain Position**
- **Sector**: {sector_analysis.get('value_chain', {}).get('sector', 'Unknown')}
- **Competitive Dynamics**: Strong position in sector value chain

### **Risk Assessment**
- **Overall Risk Level**: {sector_analysis.get('risk_register', {}).get('overall_risk_assessment', {}).get('overall_risk_level', 'Medium')}
- **Primary Risks**: {', '.join(sector_analysis.get('risk_register', {}).get('overall_risk_assessment', {}).get('primary_risks', ['Standard risks']))}
"""
    else:
        report += "\n*Industry analysis unavailable due to technical issues*\n"
    
    # Add valuation analysis if available
    if damodaran_valuation and "error" not in damodaran_valuation:
        investment_decision = damodaran_valuation.get("investment_decision", {})
        report += f"""
---

## 💰 **Damodaran Valuation Framework**

### **Investment Decision**
- **Recommendation**: {investment_decision.get('recommendation', 'HOLD')}
- **Conviction Level**: {investment_decision.get('conviction_level', 'Medium')}
- **Margin of Safety**: {investment_decision.get('margin_of_safety', 0)*100:.1f}%

### **Price Targets**
- **Bull Case**: ${investment_decision.get('price_targets', {}).get('bull_case', current_price):.2f}
- **Base Case**: ${investment_decision.get('price_targets', {}).get('base_case', current_price):.2f}
- **Bear Case**: ${investment_decision.get('price_targets', {}).get('bear_case', current_price):.2f}
"""
    else:
        report += "\n*Damodaran valuation unavailable due to technical issues*\n"
    
    # Add base analysis summary
    base_report = results.get("report", "")
    if base_report:
        report += f"""
---

## 📈 **Traditional Analysis Summary**

{base_report[:1000]}...

*[Full traditional analysis available in base results]*
"""
    
    report += f"""
---

## 🔗 **Methodology Integration**

This analysis combines:
1. **Traditional Quantitative Analysis**: Financial metrics, ratios, and technical indicators
2. **Damodaran Story-Driven Framework**: Business narrative, life cycle analysis, and story validation
3. **Industry Analysis**: Sector-specific risks, value chain positioning, and competitive dynamics
4. **Enhanced Valuation**: Story-consistent DCF modeling with scenario analysis

**Analysis completed at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report


async def analyze_stock_enhanced(ticker: str, company_name: str = None) -> Dict[str, Any]:
    """
    Main function to analyze a stock with Damodaran enhancements
    
    Args:
        ticker: Stock ticker
        company_name: Company name (optional)
        
    Returns:
        Complete enhanced analysis results
    """
    try:
        # Run enhanced analysis
        results = await run_enhanced_analysis(ticker, company_name)
        
        # Save results
        output_dir = Path("data/outputs") / ticker
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_file = output_dir / f"{ticker}_damodaran_enhanced.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        # Save enhanced report
        if "enhanced_report" in results:
            report_file = output_dir / f"{ticker}_damodaran_enhanced_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(results["enhanced_report"])
        
        logger.info(f"Enhanced analysis saved to {output_dir}")
        
        # Print summary
        print_enhanced_summary(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


def print_enhanced_summary(results: Dict[str, Any]):
    """Print a formatted summary of the enhanced analysis"""
    
    ticker = results.get("ticker", "UNKNOWN")
    
    print(f"\n{'='*60}")
    print(f"📊 DAMODARAN-ENHANCED ANALYSIS: {ticker}")
    print(f"{'='*60}")
    
    # Base metrics
    investment_score = results.get("investment_score", 5.0)
    fair_value = results.get("fair_value", 0.0)
    current_price = results.get("current_price", 0.0)
    
    print(f"\n📈 INVESTMENT METRICS:")
    print(f"   Investment Score: {investment_score:.1f}/10")
    print(f"   Fair Value: ${fair_value:.2f}")
    print(f"   Current Price: ${current_price:.2f}")
    if current_price > 0:
        gap = (fair_value / current_price - 1) * 100
        print(f"   Price Gap: {gap:.1f}%")
    
    # Damodaran components
    damodaran_analysis = results.get("damodaran_analysis", {})
    
    story_context = damodaran_analysis.get("story_context", {})
    if story_context and "error" not in story_context:
        classification = story_context.get("classification", {})
        print(f"\n🎭 BUSINESS STORY:")
        print(f"   Life Cycle: {classification.get('life_cycle_stage', 'Unknown')}")
        print(f"   Business Model: {classification.get('business_model_type', 'Unknown')}")
        print(f"   Core Business: {story_context.get('core_business', 'Not specified')[:80]}...")
    
    damodaran_valuation = damodaran_analysis.get("damodaran_valuation", {})
    if damodaran_valuation and "error" not in damodaran_valuation:
        investment_decision = damodaran_valuation.get("investment_decision", {})
        print(f"\n🎯 DAMODARAN DECISION:")
        print(f"   Recommendation: {investment_decision.get('recommendation', 'HOLD')}")
        print(f"   Conviction: {investment_decision.get('conviction_level', 'Medium')}")
        print(f"   Margin of Safety: {investment_decision.get('margin_of_safety', 0)*100:.1f}%")
    
    print(f"\n✅ ENHANCED ANALYSIS COMPLETE")
    print(f"   Framework: Traditional + Damodaran Story-Driven")
    print(f"   Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")


async def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description="Run Damodaran-Enhanced Stock Analysis")
    parser.add_argument("--ticker", "-t", required=True, help="Stock ticker to analyze")
    parser.add_argument("--company", "-c", help="Company name (optional)")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directories
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    # Run enhanced analysis
    results = await analyze_stock_enhanced(args.ticker.upper(), args.company)
    
    # Save to custom output file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results also saved to {args.output}")
    
    return results


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) == 1:
        print("📊 Damodaran-Enhanced Stock Research System")
        print("="*50)
        print("Example usage:")
        print("python run_damodaran_integrated.py --ticker META --company 'Meta Platforms Inc.'")
        print("python run_damodaran_integrated.py --ticker AAPL --verbose")
        print("")
        print("Features:")
        print("✅ Full traditional analysis (working)")
        print("✅ Damodaran story development")
        print("✅ Industry analysis framework") 
        print("✅ Enhanced valuation methodology")
        print("✅ Integrated comprehensive reporting")
        print("")
        sys.exit(1)
    
    asyncio.run(main())
