#!/usr/bin/env python3
"""
Damodaran Story Development Agent

Implements Aswath Damodaran's Phase 1: Company Classification & Story Identification
Following the established agent pattern from the deepresearch system.
"""
import os
import logging
import yaml
from typing import Dict, Any, List, Optional
from datetime import datetime

from llama_index.core.agent import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

logger = logging.getLogger(__name__)


class DamodaranStoryEngine:
    """Core engine for Damodaran story development and validation"""
    
    def __init__(self, config_path: str = "configs/scoring.yaml"):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Damodaran configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}
    
    def classify_company(self, financial_data: Dict[str, Any], sector: str) -> Dict[str, Any]:
        """
        Step 1A: Initial Company Categorization
        Implements Damodaran's company classification framework
        """
        try:
            # Extract key financial metrics
            income_statements = financial_data.get("income_statements", [])
            balance_sheets = financial_data.get("balance_sheets", [])
            
            # Ensure data is in the correct format (list)
            if not isinstance(income_statements, list):
                income_statements = []
            if not isinstance(balance_sheets, list):
                balance_sheets = []
            
            if not income_statements:
                return self._default_classification(sector)
            
            recent_income = income_statements[0]
            recent_balance = balance_sheets[0] if balance_sheets else {}
            
            # Calculate key ratios
            revenue = recent_income.get("revenue", 0)
            net_income = recent_income.get("net_income", 0)
            total_assets = recent_balance.get("total_assets", 1)
            
            # Revenue growth calculation
            revenue_growth = self._calculate_revenue_growth(income_statements)
            
            # Margin analysis
            net_margin = net_income / revenue if revenue > 0 else 0
            
            # Asset turnover
            asset_turnover = revenue / total_assets if total_assets > 0 else 0
            
            # Life cycle classification
            life_cycle = self._classify_life_cycle(revenue_growth, net_margin)
            
            # Business model classification
            business_model = self._classify_business_model(sector, asset_turnover)
            
            # Value source assessment
            value_sources = self._assess_value_sources(life_cycle, business_model)
            
            # Market position assessment
            market_position = self._assess_market_position(sector, net_margin, revenue_growth)
            
            return {
                "life_cycle_stage": life_cycle,
                "business_model_type": business_model,
                "value_sources": value_sources,
                "market_position": market_position,
                "key_metrics": {
                    "revenue_growth": revenue_growth,
                    "net_margin": net_margin,
                    "asset_turnover": asset_turnover
                },
                "recommended_approach": self._recommend_valuation_approach(life_cycle, business_model),
                "classification_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Company classification failed: {e}")
            return self._default_classification(sector)
    
    def _classify_life_cycle(self, revenue_growth: float, net_margin: float) -> str:
        """Classify company life cycle stage"""
        stages = self.config.get("life_cycle_stages", {})
        
        if revenue_growth > 0.20 and net_margin < 0.15:
            return "young_growth"
        elif revenue_growth > 0.10 and 0.10 <= net_margin <= 0.25:
            return "growth"
        elif 0.02 <= revenue_growth <= 0.08 and net_margin >= 0.15:
            return "mature"
        elif revenue_growth < 0.02:
            return "decline"
        else:
            return "growth"  # Default
    
    def _classify_business_model(self, sector: str, asset_turnover: float) -> str:
        """Classify business model type"""
        asset_heavy_sectors = ["industrials", "energy", "utilities", "materials"]
        asset_light_sectors = ["technology", "software", "services"]
        
        if sector.lower() in asset_heavy_sectors or asset_turnover < 1.5:
            return "asset_heavy"
        elif sector.lower() in asset_light_sectors or asset_turnover > 2.0:
            return "asset_light"
        else:
            return "hybrid"
    
    def _assess_value_sources(self, life_cycle: str, business_model: str) -> Dict[str, float]:
        """Assess value source percentages"""
        if life_cycle == "young_growth":
            return {"assets_in_place": 0.2, "growth_assets": 0.8}
        elif life_cycle == "growth":
            return {"assets_in_place": 0.4, "growth_assets": 0.6}
        elif life_cycle == "mature":
            return {"assets_in_place": 0.7, "growth_assets": 0.3}
        else:  # decline
            return {"assets_in_place": 0.9, "growth_assets": 0.1}
    
    def _assess_market_position(self, sector: str, net_margin: float, revenue_growth: float) -> str:
        """Assess market position"""
        if net_margin > 0.20 and revenue_growth > 0.10:
            return "market_leader_with_pricing_power"
        elif net_margin > 0.15:
            return "strong_competitor_in_oligopoly"
        elif revenue_growth > 0.15:
            return "niche_player"
        else:
            return "commodity_player"
    
    def _recommend_valuation_approach(self, life_cycle: str, business_model: str) -> str:
        """Recommend appropriate valuation methodology"""
        stages = self.config.get("life_cycle_stages", {})
        return stages.get(life_cycle, {}).get("valuation_approach", "traditional_dcf")
    
    def _calculate_revenue_growth(self, income_statements: List[Dict]) -> float:
        """Calculate 3-year revenue CAGR"""
        try:
            if len(income_statements) < 2:
                return 0.05
            
            revenues = [stmt.get("revenue", 0) for stmt in income_statements[:4]]
            revenues = [r for r in revenues if r > 0]
            
            if len(revenues) < 2:
                return 0.05
            
            years = len(revenues) - 1
            cagr = ((revenues[0] / revenues[-1]) ** (1/years)) - 1
            return max(-0.50, min(1.00, cagr))
            
        except Exception:
            return 0.05
    
    def _default_classification(self, sector: str) -> Dict[str, Any]:
        """Default classification when analysis fails"""
        return {
            "life_cycle_stage": "mature",
            "business_model_type": "hybrid",
            "value_sources": {"assets_in_place": 0.6, "growth_assets": 0.4},
            "market_position": "strong_competitor_in_oligopoly",
            "recommended_approach": "traditional_dcf",
            "classification_date": datetime.now().isoformat()
        }


def create_story_tools(story_engine: DamodaranStoryEngine) -> List[FunctionTool]:
    """Create story development tools"""
    
    async def classify_company_tool(
        financial_data_json: str,
        sector: str = "default"
    ) -> str:
        """
        Classify company using Damodaran's framework.
        
        Args:
            financial_data_json: JSON string of financial data
            sector: Company sector
            
        Returns:
            JSON string of classification results
        """
        try:
            import json
            financial_data = json.loads(financial_data_json)
            classification = story_engine.classify_company(financial_data, sector)
            return json.dumps(classification, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def validate_story_tool(
        story_elements_json: str
    ) -> str:
        """
        Validate business story using Damodaran's 3-part test.
        
        Args:
            story_elements_json: JSON string of story elements
            
        Returns:
            JSON string of validation results
        """
        try:
            import json
            story_elements = json.loads(story_elements_json)
            
            # Apply 3-part test
            validation = {
                "possible": True,  # Can this story actually happen?
                "plausible": True,  # Is it reasonable given competitive dynamics?
                "probable": True,   # Is there evidence supporting this outcome?
                "overall_score": "strong",
                "recommendations": [],
                "validation_date": datetime.now().isoformat()
            }
            
            # Story consistency checks
            core_business = story_elements.get("core_business", "")
            competitive_advantage = story_elements.get("competitive_advantage", "")
            growth_drivers = story_elements.get("growth_drivers", [])
            
            if not core_business:
                validation["recommendations"].append("Define clearer core business identity")
                validation["probable"] = False
            
            if not competitive_advantage:
                validation["recommendations"].append("Identify sustainable competitive advantages")
                validation["plausible"] = False
            
            if len(growth_drivers) == 0:
                validation["recommendations"].append("Specify concrete growth drivers")
                validation["possible"] = False
            
            # Overall assessment
            if all([validation["possible"], validation["plausible"], validation["probable"]]):
                validation["overall_score"] = "strong"
            elif sum([validation["possible"], validation["plausible"], validation["probable"]]) >= 2:
                validation["overall_score"] = "moderate"
            else:
                validation["overall_score"] = "weak"
            
            return json.dumps(validation, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    return [
        FunctionTool.from_defaults(fn=classify_company_tool),
        FunctionTool.from_defaults(fn=validate_story_tool)
    ]


def create_damodaran_story_agent(openai_api_key: str) -> FunctionAgent:
    """Create Damodaran story development agent"""
    
    llm = OpenAI(model="gpt-4o", api_key=openai_api_key)
    story_engine = DamodaranStoryEngine()
    tools = create_story_tools(story_engine)
    
    system_prompt = """You are Aswath Damodaran's AI Research Assistant, specializing in story-driven company analysis and valuation.

Your expertise covers:
1. **Company Classification**: Life cycle stage, business model, value sources, market position
2. **Business Story Development**: Core business identity, competitive advantages, growth drivers
3. **Story Validation**: Applying the 3-part test (Possible, Plausible, Probable)

**CRITICAL FRAMEWORK - Always follow this structure:**

**Phase 1A: Company Classification**
For each company, determine:
- Life Cycle Stage: Young Growth / Growth / Mature / Decline
- Business Model: Asset-heavy / Asset-light / Hybrid  
- Value Sources: % from Assets in Place vs Growth Assets
- Market Position: Market leader / Strong competitor / Niche player / Commodity player

**Phase 1B: Story Development**
Develop a clear business story answering:
- Core Business: What business is this company really in?
- Competitive Advantage: What makes this company different/better?
- Growth Drivers: What will drive future growth?
- Key Risks: What could derail this story?
- End Game: What does this company look like in 10 years?

**Validation Process:**
Apply Damodaran's 3-part test:
- Possible: Can this story actually happen?
- Plausible: Is it reasonable given market/competitive dynamics?
- Probable: Is there evidence supporting this outcome?

**Response Format:**
Always provide structured JSON output with:
1. Company classification results
2. Business story elements
3. Story validation assessment
4. Recommended valuation approach
5. Key risks and opportunities

**Quality Standards:**
- Base analysis on financial data and sector dynamics
- Ensure story consistency across all elements
- Identify concrete, measurable growth drivers
- Acknowledge risks honestly and completely
- Connect story to appropriate valuation methodology

You have access to classification and validation tools. Use them to support your analysis."""

    return FunctionAgent(
        tools=tools,
        llm=llm,
        verbose=False,
        system_prompt=system_prompt
    )


async def develop_company_story(
    agent: FunctionAgent,
    ticker: str,
    company_name: str,
    financial_data: Dict[str, Any],
    sector: str = "default"
) -> Dict[str, Any]:
    """
    Develop comprehensive company story using Damodaran's framework
    
    Args:
        agent: Story development agent
        ticker: Stock ticker
        company_name: Company name
        financial_data: Financial data from FMP
        sector: Company sector
        
    Returns:
        Complete story analysis
    """
    try:
        import json
        
        # Prepare the story development prompt
        prompt = f"""
        Develop a comprehensive business story for {company_name} ({ticker}) using Damodaran's story-driven valuation framework.
        
        Company: {company_name}
        Ticker: {ticker}
        Sector: {sector}
        
        **REQUIRED ANALYSIS:**
        
        1. **Company Classification** - Use the classify_company_tool with the financial data
        2. **Story Development** - Create a complete business story covering:
           - Core business identity and competitive positioning
           - Sustainable competitive advantages and moats
           - Primary growth drivers (market expansion, share gains, new products)
           - Key risks that could derail the story
           - 10-year vision of the mature company
        
        3. **Story Validation** - Use the validate_story_tool to apply the 3-part test
        
        **Financial Data Available:**
        {json.dumps({k: len(v) if isinstance(v, list) else str(v)[:100] for k, v in financial_data.items()}, indent=2)}
        
        Provide a complete, actionable story that will guide the valuation process.
        """
        
        # Get the story analysis
        response = await agent.run(user_msg=prompt)
        
        # Parse the response to extract structured data
        story_analysis = {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "story_response": str(response),
            "analysis_date": datetime.now().isoformat(),
            "framework": "damodaran_story_driven"
        }
        
        # Try to extract JSON from the response if present
        try:
            response_text = str(response)
            if "{" in response_text and "}" in response_text:
                # Extract JSON parts from the response
                import re
                json_matches = re.findall(r'\{[^{}]*\}', response_text)
                if json_matches:
                    for match in json_matches:
                        try:
                            parsed = json.loads(match)
                            if "life_cycle_stage" in parsed:
                                story_analysis["classification"] = parsed
                            elif "possible" in parsed:
                                story_analysis["validation"] = parsed
                        except:
                            continue
        except:
            pass
        
        logger.info(f"Developed story for {ticker}")
        return story_analysis
        
    except Exception as e:
        logger.error(f"Story development failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "error": str(e),
            "analysis_date": datetime.now().isoformat()
        }


# Standalone function for testing
def get_damodaran_story_agent() -> FunctionAgent:
    """Get the Damodaran story agent"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")
    
    return create_damodaran_story_agent(openai_api_key)


if __name__ == "__main__":
    # Test the story agent
    import asyncio
    import json
    
    async def test_story_agent():
        agent = get_damodaran_story_agent()
        
        # Mock financial data
        test_data = {
            "income_statements": [
                {"revenue": 100000000, "net_income": 15000000},
                {"revenue": 90000000, "net_income": 12000000},
                {"revenue": 80000000, "net_income": 10000000}
            ],
            "balance_sheets": [
                {"total_assets": 50000000}
            ]
        }
        
        result = await develop_company_story(
            agent, "TEST", "Test Company", test_data, "technology"
        )
        
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_story_agent())
