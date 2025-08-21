#!/usr/bin/env python3
"""
Damodaran Industry Analysis Tool

Implements Phase 2: Sector-Specific Analysis including value chain mapping,
risk registers, and accounting adjustments.
"""
import logging
import yaml
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DamodaranIndustryAnalyzer:
    """Comprehensive industry analysis following Damodaran's framework"""
    
    def __init__(self, config_path: str = "configs/scoring.yaml"):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load industry analysis configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}
    
    def value_chain_mapping(self, sector: str, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map the complete value chain and assess company positioning
        
        Args:
            sector: Company sector
            company_data: Company financial and operational data
            
        Returns:
            Value chain analysis
        """
        try:
            # Get sector-specific value chain structure
            value_chains = {
                "technology": {
                    "stages": ["R&D", "Product Development", "Manufacturing", "Distribution", "Support"],
                    "value_capture_points": ["IP Creation", "Platform Network", "Data Monetization"],
                    "cost_drivers": ["R&D", "Talent", "Infrastructure"],
                    "competitive_dynamics": "Winner-take-most with network effects"
                },
                "consumer_staples": {
                    "stages": ["Raw Materials", "Manufacturing", "Brand Marketing", "Distribution", "Retail"],
                    "value_capture_points": ["Brand Premium", "Distribution Control", "Scale Efficiency"],
                    "cost_drivers": ["Raw Materials", "Manufacturing", "Marketing"],
                    "competitive_dynamics": "Oligopoly with brand differentiation"
                },
                "healthcare": {
                    "stages": ["Research", "Development", "Clinical Trials", "Regulatory", "Commercialization"],
                    "value_capture_points": ["Patent Protection", "Regulatory Moats", "Distribution"],
                    "cost_drivers": ["R&D", "Clinical Trials", "Manufacturing"],
                    "competitive_dynamics": "Patent-protected monopolies with cliff risks"
                },
                "financials": {
                    "stages": ["Capital Formation", "Risk Assessment", "Credit Decision", "Monitoring", "Recovery"],
                    "value_capture_points": ["Interest Spread", "Fee Income", "Risk Management"],
                    "cost_drivers": ["Credit Losses", "Operations", "Regulatory Compliance"],
                    "competitive_dynamics": "Regulated oligopoly with capital barriers"
                },
                "energy": {
                    "stages": ["Exploration", "Extraction", "Refining", "Distribution", "Retail"],
                    "value_capture_points": ["Reserve Access", "Cost Efficiency", "Integration"],
                    "cost_drivers": ["Finding Costs", "Extraction", "Environmental"],
                    "competitive_dynamics": "Commodity with operational differentiation"
                },
                "industrials": {
                    "stages": ["Raw Materials", "Manufacturing", "Assembly", "Distribution", "Service"],
                    "value_capture_points": ["Operational Excellence", "Technology", "Service"],
                    "cost_drivers": ["Materials", "Labor", "Logistics"],
                    "competitive_dynamics": "Cyclical with scale advantages"
                }
            }
            
            chain_data = value_chains.get(sector.lower(), value_chains["industrials"])
            
            # Assess company position in value chain
            position_assessment = self._assess_value_chain_position(company_data, chain_data)
            
            return {
                "sector": sector,
                "value_chain_structure": chain_data,
                "company_position": position_assessment,
                "bargaining_power_analysis": self._analyze_bargaining_power(sector, company_data),
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Value chain mapping failed: {e}")
            return {"error": str(e)}
    
    def sector_risk_register(self, sector: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive sector risk analysis with quantification
        
        Args:
            sector: Company sector
            financial_data: Company financial data
            
        Returns:
            Detailed risk assessment
        """
        try:
            # Get sector-specific risks from configuration
            sector_config = self.config.get("damodaran_sectors", {}).get(sector.lower(), {})
            risk_factors = sector_config.get("risk_factors", [])
            
            # Build comprehensive risk register
            risk_register = {
                "demand_drivers": self._analyze_demand_risks(sector, financial_data),
                "pricing_power": self._analyze_pricing_risks(sector, financial_data),
                "cost_structure": self._analyze_cost_risks(sector, financial_data),
                "regulatory_environment": self._analyze_regulatory_risks(sector),
                "technology_disruption": self._analyze_tech_disruption_risks(sector),
                "esg_factors": self._analyze_esg_risks(sector),
                "cyclicality": self._analyze_cyclical_risks(sector, financial_data),
                "sector_specific_risks": risk_factors
            }
            
            # Calculate overall risk score
            risk_register["overall_risk_assessment"] = self._calculate_risk_score(risk_register)
            risk_register["analysis_date"] = datetime.now().isoformat()
            
            return risk_register
            
        except Exception as e:
            logger.error(f"Risk register analysis failed: {e}")
            return {"error": str(e)}
    
    def accounting_quirks_analysis(self, sector: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify sector-specific accounting distortions
        
        Args:
            sector: Company sector
            financial_data: Company financial statements
            
        Returns:
            Accounting adjustment recommendations
        """
        try:
            # Get sector-specific accounting issues
            sector_config = self.config.get("damodaran_sectors", {}).get(sector.lower(), {})
            accounting_adjustments = sector_config.get("accounting_adjustments", [])
            
            adjustments = {
                "sector": sector,
                "identified_quirks": accounting_adjustments,
                "recommended_adjustments": [],
                "impact_assessment": {}
            }
            
            # Analyze specific accounting issues by sector
            if sector.lower() == "technology":
                adjustments.update(self._analyze_tech_accounting(financial_data))
            elif sector.lower() == "healthcare":
                adjustments.update(self._analyze_healthcare_accounting(financial_data))
            elif sector.lower() == "energy":
                adjustments.update(self._analyze_energy_accounting(financial_data))
            elif sector.lower() == "financials":
                adjustments.update(self._analyze_financial_accounting(financial_data))
            
            adjustments["analysis_date"] = datetime.now().isoformat()
            return adjustments
            
        except Exception as e:
            logger.error(f"Accounting analysis failed: {e}")
            return {"error": str(e)}
    
    def comprehensive_sector_analysis(self, sector: str, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete Damodaran Phase 2 analysis
        
        Args:
            sector: Company sector
            company_data: Complete company data
            
        Returns:
            Comprehensive sector analysis
        """
        try:
            financial_data = company_data.get("financial_data", {})
            
            analysis = {
                "value_chain": self.value_chain_mapping(sector, company_data),
                "risk_register": self.sector_risk_register(sector, financial_data),
                "accounting_quirks": self.accounting_quirks_analysis(sector, financial_data),
                "sector_metrics": self._get_sector_key_metrics(sector, financial_data),
                "competitive_landscape": self._analyze_competitive_landscape(sector),
                "analysis_date": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Comprehensive sector analysis failed: {e}")
            return {"error": str(e)}
    
    # Helper methods for detailed analysis
    
    def _assess_value_chain_position(self, company_data: Dict, chain_data: Dict) -> Dict[str, Any]:
        """Assess company's position in the value chain"""
        return {
            "primary_stage": "Unknown",  # Would need more company-specific data
            "integration_level": "Moderate",
            "differentiation_source": chain_data.get("value_capture_points", ["Unknown"])[0],
            "competitive_strengths": ["Scale", "Brand", "Technology"]  # Generic for now
        }
    
    def _analyze_bargaining_power(self, sector: str, company_data: Dict) -> Dict[str, Any]:
        """Analyze bargaining power across the value chain"""
        return {
            "supplier_power": "Medium",
            "customer_power": "Medium", 
            "competitive_intensity": "High",
            "threat_of_substitutes": "Medium",
            "barriers_to_entry": "High" if sector.lower() in ["technology", "healthcare"] else "Medium"
        }
    
    def _analyze_demand_risks(self, sector: str, financial_data: Dict) -> Dict[str, Any]:
        """Analyze demand-side risks"""
        cyclical_sectors = ["industrials", "energy", "materials", "consumer_discretionary"]
        
        return {
            "economic_sensitivity": "High" if sector.lower() in cyclical_sectors else "Medium",
            "demographic_trends": "Favorable" if sector.lower() in ["healthcare", "technology"] else "Neutral",
            "regulatory_changes": "High" if sector.lower() in ["healthcare", "financials", "energy"] else "Medium"
        }
    
    def _analyze_pricing_risks(self, sector: str, financial_data: Dict) -> Dict[str, Any]:
        """Analyze pricing power and risks"""
        return {
            "pricing_power": "High" if sector.lower() in ["technology", "healthcare"] else "Medium",
            "cost_pass_through": "Medium",
            "competitive_pricing_pressure": "High"
        }
    
    def _analyze_cost_risks(self, sector: str, financial_data: Dict) -> Dict[str, Any]:
        """Analyze cost structure risks"""
        return {
            "input_cost_volatility": "High" if sector.lower() in ["energy", "materials"] else "Medium",
            "labor_cost_pressure": "Medium",
            "operating_leverage": "High" if sector.lower() in ["technology", "industrials"] else "Medium"
        }
    
    def _analyze_regulatory_risks(self, sector: str) -> Dict[str, Any]:
        """Analyze regulatory environment"""
        high_regulation = ["healthcare", "financials", "energy", "utilities"]
        
        return {
            "regulatory_intensity": "High" if sector.lower() in high_regulation else "Medium",
            "pending_regulations": "Multiple" if sector.lower() in ["technology", "healthcare"] else "Few",
            "compliance_costs": "Rising"
        }
    
    def _analyze_tech_disruption_risks(self, sector: str) -> Dict[str, Any]:
        """Analyze technology disruption risks"""
        disruption_risk = {
            "technology": "High",
            "financials": "High", 
            "healthcare": "Medium",
            "consumer_discretionary": "High",
            "industrials": "Medium",
            "energy": "Medium",
            "consumer_staples": "Low"
        }
        
        return {
            "disruption_risk": disruption_risk.get(sector.lower(), "Medium"),
            "automation_threat": "High" if sector.lower() in ["industrials", "financials"] else "Medium",
            "digital_transformation": "Critical"
        }
    
    def _analyze_esg_risks(self, sector: str) -> Dict[str, Any]:
        """Analyze ESG factors"""
        return {
            "environmental_risk": "High" if sector.lower() in ["energy", "materials"] else "Medium",
            "social_license": "Critical" if sector.lower() in ["healthcare", "financials"] else "Important",
            "governance_focus": "High"
        }
    
    def _analyze_cyclical_risks(self, sector: str, financial_data: Dict) -> Dict[str, Any]:
        """Analyze cyclical exposure"""
        cyclical_sectors = ["industrials", "energy", "materials", "consumer_discretionary", "financials"]
        
        return {
            "cyclical_exposure": "High" if sector.lower() in cyclical_sectors else "Low",
            "current_cycle_position": "Mid-cycle",  # Would need macro analysis
            "volatility_characteristics": "High" if sector.lower() in ["energy", "materials"] else "Medium"
        }
    
    def _calculate_risk_score(self, risk_register: Dict) -> Dict[str, Any]:
        """Calculate overall risk assessment"""
        return {
            "overall_risk_level": "Medium",  # Simplified calculation
            "primary_risks": ["Regulatory", "Competition", "Technology"],
            "risk_mitigation_factors": ["Scale", "Diversification", "Management"],
            "monitoring_priorities": ["Market Share", "Margins", "Regulatory Changes"]
        }
    
    def _get_sector_key_metrics(self, sector: str, financial_data: Dict) -> Dict[str, Any]:
        """Get sector-specific key metrics"""
        sector_config = self.config.get("damodaran_sectors", {}).get(sector.lower(), {})
        key_metrics = sector_config.get("key_metrics", [])
        
        return {
            "relevant_metrics": key_metrics,
            "benchmarking_required": True,
            "tracking_frequency": "Quarterly"
        }
    
    def _analyze_competitive_landscape(self, sector: str) -> Dict[str, Any]:
        """Analyze competitive dynamics"""
        return {
            "market_structure": "Oligopoly" if sector.lower() in ["technology", "healthcare"] else "Competitive",
            "competitive_intensity": "High",
            "differentiation_potential": "High" if sector.lower() in ["technology", "healthcare"] else "Medium",
            "scale_advantages": "Significant"
        }
    
    def _analyze_tech_accounting(self, financial_data: Dict) -> Dict[str, Any]:
        """Technology sector accounting analysis"""
        return {
            "rd_treatment": "Expense vs capitalize analysis needed",
            "stock_compensation": "Add back to cash costs", 
            "deferred_revenue": "Adjust for subscription model",
            "intangible_assets": "Assess fair value vs book value"
        }
    
    def _analyze_healthcare_accounting(self, financial_data: Dict) -> Dict[str, Any]:
        """Healthcare sector accounting analysis"""
        return {
            "rd_success_probability": "Risk-adjust pipeline value",
            "milestone_payments": "Revenue recognition timing",
            "inventory_obsolescence": "Assess reserve adequacy",
            "patent_amortization": "Cliff risk analysis"
        }
    
    def _analyze_energy_accounting(self, financial_data: Dict) -> Dict[str, Any]:
        """Energy sector accounting analysis"""
        return {
            "depletion_methods": "DD&A rate sustainability",
            "exploration_costs": "Success rate analysis", 
            "derivatives": "Mark-to-market volatility",
            "asset_retirement": "Environmental liability adequacy"
        }
    
    def _analyze_financial_accounting(self, financial_data: Dict) -> Dict[str, Any]:
        """Financial sector accounting analysis"""
        return {
            "loan_loss_provisions": "Cycle-adjusted adequacy",
            "fair_value_accounting": "Mark-to-market impact",
            "off_balance_sheet": "Hidden leverage assessment",
            "regulatory_capital": "Basel III compliance"
        }


# Tool creation functions for integration with agents
async def get_industry_analysis(sector: str, company_data_json: str) -> str:
    """
    Get comprehensive industry analysis
    
    Args:
        sector: Company sector
        company_data_json: JSON string of company data
        
    Returns:
        JSON string of industry analysis
    """
    try:
        company_data = json.loads(company_data_json)
        analyzer = DamodaranIndustryAnalyzer()
        analysis = analyzer.comprehensive_sector_analysis(sector, company_data)
        return json.dumps(analysis, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def get_value_chain_analysis(sector: str, company_data_json: str) -> str:
    """
    Get value chain mapping analysis
    
    Args:
        sector: Company sector
        company_data_json: JSON string of company data
        
    Returns:
        JSON string of value chain analysis
    """
    try:
        company_data = json.loads(company_data_json)
        analyzer = DamodaranIndustryAnalyzer()
        analysis = analyzer.value_chain_mapping(sector, company_data)
        return json.dumps(analysis, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def get_sector_risk_register(sector: str, financial_data_json: str) -> str:
    """
    Get comprehensive sector risk register
    
    Args:
        sector: Company sector
        financial_data_json: JSON string of financial data
        
    Returns:
        JSON string of risk analysis
    """
    try:
        financial_data = json.loads(financial_data_json)
        analyzer = DamodaranIndustryAnalyzer()
        analysis = analyzer.sector_risk_register(sector, financial_data)
        return json.dumps(analysis, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    # Test the industry analyzer
    import asyncio
    
    async def test_analyzer():
        analyzer = DamodaranIndustryAnalyzer()
        
        test_data = {
            "financial_data": {
                "income_statements": [{"revenue": 100000000}],
                "balance_sheets": [{"total_assets": 50000000}]
            }
        }
        
        result = analyzer.comprehensive_sector_analysis("technology", test_data)
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_analyzer())
