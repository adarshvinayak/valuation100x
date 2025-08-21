"""
Ensemble Valuation Module

Implements multiple valuation methodologies including DCF scenarios,
reverse DCF analysis, multiples cross-check, and residual income modeling.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import yaml
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ValuationEngine:
    """Comprehensive valuation engine with multiple methodologies"""
    
    def __init__(self, config_path: str = "configs/scoring.yaml"):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load valuation configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration if file not available"""
        return {
            "sector_wacc": {"default": [8.0, 11.0]},
            "terminal_growth": {"conservative": 2.0, "base": 2.5, "optimistic": 3.0},
            "dcf_scenarios": {"bear": 0.25, "base": 0.50, "bull": 0.25}
        }
    
    def _determine_sector(self, ticker: str, sector_hint: str = None) -> str:
        """Determine company sector for WACC estimation"""
        if sector_hint:
            sector_hint = sector_hint.lower()
            if "tech" in sector_hint:
                return "technology"
            elif "health" in sector_hint or "pharma" in sector_hint:
                return "healthcare"
            elif "financial" in sector_hint or "bank" in sector_hint:
                return "financials"
            elif "energy" in sector_hint or "oil" in sector_hint:
                return "energy"
            elif "consumer" in sector_hint:
                return "consumer_staples"
            elif "industrial" in sector_hint:
                return "industrials"
        
        # Simple ticker-based mapping for demo
        ticker_sectors = {
            "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
            "AMZN": "consumer_discretionary", "KO": "consumer_staples",
            "PG": "consumer_staples", "UNH": "healthcare", "JNJ": "healthcare",
            "JPM": "financials", "GS": "financials", "XOM": "energy",
            "CVX": "energy", "CAT": "industrials"
        }
        
        return ticker_sectors.get(ticker.upper(), "default")
    
    def _get_wacc_range(self, sector: str) -> Tuple[float, float]:
        """Get WACC range for sector"""
        wacc_ranges = self.config.get("sector_wacc", {})
        wacc_range = wacc_ranges.get(sector, wacc_ranges.get("default", [8.0, 11.0]))
        return wacc_range[0] / 100, wacc_range[1] / 100
    
    def dcf_valuation(self, 
                     financial_data: Dict[str, Any],
                     ticker: str,
                     current_price: float,
                     sector: str = None) -> Dict[str, Any]:
        """
        Perform DCF valuation with multiple scenarios
        
        Args:
            financial_data: Company financial data from FMP
            ticker: Stock ticker
            current_price: Current stock price
            sector: Company sector for WACC estimation
            
        Returns:
            DCF valuation results with scenarios
        """
        try:
            sector = sector or self._determine_sector(ticker)
            
            # Extract key financial metrics
            income_statements = financial_data.get("income_statements", [])
            balance_sheets = financial_data.get("balance_sheets", [])
            cash_flows = financial_data.get("cash_flows", [])
            
            if not income_statements:
                raise ValueError("No income statement data available")
            
            # Get recent financial data
            recent_income = income_statements[0]
            recent_balance = balance_sheets[0] if balance_sheets else {}
            recent_cash = cash_flows[0] if cash_flows else {}
            
            # Base metrics
            revenue = recent_income.get("revenue", 0)
            ebit = recent_income.get("operating_income", 0)
            net_income = recent_income.get("net_income", 0)
            shares_outstanding = recent_income.get("shares_outstanding", 1)
            
            # Calculate historical growth rates
            revenue_growth = self._calculate_growth_rate(income_statements, "revenue")
            ebit_growth = self._calculate_growth_rate(income_statements, "operating_income")
            
            # WACC estimation
            wacc_low, wacc_high = self._get_wacc_range(sector)
            
            # Terminal growth rates
            terminal_rates = self.config.get("terminal_growth", {})
            
            # DCF scenarios
            scenarios = {}
            scenario_weights = self.config.get("dcf_scenarios", {})
            
            # Bear case
            bear_params = {
                "revenue_growth": max(0.02, revenue_growth * 0.7),  # 70% of historical
                "ebit_margin": max(0.05, ebit / revenue * 0.9) if revenue > 0 else 0.05,
                "wacc": wacc_high,
                "terminal_growth": terminal_rates.get("conservative", 0.02),
                "years": 5
            }
            scenarios["bear"] = self._dcf_scenario(bear_params, revenue, shares_outstanding)
            
            # Base case
            base_params = {
                "revenue_growth": max(0.03, revenue_growth),
                "ebit_margin": ebit / revenue if revenue > 0 else 0.10,
                "wacc": (wacc_low + wacc_high) / 2,
                "terminal_growth": terminal_rates.get("base", 0.025),
                "years": 5
            }
            scenarios["base"] = self._dcf_scenario(base_params, revenue, shares_outstanding)
            
            # Bull case
            bull_params = {
                "revenue_growth": max(0.05, revenue_growth * 1.3),  # 130% of historical
                "ebit_margin": min(0.40, ebit / revenue * 1.1) if revenue > 0 else 0.15,
                "wacc": wacc_low,
                "terminal_growth": terminal_rates.get("optimistic", 0.03),
                "years": 5
            }
            scenarios["bull"] = self._dcf_scenario(bull_params, revenue, shares_outstanding)
            
            # Calculate weighted average fair value
            weighted_fv = 0
            scenario_values = []
            
            for scenario, weight in scenario_weights.items():
                if scenario in scenarios:
                    fv = scenarios[scenario]["fair_value_per_share"]
                    weighted_fv += fv * weight
                    scenario_values.append(fv)
            
            # Calculate statistics
            median_fv = np.median(scenario_values) if scenario_values else weighted_fv
            
            # Probability of undervaluation (based on scenarios above current price)
            above_current = sum(1 for fv in scenario_values if fv > current_price)
            p_undervalued = above_current / len(scenario_values) if scenario_values else 0.5
            
            # Price gap
            gap = (median_fv / current_price - 1) if current_price > 0 else 0
            
            return {
                "ticker": ticker,
                "current_price": current_price,
                "scenarios": scenarios,
                "weighted_fair_value": weighted_fv,
                "median_fv": median_fv,
                "scenario_values": scenario_values,
                "p_underv": p_undervalued,
                "gap": gap,
                "sector": sector,
                "wacc_range": [wacc_low, wacc_high],
                "base_metrics": {
                    "revenue": revenue,
                    "ebit": ebit,
                    "net_income": net_income,
                    "shares_outstanding": shares_outstanding,
                    "revenue_growth": revenue_growth,
                    "ebit_margin": ebit / revenue if revenue > 0 else 0
                },
                "valuation_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"DCF valuation failed for {ticker}: {e}")
            return self._get_default_valuation(ticker, current_price)
    
    def _dcf_scenario(self, params: Dict[str, Any], base_revenue: float, shares: float) -> Dict[str, Any]:
        """Calculate DCF for a single scenario"""
        try:
            years = params["years"]
            revenue_growth = params["revenue_growth"]
            ebit_margin = params["ebit_margin"]
            wacc = params["wacc"]
            terminal_growth = params["terminal_growth"]
            
            # Tax rate assumption (typical corporate rate)
            tax_rate = 0.25
            
            # Project cash flows
            projected_fcf = []
            revenue = base_revenue
            
            for year in range(1, years + 1):
                # Revenue growth (declining over time)
                growth_rate = revenue_growth * (0.9 ** (year - 1))  # Fade factor
                revenue *= (1 + growth_rate)
                
                # EBIT and taxes
                ebit = revenue * ebit_margin
                taxes = ebit * tax_rate
                nopat = ebit - taxes
                
                # Simplified: assume FCF = NOPAT - reinvestment
                reinvestment_rate = growth_rate / 0.12  # Assume 12% ROIC
                reinvestment = nopat * reinvestment_rate
                fcf = nopat - reinvestment
                
                projected_fcf.append(fcf)
            
            # Terminal value
            terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (wacc - terminal_growth)
            
            # Discount to present value
            pv_fcf = []
            for i, fcf in enumerate(projected_fcf):
                pv = fcf / ((1 + wacc) ** (i + 1))
                pv_fcf.append(pv)
            
            pv_terminal = terminal_value / ((1 + wacc) ** years)
            
            # Enterprise value and equity value
            enterprise_value = sum(pv_fcf) + pv_terminal
            
            # Simplified: assume no net debt
            equity_value = enterprise_value
            fair_value_per_share = equity_value / shares if shares > 0 else 0
            
            return {
                "projected_fcf": projected_fcf,
                "terminal_value": terminal_value,
                "pv_fcf": pv_fcf,
                "pv_terminal": pv_terminal,
                "enterprise_value": enterprise_value,
                "equity_value": equity_value,
                "fair_value_per_share": fair_value_per_share,
                "params": params
            }
            
        except Exception as e:
            logger.error(f"DCF scenario calculation failed: {e}")
            return {"fair_value_per_share": 0, "params": params}
    
    def reverse_dcf(self, current_price: float, financial_data: Dict[str, Any], 
                   ticker: str, sector: str = None) -> Dict[str, Any]:
        """
        Calculate implied growth rate from current price (reverse DCF)
        """
        try:
            sector = sector or self._determine_sector(ticker)
            wacc_low, wacc_high = self._get_wacc_range(sector)
            avg_wacc = (wacc_low + wacc_high) / 2
            
            # Base metrics
            income_statements = financial_data.get("income_statements", [])
            if not income_statements:
                return {"implied_cagr": 0.0, "error": "No financial data"}
            
            recent_income = income_statements[0]
            revenue = recent_income.get("revenue", 0)
            ebit = recent_income.get("operating_income", 0)
            shares = recent_income.get("shares_outstanding", 1)
            
            if revenue == 0 or shares == 0:
                return {"implied_cagr": 0.0, "error": "Invalid financial data"}
            
            # Current market cap
            market_cap = current_price * shares
            
            # Assume current EBIT margin
            ebit_margin = ebit / revenue if revenue > 0 else 0.10
            
            # Tax rate
            tax_rate = 0.25
            
            # Terminal growth
            terminal_growth = 0.025  # 2.5%
            
            # Solve for implied growth rate
            # This is a simplified approach - in practice would use numerical solver
            implied_growth_rates = []
            
            for test_growth in np.arange(0.0, 0.30, 0.001):  # Test 0% to 30%
                try:
                    # Project 5 years
                    total_pv = 0
                    test_revenue = revenue
                    
                    for year in range(1, 6):
                        growth_rate = test_growth * (0.9 ** (year - 1))
                        test_revenue *= (1 + growth_rate)
                        ebit_proj = test_revenue * ebit_margin
                        nopat = ebit_proj * (1 - tax_rate)
                        
                        # Simplified FCF
                        reinvestment_rate = growth_rate / 0.12
                        fcf = nopat * (1 - reinvestment_rate)
                        
                        pv_fcf = fcf / ((1 + avg_wacc) ** year)
                        total_pv += pv_fcf
                    
                    # Terminal value
                    terminal_fcf = fcf * (1 + terminal_growth)
                    terminal_value = terminal_fcf / (avg_wacc - terminal_growth)
                    pv_terminal = terminal_value / ((1 + avg_wacc) ** 5)
                    
                    total_value = total_pv + pv_terminal
                    
                    # Check if this matches current market cap
                    if abs(total_value - market_cap) < market_cap * 0.01:  # Within 1%
                        implied_growth_rates.append(test_growth)
                        break
                        
                except:
                    continue
            
            implied_cagr = implied_growth_rates[0] if implied_growth_rates else 0.05
            
            return {
                "implied_cagr": implied_cagr,
                "current_price": current_price,
                "market_cap": market_cap,
                "wacc_used": avg_wacc,
                "terminal_growth": terminal_growth,
                "base_metrics": {
                    "revenue": revenue,
                    "ebit_margin": ebit_margin
                }
            }
            
        except Exception as e:
            logger.error(f"Reverse DCF failed for {ticker}: {e}")
            return {"implied_cagr": 0.05, "error": str(e)}
    
    def multiples_valuation(self, financial_data: Dict[str, Any], 
                          ticker: str, sector: str = None) -> Dict[str, Any]:
        """
        Valuation using peer multiples (simplified approach)
        """
        try:
            # Get recent financials
            income_statements = financial_data.get("income_statements", [])
            if not income_statements:
                return {"error": "No financial data for multiples valuation"}
            
            recent_income = income_statements[0]
            revenue = recent_income.get("revenue", 0)
            net_income = recent_income.get("net_income", 0)
            ebitda = recent_income.get("ebitda", 0)
            shares = recent_income.get("shares_outstanding", 1)
            
            sector = sector or self._determine_sector(ticker)
            
            # Sector average multiples (these would come from market data in production)
            sector_multiples = {
                "technology": {"pe": 25, "ps": 8, "ev_ebitda": 20},
                "healthcare": {"pe": 22, "ps": 6, "ev_ebitda": 18},
                "financials": {"pe": 12, "ps": 3, "ev_ebitda": 10},
                "energy": {"pe": 15, "ps": 2, "ev_ebitda": 8},
                "consumer_staples": {"pe": 20, "ps": 2.5, "ev_ebitda": 12},
                "industrials": {"pe": 18, "ps": 2, "ev_ebitda": 14},
                "default": {"pe": 18, "ps": 4, "ev_ebitda": 15}
            }
            
            multiples = sector_multiples.get(sector, sector_multiples["default"])
            
            # Calculate implied values
            valuations = {}
            
            # P/E multiple
            if net_income > 0:
                eps = net_income / shares
                pe_value = eps * multiples["pe"] * shares
                valuations["pe_valuation"] = pe_value / shares
            
            # P/S multiple
            if revenue > 0:
                ps_value = revenue * multiples["ps"]
                valuations["ps_valuation"] = ps_value / shares
            
            # EV/EBITDA multiple (simplified)
            if ebitda > 0:
                ev = ebitda * multiples["ev_ebitda"]
                # Simplified: assume EV = equity value
                valuations["ev_ebitda_valuation"] = ev / shares
            
            # Average of available valuations
            valid_valuations = [v for v in valuations.values() if v > 0]
            average_valuation = sum(valid_valuations) / len(valid_valuations) if valid_valuations else 0
            
            return {
                "ticker": ticker,
                "sector": sector,
                "multiples_used": multiples,
                "individual_valuations": valuations,
                "average_valuation": average_valuation,
                "base_metrics": {
                    "revenue": revenue,
                    "net_income": net_income,
                    "ebitda": ebitda,
                    "shares": shares
                }
            }
            
        except Exception as e:
            logger.error(f"Multiples valuation failed for {ticker}: {e}")
            return {"error": str(e)}
    
    def multiples_valuation_with_crosscheck(self, financial_data: Dict[str, Any], 
                                          ticker: str, sector: str = None,
                                          sector_medians: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Enhanced multiples valuation with sector median cross-check
        """
        try:
            # Get base multiples valuation
            base_result = self.multiples_valuation(financial_data, ticker, sector)
            
            if "error" in base_result:
                return base_result
            
            # Get normalized metrics for cross-check
            normalized_metrics = financial_data.get("normalized_metrics", {})
            pe_ttm = normalized_metrics.get("pe_ttm", 0)
            ev_ebitda_ttm = normalized_metrics.get("ev_ebitda_ttm", 0)
            
            # Apply sector median cross-check if available
            if sector_medians and pe_ttm > 0:
                sector_pe_median = sector_medians.get(f"{sector}_pe_median", 0)
                sector_ev_ebitda_median = sector_medians.get(f"{sector}_ev_ebitda_median", 0)
                
                # Calculate sector-relative valuation
                if sector_pe_median > 0:
                    pe_relative_discount = 1 - (pe_ttm / sector_pe_median)
                    base_result["pe_relative_discount"] = pe_relative_discount
                
                if sector_ev_ebitda_median > 0:
                    ev_ebitda_relative_discount = 1 - (ev_ebitda_ttm / sector_ev_ebitda_median)
                    base_result["ev_ebitda_relative_discount"] = ev_ebitda_relative_discount
                
                # Adjust valuation based on sector relative position
                avg_relative_discount = np.mean([
                    pe_relative_discount if sector_pe_median > 0 else 0,
                    ev_ebitda_relative_discount if sector_ev_ebitda_median > 0 else 0
                ])
                
                if abs(avg_relative_discount) > 0.1:  # Significant discount/premium
                    adjustment_factor = 1 + (avg_relative_discount * 0.5)  # 50% weight to relative position
                    base_result["sector_adjusted_valuation"] = base_result["average_valuation"] * adjustment_factor
                    base_result["average_valuation"] = base_result["sector_adjusted_valuation"]
            
            return base_result
            
        except Exception as e:
            logger.error(f"Enhanced multiples valuation failed for {ticker}: {e}")
            return {"error": str(e)}
    
    def _apply_valuation_guardrails(self, valuation: float, current_price: float) -> float:
        """Apply valuation guardrails to prevent extreme values"""
        if valuation <= 0 or current_price <= 0:
            return valuation
        
        ratio = valuation / current_price
        
        # Guardrails: limit to 10x overvaluation or 90% undervaluation
        if ratio > 10.0:  # More than 10x current price
            return current_price * 10.0
        elif ratio < 0.1:  # Less than 10% of current price
            return current_price * 0.1
        
        return valuation
    
    def _calculate_growth_rate(self, statements: List[Dict], metric: str) -> float:
        """Calculate historical growth rate for a metric"""
        try:
            values = [stmt.get(metric, 0) for stmt in statements[:5]]  # Last 5 years
            values = [v for v in values if v > 0]
            
            if len(values) < 2:
                return 0.05  # Default 5%
            
            # Calculate CAGR
            years = len(values) - 1
            cagr = ((values[0] / values[-1]) ** (1/years)) - 1
            
            # Cap growth rates at reasonable levels
            return max(-0.20, min(0.50, cagr))
            
        except Exception as e:
            logger.warning(f"Growth rate calculation failed for {metric}: {e}")
            return 0.05
    
    def _get_default_valuation(self, ticker: str, current_price: float) -> Dict[str, Any]:
        """Default valuation when calculations fail"""
        return {
            "ticker": ticker,
            "current_price": current_price,
            "median_fv": current_price,  # Assume fair value
            "p_underv": 0.5,  # 50% probability
            "gap": 0.0,  # No gap
            "error": "Valuation calculation failed - using current price as estimate",
            "valuation_date": datetime.now().isoformat()
        }


def ensemble_valuation(financial_data: Dict[str, Any],
                      ticker: str,
                      current_price: float,
                      sector: str = None,
                      sector_medians: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Main ensemble valuation function combining multiple methodologies
    
    Args:
        financial_data: Company financial data
        ticker: Stock ticker
        current_price: Current stock price
        sector: Company sector (optional)
    
    Returns:
        Combined valuation results
    """
    try:
        engine = ValuationEngine()
        
        # Perform DCF valuation
        dcf_results = engine.dcf_valuation(financial_data, ticker, current_price, sector)
        
        # Perform reverse DCF
        reverse_dcf_results = engine.reverse_dcf(current_price, financial_data, ticker, sector)
        
        # Perform multiples valuation with sector cross-check
        multiples_results = engine.multiples_valuation_with_crosscheck(
            financial_data, ticker, sector, sector_medians
        )
        
        # Combine results with DCF/Multiples blending (70/30)
        dcf_value = dcf_results.get("median_fv", 0)
        multiples_value = multiples_results.get("average_valuation", 0)
        
        # Apply guardrails
        dcf_value = engine._apply_valuation_guardrails(dcf_value, current_price)
        multiples_value = engine._apply_valuation_guardrails(multiples_value, current_price)
        
        valuation_methods = []
        blended_fv = 0
        
        if dcf_value > 0 and multiples_value > 0:
            # 70/30 DCF/Multiples blend
            blended_fv = dcf_value * 0.7 + multiples_value * 0.3
            valuation_methods = [dcf_value, multiples_value, blended_fv]
        elif dcf_value > 0:
            blended_fv = dcf_value
            valuation_methods = [dcf_value]
        elif multiples_value > 0:
            blended_fv = multiples_value
            valuation_methods = [multiples_value]
        
        # Calculate ensemble metrics
        if valuation_methods:
            median_fv = blended_fv if blended_fv > 0 else np.median(valuation_methods)
            p_underv = sum(1 for fv in valuation_methods if fv > current_price) / len(valuation_methods)
            gap = (median_fv / current_price - 1) if current_price > 0 else 0
        else:
            median_fv = current_price
            p_underv = 0.5
            gap = 0.0
        
        # Get implied CAGR
        implied_cagr = reverse_dcf_results.get("implied_cagr", 0.05)
        
        return {
            "ticker": ticker,
            "median_fv": float(median_fv),
            "p_underv": float(p_underv),
            "gap": float(gap),
            "implied_cagr": float(implied_cagr),
            "valuation_methods": valuation_methods,
            "dcf_analysis": dcf_results,
            "reverse_dcf": reverse_dcf_results,
            "multiples_analysis": multiples_results,
            "ensemble_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ensemble valuation failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "median_fv": float(current_price),
            "p_underv": 0.5,
            "gap": 0.0,
            "implied_cagr": 0.05,
            "error": str(e),
            "ensemble_date": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test valuation engine
    import json
    
    # Sample financial data
    sample_data = {
        "income_statements": [
            {
                "date": "2023-12-31",
                "revenue": 100000000,
                "operating_income": 20000000,
                "net_income": 15000000,
                "shares_outstanding": 10000000,
                "ebitda": 25000000
            }
        ]
    }
    
    # Test ensemble valuation
    result = ensemble_valuation(sample_data, "TEST", 50.0, "technology")
    
    print("Ensemble Valuation Test Results:")
    print(json.dumps(result, indent=2))
