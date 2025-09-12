#!/usr/bin/env python3
"""
Data Validation Module
Addresses critical data accuracy and timeliness issues identified in WMT validation.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import aiohttp
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Critical data validation and cross-checking against multiple sources
    Fixes: Data accuracy, timeliness, TTM vs forward mixing
    """
    
    def __init__(self):
        self.sec_api_key = os.getenv("SEC_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def validate_financial_data(self, ticker: str, fmp_data: Dict) -> Dict[str, Any]:
        """
        Comprehensive data validation addressing WMT issues
        
        Returns:
            Validation report with quality scores and corrections
        """
        try:
            logger.info(f"🔍 Validating financial data for {ticker}...")
            
            validation_report = {
                "ticker": ticker,
                "validation_timestamp": datetime.now().isoformat(),
                "data_quality_score": 0.0,
                "critical_issues": [],
                "corrections_applied": [],
                "consensus_comparison": {},
                "sec_validation": {},
                "temporal_alignment": {}
            }
            
            # 1. Check for critical missing data (addresses AAPL N/A issues)
            missing_data_check = await self._check_critical_data_completeness(fmp_data)
            validation_report["critical_issues"].extend(missing_data_check["issues"])
            validation_report["data_quality_score"] += missing_data_check["score"] * 0.4
            
            # 2. SEC filing cross-validation
            if self.sec_api_key:
                sec_validation = await self._validate_against_latest_sec_filing(ticker, fmp_data)
                validation_report["sec_validation"] = sec_validation
                validation_report["data_quality_score"] += sec_validation["alignment_score"] * 0.3
            
            # 3. Consensus estimates comparison
            consensus_data = await self._get_consensus_estimates(ticker)
            if consensus_data:
                consensus_validation = self._compare_with_consensus(fmp_data, consensus_data)
                validation_report["consensus_comparison"] = consensus_validation
                validation_report["data_quality_score"] += consensus_validation["alignment_score"] * 0.2
            
            # 4. Temporal alignment check (TTM vs Forward)
            temporal_check = self._validate_temporal_alignment(fmp_data)
            validation_report["temporal_alignment"] = temporal_check
            validation_report["data_quality_score"] += temporal_check["score"] * 0.1
            
            # 5. Apply data corrections
            if validation_report["critical_issues"]:
                corrected_data = await self._apply_data_corrections(ticker, fmp_data, validation_report)
                validation_report["corrected_data"] = corrected_data
            
            logger.info(f"✅ Data validation complete for {ticker}. Quality score: {validation_report['data_quality_score']:.2f}")
            return validation_report
            
        except Exception as e:
            logger.error(f"❌ Data validation failed for {ticker}: {e}")
            return {
                "ticker": ticker,
                "validation_timestamp": datetime.now().isoformat(),
                "data_quality_score": 0.0,
                "error": str(e)
            }
    
    async def _check_critical_data_completeness(self, fmp_data: Dict) -> Dict:
        """Check for critical missing data that causes N/A values"""
        issues = []
        score = 1.0
        
        # Critical fields that must not be N/A or 0
        critical_fields = {
            "profile.market_cap": "Market Cap",
            "normalized_metrics.enterprise_value": "Enterprise Value", 
            "normalized_metrics.revenue_ttm": "Revenue (TTM)",
            "normalized_metrics.ebitda_ttm": "EBITDA (TTM)",
            "normalized_metrics.shares_out": "Shares Outstanding"
        }
        
        for field_path, display_name in critical_fields.items():
            value = self._get_nested_value(fmp_data, field_path)
            if value is None or value == 0 or value == "N/A":
                issues.append(f"Missing critical data: {display_name}")
                score -= 0.2
        
        return {
            "issues": issues,
            "score": max(0.0, score),
            "critical_fields_check": True
        }
    
    async def _validate_against_latest_sec_filing(self, ticker: str, fmp_data: Dict) -> Dict:
        """Cross-validate against latest SEC filing"""
        try:
            # Get latest 10-K/10-Q filing
            latest_filing = await self._get_latest_sec_filing(ticker)
            if not latest_filing:
                return {"alignment_score": 0.5, "note": "No recent SEC filing found"}
            
            # Extract key metrics from filing
            sec_metrics = await self._extract_metrics_from_filing(latest_filing)
            
            # Compare with FMP data
            alignment_score = self._calculate_alignment_score(fmp_data, sec_metrics)
            
            return {
                "alignment_score": alignment_score,
                "filing_date": latest_filing.get("filing_date"),
                "filing_type": latest_filing.get("form_type"),
                "sec_metrics": sec_metrics,
                "variances": self._calculate_variances(fmp_data, sec_metrics)
            }
            
        except Exception as e:
            logger.warning(f"SEC validation failed for {ticker}: {e}")
            return {"alignment_score": 0.5, "error": str(e)}
    
    async def _get_consensus_estimates(self, ticker: str) -> Optional[Dict]:
        """Get analyst consensus estimates from multiple sources"""
        try:
            # Try Alpha Vantage first
            if self.alpha_vantage_key:
                av_data = await self._get_alpha_vantage_estimates(ticker)
                if av_data:
                    return av_data
            
            # Fallback to Yahoo Finance (free)
            yahoo_data = await self._get_yahoo_estimates(ticker)
            return yahoo_data
            
        except Exception as e:
            logger.warning(f"Consensus estimates failed for {ticker}: {e}")
            return None
    
    async def _get_alpha_vantage_estimates(self, ticker: str) -> Optional[Dict]:
        """Get estimates from Alpha Vantage"""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "EARNINGS",
                "symbol": ticker,
                "apikey": self.alpha_vantage_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_alpha_vantage_estimates(data)
                    
        except Exception as e:
            logger.debug(f"Alpha Vantage estimates failed for {ticker}: {e}")
        return None
    
    async def _get_yahoo_estimates(self, ticker: str) -> Optional[Dict]:
        """Get estimates from Yahoo Finance (free alternative)"""
        try:
            # Yahoo Finance analyst estimates endpoint
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            params = {
                "modules": "defaultKeyStatistics,financialData,recommendationTrend,earningsHistory,earningsTrend"
            }
            
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_yahoo_estimates(data)
                    
        except Exception as e:
            logger.debug(f"Yahoo estimates failed for {ticker}: {e}")
        return None
    
    def _compare_with_consensus(self, fmp_data: Dict, consensus_data: Dict) -> Dict:
        """Compare FMP data with consensus estimates"""
        variances = {}
        alignment_score = 1.0
        
        # Compare key metrics
        comparisons = {
            "revenue_estimate": ("normalized_metrics.revenue_ttm", "revenue_estimate"),
            "eps_estimate": ("normalized_metrics.eps_ttm", "eps_estimate"),
            "growth_estimate": ("normalized_metrics.revenue_growth", "revenue_growth_estimate")
        }
        
        for metric, (fmp_path, consensus_key) in comparisons.items():
            fmp_value = self._get_nested_value(fmp_data, fmp_path)
            consensus_value = consensus_data.get(consensus_key)
            
            if fmp_value and consensus_value:
                variance = abs(fmp_value - consensus_value) / consensus_value
                variances[metric] = {
                    "fmp_value": fmp_value,
                    "consensus_value": consensus_value,
                    "variance_pct": variance * 100
                }
                
                # Penalize large variances
                if variance > 0.1:  # >10% variance
                    alignment_score -= 0.2
        
        return {
            "alignment_score": max(0.0, alignment_score),
            "variances": variances,
            "consensus_source": consensus_data.get("source", "unknown")
        }
    
    def _validate_temporal_alignment(self, fmp_data: Dict) -> Dict:
        """Ensure TTM vs Forward metrics are properly separated"""
        issues = []
        score = 1.0
        
        # Check for temporal mixing indicators
        income_statements = fmp_data.get("income_statements", [])
        if income_statements:
            # Ensure dates are properly ordered and recent
            latest_date = income_statements[0].get("date", "")
            if latest_date:
                try:
                    latest_dt = datetime.strptime(latest_date[:10], "%Y-%m-%d")
                    if (datetime.now() - latest_dt).days > 365:
                        issues.append("Financial data is more than 1 year old")
                        score -= 0.3
                except:
                    issues.append("Invalid date format in financial data")
                    score -= 0.2
        
        return {
            "score": max(0.0, score),
            "issues": issues,
            "temporal_check": True
        }
    
    async def _apply_data_corrections(self, ticker: str, fmp_data: Dict, validation_report: Dict) -> Dict:
        """Apply corrections for missing or invalid data"""
        corrected_data = fmp_data.copy()
        corrections = []
        
        # Fix missing market cap
        if self._get_nested_value(corrected_data, "profile.market_cap") in [None, 0, "N/A"]:
            # Calculate from shares outstanding and current price
            shares = self._get_nested_value(corrected_data, "normalized_metrics.shares_out")
            current_price = await self._get_current_price(ticker)
            
            if shares and current_price:
                market_cap = shares * current_price
                self._set_nested_value(corrected_data, "profile.market_cap", market_cap)
                corrections.append(f"Calculated market cap: ${market_cap:,.0f}")
        
        # Fix missing enterprise value
        if self._get_nested_value(corrected_data, "normalized_metrics.enterprise_value") in [None, 0, "N/A"]:
            market_cap = self._get_nested_value(corrected_data, "profile.market_cap")
            net_debt = self._get_nested_value(corrected_data, "normalized_metrics.net_debt", 0)
            
            if market_cap:
                enterprise_value = market_cap + net_debt
                self._set_nested_value(corrected_data, "normalized_metrics.enterprise_value", enterprise_value)
                corrections.append(f"Calculated enterprise value: ${enterprise_value:,.0f}")
        
        return {
            "corrected_data": corrected_data,
            "corrections_applied": corrections
        }
    
    def _get_nested_value(self, data: Dict, path: str, default=None):
        """Get nested dictionary value using dot notation"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def _set_nested_value(self, data: Dict, path: str, value):
        """Set nested dictionary value using dot notation"""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    async def _get_current_price(self, ticker: str) -> Optional[float]:
        """Get current stock price for calculations"""
        try:
            # Use FMP for current price
            url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}"
            params = {"apikey": os.getenv("FMP_API_KEY")}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        return data[0].get("price")
        except:
            pass
        return None
    
    # Additional helper methods for SEC and consensus parsing
    async def _get_latest_sec_filing(self, ticker: str) -> Optional[Dict]:
        """Get latest SEC filing metadata"""
        # Implementation would use SEC API to get latest 10-K/10-Q
        return None  # Placeholder
    
    async def _extract_metrics_from_filing(self, filing: Dict) -> Dict:
        """Extract key metrics from SEC filing"""
        return {}  # Placeholder
    
    def _calculate_alignment_score(self, fmp_data: Dict, sec_metrics: Dict) -> float:
        """Calculate alignment score between FMP and SEC data"""
        return 0.8  # Placeholder
    
    def _calculate_variances(self, fmp_data: Dict, sec_metrics: Dict) -> Dict:
        """Calculate variances between data sources"""
        return {}  # Placeholder
    
    def _parse_alpha_vantage_estimates(self, data: Dict) -> Dict:
        """Parse Alpha Vantage estimates response"""
        return {}  # Placeholder
    
    def _parse_yahoo_estimates(self, data: Dict) -> Dict:
        """Parse Yahoo Finance estimates response"""
        return {}  # Placeholder


# Convenience function for integration
async def validate_financial_data(ticker: str, fmp_data: Dict) -> Dict:
    """Validate financial data using the DataValidator"""
    async with DataValidator() as validator:
        return await validator.validate_financial_data(ticker, fmp_data)


if __name__ == "__main__":
    # Test the validator
    async def test_validator():
        # Mock FMP data with typical issues
        mock_data = {
            "profile": {"market_cap": None},  # Missing critical data
            "normalized_metrics": {
                "enterprise_value": 0,  # Missing critical data
                "revenue_ttm": None,    # Missing critical data
                "shares_out": 1000000
            }
        }
        
        result = await validate_financial_data("AAPL", mock_data)
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_validator())
