"""
Financial Modeling Prep (FMP) API Integration

Fetches financial data including company profile, ratios, income statements,
balance sheets, and cash flow statements.
"""
import asyncio
import os
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from .cache import get_cache

logger = logging.getLogger(__name__)


class FMPClient:
    """Financial Modeling Prep API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.session = None
        self.cache = get_cache()
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with retries"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}/{endpoint}"
        request_params = {"apikey": self.api_key}
        if params:
            request_params.update(params)
        
        # Check cache first
        cache_key = f"fmp:{endpoint}:{str(sorted(request_params.items()))}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for FMP request: {endpoint}")
            return cached_result
        
        async with self.session.get(url, params=request_params) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Cache the result
            self.cache.set(cache_key, data, ttl_hours=24)
            
            return data
    
    async def get_company_profile(self, ticker: str) -> Dict[str, Any]:
        """Get company profile information"""
        try:
            data = await self._make_request(f"profile/{ticker}")
            
            if not data or not isinstance(data, list) or len(data) == 0:
                return {}
            
            profile = data[0]
            
            # Normalize the response
            return {
                "ticker": ticker,
                "company_name": profile.get("companyName", ""),
                "sector": profile.get("sector", ""),
                "industry": profile.get("industry", ""),
                "market_cap": profile.get("mktCap", 0),
                "price": profile.get("price", 0),
                "beta": profile.get("beta", 0),
                "country": profile.get("country", ""),
                "exchange": profile.get("exchange", ""),
                "currency": profile.get("currency", "USD"),
                "description": profile.get("description", ""),
                "website": profile.get("website", ""),
                "ipo_date": profile.get("ipoDate", ""),
                "full_time_employees": profile.get("fullTimeEmployees", 0),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get company profile for {ticker}: {e}")
            return {}
    
    async def get_key_metrics(self, ticker: str, period: str = "annual", limit: int = 5) -> List[Dict[str, Any]]:
        """Get key financial metrics and ratios"""
        try:
            data = await self._make_request(
                f"key-metrics-ttm/{ticker}" if period == "ttm" else f"key-metrics/{ticker}",
                {"period": period, "limit": limit}
            )
            
            if not data or not isinstance(data, list):
                return []
            
            normalized_metrics = []
            for metric in data:
                normalized_metric = {
                    "ticker": ticker,
                    "date": metric.get("date", ""),
                    "period": metric.get("period", period),
                    
                    # Valuation ratios
                    "pe_ratio": metric.get("peRatio", 0),
                    "peg_ratio": metric.get("pegRatio", 0),
                    "price_to_book": metric.get("priceToBookRatio", 0),
                    "price_to_sales": metric.get("priceToSalesRatio", 0),
                    "enterprise_value": metric.get("enterpriseValue", 0),
                    "ev_to_revenue": metric.get("enterpriseValueOverEBITDA", 0),
                    
                    # Profitability ratios
                    "roe": metric.get("returnOnEquity", 0),
                    "roa": metric.get("returnOnAssets", 0),
                    "roic": metric.get("returnOnCapitalEmployed", 0),
                    "gross_profit_margin": metric.get("grossProfitMargin", 0),
                    "operating_margin": metric.get("operatingProfitMargin", 0),
                    "net_profit_margin": metric.get("netProfitMargin", 0),
                    
                    # Liquidity ratios
                    "current_ratio": metric.get("currentRatio", 0),
                    "quick_ratio": metric.get("quickRatio", 0),
                    "cash_ratio": metric.get("cashRatio", 0),
                    
                    # Leverage ratios
                    "debt_to_equity": metric.get("debtToEquity", 0),
                    "debt_to_assets": metric.get("debtToAssets", 0),
                    "interest_coverage": metric.get("interestCoverage", 0),
                    
                    # Growth metrics
                    "revenue_growth": metric.get("revenueGrowth", 0),
                    "earnings_growth": metric.get("epsgrowth", 0),
                    
                    # Market metrics
                    "book_value_per_share": metric.get("bookValuePerShare", 0),
                    "tangible_book_value": metric.get("tangibleBookValuePerShare", 0),
                    "working_capital": metric.get("workingCapital", 0),
                    
                    "last_updated": datetime.now().isoformat()
                }
                normalized_metrics.append(normalized_metric)
            
            return normalized_metrics
            
        except Exception as e:
            logger.error(f"Failed to get key metrics for {ticker}: {e}")
            return []
    
    async def get_income_statement(self, ticker: str, period: str = "annual", limit: int = 5) -> List[Dict[str, Any]]:
        """Get income statement data"""
        try:
            data = await self._make_request(
                f"income-statement/{ticker}",
                {"period": period, "limit": limit}
            )
            
            if not data or not isinstance(data, list):
                return []
            
            normalized_statements = []
            for statement in data:
                normalized_statement = {
                    "ticker": ticker,
                    "date": statement.get("date", ""),
                    "period": statement.get("period", period),
                    
                    # Revenue
                    "revenue": statement.get("revenue", 0),
                    "cost_of_revenue": statement.get("costOfRevenue", 0),
                    "gross_profit": statement.get("grossProfit", 0),
                    
                    # Operating expenses
                    "operating_expenses": statement.get("operatingExpenses", 0),
                    "rd_expenses": statement.get("researchAndDevelopmentExpenses", 0),
                    "sga_expenses": statement.get("sellingGeneralAndAdministrativeExpenses", 0),
                    
                    # Operating income
                    "operating_income": statement.get("operatingIncome", 0),
                    "ebitda": statement.get("ebitda", 0),
                    
                    # Non-operating
                    "interest_expense": statement.get("interestExpense", 0),
                    "interest_income": statement.get("interestIncome", 0),
                    "other_income": statement.get("otherIncomeExpenseNet", 0),
                    
                    # Pre-tax and taxes
                    "income_before_tax": statement.get("incomeBeforeTax", 0),
                    "tax_expense": statement.get("incomeTaxExpense", 0),
                    
                    # Net income
                    "net_income": statement.get("netIncome", 0),
                    "eps": statement.get("eps", 0),
                    "eps_diluted": statement.get("epsdiluted", 0),
                    
                    # Share counts
                    "shares_outstanding": statement.get("weightedAverageShsOut", 0),
                    "shares_outstanding_diluted": statement.get("weightedAverageShsOutDil", 0),
                    
                    "last_updated": datetime.now().isoformat()
                }
                normalized_statements.append(normalized_statement)
            
            return normalized_statements
            
        except Exception as e:
            logger.error(f"Failed to get income statement for {ticker}: {e}")
            return []
    
    async def get_balance_sheet(self, ticker: str, period: str = "annual", limit: int = 5) -> List[Dict[str, Any]]:
        """Get balance sheet data"""
        try:
            data = await self._make_request(
                f"balance-sheet-statement/{ticker}",
                {"period": period, "limit": limit}
            )
            
            if not data or not isinstance(data, list):
                return []
            
            normalized_statements = []
            for statement in data:
                normalized_statement = {
                    "ticker": ticker,
                    "date": statement.get("date", ""),
                    "period": statement.get("period", period),
                    
                    # Current assets
                    "cash_and_equivalents": statement.get("cashAndCashEquivalents", 0),
                    "short_term_investments": statement.get("shortTermInvestments", 0),
                    "accounts_receivable": statement.get("netReceivables", 0),
                    "inventory": statement.get("inventory", 0),
                    "current_assets": statement.get("totalCurrentAssets", 0),
                    
                    # Non-current assets
                    "ppe_net": statement.get("propertyPlantEquipmentNet", 0),
                    "goodwill": statement.get("goodwill", 0),
                    "intangible_assets": statement.get("intangibleAssets", 0),
                    "long_term_investments": statement.get("longTermInvestments", 0),
                    "total_assets": statement.get("totalAssets", 0),
                    
                    # Current liabilities
                    "accounts_payable": statement.get("accountPayables", 0),
                    "short_term_debt": statement.get("shortTermDebt", 0),
                    "current_liabilities": statement.get("totalCurrentLiabilities", 0),
                    
                    # Non-current liabilities
                    "long_term_debt": statement.get("longTermDebt", 0),
                    "total_debt": statement.get("totalDebt", 0),
                    "total_liabilities": statement.get("totalLiabilities", 0),
                    
                    # Equity
                    "shareholders_equity": statement.get("totalShareholdersEquity", 0),
                    "retained_earnings": statement.get("retainedEarnings", 0),
                    "common_stock": statement.get("commonStock", 0),
                    
                    "last_updated": datetime.now().isoformat()
                }
                normalized_statements.append(normalized_statement)
            
            return normalized_statements
            
        except Exception as e:
            logger.error(f"Failed to get balance sheet for {ticker}: {e}")
            return []
    
    async def get_cash_flow(self, ticker: str, period: str = "annual", limit: int = 5) -> List[Dict[str, Any]]:
        """Get cash flow statement data"""
        try:
            data = await self._make_request(
                f"cash-flow-statement/{ticker}",
                {"period": period, "limit": limit}
            )
            
            if not data or not isinstance(data, list):
                return []
            
            normalized_statements = []
            for statement in data:
                normalized_statement = {
                    "ticker": ticker,
                    "date": statement.get("date", ""),
                    "period": statement.get("period", period),
                    
                    # Operating activities
                    "net_income": statement.get("netIncome", 0),
                    "depreciation": statement.get("depreciationAndAmortization", 0),
                    "stock_compensation": statement.get("stockBasedCompensation", 0),
                    "working_capital_change": statement.get("changeInWorkingCapital", 0),
                    "operating_cash_flow": statement.get("operatingCashFlow", 0),
                    
                    # Investing activities
                    "capex": statement.get("capitalExpenditure", 0),
                    "acquisitions": statement.get("acquisitionsNet", 0),
                    "investments_change": statement.get("investmentsInPropertyPlantAndEquipment", 0),
                    "investing_cash_flow": statement.get("netCashUsedForInvestingActivites", 0),
                    
                    # Financing activities
                    "debt_repayment": statement.get("debtRepayment", 0),
                    "common_stock_issued": statement.get("commonStockIssued", 0),
                    "common_stock_repurchased": statement.get("commonStockRepurchased", 0),
                    "dividends_paid": statement.get("dividendsPaid", 0),
                    "financing_cash_flow": statement.get("netCashUsedProvidedByFinancingActivities", 0),
                    
                    # Net change
                    "net_change_in_cash": statement.get("netChangeInCash", 0),
                    "free_cash_flow": statement.get("freeCashFlow", 0),
                    
                    "last_updated": datetime.now().isoformat()
                }
                normalized_statements.append(normalized_statement)
            
            return normalized_statements
            
        except Exception as e:
            logger.error(f"Failed to get cash flow for {ticker}: {e}")
            return []
    
def _normalize_financial_metrics(profile: Dict, ratios: Dict, 
                               income: Dict, balance: Dict) -> Dict[str, Any]:
        """Normalize financial metrics to standard format"""
        try:
            # Extract basic data
            shares_out = income.get("shares_outstanding", 0) or profile.get("shares_outstanding", 0)
            market_cap = profile.get("market_cap", 0)
            revenue_ttm = income.get("revenue", 0)
            ebitda_ttm = income.get("ebitda", 0)
            net_income_ttm = income.get("net_income", 0)
            
            # Balance sheet items
            total_debt = balance.get("total_debt", 0)
            cash_and_equivalents = balance.get("cash_and_equivalents", 0)
            total_assets = balance.get("total_assets", 0)
            shareholders_equity = balance.get("shareholders_equity", 0)
            
            # Calculate derived metrics
            net_debt = max(0, total_debt - cash_and_equivalents)
            enterprise_value = market_cap + net_debt if market_cap > 0 else 0
            
            # Ratios
            pe_ttm = ratios.get("pe_ratio", 0)
            ev_ebitda_ttm = enterprise_value / ebitda_ttm if ebitda_ttm > 0 else 0
            gross_margin_ttm = ratios.get("gross_profit_margin", 0)
            op_margin_ttm = ratios.get("operating_margin", 0)
            
            # ROIC estimate (simplified)
            # ROIC = NOPAT / Invested Capital
            # NOPAT ≈ Operating Income * (1 - Tax Rate)
            operating_income = income.get("operating_income", 0)
            tax_rate = 0.25  # Assumption
            nopat = operating_income * (1 - tax_rate)
            invested_capital = shareholders_equity + net_debt if shareholders_equity > 0 else total_assets
            roic_estimate = nopat / invested_capital if invested_capital > 0 else 0
            
            return {
                "shares_out": float(shares_out),
                "market_cap": float(market_cap),
                "enterprise_value": float(enterprise_value),
                "revenue_ttm": float(revenue_ttm),
                "ebitda_ttm": float(ebitda_ttm),
                "net_debt": float(net_debt),
                "pe_ttm": float(pe_ttm),
                "ev_ebitda_ttm": float(ev_ebitda_ttm),
                "gross_margin_ttm": float(gross_margin_ttm),
                "op_margin_ttm": float(op_margin_ttm),
                "roic_estimate": float(roic_estimate),
                "calculation_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Failed to normalize financial metrics: {e}")
            return {
                "shares_out": 0.0,
                "market_cap": 0.0,
                "enterprise_value": 0.0,
                "revenue_ttm": 0.0,
                "ebitda_ttm": 0.0,
                "net_debt": 0.0,
                "pe_ttm": 0.0,
                "ev_ebitda_ttm": 0.0,
                "gross_margin_ttm": 0.0,
                "op_margin_ttm": 0.0,
                "roic_estimate": 0.0,
                "calculation_date": datetime.now().isoformat()
            }


async def get_financials_fmp(ticker: str) -> Dict[str, Any]:
    """
    Main function to get comprehensive financial data for a ticker
    
    Returns:
        Dictionary with profile, ratios, income, balance, and cash flow data
    """
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        logger.error("FMP_API_KEY not found in environment")
        return {}
    
    try:
        async with FMPClient(api_key) as client:
            # Fetch all financial data concurrently
            profile_task = client.get_company_profile(ticker)
            ratios_task = client.get_key_metrics(ticker, "ttm", 1)  # TTM ratios
            income_task = client.get_income_statement(ticker, "annual", 5)
            balance_task = client.get_balance_sheet(ticker, "annual", 5)
            cash_task = client.get_cash_flow(ticker, "annual", 5)
            
            # Wait for all requests to complete
            profile, ratios_ttm, income, balance, cash_flow = await asyncio.gather(
                profile_task, ratios_task, income_task, balance_task, cash_task,
                return_exceptions=True
            )
            
            # Handle any exceptions
            if isinstance(profile, Exception):
                logger.error(f"Profile fetch failed: {profile}")
                profile = {}
            if isinstance(ratios_ttm, Exception):
                logger.error(f"Ratios fetch failed: {ratios_ttm}")
                ratios_ttm = []
            if isinstance(income, Exception):
                logger.error(f"Income statement fetch failed: {income}")
                income = []
            if isinstance(balance, Exception):
                logger.error(f"Balance sheet fetch failed: {balance}")
                balance = []
            if isinstance(cash_flow, Exception):
                logger.error(f"Cash flow fetch failed: {cash_flow}")
                cash_flow = []
            
            # Calculate normalized metrics
            normalized_metrics = _normalize_financial_metrics(
                profile, ratios_ttm[0] if ratios_ttm else {}, 
                income[0] if income else {}, balance[0] if balance else {}
            )
            
            return {
                "ticker": ticker,
                "profile": profile,
                "ratios_ttm": ratios_ttm[0] if ratios_ttm else {},
                "income_statements": income,
                "balance_sheets": balance,
                "cash_flows": cash_flow,
                "normalized_metrics": normalized_metrics,
                "last_updated": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Failed to get financials for {ticker}: {e}")
        return {}


async def main():
    """CLI entry point for testing FMP integration"""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test FMP API integration")
    parser.add_argument("--ticker", required=True, help="Stock ticker")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Test the integration
    result = await get_financials_fmp(args.ticker)
    
    print(f"\nFinancial data for {args.ticker}:")
    print("=" * 50)
    
    if result.get("profile"):
        profile = result["profile"]
        print(f"Company: {profile.get('company_name')}")
        print(f"Sector: {profile.get('sector')}")
        print(f"Market Cap: ${profile.get('market_cap', 0):,.0f}")
    
    if result.get("ratios_ttm"):
        ratios = result["ratios_ttm"]
        print(f"\nKey Ratios (TTM):")
        print(f"P/E Ratio: {ratios.get('pe_ratio', 0):.2f}")
        print(f"ROE: {ratios.get('roe', 0):.2%}")
        print(f"Debt/Equity: {ratios.get('debt_to_equity', 0):.2f}")
    
    print(f"\nIncome Statements: {len(result.get('income_statements', []))}")
    print(f"Balance Sheets: {len(result.get('balance_sheets', []))}")
    print(f"Cash Flows: {len(result.get('cash_flows', []))}")


if __name__ == "__main__":
    asyncio.run(main())
