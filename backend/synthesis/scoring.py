"""
Investment Scoring Module

Combines valuation, quality, sentiment, and technical analysis into a
comprehensive 1-10 investment score with component breakdown.
"""
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


class InvestmentScorer:
    """Comprehensive investment scoring system"""
    
    def __init__(self, config_path: str = "configs/scoring.yaml"):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load scoring configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default scoring configuration"""
        return {
            "scoring_weights": {
                "valuation": 0.40,
                "quality": 0.25,
                "sentiment": 0.20,
                "technicals": 0.15
            },
            "quality_thresholds": {
                "roe_excellent": 0.20,
                "roe_good": 0.15,
                "debt_to_equity_low": 0.30,
                "debt_to_equity_high": 1.00,
                "current_ratio_good": 1.50,
                "profit_margin_excellent": 0.20,
                "profit_margin_good": 0.10
            },
            "technical_thresholds": {
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "volatility_high": 0.30,
                "momentum_strong": 0.10
            },
            "sentiment_thresholds": {
                "very_positive": 0.6,
                "positive": 0.2,
                "negative": -0.2,
                "very_negative": -0.6
            }
        }
    
    def score_valuation(self, valuation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score valuation component (0-10 scale)
        
        Args:
            valuation_data: Results from ensemble valuation
            
        Returns:
            Valuation score and details
        """
        try:
            # Key metrics
            p_undervalued = valuation_data.get("p_underv", 0.5)
            gap = valuation_data.get("gap", 0.0)  # Price gap (positive = undervalued)
            implied_cagr = valuation_data.get("implied_cagr", 0.05)
            
            # Probability-based scoring (0-10)
            prob_score = p_undervalued * 10
            
            # Gap-based scoring
            if gap > 0.5:  # >50% undervalued
                gap_score = 10
            elif gap > 0.3:  # >30% undervalued
                gap_score = 8
            elif gap > 0.1:  # >10% undervalued
                gap_score = 6
            elif gap > -0.1:  # Within ±10%
                gap_score = 5
            elif gap > -0.3:  # <30% overvalued
                gap_score = 3
            else:  # >30% overvalued
                gap_score = 1
            
            # Implied growth reasonableness
            if 0.05 <= implied_cagr <= 0.15:  # 5-15% is reasonable
                growth_score = 8
            elif 0.03 <= implied_cagr <= 0.20:  # 3-20% is acceptable
                growth_score = 6
            elif implied_cagr <= 0.25:  # Up to 25% is possible
                growth_score = 4
            else:  # >25% is unrealistic
                growth_score = 2
            
            # Weighted average
            final_score = (prob_score * 0.4 + gap_score * 0.4 + growth_score * 0.2)
            
            return {
                "score": float(min(10, max(1, final_score))),
                "components": {
                    "probability_score": float(prob_score),
                    "gap_score": float(gap_score),
                    "growth_reasonableness": float(growth_score)
                },
                "metrics": {
                    "p_undervalued": float(p_undervalued),
                    "price_gap": float(gap),
                    "implied_cagr": float(implied_cagr)
                }
            }
            
        except Exception as e:
            logger.error(f"Valuation scoring failed: {e}")
            return {"score": 5.0, "error": str(e)}
    
    def score_quality(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score financial quality component (0-10 scale)
        
        Args:
            financial_data: Financial data from FMP
            
        Returns:
            Quality score and details
        """
        try:
            # Get thresholds from config
            thresholds = self.config.get("quality_thresholds", {})
            
            # Extract key ratios
            ratios = financial_data.get("ratios_ttm", {})
            if not ratios:
                return {"score": 5.0, "error": "No ratios data available"}
            
            # Quality metrics
            roe = ratios.get("roe", 0)
            roa = ratios.get("roa", 0)
            debt_to_equity = ratios.get("debt_to_equity", 0)
            current_ratio = ratios.get("current_ratio", 0)
            profit_margin = ratios.get("net_profit_margin", 0)
            gross_margin = ratios.get("gross_profit_margin", 0)
            interest_coverage = ratios.get("interest_coverage", 0)
            
            scores = {}
            
            # ROE scoring
            if roe >= thresholds.get("roe_excellent", 0.20):
                scores["roe"] = 10
            elif roe >= thresholds.get("roe_good", 0.15):
                scores["roe"] = 8
            elif roe >= 0.10:
                scores["roe"] = 6
            elif roe >= 0.05:
                scores["roe"] = 4
            else:
                scores["roe"] = 2
            
            # ROA scoring
            if roa >= 0.15:
                scores["roa"] = 10
            elif roa >= 0.10:
                scores["roa"] = 8
            elif roa >= 0.05:
                scores["roa"] = 6
            elif roa >= 0.02:
                scores["roa"] = 4
            else:
                scores["roa"] = 2
            
            # Debt scoring (lower is better)
            if debt_to_equity <= thresholds.get("debt_to_equity_low", 0.30):
                scores["debt"] = 10
            elif debt_to_equity <= 0.50:
                scores["debt"] = 8
            elif debt_to_equity <= thresholds.get("debt_to_equity_high", 1.00):
                scores["debt"] = 6
            elif debt_to_equity <= 2.00:
                scores["debt"] = 4
            else:
                scores["debt"] = 2
            
            # Liquidity scoring
            if current_ratio >= thresholds.get("current_ratio_good", 1.50):
                scores["liquidity"] = 10
            elif current_ratio >= 1.25:
                scores["liquidity"] = 8
            elif current_ratio >= 1.00:
                scores["liquidity"] = 6
            elif current_ratio >= 0.75:
                scores["liquidity"] = 4
            else:
                scores["liquidity"] = 2
            
            # Profitability scoring
            if profit_margin >= thresholds.get("profit_margin_excellent", 0.20):
                scores["profitability"] = 10
            elif profit_margin >= thresholds.get("profit_margin_good", 0.10):
                scores["profitability"] = 8
            elif profit_margin >= 0.05:
                scores["profitability"] = 6
            elif profit_margin >= 0.02:
                scores["profitability"] = 4
            else:
                scores["profitability"] = 2
            
            # Interest coverage
            if interest_coverage >= 10:
                scores["interest_coverage"] = 10
            elif interest_coverage >= 5:
                scores["interest_coverage"] = 8
            elif interest_coverage >= 2.5:
                scores["interest_coverage"] = 6
            elif interest_coverage >= 1.5:
                scores["interest_coverage"] = 4
            else:
                scores["interest_coverage"] = 2
            
            # Calculate weighted average
            weights = {
                "roe": 0.25,
                "roa": 0.20,
                "debt": 0.20,
                "liquidity": 0.15,
                "profitability": 0.15,
                "interest_coverage": 0.05
            }
            
            final_score = sum(scores[metric] * weights[metric] for metric in scores)
            
            return {
                "score": float(min(10, max(1, final_score))),
                "components": scores,
                "metrics": {
                    "roe": float(roe),
                    "roa": float(roa),
                    "debt_to_equity": float(debt_to_equity),
                    "current_ratio": float(current_ratio),
                    "profit_margin": float(profit_margin),
                    "gross_margin": float(gross_margin),
                    "interest_coverage": float(interest_coverage)
                }
            }
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            return {"score": 5.0, "error": str(e)}
    
    def score_sentiment(self, sentiment_data: Dict[str, Any], 
                       news_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Score sentiment component (0-10 scale)
        
        Args:
            sentiment_data: Sentiment analysis results
            news_data: Recent news articles (optional)
            
        Returns:
            Sentiment score and details
        """
        try:
            thresholds = self.config.get("sentiment_thresholds", {})
            
            # Extract sentiment metrics
            overall_sentiment = sentiment_data.get("average_sentiment", 0.0)
            transcript_analysis = sentiment_data.get("transcript_analysis", {})
            transcript_sentiment = transcript_analysis.get("overall_sentiment", 0.0)
            uncertainty_index = transcript_analysis.get("uncertainty_index", 0.5)
            
            scores = {}
            
            # Overall sentiment scoring
            if overall_sentiment >= thresholds.get("very_positive", 0.6):
                scores["overall"] = 10
            elif overall_sentiment >= thresholds.get("positive", 0.2):
                scores["overall"] = 8
            elif overall_sentiment >= thresholds.get("negative", -0.2):
                scores["overall"] = 5
            elif overall_sentiment >= thresholds.get("very_negative", -0.6):
                scores["overall"] = 3
            else:
                scores["overall"] = 1
            
            # Transcript sentiment scoring
            if transcript_sentiment >= 0.5:
                scores["transcript"] = 10
            elif transcript_sentiment >= 0.2:
                scores["transcript"] = 8
            elif transcript_sentiment >= -0.2:
                scores["transcript"] = 5
            elif transcript_sentiment >= -0.5:
                scores["transcript"] = 3
            else:
                scores["transcript"] = 1
            
            # Uncertainty scoring (lower uncertainty is better)
            if uncertainty_index <= 0.2:
                scores["uncertainty"] = 10
            elif uncertainty_index <= 0.4:
                scores["uncertainty"] = 8
            elif uncertainty_index <= 0.6:
                scores["uncertainty"] = 6
            elif uncertainty_index <= 0.8:
                scores["uncertainty"] = 4
            else:
                scores["uncertainty"] = 2
            
            # News sentiment scoring (if available)
            news_score = 5  # Default neutral
            if news_data:
                positive_news = sum(1 for article in news_data 
                                  if "upgrade" in article.get("title", "").lower() or 
                                     "beat" in article.get("title", "").lower())
                negative_news = sum(1 for article in news_data 
                                  if "downgrade" in article.get("title", "").lower() or 
                                     "miss" in article.get("title", "").lower())
                
                if positive_news > negative_news:
                    news_score = min(10, 5 + positive_news * 2)
                elif negative_news > positive_news:
                    news_score = max(1, 5 - negative_news * 2)
            
            scores["news"] = news_score
            
            # Weighted average
            weights = {
                "overall": 0.3,
                "transcript": 0.3,
                "uncertainty": 0.2,
                "news": 0.2
            }
            
            final_score = sum(scores[metric] * weights[metric] for metric in scores)
            
            return {
                "score": float(min(10, max(1, final_score))),
                "components": scores,
                "metrics": {
                    "overall_sentiment": float(overall_sentiment),
                    "transcript_sentiment": float(transcript_sentiment),
                    "uncertainty_index": float(uncertainty_index),
                    "news_articles_analyzed": len(news_data) if news_data else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Sentiment scoring failed: {e}")
            return {"score": 5.0, "error": str(e)}
    
    def score_technicals(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score technical analysis component (0-10 scale)
        
        Args:
            technical_data: Technical indicators results
            
        Returns:
            Technical score and details
        """
        try:
            thresholds = self.config.get("technical_thresholds", {})
            
            # Extract technical metrics
            indicators = technical_data.get("indicators", {})
            if not indicators:
                return {"score": 5.0, "error": "No technical indicators available"}
            
            # Extract values with null safety
            rsi = indicators.get("rsi_14") 
            rsi = float(rsi) if rsi is not None else 50.0
            
            price_vs_sma_50 = indicators.get("price_vs_sma_50")
            price_vs_sma_50 = float(price_vs_sma_50) if price_vs_sma_50 is not None else 0.0
            
            price_vs_sma_200 = indicators.get("price_vs_sma_200")
            price_vs_sma_200 = float(price_vs_sma_200) if price_vs_sma_200 is not None else 0.0
            
            trend_direction = indicators.get("trend_direction") or "sideways"
            
            volatility_raw = indicators.get("volatility_30d")
            volatility = (float(volatility_raw) / 100) if volatility_raw is not None else 0.2
            
            return_1m_raw = indicators.get("return_1m")
            return_1m = (float(return_1m_raw) / 100) if return_1m_raw is not None else 0.0
            
            return_3m_raw = indicators.get("return_3m")
            return_3m = (float(return_3m_raw) / 100) if return_3m_raw is not None else 0.0
            
            max_drawdown_raw = indicators.get("max_drawdown")
            max_drawdown = (float(max_drawdown_raw) / 100) if max_drawdown_raw is not None else 0.0
            
            scores = {}
            
            # RSI scoring (50 is neutral, oversold/overbought conditions)
            if thresholds.get("rsi_oversold", 30) <= rsi <= 40:  # Oversold but recovering
                scores["rsi"] = 8
            elif 40 < rsi <= 60:  # Neutral zone
                scores["rsi"] = 6
            elif 60 < rsi <= thresholds.get("rsi_overbought", 70):  # Strong but not overbought
                scores["rsi"] = 7
            elif rsi > 70:  # Overbought
                scores["rsi"] = 4
            else:  # Very oversold
                scores["rsi"] = 5
            
            # Moving average trends
            sma_score = 5  # Default
            if price_vs_sma_50 > 0 and price_vs_sma_200 > 0:
                sma_score = 8  # Above both MAs
            elif price_vs_sma_50 > 0:
                sma_score = 6  # Above short-term MA
            elif price_vs_sma_200 > 0:
                sma_score = 5  # Above long-term MA only
            else:
                sma_score = 3  # Below both MAs
            
            scores["moving_averages"] = sma_score
            
            # Trend scoring
            if trend_direction == "uptrend":
                scores["trend"] = 8
            elif trend_direction == "sideways":
                scores["trend"] = 5
            else:  # downtrend
                scores["trend"] = 3
            
            # Volatility scoring (lower volatility is generally better for scoring)
            if volatility <= 0.15:  # Low volatility
                scores["volatility"] = 8
            elif volatility <= 0.25:  # Moderate volatility
                scores["volatility"] = 6
            elif volatility <= thresholds.get("volatility_high", 0.30):  # High volatility
                scores["volatility"] = 4
            else:  # Very high volatility
                scores["volatility"] = 2
            
            # Recent performance scoring
            if return_1m >= thresholds.get("momentum_strong", 0.10):  # Strong momentum
                scores["momentum"] = 9
            elif return_1m >= 0.05:  # Good momentum
                scores["momentum"] = 7
            elif return_1m >= 0:  # Positive
                scores["momentum"] = 6
            elif return_1m >= -0.05:  # Slight decline
                scores["momentum"] = 4
            else:  # Poor performance
                scores["momentum"] = 2
            
            # Drawdown scoring (lower drawdown is better)
            if abs(max_drawdown) <= 0.05:  # Low drawdown
                scores["drawdown"] = 10
            elif abs(max_drawdown) <= 0.10:  # Moderate drawdown
                scores["drawdown"] = 8
            elif abs(max_drawdown) <= 0.20:  # Significant drawdown
                scores["drawdown"] = 6
            elif abs(max_drawdown) <= 0.30:  # Large drawdown
                scores["drawdown"] = 4
            else:  # Very large drawdown
                scores["drawdown"] = 2
            
            # Weighted average
            weights = {
                "rsi": 0.15,
                "moving_averages": 0.20,
                "trend": 0.25,
                "volatility": 0.15,
                "momentum": 0.15,
                "drawdown": 0.10
            }
            
            final_score = sum(scores[metric] * weights[metric] for metric in scores)
            
            return {
                "score": float(min(10, max(1, final_score))),
                "components": scores,
                "metrics": {
                    "rsi_14": float(rsi),
                    "price_vs_sma_50": float(price_vs_sma_50),
                    "price_vs_sma_200": float(price_vs_sma_200),
                    "trend_direction": trend_direction,
                    "volatility_30d": float(volatility * 100),
                    "return_1m": float(return_1m * 100),
                    "max_drawdown": float(max_drawdown * 100)
                }
            }
            
        except Exception as e:
            logger.error(f"Technical scoring failed: {e}")
            return {"score": 5.0, "error": str(e)}
    
    def score_1_to_10(self, 
                     valuation_data: Dict[str, Any],
                     quality_data: Dict[str, Any],
                     sentiment_data: Dict[str, Any],
                     technical_data: Dict[str, Any],
                     coverage_adjustment: float = 0.0) -> Dict[str, Any]:
        """
        Calculate comprehensive 1-10 investment score
        
        Args:
            valuation_data: Valuation analysis results
            quality_data: Financial quality metrics
            sentiment_data: Sentiment analysis results
            technical_data: Technical analysis results
            coverage_adjustment: Additional adjustment for coverage quality
            
        Returns:
            Final score with component breakdown
        """
        try:
            # Score each component
            valuation_score = self.score_valuation(valuation_data)
            quality_score = self.score_quality(quality_data)
            sentiment_score = self.score_sentiment(sentiment_data)
            technical_score = self.score_technicals(technical_data)
            
            # Get weights from config
            weights = self.config.get("scoring_weights", {})
            
            # Calculate weighted score
            component_scores = {
                "valuation": valuation_score.get("score", 5.0),
                "quality": quality_score.get("score", 5.0),
                "sentiment": sentiment_score.get("score", 5.0),
                "technicals": technical_score.get("score", 5.0)
            }
            
            weighted_score = (
                component_scores["valuation"] * weights.get("valuation", 0.40) +
                component_scores["quality"] * weights.get("quality", 0.25) +
                component_scores["sentiment"] * weights.get("sentiment", 0.20) +
                component_scores["technicals"] * weights.get("technicals", 0.15)
            )
            
            # Apply coverage adjustment
            final_score = weighted_score + coverage_adjustment
            
            # Ensure score is between 1 and 10
            final_score = max(1.0, min(10.0, final_score))
            
            # Calculate confidence based on data coverage
            confidence = self._calculate_scoring_confidence(
                valuation_data, quality_data, sentiment_data, technical_data
            )
            
            return {
                "score": float(final_score),  # Main score field
                "final_score": float(final_score),  # Legacy compatibility
                "components": {
                    "valuation": {
                        "score": component_scores["valuation"],
                        "weight": weights.get("valuation", 0.40),
                        "details": valuation_score
                    },
                    "quality": {
                        "score": component_scores["quality"],
                        "weight": weights.get("quality", 0.25),
                        "details": quality_score
                    },
                    "sentiment": {
                        "score": component_scores["sentiment"],
                        "weight": weights.get("sentiment", 0.20),
                        "details": sentiment_score
                    },
                    "technicals": {
                        "score": component_scores["technicals"],
                        "weight": weights.get("technicals", 0.15),
                        "details": technical_score
                    }
                },
                "confidence": float(confidence),
                "component_scores": component_scores,  # Legacy compatibility
                "component_weights": weights,
                "coverage_adjustment": float(coverage_adjustment),
                "component_details": {
                    "valuation": valuation_score,
                    "quality": quality_score,
                    "sentiment": sentiment_score,
                    "technicals": technical_score
                },
                "scoring_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Investment scoring failed: {e}")
            return {
                "score": 5.0,
                "final_score": 5.0,
                "confidence": 0.0,
                "error": str(e),
                "scoring_date": datetime.now().isoformat()
            }
    
    def _calculate_scoring_confidence(self, valuation_data: Dict[str, Any],
                                    quality_data: Dict[str, Any],
                                    sentiment_data: Dict[str, Any],
                                    technical_data: Dict[str, Any]) -> float:
        """
        Calculate confidence based on data coverage and quality (0-1 scale)
        """
        try:
            confidence_factors = []
            
            # Valuation data coverage
            val_coverage = 0.0
            if valuation_data.get("median_fv", 0) > 0:
                val_coverage += 0.4  # DCF available
            if valuation_data.get("implied_cagr", 0) > 0:
                val_coverage += 0.3  # Reverse DCF available
            if "valuation_methods" in valuation_data and len(valuation_data["valuation_methods"]) > 1:
                val_coverage += 0.3  # Multiple methods available
            confidence_factors.append(min(1.0, val_coverage))
            
            # Quality data coverage
            qual_coverage = 0.0
            ratios = quality_data.get("ratios_ttm", {})
            key_ratios = ["roe", "roa", "debt_to_equity", "current_ratio", "net_profit_margin"]
            available_ratios = sum(1 for ratio in key_ratios if ratios.get(ratio, 0) != 0)
            qual_coverage = available_ratios / len(key_ratios)
            confidence_factors.append(qual_coverage)
            
            # Sentiment data coverage
            sent_coverage = 0.0
            if sentiment_data.get("average_sentiment") is not None:
                sent_coverage += 0.5
            if sentiment_data.get("transcript_analysis", {}).get("overall_sentiment") is not None:
                sent_coverage += 0.5
            confidence_factors.append(sent_coverage)
            
            # Technical data coverage
            tech_coverage = 0.0
            indicators = technical_data.get("indicators", {})
            key_indicators = ["rsi_14", "price_vs_sma_50", "return_1m", "volatility_30d"]
            available_indicators = sum(1 for indicator in key_indicators 
                                     if indicators.get(indicator) is not None)
            tech_coverage = available_indicators / len(key_indicators)
            confidence_factors.append(tech_coverage)
            
            # Calculate weighted average confidence
            weights = [0.4, 0.25, 0.2, 0.15]  # Same as scoring weights
            overall_confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
            
            # Apply penalty for missing critical data
            if valuation_data.get("median_fv", 0) <= 0:
                overall_confidence *= 0.7  # Major penalty for missing valuation
            
            if not ratios:
                overall_confidence *= 0.8  # Penalty for missing financial ratios
            
            return max(0.1, min(1.0, overall_confidence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate scoring confidence: {e}")
            return 0.5  # Default neutral confidence


# Main scoring function for external use
def score_1_to_10(valuation_data: Dict[str, Any],
                 quality_data: Dict[str, Any], 
                 sentiment_data: Dict[str, Any],
                 technical_data: Dict[str, Any],
                 coverage_adjustment: float = 0.0) -> Dict[str, Any]:
    """
    Main function for comprehensive investment scoring
    
    Returns score from 1 (overvalued) to 10 (undervalued)
    """
    scorer = InvestmentScorer()
    return scorer.score_1_to_10(valuation_data, quality_data, sentiment_data, 
                               technical_data, coverage_adjustment)


if __name__ == "__main__":
    # Test scoring system
    import json
    
    # Sample data for testing
    sample_valuation = {
        "p_underv": 0.75,
        "gap": 0.25,
        "implied_cagr": 0.12
    }
    
    sample_quality = {
        "ratios_ttm": {
            "roe": 0.18,
            "roa": 0.12,
            "debt_to_equity": 0.25,
            "current_ratio": 1.8,
            "net_profit_margin": 0.15,
            "gross_profit_margin": 0.35,
            "interest_coverage": 8.5
        }
    }
    
    sample_sentiment = {
        "average_sentiment": 0.3,
        "transcript_analysis": {
            "overall_sentiment": 0.4,
            "uncertainty_index": 0.3
        }
    }
    
    sample_technical = {
        "indicators": {
            "rsi_14": 45,
            "price_vs_sma_50": 5.2,
            "price_vs_sma_200": 12.8,
            "trend_direction": "uptrend",
            "volatility_30d": 18.5,
            "return_1m": 8.2,
            "max_drawdown": -8.5
        }
    }
    
    # Test scoring
    result = score_1_to_10(sample_valuation, sample_quality, sample_sentiment, sample_technical)
    
    print("Investment Scoring Test Results:")
    print(json.dumps(result, indent=2))
