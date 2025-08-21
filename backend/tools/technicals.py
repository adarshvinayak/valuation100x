"""
Technical Analysis Indicators

Computes technical indicators from price data including moving averages,
RSI, returns, drawdowns, and volatility metrics.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def calculate_sma(prices: List[float], window: int) -> Optional[float]:
    """
    Calculate Simple Moving Average
    
    Args:
        prices: List of price values
        window: Number of periods for SMA
    
    Returns:
        SMA value or None if insufficient data
    """
    if len(prices) < window:
        return None
    
    return sum(prices[-window:]) / window


def calculate_volatility(prices: List[float], window: int) -> float:
    """
    Calculate annualized volatility from price list
    
    Args:
        prices: List of price values
        window: Number of periods
    
    Returns:
        Annualized volatility as percentage
    """
    if len(prices) < window:
        return 0.0
    
    # Calculate daily returns
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate standard deviation and annualize
    std_dev = np.std(returns)
    annualized_vol = std_dev * np.sqrt(252) * 100  # 252 trading days per year
    
    return annualized_vol


def compute_indicators(prices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute technical indicators from price data
    
    Args:
        prices: List of price dictionaries with date, close, high, low, volume
    
    Returns:
        Dictionary containing computed technical indicators
    """
    if not prices or len(prices) < 2:
        logger.warning("Insufficient price data for technical analysis")
        return {}
    
    try:
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(prices)
        
        # Ensure we have required columns
        required_cols = ['date', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error("Missing required columns in price data")
            return {}
        
        # Sort by date (oldest first for calculations)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Convert price columns to numeric
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid data
        df = df.dropna(subset=['close'])
        
        if len(df) < 2:
            logger.warning("Insufficient valid price data after cleaning")
            return {}
        
        # Calculate indicators
        indicators = {}
        
        # Current price info
        latest_price = df.iloc[-1]
        indicators['current_price'] = float(latest_price['close'])
        indicators['current_date'] = latest_price['date'].strftime('%Y-%m-%d')
        
        # Moving averages
        if len(df) >= 50:
            indicators['sma_50'] = float(df['close'].rolling(window=50).mean().iloc[-1])
            indicators['price_vs_sma_50'] = (indicators['current_price'] / indicators['sma_50'] - 1) * 100
        else:
            indicators['sma_50'] = None
            indicators['price_vs_sma_50'] = None
        
        if len(df) >= 200:
            indicators['sma_200'] = float(df['close'].rolling(window=200).mean().iloc[-1])
            indicators['price_vs_sma_200'] = (indicators['current_price'] / indicators['sma_200'] - 1) * 100
        else:
            indicators['sma_200'] = None
            indicators['price_vs_sma_200'] = None
        
        # RSI (14-day)
        if len(df) >= 15:
            indicators['rsi_14'] = calculate_rsi(df['close'], 14)
        else:
            indicators['rsi_14'] = None
        
        # Price returns
        indicators.update(calculate_returns(df))
        
        # Drawdown analysis
        indicators.update(calculate_drawdown(df))
        
        # Volatility (enhanced)
        indicators.update(calculate_volatility(df))
        
        # Enhanced returns
        indicators.update(calculate_enhanced_returns(df))
        
        # Support and resistance (simple version)
        indicators.update(calculate_support_resistance(df))
        
        # Volume analysis
        if 'volume' in df.columns:
            indicators.update(calculate_volume_indicators(df))
        
        # Trend analysis
        indicators.update(calculate_trend_indicators(df))
        
        indicators['calculation_date'] = datetime.now().isoformat()
        indicators['data_points'] = len(df)
        
        # Return None for insufficient data instead of empty dict
        if len(df) < 2:
            return {k: None for k in indicators.keys()}
        
        return indicators
        
    except Exception as e:
        logger.error(f"Failed to compute technical indicators: {e}")
        return {}


def calculate_rsi(prices: pd.Series, window: int = 14) -> Optional[float]:
    """Calculate Relative Strength Index"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1])
    except Exception as e:
        logger.warning(f"Failed to calculate RSI: {e}")
        return None


def calculate_returns(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate various return periods"""
    returns = {}
    
    try:
        closes = df['close'].values
        dates = df['date']
        
        # 1-month return (approximately 22 trading days)
        if len(closes) >= 22:
            returns['return_1m'] = (closes[-1] / closes[-22] - 1) * 100
        
        # 3-month return (approximately 66 trading days)
        if len(closes) >= 66:
            returns['return_3m'] = (closes[-1] / closes[-66] - 1) * 100
        
        # 6-month return (approximately 132 trading days)
        if len(closes) >= 132:
            returns['return_6m'] = (closes[-1] / closes[-132] - 1) * 100
        
        # 1-year return (approximately 252 trading days)
        if len(closes) >= 252:
            returns['return_1y'] = (closes[-1] / closes[-252] - 1) * 100
        
        # Year-to-date return
        current_year = dates.iloc[-1].year
        ytd_start = df[df['date'].dt.year == current_year]
        if len(ytd_start) > 0:
            ytd_start_price = ytd_start.iloc[0]['close']
            returns['return_ytd'] = (closes[-1] / ytd_start_price - 1) * 100
        
    except Exception as e:
        logger.warning(f"Failed to calculate returns: {e}")
    
    return returns


def calculate_drawdown(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate drawdown metrics"""
    drawdown_metrics = {}
    
    try:
        closes = df['close'].values
        
        # Calculate rolling maximum (peak)
        rolling_max = pd.Series(closes).expanding().max()
        
        # Calculate drawdown as percentage from peak
        drawdown = (pd.Series(closes) / rolling_max - 1) * 100
        
        # Current drawdown
        drawdown_metrics['current_drawdown'] = float(drawdown.iloc[-1])
        
        # Maximum drawdown
        drawdown_metrics['max_drawdown'] = float(drawdown.min())
        
        # Days since peak
        peak_idx = rolling_max.idxmax()
        if peak_idx < len(df) - 1:
            drawdown_metrics['days_from_peak'] = len(df) - 1 - peak_idx
        else:
            drawdown_metrics['days_from_peak'] = 0
        
    except Exception as e:
        logger.warning(f"Failed to calculate drawdown: {e}")
    
    return drawdown_metrics


def calculate_volatility(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate enhanced volatility metrics"""
    volatility_metrics = {}
    
    try:
        # Daily returns
        daily_returns = df['close'].pct_change().dropna()
        
        # Realized volatility (annualized)
        if len(daily_returns) >= 30:
            # 30-day realized volatility
            vol_30d = daily_returns.tail(30).std() * np.sqrt(252) * 100
            volatility_metrics['realized_volatility_30d'] = float(vol_30d)
        else:
            volatility_metrics['realized_volatility_30d'] = None
        
        if len(daily_returns) >= 90:
            # 90-day realized volatility
            vol_90d = daily_returns.tail(90).std() * np.sqrt(252) * 100
            volatility_metrics['realized_volatility_90d'] = float(vol_90d)
        else:
            volatility_metrics['realized_volatility_90d'] = None
        
        # Legacy naming for backward compatibility
        volatility_metrics['volatility_30d'] = volatility_metrics['realized_volatility_30d']
        volatility_metrics['volatility_90d'] = volatility_metrics['realized_volatility_90d']
        
        # Average true range (if high/low data available)
        if all(col in df.columns for col in ['high', 'low']):
            atr = calculate_atr(df)
            if atr is not None:
                volatility_metrics['atr_14'] = atr
        
    except Exception as e:
        logger.warning(f"Failed to calculate volatility: {e}")
    
    return volatility_metrics

def calculate_enhanced_returns(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate enhanced return metrics including 1y max drawdown"""
    returns_metrics = {}
    
    try:
        closes = df['close'].values
        dates = df['date']
        
        # 1-month return (approximately 22 trading days)
        if len(closes) >= 22:
            returns_metrics['return_1m'] = (closes[-1] / closes[-22] - 1) * 100
        else:
            returns_metrics['return_1m'] = None
        
        # 3-month return (approximately 66 trading days)
        if len(closes) >= 66:
            returns_metrics['return_3m'] = (closes[-1] / closes[-66] - 1) * 100
        else:
            returns_metrics['return_3m'] = None
        
        # 6-month return (approximately 132 trading days)
        if len(closes) >= 132:
            returns_metrics['return_6m'] = (closes[-1] / closes[-132] - 1) * 100
        else:
            returns_metrics['return_6m'] = None
        
        # 1-year return (approximately 252 trading days)
        if len(closes) >= 252:
            returns_metrics['return_1y'] = (closes[-1] / closes[-252] - 1) * 100
        else:
            returns_metrics['return_1y'] = None
        
        # Max drawdown over 1 year
        if len(closes) >= 252:
            year_closes = closes[-252:]
            returns_metrics['max_drawdown_1y'] = float(calculate_max_drawdown_period(year_closes))
        else:
            returns_metrics['max_drawdown_1y'] = None
        
        # Year-to-date return
        current_year = dates.iloc[-1].year
        ytd_start = df[df['date'].dt.year == current_year]
        if len(ytd_start) > 0:
            ytd_start_price = ytd_start.iloc[0]['close']
            returns_metrics['return_ytd'] = (closes[-1] / ytd_start_price - 1) * 100
        else:
            returns_metrics['return_ytd'] = None
        
    except Exception as e:
        logger.warning(f"Failed to calculate enhanced returns: {e}")
    
    return returns_metrics

def calculate_max_drawdown_period(prices: np.ndarray) -> float:
    """Calculate maximum drawdown for a given price series"""
    try:
        # Calculate rolling maximum (peak)
        rolling_max = pd.Series(prices).expanding().max()
        
        # Calculate drawdown as percentage from peak
        drawdown = (pd.Series(prices) / rolling_max - 1) * 100
        
        # Return maximum (most negative) drawdown
        return float(drawdown.min())
        
    except Exception as e:
        logger.warning(f"Failed to calculate max drawdown: {e}")
        return 0.0


def calculate_atr(df: pd.DataFrame, window: int = 14) -> Optional[float]:
    """Calculate Average True Range"""
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return float(atr.iloc[-1])
    except Exception as e:
        logger.warning(f"Failed to calculate ATR: {e}")
        return None


def calculate_support_resistance(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate basic support and resistance levels"""
    levels = {}
    
    try:
        # Use recent 50 days or all available data
        recent_df = df.tail(min(50, len(df)))
        
        # Simple support/resistance based on recent highs and lows
        levels['resistance_level'] = float(recent_df['high'].max())
        levels['support_level'] = float(recent_df['low'].min())
        
        # Distance from current price
        current_price = df.iloc[-1]['close']
        levels['distance_to_resistance'] = ((levels['resistance_level'] / current_price) - 1) * 100
        levels['distance_to_support'] = ((current_price / levels['support_level']) - 1) * 100
        
    except Exception as e:
        logger.warning(f"Failed to calculate support/resistance: {e}")
    
    return levels


def calculate_volume_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate volume-based indicators"""
    volume_indicators = {}
    
    try:
        if 'volume' not in df.columns:
            return {}
        
        volume = df['volume']
        
        # Average volume
        if len(volume) >= 30:
            avg_volume_30d = volume.tail(30).mean()
            current_volume = volume.iloc[-1]
            volume_indicators['avg_volume_30d'] = float(avg_volume_30d)
            volume_indicators['current_vs_avg_volume'] = (current_volume / avg_volume_30d - 1) * 100
        
        # Volume trend (increasing/decreasing)
        if len(volume) >= 10:
            recent_avg = volume.tail(5).mean()
            previous_avg = volume.tail(10).head(5).mean()
            volume_indicators['volume_trend'] = (recent_avg / previous_avg - 1) * 100
        
    except Exception as e:
        logger.warning(f"Failed to calculate volume indicators: {e}")
    
    return volume_indicators


def calculate_trend_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate trend indicators"""
    trend_indicators = {}
    
    try:
        closes = df['close']
        
        # Simple trend based on recent slope
        if len(closes) >= 20:
            # Linear regression on recent 20 days
            recent_closes = closes.tail(20).values
            x = np.arange(len(recent_closes))
            slope, _ = np.polyfit(x, recent_closes, 1)
            
            # Normalize slope as percentage per day
            avg_price = recent_closes.mean()
            trend_indicators['trend_slope'] = (slope / avg_price) * 100
            
            # Classify trend
            if abs(slope / avg_price) < 0.001:  # Less than 0.1% per day
                trend_indicators['trend_direction'] = 'sideways'
            elif slope > 0:
                trend_indicators['trend_direction'] = 'uptrend'
            else:
                trend_indicators['trend_direction'] = 'downtrend'
        
        # Consecutive up/down days
        daily_changes = closes.diff()
        if len(daily_changes) > 1:
            # Count consecutive up days
            up_streak = 0
            down_streak = 0
            
            for change in reversed(daily_changes.dropna().tail(10)):
                if change > 0:
                    up_streak += 1
                    break
                elif change < 0:
                    down_streak += 1
                else:
                    break
            
            trend_indicators['consecutive_up_days'] = up_streak
            trend_indicators['consecutive_down_days'] = down_streak
        
    except Exception as e:
        logger.warning(f"Failed to calculate trend indicators: {e}")
    
    return trend_indicators


if __name__ == "__main__":
    # Test with sample data
    import random
    from datetime import timedelta
    
    # Generate sample price data
    base_date = datetime.now() - timedelta(days=300)
    base_price = 100.0
    
    sample_prices = []
    for i in range(300):
        date = base_date + timedelta(days=i)
        # Random walk with slight upward bias
        change = random.normalvariate(0.001, 0.02)  # 0.1% mean, 2% std
        base_price *= (1 + change)
        
        sample_prices.append({
            'date': date.strftime('%Y-%m-%d'),
            'open': base_price * random.uniform(0.99, 1.01),
            'high': base_price * random.uniform(1.00, 1.03),
            'low': base_price * random.uniform(0.97, 1.00),
            'close': base_price,
            'volume': random.randint(1000000, 10000000)
        })
    
    # Test the indicators
    indicators = compute_indicators(sample_prices)
    
    print("Technical Indicators Test:")
    print("=" * 40)
    for key, value in indicators.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
