"""
Sentiment Analysis Tools

Provides sentiment analysis using FinBERT model for financial text,
with fallback to hosted API or simple rule-based analysis.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union
import re
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Global model cache
_finbert_model = None
_finbert_tokenizer = None


def load_finbert_model():
    """Load FinBERT model and tokenizer"""
    global _finbert_model, _finbert_tokenizer
    
    if _finbert_model is not None:
        return _finbert_model, _finbert_tokenizer
    
    try:
        # Check if PyTorch is available first
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not found. Install with: pip install torch>=2.0.0")
            return None, None
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "ProsusAI/finbert"
        logger.info(f"Loading FinBERT model: {model_name}")
        
        _finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        logger.info("FinBERT model loaded successfully")
        return _finbert_model, _finbert_tokenizer
        
    except ImportError as e:
        logger.warning(f"Missing dependencies for FinBERT: {e}")
        logger.info("Install with: pip install torch>=2.0.0 transformers>=4.30.0")
        return None, None
    except Exception as e:
        logger.warning(f"Failed to load FinBERT model: {e}")
        return None, None


def finbert_sentiment_local(texts: List[str]) -> List[float]:
    """
    Analyze sentiment using local FinBERT model
    
    Args:
        texts: List of text strings to analyze
    
    Returns:
        List of sentiment scores [-1, 1] where -1 is very negative, 1 is very positive
    """
    if not texts:
        return []
    
    try:
        model, tokenizer = load_finbert_model()
        
        if model is None or tokenizer is None:
            logger.warning("FinBERT model not available, falling back to rule-based sentiment")
            return rule_based_sentiment(texts)
        
        import torch
        import torch.nn.functional as F
        
        sentiments = []
        
        for text in texts:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:500] + "..."
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                             padding=True, max_length=512)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
            
            # FinBERT outputs: [negative, neutral, positive]
            neg_prob = probabilities[0][0].item()
            neu_prob = probabilities[0][1].item()
            pos_prob = probabilities[0][2].item()
            
            # Convert to -1 to 1 scale
            # Weighted average: negative=-1, neutral=0, positive=1
            sentiment_score = (pos_prob * 1.0) + (neu_prob * 0.0) + (neg_prob * -1.0)
            
            sentiments.append(sentiment_score)
        
        logger.info(f"Analyzed sentiment for {len(texts)} texts using FinBERT")
        return sentiments
        
    except Exception as e:
        logger.error(f"FinBERT sentiment analysis failed: {e}")
        return rule_based_sentiment(texts)


def rule_based_sentiment(texts: List[str]) -> List[float]:
    """
    Simple rule-based sentiment analysis as fallback
    
    Args:
        texts: List of text strings to analyze
    
    Returns:
        List of sentiment scores [-1, 1]
    """
    # Define positive and negative financial keywords
    positive_keywords = [
        'strong', 'growth', 'profit', 'revenue', 'beat', 'exceed', 'outperform',
        'positive', 'bullish', 'buy', 'upgrade', 'increase', 'rise', 'gain',
        'improve', 'better', 'success', 'opportunity', 'optimistic', 'confident',
        'expansion', 'momentum', 'robust', 'solid', 'excellent', 'outstanding',
        'confidence', 'pleased', 'good', 'great', 'performance', 'prospects',
        'future', 'analyst', 'expectations', 'quarter', 'earnings', 'margin'
    ]
    
    negative_keywords = [
        'weak', 'decline', 'loss', 'miss', 'underperform', 'negative', 'bearish',
        'sell', 'downgrade', 'decrease', 'fall', 'drop', 'worsen', 'worse',
        'concern', 'risk', 'problem', 'challenge', 'pessimistic', 'uncertain',
        'contraction', 'slow', 'poor', 'bad', 'disappointing', 'warning',
        'concerns', 'declining', 'weak', 'guidance', 'headwinds', 'pressure',
        'affecting', 'significant', 'profitability'
    ]
    
    sentiments = []
    
    for text in texts:
        if not text or not isinstance(text, str):
            sentiments.append(0.0)
            continue
        
        text_lower = text.lower()
        
        # Count positive and negative words
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        
        # Enhanced scoring algorithm
        total_words = len(text_lower.split())
        if total_words == 0:
            sentiment = 0.0
        else:
            # More sensitive scoring with better scaling
            if pos_count > 0 or neg_count > 0:
                # Use ratio-based scoring for better sensitivity
                pos_ratio = pos_count / max(total_words, 1)
                neg_ratio = neg_count / max(total_words, 1)
                
                # Amplify the signal for short texts
                amplification = min(3.0, 20.0 / max(total_words, 5))
                
                pos_score = pos_ratio * amplification
                neg_score = neg_ratio * amplification
                
                # Calculate net sentiment
                net_sentiment = pos_score - neg_score
                
                # Apply sigmoid-like scaling for better distribution
                sentiment = max(-1.0, min(1.0, net_sentiment * 2))
            else:
                sentiment = 0.0
        
        sentiments.append(sentiment)
    
    logger.info(f"Analyzed sentiment for {len(texts)} texts using rule-based method")
    return sentiments


def finbert_sentiment(texts: Union[str, List[str]], 
                     use_local: bool = True) -> Union[float, List[float]]:
    """
    Main sentiment analysis function
    
    Args:
        texts: Single text string or list of text strings
        use_local: Whether to use local FinBERT model (True) or hosted API (False)
    
    Returns:
        Single sentiment score or list of scores [-1, 1]
    """
    # Handle single text input
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]
    
    if not texts:
        return [] if not single_input else 0.0
    
    try:
        if use_local:
            sentiments = finbert_sentiment_local(texts)
        else:
            # TODO: Implement hosted API fallback
            logger.warning("Hosted sentiment API not implemented, using local model")
            sentiments = finbert_sentiment_local(texts)
        
        return sentiments[0] if single_input else sentiments
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        fallback_sentiments = [0.0] * len(texts)
        return fallback_sentiments[0] if single_input else fallback_sentiments


def uncertainty_index(texts: List[str]) -> float:
    """
    Calculate uncertainty index from text content
    
    Args:
        texts: List of text strings to analyze
    
    Returns:
        Uncertainty score [0, 1] where 1 is high uncertainty
    """
    uncertainty_keywords = [
        'uncertain', 'uncertainty', 'unclear', 'volatile', 'volatility',
        'risk', 'risky', 'unpredictable', 'variable', 'unstable',
        'fluctuate', 'maybe', 'perhaps', 'might', 'could', 'possible',
        'potential', 'may', 'unclear', 'unknown', 'unsure', 'doubt',
        'question', 'challenge', 'difficult', 'complex', 'turbulent'
    ]
    
    if not texts:
        return 0.0
    
    try:
        total_uncertainty = 0.0
        total_words = 0
        
        for text in texts:
            if not text or not isinstance(text, str):
                continue
            
            text_lower = text.lower()
            words = text_lower.split()
            total_words += len(words)
            
            # Count uncertainty keywords
            uncertainty_count = sum(1 for word in uncertainty_keywords 
                                  if word in text_lower)
            
            total_uncertainty += uncertainty_count
        
        if total_words == 0:
            return 0.0
        
        # Normalize and scale
        uncertainty_ratio = total_uncertainty / total_words
        uncertainty_score = min(1.0, uncertainty_ratio * 20)  # Scale factor
        
        logger.info(f"Calculated uncertainty index: {uncertainty_score:.3f}")
        return uncertainty_score
        
    except Exception as e:
        logger.error(f"Failed to calculate uncertainty index: {e}")
        return 0.0


def analyze_transcript_sentiment(transcript: str) -> Dict[str, Any]:
    """
    Analyze sentiment of an earnings call transcript
    
    Args:
        transcript: Full transcript text
    
    Returns:
        Dictionary with sentiment analysis results
    """
    if not transcript:
        return {
            "overall_sentiment": 0.0,
            "uncertainty_index": 0.0,
            "segment_count": 0,
            "analysis_date": datetime.now().isoformat()
        }
    
    try:
        # Split transcript into segments (by paragraph breaks and sentences)
        # First try paragraph breaks
        segments = re.split(r'\n\s*\n', transcript)
        segments = [seg.strip() for seg in segments if seg.strip()]
        
        # If no paragraph breaks, split by sentences
        if len(segments) <= 1:
            segments = re.split(r'[.!?]+\s+', transcript)
            segments = [seg.strip() for seg in segments if seg.strip()]
        
        # Filter segments that are too short (but lower threshold)
        segments = [seg for seg in segments if len(seg.strip()) > 20]
        
        if not segments:
            segments = [transcript]  # Use full text if splitting fails
        
        # Analyze sentiment for each segment
        segment_sentiments = finbert_sentiment(segments)
        
        # Calculate overall metrics
        overall_sentiment = sum(segment_sentiments) / len(segment_sentiments)
        uncertainty = uncertainty_index(segments)
        
        # Find most positive and negative segments
        max_sentiment_idx = max(range(len(segment_sentiments)), 
                               key=lambda i: segment_sentiments[i])
        min_sentiment_idx = min(range(len(segment_sentiments)), 
                               key=lambda i: segment_sentiments[i])
        
        return {
            "overall_sentiment": overall_sentiment,
            "uncertainty_index": uncertainty,
            "segment_count": len(segments),
            "sentiment_std": float(np.std(segment_sentiments)) if len(segment_sentiments) > 1 else 0.0,
            "most_positive_segment": {
                "sentiment": segment_sentiments[max_sentiment_idx],
                "text": segments[max_sentiment_idx][:200] + "..." if len(segments[max_sentiment_idx]) > 200 else segments[max_sentiment_idx]
            },
            "most_negative_segment": {
                "sentiment": segment_sentiments[min_sentiment_idx],
                "text": segments[min_sentiment_idx][:200] + "..." if len(segments[min_sentiment_idx]) > 200 else segments[min_sentiment_idx]
            },
            "analysis_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze transcript sentiment: {e}")
        return {
            "overall_sentiment": 0.0,
            "uncertainty_index": 0.0,
            "segment_count": 0,
            "error": str(e),
            "analysis_date": datetime.now().isoformat()
        }


# For backward compatibility
def get_sentiment_score(text: str) -> float:
    """Simple function to get sentiment score for a single text"""
    return finbert_sentiment(text)


if __name__ == "__main__":
    # Test sentiment analysis
    test_texts = [
        "The company reported strong revenue growth and exceeded analyst expectations.",
        "We are concerned about declining margins and increased competition in the market.",
        "The outlook remains uncertain due to volatile market conditions and regulatory challenges.",
        "Management is confident about the expansion strategy and future opportunities."
    ]
    
    print("Sentiment Analysis Test:")
    print("=" * 50)
    
    sentiments = finbert_sentiment(test_texts)
    
    for i, (text, sentiment) in enumerate(zip(test_texts, sentiments)):
        print(f"\n{i+1}. Text: {text[:60]}...")
        print(f"   Sentiment: {sentiment:.3f}")
    
    # Test uncertainty index
    uncertainty = uncertainty_index(test_texts)
    print(f"\nOverall Uncertainty Index: {uncertainty:.3f}")
    
    # Test transcript analysis
    sample_transcript = " ".join(test_texts * 3)  # Simulate longer transcript
    transcript_analysis = analyze_transcript_sentiment(sample_transcript)
    print(f"\nTranscript Analysis:")
    print(f"Overall Sentiment: {transcript_analysis['overall_sentiment']:.3f}")
    print(f"Uncertainty Index: {transcript_analysis['uncertainty_index']:.3f}")
    print(f"Segments: {transcript_analysis['segment_count']}")
