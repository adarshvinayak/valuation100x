#!/usr/bin/env python3
"""
Create FAISS indexes for popular tickers to improve analysis performance

This script creates vector indexes for the most commonly analyzed stocks
so users don't see "No FAISS index found" warnings.
"""
import asyncio
import logging
from pathlib import Path

from ingestion.enhanced_embed_index_v2 import EnhancedVectorIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Popular tickers to pre-index
POPULAR_TICKERS = [
    "AAPL",  # Apple
    "TSLA",  # Tesla  
    "MSFT",  # Microsoft
    "NVDA",  # NVIDIA
    "GOOGL", # Google
    "AMZN",  # Amazon
    "META",  # Meta
    "NFLX",  # Netflix
    "AMD",   # AMD
    "INTC"   # Intel
]

async def create_ticker_index(ticker: str, embedding_provider: str = "openai"):
    """Create FAISS index for a specific ticker"""
    try:
        logger.info(f"🔍 Creating index for {ticker}...")
        
        # Check if index already exists
        index_path = Path(f"data/index/{ticker}/faiss_index")
        if index_path.exists():
            logger.info(f"✅ Index for {ticker} already exists, skipping")
            return True
        
        # Create the indexer
        indexer = EnhancedVectorIndexer(
            embedding_provider=embedding_provider,
            verbose=True
        )
        
        # Create index
        success = await indexer.create_enhanced_index(ticker)
        
        if success:
            logger.info(f"✅ Successfully created index for {ticker}")
            return True
        else:
            logger.error(f"❌ Failed to create index for {ticker}")
            return False
            
    except Exception as e:
        logger.error(f"💥 Error creating index for {ticker}: {e}")
        return False

async def create_all_indexes():
    """Create indexes for all popular tickers"""
    logger.info(f"🚀 Creating FAISS indexes for {len(POPULAR_TICKERS)} popular tickers...")
    
    # Determine embedding provider (use OpenAI for Railway, local for development)
    import os
    is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
    embedding_provider = "openai" if is_railway else "local"
    
    logger.info(f"🧠 Using {embedding_provider} embeddings")
    
    success_count = 0
    failed_tickers = []
    
    for ticker in POPULAR_TICKERS:
        success = await create_ticker_index(ticker, embedding_provider)
        if success:
            success_count += 1
        else:
            failed_tickers.append(ticker)
    
    # Summary
    logger.info(f"\n📊 Index Creation Summary:")
    logger.info(f"✅ Successfully created: {success_count}/{len(POPULAR_TICKERS)} indexes")
    
    if failed_tickers:
        logger.warning(f"❌ Failed tickers: {', '.join(failed_tickers)}")
    else:
        logger.info(f"🎉 All indexes created successfully!")
    
    return success_count, failed_tickers

if __name__ == "__main__":
    asyncio.run(create_all_indexes())
