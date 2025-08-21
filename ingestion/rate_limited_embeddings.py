"""
Rate-limited OpenAI embeddings to prevent 429 errors during large document processing
"""
import asyncio
import time
from typing import List, Optional
import logging
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)


class RateLimitedOpenAIEmbedding(OpenAIEmbedding):
    """OpenAI embedding client with built-in rate limiting to prevent 429 errors"""
    
    def __init__(self, 
                 api_key: str,
                 model: str = "text-embedding-ada-002",
                 batch_size: int = 5,
                 delay_between_batches: float = 2.0,
                 max_retries: int = 5,
                 **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.max_retries = max_retries
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.delay_between_batches:
            wait_time = self.delay_between_batches - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def get_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings for a batch of texts with rate limiting"""
        if len(texts) <= self.batch_size:
            return self._get_embeddings_with_retry(texts, **kwargs)
        
        # Process in smaller batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.debug(f"Processing embedding batch {i//self.batch_size + 1} ({len(batch)} texts)")
            
            # Apply rate limiting before each batch (except the first)
            if i > 0:
                self._rate_limit()
            
            batch_embeddings = self._get_embeddings_with_retry(batch, **kwargs)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _get_embeddings_with_retry(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings with retry logic for rate limit errors"""
        for attempt in range(self.max_retries):
            try:
                return super().get_text_embedding_batch(texts, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff for rate limit errors
                        wait_time = (2 ** attempt) * self.delay_between_batches
                        logger.warning(f"Rate limit hit (attempt {attempt + 1}), waiting {wait_time:.1f}s before retry")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for rate limit error: {e}")
                        raise
                else:
                    # Non-rate-limit error, raise immediately
                    logger.error(f"Embedding error (non-rate-limit): {e}")
                    raise
        
        return []  # Should never reach here
    
    def get_text_embedding(self, text: str, **kwargs) -> List[float]:
        """Get embedding for a single text with rate limiting"""
        self._rate_limit()
        return self._get_embeddings_with_retry([text], **kwargs)[0]


async def create_rate_limited_embedding_model(api_key: str, 
                                             model: str = "text-embedding-ada-002",
                                             batch_size: int = 5,
                                             delay_between_batches: float = 2.0) -> RateLimitedOpenAIEmbedding:
    """Factory function to create a rate-limited embedding model"""
    return RateLimitedOpenAIEmbedding(
        api_key=api_key,
        model=model,
        batch_size=batch_size,
        delay_between_batches=delay_between_batches
    )

