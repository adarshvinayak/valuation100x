"""
Local embeddings using sentence-transformers - LlamaIndex Compatible
Runs on your server/PC - no external API calls, completely FREE!
"""
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import time

try:
    from llama_index.core.embeddings import BaseEmbedding
    from pydantic import Field
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    BaseEmbedding = object  # Fallback
    def Field(**kwargs):
        return None

logger = logging.getLogger(__name__)


class LocalSentenceTransformerEmbedding(BaseEmbedding):
    """
    Local embeddings using sentence-transformers - LlamaIndex Compatible
    - Runs on your PC/server (no external API calls)
    - Completely FREE (no API costs)
    - Fast inference after model download
    - Compatible with hosted backends
    """
    
    # Declare Pydantic fields
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Sentence-transformers model name")
    cache_folder: str = Field(default="", description="Where to cache the model files")  
    device: str = Field(default="cpu", description="Device to run on (cpu or cuda)")
    
    # Internal fields
    model: Optional[Any] = Field(default=None, description="The actual sentence transformer model")
    embedding_dim: Optional[int] = Field(default=None, description="Embedding dimension")
    model_loaded: bool = Field(default=False, description="Whether model is loaded")
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_folder: Optional[str] = None,
                 device: str = "cpu",
                 **kwargs):
        """
        Initialize local embedding model
        
        Args:
            model_name: Hugging Face model name
            cache_folder: Where to cache the model (defaults to ./models)
            device: "cpu" or "cuda" (auto-detected if available)
        """
        # Set cache folder before calling super
        if cache_folder is None:
            cache_folder = os.path.join(os.getcwd(), "models", "embeddings")
        
        super().__init__(
            model_name=model_name,
            cache_folder=cache_folder,
            device=device,
            model=None,
            embedding_dim=None,
            model_loaded=False,
            **kwargs
        )
        
        # Ensure cache directory exists
        Path(self.cache_folder).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🆓 Local Embedding Setup:")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Cache: {self.cache_folder}")
        logger.info(f"   Device: {device}")
    
    def _load_model(self):
        """Lazy load the model when first needed"""
        if self.model_loaded:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required for local embeddings.\n"
                "Install with: pip install sentence-transformers"
            )
        
        logger.info(f"🔮 Loading local embedding model: {self.model_name} (this may take 10-15 seconds)")
        start_time = time.time()
        
        # Load model (downloads on first use, then cached)
        self.model = SentenceTransformer(
            self.model_name, 
            cache_folder=self.cache_folder,
            device=self.device
        )
        
        # Get model info
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        load_time = time.time() - start_time
        
        logger.info(f"✅ Local embedding model loaded successfully!")
        logger.info(f"   Load time: {load_time:.1f}s")
        logger.info(f"   Dimensions: {self.embedding_dim}")
        logger.info(f"   Max sequence length: {self.model.max_seq_length}")
        logger.info(f"   💰 Cost: $0.00 (FREE!)")
        
        self.model_loaded = True
    
    @classmethod
    def class_name(cls) -> str:
        """Return class name for LlamaIndex"""
        return "LocalSentenceTransformerEmbedding"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query (required by BaseEmbedding)"""
        return self.get_text_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text (required by BaseEmbedding)"""
        self._load_model()
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (required by BaseEmbedding)"""
        return self.get_text_embedding_batch(texts)
    
    # Async methods required by BaseEmbedding
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of _get_query_embedding"""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of _get_text_embedding"""
        return self._get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async version of _get_text_embeddings"""
        return self._get_text_embeddings(texts)
    
    # Public API methods
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        return self._get_text_embedding(text)
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query"""
        return self._get_query_embedding(query)
    
    def get_text_embedding_batch(self, texts: List[str], 
                                show_progress: bool = False) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        self._load_model()
        
        # Process in batches to manage memory efficiently
        batch_size = 32  # Adjust based on available RAM
        all_embeddings = []
        
        if show_progress:
            logger.info(f"🔮 Processing {len(texts)} texts in batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if show_progress and len(texts) > batch_size:
                batch_num = (i // batch_size) + 1
                total_batches = (len(texts) + batch_size - 1) // batch_size
                logger.info(f"   Batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            # Get embeddings for this batch
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,  # Disable internal progress bar
                convert_to_numpy=True,
                batch_size=min(32, len(batch))  # Internal batch size
            )
            
            all_embeddings.extend(batch_embeddings.tolist())
        
        return all_embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        self._load_model()
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.model.max_seq_length,
            "cache_folder": self.cache_folder,
            "device": self.device,
            "cost_per_token": 0.0,  # FREE!
            "rate_limit": "unlimited",
            "provider": "local",
            "api_calls_required": False
        }


# Model configurations with different trade-offs
EMBEDDING_MODEL_OPTIONS = {
    "tiny": {
        "model": "all-MiniLM-L12-v2",
        "dimensions": 384,
        "speed": "very fast",
        "quality": "good",
        "size_mb": 120,
        "description": "Fastest option, good for development"
    },
    "fast": {
        "model": "all-MiniLM-L6-v2",
        "dimensions": 384, 
        "speed": "very fast",
        "quality": "good",
        "size_mb": 90,
        "description": "Best balance of speed and resources (RECOMMENDED)"
    },
    "balanced": {
        "model": "all-mpnet-base-v2",
        "dimensions": 768,
        "speed": "fast", 
        "quality": "high",
        "size_mb": 420,
        "description": "Good balance of speed and quality"
    },
    "quality": {
        "model": "sentence-transformers/all-roberta-large-v1",
        "dimensions": 1024,
        "speed": "medium",
        "quality": "very high",
        "size_mb": 1350,
        "description": "Best quality, requires more resources"
    },
    "financial": {
        "model": "sentence-transformers/paraphrase-distilroberta-base-v1",
        "dimensions": 768,
        "speed": "fast",
        "quality": "high",
        "size_mb": 290,
        "description": "Optimized for financial/business documents"
    }
}


class LocalEmbeddingFactory:
    """Factory for creating local embedding models with different configurations"""
    
    @staticmethod
    def create_model(model_type: str = "fast", 
                    cache_folder: Optional[str] = None,
                    device: str = "cpu") -> LocalSentenceTransformerEmbedding:
        """
        Create local embedding model with specified configuration
        
        Args:
            model_type: One of "tiny", "fast", "balanced", "quality", "financial"
            cache_folder: Custom cache location (optional)
            device: "cpu" or "cuda"
        """
        if model_type not in EMBEDDING_MODEL_OPTIONS:
            available = list(EMBEDDING_MODEL_OPTIONS.keys())
            raise ValueError(f"Unknown model type: {model_type}. Choose from: {available}")
        
        config = EMBEDDING_MODEL_OPTIONS[model_type]
        model_name = config["model"]
        
        logger.info(f"🆓 Creating local embedding model: {model_type}")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Quality: {config['quality']}")
        logger.info(f"   Speed: {config['speed']}")
        logger.info(f"   Size: ~{config['size_mb']}MB")
        logger.info(f"   Description: {config['description']}")
        
        return LocalSentenceTransformerEmbedding(
            model_name=model_name,
            cache_folder=cache_folder,
            device=device
        )
    
    @staticmethod
    def list_available_models() -> Dict[str, Dict[str, Any]]:
        """List all available model configurations"""
        return EMBEDDING_MODEL_OPTIONS
    
    @staticmethod
    def recommend_model(priority: str = "speed") -> str:
        """
        Recommend a model based on priority
        
        Args:
            priority: "speed", "quality", "balanced", or "resource_efficient"
        """
        recommendations = {
            "speed": "fast",
            "quality": "quality", 
            "balanced": "balanced",
            "resource_efficient": "tiny",
            "financial": "financial"
        }
        
        return recommendations.get(priority, "fast")


def test_local_embeddings(model_type: str = "fast", num_test_texts: int = 5):
    """Test local embeddings with sample financial texts"""
    print(f"🧪 Testing Local Embeddings (Model: {model_type})")
    print("=" * 60)
    
    # Sample financial texts for testing
    test_texts = [
        "Apple Inc. reported strong quarterly earnings with revenue growth of 15%.",
        "The Federal Reserve announced a 0.25% interest rate increase.",
        "Tesla's stock price surged after better-than-expected delivery numbers.",
        "Microsoft Azure cloud revenue continued its upward trajectory.",
        "Goldman Sachs upgraded their price target for NVIDIA stock.",
        "The S&P 500 index reached a new all-time high this week.",
        "Warren Buffett's Berkshire Hathaway increased its Apple holdings.",
        "Cryptocurrency markets showed high volatility amid regulatory uncertainty."
    ][:num_test_texts]
    
    try:
        # Create model
        print(f"🔮 Creating {model_type} model...")
        embedder = LocalEmbeddingFactory.create_model(model_type)
        
        # Show model info
        info = embedder.get_model_info()
        print(f"📊 Model Info:")
        print(f"    Dimensions: {info['embedding_dimension']}")
        print(f"    Max length: {info['max_sequence_length']}")
        print(f"    Cost: ${info['cost_per_token']:.2f} (FREE!)")
        print()
        
        # Test embeddings
        print(f"⚡ Processing {len(test_texts)} test texts...")
        start_time = time.time()
        
        embeddings = embedder.get_text_embedding_batch(test_texts, show_progress=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Results
        print(f"✅ Results:")
        print(f"    Texts processed: {len(embeddings)}")
        print(f"    Embedding dimensions: {len(embeddings[0])}")
        print(f"    Processing time: {processing_time:.2f}s")
        print(f"    Speed: {len(test_texts)/processing_time:.1f} texts/second")
        print(f"    Total cost: $0.00 (FREE!)")
        
        # Show sample embedding
        print(f"🔢 Sample embedding (first 5 dimensions): {embeddings[0][:5]}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Solution: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Show available models
    print("🚀 Available Local Embedding Models:")
    print("=" * 50)
    factory = LocalEmbeddingFactory()
    for name, config in factory.list_available_models().items():
        print(f"{name:12} | {config['dimensions']:4}D | {config['size_mb']:4}MB | {config['description']}")
    print()
    
    # Test the recommended model
    recommended = factory.recommend_model("speed")
    print(f"🎯 Testing recommended model: {recommended}")
    test_local_embeddings(recommended)
