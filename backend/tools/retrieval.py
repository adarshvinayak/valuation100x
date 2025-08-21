"""
Document Retrieval Tool

Loads per-ticker vector indexes and provides semantic search functionality.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import json

import faiss
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from ingestion.local_embeddings import LocalEmbeddingFactory
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Handles document retrieval from vector indexes"""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 base_index_path: str = "data/index",
                 embedding_provider: str = "auto"):
        self.base_index_path = Path(base_index_path)
        self.loaded_indexes = {}  # Cache loaded indexes
        self.openai_api_key = openai_api_key
        self.embedding_provider = embedding_provider
        
        # Initialize embedding model based on provider
        self._setup_embedding_model()
    
    def _setup_embedding_model(self):
        """Setup embedding model based on provider or auto-detect from index metadata"""
        if self.embedding_provider == "local":
            # Use local embeddings
            Settings.embed_model = LocalEmbeddingFactory.create_model("fast")
            logger.info("Using local embeddings for retrieval")
        elif self.embedding_provider == "openai":
            # Use OpenAI embeddings
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            Settings.embed_model = OpenAIEmbedding(
                api_key=self.openai_api_key,
                model="text-embedding-ada-002"
            )
            logger.info("Using OpenAI embeddings for retrieval")
        else:
            # Auto-detect: Try local first, fallback to OpenAI
            try:
                Settings.embed_model = LocalEmbeddingFactory.create_model("fast")
                logger.info("Auto-detected: Using local embeddings for retrieval")
            except Exception:
                if self.openai_api_key:
                    Settings.embed_model = OpenAIEmbedding(
                        api_key=self.openai_api_key,
                        model="text-embedding-ada-002"
                    )
                    logger.info("Auto-detected: Using OpenAI embeddings for retrieval")
                else:
                    logger.warning("No embedding model available - retrieval may fail")
    
    def _detect_index_embedding_provider(self, ticker: str) -> str:
        """Detect embedding provider from index metadata"""
        metadata_path = self.base_index_path / ticker.upper() / "index_metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Check for direct embedding provider info
                if "embedding_provider" in metadata:
                    provider = metadata["embedding_provider"]
                    if provider in ["local", "openai"]:
                        return provider
                
                # Fallback: Check for embedding provider in indexing_stats
                if "indexing_stats" in metadata:
                    provider = metadata["indexing_stats"].get("embedding_provider", "unknown")
                    if provider in ["local", "openai"]:
                        return provider
                
                # Check embedding model name
                embedding_model = metadata.get("embedding_model", "") or metadata.get("indexing_stats", {}).get("embedding_model", "")
                if "MiniLM" in embedding_model or "fast" in embedding_model or "sentence-transformers" in embedding_model:
                    return "local"
                elif "text-embedding" in embedding_model or "ada" in embedding_model:
                    return "openai"
                    
            except Exception as e:
                logger.debug(f"Could not read metadata for {ticker}: {e}")
        
        return "unknown"
    
    def _load_manual_index(self, ticker: str, faiss_path: Path, chunk_metadata_path: Path) -> Optional[VectorStoreIndex]:
        """Load manually created FAISS index with separate metadata"""
        try:
            # Load FAISS index
            faiss_index = faiss.read_index(str(faiss_path))
            
            # Load chunk metadata
            with open(chunk_metadata_path, 'r', encoding='utf-8') as f:
                chunk_metadata = json.load(f)
            
            # Create nodes from metadata
            from llama_index.core.schema import TextNode
            nodes = []
            for chunk_data in chunk_metadata:
                node = TextNode(
                    text=chunk_data["text"],
                    metadata=chunk_data.get("metadata", {}),
                    node_id=chunk_data.get("node_id", f"chunk_{chunk_data['index']}")
                )
                nodes.append(node)
            
            # Create vector store with pre-populated FAISS index
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            
            # Create index from nodes and vector store
            # This bypasses the text storage requirement
            index = VectorStoreIndex(nodes=nodes, vector_store=vector_store)
            
            # Cache the loaded index
            self.loaded_indexes[ticker] = index
            
            logger.info(f"✅ Successfully loaded manual index for {ticker} ({len(nodes)} chunks)")
            return index
            
        except Exception as e:
            logger.error(f"❌ Failed to load manual index for {ticker}: {e}")
            return None
    
    def _load_index(self, ticker: str) -> Optional[VectorStoreIndex]:
        """Load vector index for a ticker with auto-detection of embedding provider"""
        if ticker in self.loaded_indexes:
            return self.loaded_indexes[ticker]
        
        ticker_index_dir = self.base_index_path / ticker.upper()
        faiss_path = ticker_index_dir / "faiss_index"
        chunk_metadata_path = ticker_index_dir / "chunk_metadata.json"
        
        if not faiss_path.exists():
            logger.error(f"No FAISS index found for {ticker} at {faiss_path}")
            return None
        
        # Auto-detect embedding provider if in auto mode
        if self.embedding_provider == "auto":
            detected_provider = self._detect_index_embedding_provider(ticker)
            if detected_provider != "unknown":
                logger.info(f"Detected {detected_provider} embeddings for {ticker} index")
                # Temporarily update embedding model
                if detected_provider == "local":
                    Settings.embed_model = LocalEmbeddingFactory.create_model("fast")
                elif detected_provider == "openai" and self.openai_api_key:
                    Settings.embed_model = OpenAIEmbedding(
                        api_key=self.openai_api_key,
                        model="text-embedding-ada-002"
                    )
        
        try:
            # Check if this is our new manual index format
            if chunk_metadata_path.exists():
                # Load our manual FAISS index with separate metadata
                return self._load_manual_index(ticker, faiss_path, chunk_metadata_path)
            else:
                # Try to load traditional LlamaIndex format
                faiss_index = faiss.read_index(str(faiss_path))
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                
                # Create VectorStoreIndex
                index = VectorStoreIndex.from_vector_store(vector_store)
                
                # Cache the loaded index
                self.loaded_indexes[ticker] = index
                
                logger.info(f"Successfully loaded traditional index for {ticker}")
                return index
            
        except Exception as e:
            logger.error(f"Failed to load index for {ticker}: {e}")
            # If auto-detection failed, try clearing cache and recreating
            if self.embedding_provider == "auto":
                logger.info(f"Index loading failed for {ticker}, may need rebuilding with correct embeddings")
            return None
    
    def _format_retrieval_result(self, node: NodeWithScore, ticker: str) -> Dict[str, Any]:
        """Format a retrieval result"""
        metadata = node.node.metadata or {}
        
        # Extract relevant information
        result = {
            "text": node.node.text,
            "score": float(node.score) if node.score is not None else 0.0,
            "ticker": ticker,
            "source_path": metadata.get("source_path", "unknown"),
            "form_type": metadata.get("form_type", "unknown"),
            "filed_date": metadata.get("filed_date", "unknown"),
            "node_id": node.node.node_id
        }
        
        # Add date if available
        if "filed_date" in metadata:
            result["date"] = metadata["filed_date"]
        
        return result
    
    async def retrieve_docs(self, 
                          ticker: str, 
                          query: str, 
                          k: int = 8,
                          similarity_top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            ticker: Stock ticker symbol
            query: Search query
            k: Number of documents to retrieve
            similarity_top_k: Override for similarity search (defaults to k)
            
        Returns:
            List of retrieval results with text, score, and metadata
        """
        # Add progress indication for user feedback
        logger.info(f"🔍 Searching SEC documents for: {query[:50]}...")
        
        # Load index for ticker
        index = self._load_index(ticker)
        if index is None:
            logger.error(f"Could not load index for {ticker}")
            return []
        
        try:
            # Create retriever
            retriever = index.as_retriever(
                similarity_top_k=similarity_top_k or k
            )
            
            # Perform retrieval
            nodes = await retriever.aretrieve(query)
            
            # Format results
            results = []
            for node in nodes[:k]:  # Ensure we don't exceed k
                result = self._format_retrieval_result(node, ticker)
                results.append(result)
            
            logger.info(f"📄 Retrieved {len(results)} relevant SEC documents for '{query[:30]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents for {ticker}: {e}")
            return []
    
    def get_available_tickers(self) -> List[str]:
        """Get list of tickers with available indexes"""
        tickers = []
        
        if not self.base_index_path.exists():
            return tickers
        
        for ticker_dir in self.base_index_path.iterdir():
            if ticker_dir.is_dir():
                faiss_path = ticker_dir / "faiss_index"
                if faiss_path.exists():
                    tickers.append(ticker_dir.name)
        
        return sorted(tickers)
    
    def get_index_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get information about a ticker's index"""
        ticker_index_dir = self.base_index_path / ticker.upper()
        metadata_path = ticker_index_dir / "index_metadata.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Add current status
            faiss_path = ticker_index_dir / "faiss_index"
            metadata["index_exists"] = faiss_path.exists()
            metadata["index_loaded"] = ticker in self.loaded_indexes
            
            if faiss_path.exists():
                metadata["index_size_mb"] = faiss_path.stat().st_size / (1024 * 1024)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get index info for {ticker}: {e}")
            return None
    
    def clear_cache(self):
        """Clear the loaded indexes cache"""
        self.loaded_indexes.clear()
        logger.info("Cleared index cache")


# Global retriever instance
_retriever = None

def get_retriever(openai_api_key: Optional[str] = None, 
                 base_index_path: str = "data/index",
                 embedding_provider: str = "auto") -> DocumentRetriever:
    """Get global retriever instance with auto-detection of embedding provider"""
    global _retriever
    
    if _retriever is None:
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            # Don't require OpenAI key if using local embeddings
            if not openai_api_key and embedding_provider == "openai":
                raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        
        _retriever = DocumentRetriever(openai_api_key, base_index_path, embedding_provider)
    
    return _retriever


async def retrieve_docs(ticker: str, query: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    Convenience function for document retrieval
    
    This is the main function that will be used by agents.
    """
    retriever = get_retriever()
    return await retriever.retrieve_docs(ticker, query, k)


async def main():
    """CLI entry point for testing retrieval"""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test document retrieval")
    parser.add_argument("--ticker", required=True, help="Stock ticker")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Test retrieval
    results = await retrieve_docs(args.ticker, args.query, args.k)
    
    print(f"\nRetrieved {len(results)} documents for '{args.query}':")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Source: {result['form_type']} - {result['filed_date']}")
        print(f"   Text: {result['text'][:200]}...")
        print("-" * 40)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
