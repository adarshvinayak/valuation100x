"""
Enhanced Embedding and Vector Index Creation with LOCAL and OpenAI Support

Now supports:
- FREE local embeddings (sentence-transformers) - NO API COSTS!
- OpenAI embeddings (rate-limited and cheaper models)
- Seamless switching between providers
- Compatible with both PC and hosted backend
"""
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    Document = None
    VectorStoreIndex = None

try:
    from rich.console import Console
    from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

# Import our modules
from .enhanced_sec_ingest import EnhancedSECIngestor, SECDocument, convert_sec_documents_to_llama_documents
from .rate_limited_embeddings import RateLimitedOpenAIEmbedding
from .local_embeddings import LocalEmbeddingFactory

logger = logging.getLogger(__name__)

if RICH_AVAILABLE:
    console = Console()
else:
    console = None


class EnhancedEmbeddingIndexer:
    """
    Enhanced document embedding and vector index creation with LOCAL and OpenAI support
    
    Features:
    - FREE local embeddings (sentence-transformers) - $0 cost!
    - OpenAI embeddings with rate limiting and cheaper models
    - Seamless provider switching
    - Compatible with PC and hosted backends
    """
    
    def __init__(self, 
                 # Embedding provider settings
                 embedding_provider: str = "local",  # "local" or "openai" 
                 local_model_type: str = "fast",     # For local: "tiny", "fast", "balanced", "quality", "financial"
                 openai_model: str = "text-embedding-3-small",  # For OpenAI (5x cheaper than ada-002!)
                 openai_api_key: Optional[str] = None,
                 
                 # Index settings
                 base_filings_path: str = "data/filings_raw",
                 base_index_path: str = "data/index",
                 chunk_size: int = 1200,
                 chunk_overlap: int = 200,
                 verbose: bool = True,
                 
                 # Performance settings
                 embedding_batch_size: int = 32,     # Higher for local (no rate limits)
                 embedding_delay: float = 0.1):     # Lower for local (no rate limits)
        
        # Store parameters
        self.embedding_provider = embedding_provider
        self.local_model_type = local_model_type
        self.openai_model = openai_model
        self.openai_api_key = openai_api_key
        self.base_filings_path = Path(base_filings_path)
        self.base_index_path = Path(base_index_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        self.embedding_batch_size = embedding_batch_size
        self.embedding_delay = embedding_delay
        
        # Validate dependencies
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not available. Install with: pip install faiss-cpu")
        
        if not LLAMA_INDEX_AVAILABLE:
            raise RuntimeError("LlamaIndex is not available. Install with: pip install llama-index")
        
        # Processing stats for reports (initialize BEFORE embedding setup)
        self.indexing_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "embedding_time": 0.0,
            "indexing_time": 0.0,
            "index_size_mb": 0.0,
            "documents_processed": [],
            "embedding_provider": embedding_provider,
            "embedding_model": local_model_type if embedding_provider == "local" else openai_model,
            "embedding_cost_estimate": 0.0,
            "embedding_dimension": None
        }
        
        # Initialize embedding model based on provider
        self._setup_embedding_provider()
        
        # Initialize LLM (only if OpenAI key provided)
        if openai_api_key:
            Settings.llm = OpenAI(api_key=openai_api_key, model="gpt-5")
        else:
            Settings.llm = None
        
        # Initialize text splitter
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def _setup_embedding_provider(self):
        """Setup embedding provider (local or OpenAI)"""
        if self.embedding_provider == "local":
            self._setup_local_embeddings()
        elif self.embedding_provider == "openai":
            self._setup_openai_embeddings()
        else:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider}")
    
    def _setup_local_embeddings(self):
        """Setup FREE local embeddings"""
        try:
            # Create local embedding model
            local_embedder = LocalEmbeddingFactory.create_model(
                model_type=self.local_model_type,
                cache_folder=os.path.join(os.getcwd(), "models", "embeddings")
            )
            
            # Set as LlamaIndex embedding model
            Settings.embed_model = local_embedder
            
            # Get model info
            model_info = local_embedder.get_model_info()
            self.embedding_dimension = model_info["embedding_dimension"]
            self.indexing_stats["embedding_dimension"] = self.embedding_dimension
            self.indexing_stats["embedding_cost_estimate"] = 0.0  # FREE!
            
            if self.verbose:
                logger.info(" LOCAL EMBEDDINGS CONFIGURED (FREE!)")
                logger.info(f"   Provider: Local (sentence-transformers)")
                logger.info(f"   Model: {self.local_model_type}")
                logger.info(f"   Dimensions: {self.embedding_dimension}")
                logger.info(f"   Cost: $0.00 (COMPLETELY FREE!)")
                logger.info(f"   Rate limit: Unlimited")
                logger.info(f"   Cache: {model_info['cache_folder']}")
                
        except ImportError:
            raise ImportError(
                "sentence-transformers required for local embeddings.\n"
                "Install with: pip install sentence-transformers\n"
                "Or switch to OpenAI provider with: embedding_provider='openai'"
            )
    
    def _setup_openai_embeddings(self):
        """Setup OpenAI embeddings with rate limiting"""
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key required for OpenAI embeddings.\n"
                "Provide openai_api_key parameter or switch to local provider."
            )
        
        # Create rate-limited OpenAI embedding model
        Settings.embed_model = RateLimitedOpenAIEmbedding(
            api_key=self.openai_api_key,
            model=self.openai_model,
            batch_size=self.embedding_batch_size,
            delay_between_batches=self.embedding_delay,
            max_retries=5
        )
        
        # Set embedding dimensions based on model
        if "3-small" in self.openai_model:
            self.embedding_dimension = 1536
            cost_per_1k = 0.00002
        elif "3-large" in self.openai_model:
            self.embedding_dimension = 3072
            cost_per_1k = 0.00013
        elif "ada-002" in self.openai_model:
            self.embedding_dimension = 1536
            cost_per_1k = 0.0001
        else:
            self.embedding_dimension = 1536
            cost_per_1k = 0.0001
        
        self.indexing_stats["embedding_dimension"] = self.embedding_dimension
        self.indexing_stats["cost_per_1k_tokens"] = cost_per_1k
        
        if self.verbose:
            logger.info(" OPENAI EMBEDDINGS CONFIGURED")
            logger.info(f"   Provider: OpenAI")
            logger.info(f"   Model: {self.openai_model}")
            logger.info(f"   Dimensions: {self.embedding_dimension}")
            logger.info(f"   Cost: ${cost_per_1k:.5f} per 1K tokens")
            logger.info(f"   Rate limit: {self.embedding_batch_size} batch, {self.embedding_delay}s delay")
    
    def _display_indexing_banner(self, ticker: str):
        """Display enhanced indexing banner with provider info"""
        if not self.verbose:
            return
        
        provider_info = " FREE Local" if self.embedding_provider == "local" else " OpenAI"
        model_name = self.local_model_type if self.embedding_provider == "local" else self.openai_model
        cost_info = "$0.00 (FREE!)" if self.embedding_provider == "local" else f"${self.indexing_stats.get('cost_per_1k_tokens', 0):.5f}/1K tokens"
        
        if RICH_AVAILABLE:
            banner = Panel(
                f" [bold blue]Enhanced Vector Index Creation[/bold blue]\n"
                f" Ticker: [bold green]{ticker.upper()}[/bold green]\n"
                f" Embedding Provider: [bold]{provider_info}[/bold]\n"
                f" Model: {model_name}\n"
                f" Dimensions: {self.embedding_dimension}\n"
                f" Cost: {cost_info}\n"
                f" Chunk Size: {self.chunk_size} chars\n"
                f" Overlap: {self.chunk_overlap} chars\n"
                f" Vector Store: FAISS",
                title=" DeepResearch Vector Indexing",
                border_style="green" if self.embedding_provider == "local" else "blue"
            )
            console.print(banner)
        else:
            print(f" Enhanced Vector Index Creation for {ticker}")
            print(f" Embedding Provider: {provider_info}")
            print(f" Model: {model_name}")
            print(f" Cost: {cost_info}")
    
    async def create_enhanced_index_from_sec_documents(self, 
                                                     ticker: str,
                                                     sec_api_key: str,
                                                     years_back: int = 3,
                                                     force_rebuild: bool = False,
                                                     priority_only: bool = False) -> Tuple[str, Dict]:
        """Create enhanced vector index with fresh SEC documents"""
        start_time = datetime.now()
        
        if self.verbose:
            self._display_indexing_banner(ticker)
        
        # Check for existing index
        if not force_rebuild:
            existing_index_path = self._get_index_path(ticker)
            if existing_index_path.exists():
                if self.verbose:
                    print(f" Existing index found for {ticker}. Use force_rebuild=True to recreate.")
                return str(existing_index_path), {"status": "existing_index_used"}
        
        # Step 1: Download SEC documents
        if self.verbose:
            print(f"\n Step 1: Downloading SEC Documents")
        
        async with EnhancedSECIngestor(sec_api_key, verbose=self.verbose) as ingestor:
            sec_documents = await ingestor.download_all_documents(
                ticker, 
                years_back=years_back,
                priority_only=priority_only
            )
            sec_summary = ingestor.get_document_summary_for_report()
        
        if not sec_documents:
            raise ValueError(f"No SEC documents found for {ticker}")
        
        # Step 2: Convert to LlamaIndex documents
        if self.verbose:
            print(f"\n Step 2: Converting Documents to Vector Format")
        
        llama_documents = convert_sec_documents_to_llama_documents(sec_documents)
        self.indexing_stats["total_documents"] = len(llama_documents)
        
        # Step 3: Create chunks
        if self.verbose:
            print(f"\n Step 3: Creating Document Chunks")
        
        chunks = await self._create_chunks_with_progress(llama_documents, ticker)
        
        # Estimate cost for OpenAI
        if self.embedding_provider == "openai":
            estimated_tokens = sum(len(chunk.get_content().split()) for chunk in chunks) * 1.3  # Rough token estimate
            estimated_cost = (estimated_tokens / 1000) * self.indexing_stats.get("cost_per_1k_tokens", 0)
            self.indexing_stats["embedding_cost_estimate"] = estimated_cost
            if self.verbose:
                print(f" Estimated embedding cost: ${estimated_cost:.4f}")
        
        # Step 4: Create embeddings and FAISS index
        if self.verbose:
            provider_msg = "FREE Local" if self.embedding_provider == "local" else "OpenAI"
            print(f"\n Step 4: Creating {provider_msg} Vector Embeddings")
        
        index_path = await self._create_vector_index_with_progress(chunks, ticker)
        
        # Step 5: Save metadata
        self.indexing_stats["indexing_time"] = (datetime.now() - start_time).total_seconds()
        metadata = self._create_comprehensive_metadata(ticker, sec_summary)
        self._save_index_metadata(ticker, metadata)
        
        if self.verbose:
            self._display_indexing_completion(ticker)
        
        return index_path, {
            "sec_documents": sec_summary,
            "indexing_stats": self.indexing_stats,
            "index_path": index_path
        }
    
    async def create_enhanced_index_from_documents(self, 
                                                 ticker: str,
                                                 sec_documents: List,
                                                 sec_summary: Dict,
                                                 force_rebuild: bool = False) -> Tuple[str, Dict]:
        """Create enhanced vector index from already downloaded SEC documents (no re-download)"""
        start_time = datetime.now()
        
        if self.verbose:
            self._display_indexing_banner(ticker)
        
        # Check for existing index
        if not force_rebuild:
            existing_index_path = self._get_index_path(ticker)
            if existing_index_path.exists():
                if self.verbose:
                    print(f" Existing index found for {ticker}. Use force_rebuild=True to recreate.")
                return str(existing_index_path), {"status": "existing_index_used"}
        
        # Skip Step 1 (documents already downloaded)
        if self.verbose:
            print(f"\n Step 1: Using Pre-Downloaded SEC Documents ({len(sec_documents)} documents)")
        
        # Step 2: Convert to LlamaIndex documents
        if self.verbose:
            print(f"\n Step 2: Converting Documents to Vector Format")
        
        llama_documents = convert_sec_documents_to_llama_documents(sec_documents)
        self.indexing_stats["total_documents"] = len(llama_documents)
        
        # Step 3: Create chunks
        if self.verbose:
            print(f"\n Step 3: Creating Document Chunks")
        
        chunks = await self._create_chunks_with_progress(llama_documents, ticker)
        
        # Estimate cost for OpenAI
        if self.embedding_provider == "openai":
            estimated_tokens = sum(len(chunk.get_content().split()) for chunk in chunks) * 1.3  # Rough token estimate
            estimated_cost = (estimated_tokens / 1000) * self.indexing_stats.get("cost_per_1k_tokens", 0)
            self.indexing_stats["embedding_cost_estimate"] = estimated_cost
        
        # Step 4: Create vector index with embeddings
        if self.verbose:
            provider_desc = "FREE Local" if self.embedding_provider == "local" else "OpenAI"
            print(f"\n Step 4: Creating {provider_desc} Vector Embeddings")
        
        # Use manual FAISS creation to ensure proper binary format
        index_path = await self._save_faiss_index_manual(chunks, ticker)
        
        # Step 5: Save metadata
        metadata = self._create_comprehensive_metadata(ticker, sec_summary)
        self._save_index_metadata(ticker, metadata)
        
        self.indexing_stats["indexing_time"] = (datetime.now() - start_time).total_seconds()
        
        if self.verbose:
            self._display_indexing_completion(ticker)
        
        return index_path, {
            "sec_documents": sec_summary,
            "indexing_stats": self.indexing_stats,
            "index_path": index_path
        }
    
    async def _create_chunks_with_progress(self, documents: List[Document], ticker: str) -> List:
        """Create document chunks with progress tracking"""
        chunks = []
        
        if self.verbose and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(" Chunking documents...", total=len(documents))
                
                for doc in documents:
                    form_type = doc.metadata.get("form_type", "Unknown")
                    accession = doc.metadata.get("accession_no", "")[:10]
                    
                    progress.update(task, description=f" Chunking {form_type} {accession}...")
                    
                    doc_chunks = self.splitter.get_nodes_from_documents([doc])
                    chunks.extend(doc_chunks)
                    
                    self.indexing_stats["documents_processed"].append({
                        "form_type": form_type,
                        "accession_no": doc.metadata.get("accession_no", ""),
                        "chunks_created": len(doc_chunks),
                        "original_size_kb": len(doc.text) / 1024
                    })
                    
                    progress.advance(task)
        else:
            if self.verbose:
                print(f" Chunking {len(documents)} documents...")
            
            for i, doc in enumerate(documents):
                if self.verbose and i % 5 == 0:
                    print(f"  Processing document {i+1}/{len(documents)}")
                
                doc_chunks = self.splitter.get_nodes_from_documents([doc])
                chunks.extend(doc_chunks)
                
                self.indexing_stats["documents_processed"].append({
                    "form_type": doc.metadata.get("form_type", "Unknown"),
                    "accession_no": doc.metadata.get("accession_no", ""),
                    "chunks_created": len(doc_chunks),
                    "original_size_kb": len(doc.text) / 1024
                })
        
        self.indexing_stats["total_chunks"] = len(chunks)
        
        if self.verbose:
            print(f" Created {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks
    
    async def _create_vector_index_with_progress(self, chunks: List, ticker: str) -> str:
        """Create FAISS vector index with progress tracking"""
        embedding_start = datetime.now()
        
        provider_desc = " FREE local embeddings" if self.embedding_provider == "local" else " OpenAI embeddings (rate limited)"
        
        if self.verbose and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                embed_task = progress.add_task(f" Creating {provider_desc}...", total=100)
                
                # Create vector store
                faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                
                progress.update(embed_task, advance=20, description=" Initializing FAISS index...")
                
                # Create vector index
                progress.update(embed_task, advance=30, description=f" Processing {provider_desc}...")
                
                index = VectorStoreIndex(
                    nodes=chunks,
                    vector_store=vector_store
                )
                
                progress.update(embed_task, advance=40, description=" Building index structure...")
                
                # Save index
                index_path = self._save_faiss_index(index, ticker)
                
                progress.update(embed_task, advance=10, description=" Saving index to disk...")
                progress.update(embed_task, completed=100, description=" Vector index complete!")
        else:
            if self.verbose:
                print(f" Creating FAISS vector index with {provider_desc}...")
                print("  This may take a few minutes depending on document size...")
            
            faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            
            index = VectorStoreIndex(
                nodes=chunks,
                vector_store=vector_store
            )
            
            index_path = self._save_faiss_index(index, ticker)
            
            if self.verbose:
                print(" Vector index creation complete!")
        
        self.indexing_stats["embedding_time"] = (datetime.now() - embedding_start).total_seconds()
        
        return index_path
    
    def _get_index_path(self, ticker: str) -> Path:
        """Get index directory path for ticker"""
        return self.base_index_path / ticker.upper()
    
    async def _save_faiss_index_manual(self, chunks: List, ticker: str) -> str:
        """Create and save FAISS index manually in proper binary format"""
        ticker_index_dir = self._get_index_path(ticker)
        ticker_index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index in binary format
        faiss_path = ticker_index_dir / "faiss_index"
        
        try:
            logger.info(f"🔧 Creating FAISS index manually for {ticker}")
            
            # Extract embeddings from chunks
            embeddings = []
            for chunk in chunks:
                if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                    embeddings.append(chunk.embedding)
                else:
                    # Generate embedding if not present
                    embedding = Settings.embed_model.get_text_embedding(chunk.get_content())
                    embeddings.append(embedding)
            
            if not embeddings:
                raise ValueError(f"No embeddings found for {ticker}")
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            d = embeddings_array.shape[1]  # embedding dimension
            
            logger.info(f"📊 Creating FAISS index: {len(embeddings)} vectors × {d} dimensions")
            
            # Create FAISS index
            faiss_index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
            faiss_index.add(embeddings_array)
            
            # Save in proper binary format
            faiss.write_index(faiss_index, str(faiss_path))
            
            # Save chunk metadata separately for retrieval
            metadata_path = ticker_index_dir / "chunk_metadata.json"
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                chunk_metadata.append({
                    "index": i,
                    "text": chunk.get_content(),
                    "metadata": chunk.metadata or {},
                    "node_id": chunk.node_id if hasattr(chunk, 'node_id') else f"chunk_{i}"
                })
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)
            
            # Calculate index size
            if faiss_path.exists():
                self.indexing_stats["index_size_mb"] = faiss_path.stat().st_size / (1024 * 1024)
            
            logger.info(f"✅ Manual FAISS index created successfully for {ticker}")
            logger.info(f"📁 Index file: {faiss_path.stat().st_size / 1024:.1f} KB")
            logger.info(f"📁 Metadata file: {metadata_path.stat().st_size / 1024:.1f} KB")
            
            # Upload to Supabase Storage if available
            try:
                from database.supabase_client import supabase_manager
                if supabase_manager.initialized:
                    # Upload FAISS index
                    with open(faiss_path, 'rb') as f:
                        index_data = f.read()
                    
                    upload_url = await supabase_manager.upload_vector_index(ticker, index_data)
                    if upload_url:
                        logger.info(f"📤 Uploaded FAISS index to Supabase: {upload_url}")
                    
                    # Upload metadata
                    with open(metadata_path, 'rb') as f:
                        metadata_data = f.read()
                    
                    metadata_url = await supabase_manager.upload_file(
                        bucket='vector-indexes',
                        file_path=f"{ticker}/chunk_metadata.json",
                        file_data=metadata_data,
                        content_type='application/json'
                    )
                    if metadata_url:
                        logger.info(f"📤 Uploaded index metadata to Supabase: {metadata_url}")
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to upload index to Supabase: {e}")
            
            return str(ticker_index_dir)
            
        except Exception as e:
            logger.error(f"❌ Failed to create manual FAISS index for {ticker}: {e}")
            raise
    
    def _create_comprehensive_metadata(self, ticker: str, sec_summary: Dict) -> Dict:
        """Create comprehensive metadata for the index"""
        return {
            "ticker": ticker,
            "created_at": datetime.now().isoformat(),
            "indexing_method": "enhanced_sec_integration_v2",
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.local_model_type if self.embedding_provider == "local" else self.openai_model,
            "embedding_dimension": self.embedding_dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vector_store": "faiss",
            "sec_documents": sec_summary,
            "indexing_stats": self.indexing_stats,
            "index_version": "3.0"
        }
    
    def _save_index_metadata(self, ticker: str, metadata: Dict):
        """Save comprehensive index metadata"""
        ticker_index_dir = self._get_index_path(ticker)
        metadata_path = ticker_index_dir / "index_metadata.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _display_indexing_completion(self, ticker: str):
        """Display indexing completion summary"""
        stats = self.indexing_stats
        provider_info = " FREE Local" if self.embedding_provider == "local" else " OpenAI"
        cost_info = "$0.00 (FREE!)" if self.embedding_provider == "local" else f"${stats.get('embedding_cost_estimate', 0):.4f}"
        
        if RICH_AVAILABLE:
            summary_panel = Panel(
                f" [bold green]Vector Index Creation Complete[/bold green]\n\n"
                f" [bold]Statistics for {ticker.upper()}:[/bold]\n"
                f"    Provider: {provider_info}\n"
                f"    Documents: {stats['total_documents']}\n"
                f"    Chunks: {stats['total_chunks']}\n"
                f"    Dimensions: {stats['embedding_dimension']}\n"
                f"    Embedding Time: {stats['embedding_time']:.1f}s\n"
                f"    Total Time: {stats['indexing_time']:.1f}s\n"
                f"    Index Size: {stats['index_size_mb']:.1f} MB\n"
                f"    Total Cost: {cost_info}\n\n"
                f" [bold]Search Capabilities:[/bold]\n"
                f"    Semantic Search: \n"
                f"    Multi-Document Context: \n"
                f"    Source Attribution: \n"
                f"    Temporal Filtering: ",
                title=" DeepResearch Vector Index Ready",
                border_style="green" if self.embedding_provider == "local" else "blue"
            )
            console.print(summary_panel)
        else:
            print(f" Vector Index Creation Complete for {ticker.upper()}")
            print(f" Statistics:")
            print(f"    Provider: {provider_info}")
            print(f"    Documents: {stats['total_documents']}")
            print(f"    Chunks: {stats['total_chunks']}")
            print(f"    Total Cost: {cost_info}")
    
    def get_indexing_summary_for_report(self) -> Dict:
        """Get indexing summary for inclusion in analysis reports"""
        return {
            "indexing_method": "enhanced_sec_integration_v2",
            "embedding_provider": self.embedding_provider,
            "processing_stats": self.indexing_stats,
            "capabilities": {
                "semantic_search": True,
                "multi_document_context": True,
                "source_attribution": True,
                "temporal_filtering": True
            },
            "technical_details": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": self.indexing_stats["embedding_model"],
                "embedding_dimension": self.embedding_dimension,
                "vector_store": "faiss",
                "cost_per_analysis": self.indexing_stats.get("embedding_cost_estimate", 0.0)
            }
        }


# Updated integration function with local embedding support
async def create_enhanced_index_for_ticker(ticker: str,
                                         sec_api_key: str,
                                         embedding_provider: str = "local",  # "local" or "openai"
                                         local_model_type: str = "fast",     # For local embeddings
                                         openai_model: str = "text-embedding-3-small",  # For OpenAI
                                         openai_api_key: Optional[str] = None,
                                         years_back: int = 3,
                                         force_rebuild: bool = False,
                                         verbose: bool = True) -> Tuple[str, Dict]:
    """
    Create enhanced vector index with LOCAL or OpenAI embeddings
    
    Args:
        embedding_provider: "local" (FREE!) or "openai" 
        local_model_type: "tiny", "fast", "balanced", "quality", "financial"
        openai_model: "text-embedding-3-small" (5x cheaper) or others
        openai_api_key: Required only for OpenAI provider
    """
    indexer = EnhancedEmbeddingIndexer(
        embedding_provider=embedding_provider,
        local_model_type=local_model_type,
        openai_model=openai_model,
        openai_api_key=openai_api_key,
        verbose=verbose
    )
    
    return await indexer.create_enhanced_index_from_sec_documents(
        ticker=ticker,
        sec_api_key=sec_api_key,
        years_back=years_back,
        force_rebuild=force_rebuild
    )
