"""
Enhanced SEC Document Ingestion with SEC-API.io Integration

Automatically downloads all relevant SEC documents for a given ticker and integrates
them into the existing FAISS vector indexing system with live progress tracking.
"""
import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import hashlib
from dataclasses import dataclass
import sys

# SEC-API.io SDK imports
try:
    from sec_api import QueryApi, RenderApi
    SEC_API_AVAILABLE = True
except ImportError:
    SEC_API_AVAILABLE = False
    QueryApi = None
    RenderApi = None

try:
    from rich.console import Console
    from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

logger = logging.getLogger(__name__)

if RICH_AVAILABLE:
    console = Console()
else:
    console = None


@dataclass
class SECDocument:
    """SEC Document metadata and content"""
    ticker: str
    form_type: str
    accession_no: str
    filing_date: str
    period_end_date: str
    company_name: str
    cik: str
    url: str
    content: str
    document_id: str
    file_size: int = 0
    processed_date: str = None


class EnhancedSECIngestor:
    """Enhanced SEC filings ingestion via SEC-API.io with comprehensive document coverage"""
    
    # Comprehensive list of relevant SEC form types
    PRIORITY_FORMS = [
        "10-K",      # Annual report
        "10-Q",      # Quarterly report
        "8-K",       # Current report
        "DEF 14A",   # Proxy statements
        "S-1",       # Registration statement
        "S-3",       # Registration statement
        "10-K/A",    # Annual report amendment
        "10-Q/A",    # Quarterly report amendment
    ]
    
    ADDITIONAL_FORMS = [
        "SC 13D",    # Beneficial ownership
        "SC 13G",    # Beneficial ownership (passive)
        "4",         # Statement of changes in beneficial ownership
        "3",         # Initial statement of beneficial ownership
        "11-K",      # Employee stock purchase plan annual report
        "20-F",      # Annual report (foreign companies)
        "6-K",       # Interim report (foreign companies)
        "F-1",       # Registration statement (foreign companies)
    ]
    
    def __init__(self, 
                 api_key: str, 
                 base_path: str = "data/filings_raw",
                 cache_ttl_days: int = 7,
                 max_concurrent_downloads: int = 5,
                 verbose: bool = True):
        self.api_key = api_key
        self.base_path = Path(base_path)
        self.base_url = "https://api.sec-api.io"
        self.cache_ttl_days = cache_ttl_days
        self.max_concurrent_downloads = max_concurrent_downloads
        self.verbose = verbose
        self.session = None
        
        # Initialize SEC-API.io SDK clients
        if not SEC_API_AVAILABLE:
            raise ImportError("sec-api package is required. Install with: pip install sec-api")
        
        self.query_api = QueryApi(api_key=api_key)
        self.render_api = RenderApi(api_key=api_key)
        
        # Document tracking for reports
        self.downloaded_documents = []
        self.processing_stats = {
            "total_found": 0,
            "downloaded": 0,
            "cached": 0,
            "failed": 0,
            "total_size_mb": 0.0,
            "processing_time": 0.0
        }
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting (SEC-API.io allows 100 requests/minute for paid plans)
        self.request_semaphore = asyncio.Semaphore(max_concurrent_downloads)
        self.last_request_time = None
        self.min_request_interval = 0.6  # 100 requests per minute = 0.6 seconds between requests
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _display_banner(self, ticker: str):
        """Display enhanced SEC ingestion banner"""
        if not self.verbose or not RICH_AVAILABLE:
            print(f"🏛️ Enhanced SEC Document Ingestion for {ticker}")
            return
            
        banner = Panel(
            f"🏛️ [bold blue]Enhanced SEC Document Ingestion[/bold blue]\n"
            f"📊 Ticker: [bold green]{ticker.upper()}[/bold green]\n"
            f"🔗 Source: SEC-API.io\n"
            f"📄 Document Types: {len(self.PRIORITY_FORMS + self.ADDITIONAL_FORMS)} forms\n"
            f"⚡ Max Concurrent: {self.max_concurrent_downloads}",
            title="🚀 DeepResearch SEC Integration",
            border_style="blue"
        )
        console.print(banner)
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        async with self.request_semaphore:
            if self.last_request_time:
                elapsed = asyncio.get_event_loop().time() - self.last_request_time
                if elapsed < self.min_request_interval:
                    await asyncio.sleep(self.min_request_interval - elapsed)
            self.last_request_time = asyncio.get_event_loop().time()
    
    def _get_cache_path(self, ticker: str) -> Path:
        """Get cache directory for ticker"""
        return self.base_path / ticker.upper()
    
    def _get_document_cache_path(self, ticker: str, accession_no: str, form_type: str) -> Path:
        """Get cache path for specific document"""
        cache_dir = self._get_cache_path(ticker)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename
        safe_accession = accession_no.replace("-", "").replace("/", "_")
        safe_form = form_type.replace("/", "_").replace(" ", "_")
        return cache_dir / f"{safe_form}_{safe_accession}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached document is still valid"""
        if not cache_path.exists():
            return False
        
        try:
            stat = cache_path.stat()
            cache_age = datetime.now().timestamp() - stat.st_mtime
            return cache_age < (self.cache_ttl_days * 24 * 3600)
        except:
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _fetch_filings_metadata(self, 
                                     ticker: str, 
                                     form_types: List[str], 
                                     years_back: int = 3,
                                     max_results: int = 100) -> List[Dict]:
        """Fetch comprehensive filings metadata for a ticker using SEC-API.io SDK"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        if self.verbose:
            print(f"🔍 Searching SEC filings for {ticker}...")
        
        try:
            # Build search query using SEC-API.io format
            form_query = " OR ".join([f'formType:"{ft}"' for ft in form_types])
            date_range = f'filedAt:[{start_date.strftime("%Y-%m-%d")} TO {end_date.strftime("%Y-%m-%d")}]'
            
            search_query = {
                "query": f'ticker:{ticker.upper()} AND ({form_query}) AND {date_range}',
                "from": "0",
                "size": str(min(max_results, 200)),  # SEC-API.io max is 200
                "sort": [{"filedAt": {"order": "desc"}}]
            }
            
            # Use SDK to fetch filings (this is synchronous)
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.query_api.get_filings, 
                search_query
            )
            
            filings = response.get("filings", [])
            self.processing_stats["total_found"] = len(filings)
            
            if self.verbose:
                print(f"📋 Found {len(filings)} SEC filings for {ticker}")
            
            return filings
                
        except Exception as e:
            logger.error(f"Error fetching filings metadata for {ticker}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _fetch_filing_content(self, filing_url: str, accession_no: str) -> str:
        """Fetch the full text content of a filing using SEC-API.io SDK"""
        
        try:
            # Use SEC-API.io SDK to fetch and render filing content
            content = await asyncio.get_event_loop().run_in_executor(
                None,
                self.render_api.get_filing,
                filing_url
            )
            
            if len(content.strip()) < 100:
                logger.warning(f"Suspiciously short content for {accession_no}: {len(content)} characters")
            
            return content
                
        except Exception as e:
            logger.error(f"Error fetching filing content for {accession_no}: {e}")
            # Fallback: try to fetch raw filing directly
            try:
                if self.session:
                    async with self.session.get(filing_url) as response:
                        response.raise_for_status()
                        return await response.text()
                else:
                    # If no session available, create a temporary one
                    async with aiohttp.ClientSession() as temp_session:
                        async with temp_session.get(filing_url) as response:
                            response.raise_for_status()
                            return await response.text()
            except Exception as fallback_error:
                logger.error(f"Fallback also failed for {accession_no}: {fallback_error}")
                raise e
    
    async def _save_document_cache(self, document: SECDocument):
        """Save document to cache"""
        cache_path = self._get_document_cache_path(
            document.ticker, 
            document.accession_no, 
            document.form_type
        )
        
        document_data = {
            "ticker": document.ticker,
            "form_type": document.form_type,
            "accession_no": document.accession_no,
            "filing_date": document.filing_date,
            "period_end_date": document.period_end_date,
            "company_name": document.company_name,
            "cik": document.cik,
            "url": document.url,
            "content": document.content,
            "document_id": document.document_id,
            "file_size": len(document.content),
            "processed_date": datetime.now().isoformat(),
            "cache_version": "2.0"
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving document cache {cache_path}: {e}")
    
    def _load_document_cache(self, ticker: str, accession_no: str, form_type: str) -> Optional[SECDocument]:
        """Load document from cache"""
        cache_path = self._get_document_cache_path(ticker, accession_no, form_type)
        
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return SECDocument(
                ticker=data["ticker"],
                form_type=data["form_type"],
                accession_no=data["accession_no"],
                filing_date=data["filing_date"],
                period_end_date=data.get("period_end_date", ""),
                company_name=data["company_name"],
                cik=data["cik"],
                url=data["url"],
                content=data["content"],
                document_id=data["document_id"],
                file_size=data.get("file_size", 0),
                processed_date=data.get("processed_date")
            )
        except Exception as e:
            logger.error(f"Error loading document cache {cache_path}: {e}")
            return None
    
    def _generate_document_id(self, accession_no: str, form_type: str) -> str:
        """Generate unique document ID"""
        content = f"{accession_no}_{form_type}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _process_filing(self, filing_metadata: Dict, ticker: str, progress=None, task_id=None) -> Optional[SECDocument]:
        """Process a single filing and return SECDocument"""
        accession_no = filing_metadata.get("accessionNo", "")
        form_type = filing_metadata.get("formType", "")
        
        # Check cache first
        cached_doc = self._load_document_cache(ticker, accession_no, form_type)
        if cached_doc:
            if self.verbose and progress and task_id:
                if RICH_AVAILABLE:
                    progress.update(task_id, advance=1, description=f"📁 [cyan]Cached: {form_type} {accession_no[:10]}...[/cyan]")
                else:
                    print(f"📁 Cached: {form_type} {accession_no[:10]}...")
            self.processing_stats["cached"] += 1
            return cached_doc
        
        # Extract filing details
        filing_url = filing_metadata.get("linkToFilingDetails", "")
        if not filing_url:
            logger.warning(f"No filing URL for {accession_no}")
            self.processing_stats["failed"] += 1
            return None
        
        try:
            if self.verbose and progress and task_id:
                if RICH_AVAILABLE:
                    progress.update(task_id, advance=0, description=f"⬇️ [yellow]Downloading: {form_type} {accession_no[:10]}...[/yellow]")
                else:
                    print(f"⬇️ Downloading: {form_type} {accession_no[:10]}...")
            
            # Fetch content
            content = await self._fetch_filing_content(filing_url, accession_no)
            
            # Create document object
            document = SECDocument(
                ticker=ticker.upper(),
                form_type=form_type,
                accession_no=accession_no,
                filing_date=filing_metadata.get("filedAt", ""),
                period_end_date=filing_metadata.get("periodOfReport", ""),
                company_name=filing_metadata.get("companyName", ""),
                cik=filing_metadata.get("cik", ""),
                url=filing_url,
                content=content,
                document_id=self._generate_document_id(accession_no, form_type),
                file_size=len(content)
            )
            
            # Cache the document
            await self._save_document_cache(document)
            
            if self.verbose:
                size_mb = len(content) / (1024 * 1024)
                if progress and task_id:
                    if RICH_AVAILABLE:
                        progress.update(task_id, advance=1, description=f"✅ [green]Completed: {form_type} ({size_mb:.1f}MB)[/green]")
                    else:
                        print(f"✅ Completed: {form_type} ({size_mb:.1f}MB)")
                else:
                    print(f"✅ Completed: {form_type} ({size_mb:.1f}MB)")
            
            self.processing_stats["downloaded"] += 1
            self.processing_stats["total_size_mb"] += len(content) / (1024 * 1024)
            
            return document
            
        except Exception as e:
            if self.verbose and progress and task_id:
                if RICH_AVAILABLE:
                    progress.update(task_id, advance=1, description=f"❌ [red]Failed: {form_type} {accession_no[:10]}...[/red]")
                else:
                    print(f"❌ Failed: {form_type} {accession_no[:10]}...")
            logger.error(f"Error processing filing {accession_no}: {e}")
            self.processing_stats["failed"] += 1
            return None
    
    async def download_all_documents(self, 
                                   ticker: str, 
                                   years_back: int = 3,
                                   priority_only: bool = False) -> List[SECDocument]:
        """Download all relevant SEC documents for a ticker with live progress"""
        start_time = datetime.now()
        
        if self.verbose:
            self._display_banner(ticker)
        
        # Determine which form types to fetch
        form_types = self.PRIORITY_FORMS.copy()
        if not priority_only:
            form_types.extend(self.ADDITIONAL_FORMS)
        
        # Fetch filings metadata
        try:
            filings_metadata = await self._fetch_filings_metadata(
                ticker, 
                form_types, 
                years_back=years_back
            )
        except Exception as e:
            logger.error(f"Failed to fetch filings metadata for {ticker}: {e}")
            return []
        
        if not filings_metadata:
            if self.verbose:
                print(f"⚠️ No SEC filings found for {ticker}")
            return []
        
        # Display filing summary
        if self.verbose:
            self._display_filing_summary(filings_metadata)
        
        # Process filings with progress bar
        documents = []
        
        if self.verbose and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("📥 Processing SEC documents...", total=len(filings_metadata))
                
                # Process filings with concurrency control
                semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
                
                async def process_with_semaphore(filing_metadata):
                    async with semaphore:
                        return await self._process_filing(filing_metadata, ticker, progress, task)
                
                # Process all filings
                tasks = [process_with_semaphore(filing) for filing in filings_metadata]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter successful results
                for result in results:
                    if isinstance(result, SECDocument):
                        documents.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Document processing failed: {result}")
        else:
            # Process without progress bar for non-verbose mode or when Rich not available
            semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
            
            async def process_with_semaphore(filing_metadata):
                async with semaphore:
                    return await self._process_filing(filing_metadata, ticker, None, None)
            
            if self.verbose:
                print(f"📥 Processing {len(filings_metadata)} SEC documents...")
            
            tasks = [process_with_semaphore(filing) for filing in filings_metadata]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, SECDocument):
                    documents.append(result)
        
        # Update processing stats
        self.processing_stats["processing_time"] = (datetime.now() - start_time).total_seconds()
        
        # Sort by filing date (newest first)
        documents.sort(key=lambda x: x.filing_date, reverse=True)
        
        # Store for report generation
        self.downloaded_documents = documents
        
        if self.verbose:
            self._display_completion_summary(ticker, documents)
        
        return documents
    
    def _display_filing_summary(self, filings_metadata: List[Dict]):
        """Display summary of found filings"""
        # Group by form type
        form_counts = {}
        for filing in filings_metadata:
            form_type = filing.get("formType", "Unknown")
            form_counts[form_type] = form_counts.get(form_type, 0) + 1
        
        if RICH_AVAILABLE:
            table = Table(title="📋 Found SEC Filings by Type")
            table.add_column("Form Type", style="cyan", no_wrap=True)
            table.add_column("Count", style="magenta")
            table.add_column("Description", style="white")
            
            form_descriptions = {
                "10-K": "Annual Report",
                "10-Q": "Quarterly Report", 
                "8-K": "Current Report",
                "DEF 14A": "Proxy Statement",
                "S-1": "Registration Statement",
                "S-3": "Registration Statement",
                "10-K/A": "Annual Report Amendment",
                "10-Q/A": "Quarterly Report Amendment",
                "SC 13D": "Beneficial Ownership",
                "SC 13G": "Beneficial Ownership (Passive)",
                "4": "Insider Trading Report",
                "3": "Initial Beneficial Ownership",
            }
            
            for form_type, count in sorted(form_counts.items()):
                description = form_descriptions.get(form_type, "Other SEC Filing")
                table.add_row(form_type, str(count), description)
            
            console.print(table)
        else:
            print("📋 Found SEC Filings by Type:")
            for form_type, count in sorted(form_counts.items()):
                print(f"  {form_type}: {count}")
    
    def _display_completion_summary(self, ticker: str, documents: List[SECDocument]):
        """Display completion summary"""
        stats = self.processing_stats
        
        if RICH_AVAILABLE:
            summary_panel = Panel(
                f"✅ [bold green]SEC Document Download Complete[/bold green]\n\n"
                f"📊 [bold]Statistics for {ticker.upper()}:[/bold]\n"
                f"   • Total Found: {stats['total_found']}\n"
                f"   • Downloaded: {stats['downloaded']}\n"
                f"   • From Cache: {stats['cached']}\n"
                f"   • Failed: {stats['failed']}\n"
                f"   • Total Size: {stats['total_size_mb']:.1f} MB\n"
                f"   • Processing Time: {stats['processing_time']:.1f}s\n\n"
                f"📁 [bold]Document Coverage:[/bold]\n"
                f"   • Recent 10-K: {'✅' if any(d.form_type == '10-K' for d in documents[:5]) else '❌'}\n"
                f"   • Recent 10-Q: {'✅' if any(d.form_type == '10-Q' for d in documents[:5]) else '❌'}\n"
                f"   • Recent 8-K: {'✅' if any(d.form_type == '8-K' for d in documents[:10]) else '❌'}\n"
                f"   • Proxy Statements: {'✅' if any(d.form_type == 'DEF 14A' for d in documents) else '❌'}",
                title="📈 DeepResearch SEC Integration Complete",
                border_style="green"
            )
            console.print(summary_panel)
        else:
            print(f"✅ SEC Document Download Complete for {ticker.upper()}")
            print(f"📊 Statistics:")
            print(f"   • Total Found: {stats['total_found']}")
            print(f"   • Downloaded: {stats['downloaded']}")
            print(f"   • From Cache: {stats['cached']}")
            print(f"   • Failed: {stats['failed']}")
            print(f"   • Total Size: {stats['total_size_mb']:.1f} MB")
            print(f"   • Processing Time: {stats['processing_time']:.1f}s")
    
    def get_document_summary_for_report(self) -> Dict[str, any]:
        """Get document summary for inclusion in analysis reports"""
        if not self.downloaded_documents:
            return {"error": "No documents processed"}
        
        # Group documents by form type
        by_form_type = {}
        total_size = 0
        date_range = {"earliest": None, "latest": None}
        
        for doc in self.downloaded_documents:
            form_type = doc.form_type
            if form_type not in by_form_type:
                by_form_type[form_type] = []
            
            by_form_type[form_type].append({
                "accession_no": doc.accession_no,
                "filing_date": doc.filing_date,
                "period_end_date": doc.period_end_date,
                "url": doc.url,
                "size_kb": round(doc.file_size / 1024, 1),
                "document_id": doc.document_id
            })
            
            total_size += doc.file_size
            
            # Track date range
            if not date_range["earliest"] or doc.filing_date < date_range["earliest"]:
                date_range["earliest"] = doc.filing_date
            if not date_range["latest"] or doc.filing_date > date_range["latest"]:
                date_range["latest"] = doc.filing_date
        
        return {
            "processing_summary": self.processing_stats,
            "document_count": len(self.downloaded_documents),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "date_range": date_range,
            "documents_by_type": by_form_type,
            "coverage_quality": self._calculate_coverage_quality(),
            "document_list": [
                {
                    "form_type": doc.form_type,
                    "filing_date": doc.filing_date,
                    "accession_no": doc.accession_no,
                    "company_name": doc.company_name,
                    "size_mb": round(doc.file_size / (1024 * 1024), 2),
                    "url": doc.url
                }
                for doc in self.downloaded_documents
            ]
        }
    
    def _calculate_coverage_quality(self) -> Dict[str, any]:
        """Calculate quality score for document coverage"""
        essential_forms = ["10-K", "10-Q", "8-K"]
        important_forms = ["DEF 14A", "S-1", "S-3"]
        
        coverage = {
            "essential_coverage": 0,
            "important_coverage": 0,
            "total_score": 0,
            "missing_essential": [],
            "recent_coverage": False
        }
        
        form_types_found = set(doc.form_type for doc in self.downloaded_documents)
        
        # Check essential forms
        essential_found = 0
        for form in essential_forms:
            if form in form_types_found:
                essential_found += 1
            else:
                coverage["missing_essential"].append(form)
        
        coverage["essential_coverage"] = essential_found / len(essential_forms)
        
        # Check important forms
        important_found = sum(1 for form in important_forms if form in form_types_found)
        coverage["important_coverage"] = important_found / len(important_forms)
        
        # Check recent coverage (filings in last 12 months)
        recent_cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        recent_docs = [doc for doc in self.downloaded_documents if doc.filing_date >= recent_cutoff]
        coverage["recent_coverage"] = len(recent_docs) > 0
        
        # Calculate total score
        coverage["total_score"] = (
            coverage["essential_coverage"] * 0.6 +
            coverage["important_coverage"] * 0.3 +
            (1.0 if coverage["recent_coverage"] else 0.0) * 0.1
        )
        
        return coverage


def convert_sec_documents_to_llama_documents(sec_documents: List[SECDocument]) -> List:
    """Convert SECDocument objects to LlamaIndex Document objects"""
    try:
        from llama_index.core import Document
    except ImportError:
        logger.error("LlamaIndex not available for document conversion")
        return []
    
    llama_documents = []
    
    for sec_doc in sec_documents:
        # Create metadata
        metadata = {
            "ticker": sec_doc.ticker,
            "form_type": sec_doc.form_type,
            "accession_no": sec_doc.accession_no,
            "filing_date": sec_doc.filing_date,
            "period_end_date": sec_doc.period_end_date,
            "company_name": sec_doc.company_name,
            "cik": sec_doc.cik,
            "url": sec_doc.url,
            "document_id": sec_doc.document_id,
            "file_size": sec_doc.file_size,
            "processed_date": sec_doc.processed_date or datetime.now().isoformat(),
            "source": "SEC-API.io"
        }
        
        # Create LlamaIndex document
        llama_doc = Document(
            text=sec_doc.content,
            metadata=metadata,
            id_=sec_doc.document_id
        )
        
        llama_documents.append(llama_doc)
    
    return llama_documents


# Integration function for existing system
async def download_sec_documents_for_ticker(ticker: str, 
                                          api_key: str,
                                          years_back: int = 3,
                                          priority_only: bool = False,
                                          verbose: bool = True) -> Tuple[List[SECDocument], Dict]:
    """Convenience function to download SEC documents for a ticker"""
    async with EnhancedSECIngestor(api_key, verbose=verbose) as ingestor:
        documents = await ingestor.download_all_documents(
            ticker, 
            years_back=years_back,
            priority_only=priority_only
        )
        summary = ingestor.get_document_summary_for_report()
        return documents, summary
