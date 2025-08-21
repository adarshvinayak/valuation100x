"""
CLI Runner for Stock Research System

Main command-line interface for running the comprehensive stock research system.
Supports ingestion, index building, and analysis with configurable options.
"""
import asyncio
import argparse
import logging
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import time

from dotenv import load_dotenv

# Import system components
from workflow import run_stock_analysis
from ingestion.sec_ingest import SECIngestor
from ingestion.embed_index import EmbeddingIndexer
from tools.cache import get_cache
from utils.markdown_formatter import format_comprehensive_markdown_report, generate_summary_table


class StockResearchCLI:
    """Command-line interface for the stock research system"""
    
    def __init__(self):
        self.setup_logging()
        # Load environment variables from the correct path
        env_path = r'C:\Users\rnuser\Documents\deepresearch\.env'
        if os.path.exists(env_path):
            load_dotenv(env_path)
            os.environ['ENV_FILE_PATH'] = env_path
        
    def setup_logging(self):
        """Configure logging for the CLI"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/stock_research.log')
            ]
        )
        
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        
        # Set up rate limiting for OpenAI API
        self.api_call_count = 0
        self.last_api_call = 0
        self.min_api_interval = 2.0  # Minimum 2 seconds between calls
        
        self.logger = logging.getLogger(__name__)
    
    async def ingest_filings(self, tickers: List[str], years: int = 3) -> Dict[str, Any]:
        """Ingest SEC filings for tickers"""
        self.logger.info(f"Starting SEC filings ingestion for {len(tickers)} tickers")
        
        import os
        sec_api_key = os.getenv("SEC_API_KEY")
        if not sec_api_key:
            raise ValueError("SEC_API_KEY not found in environment variables")
        
        results = {}
        
        async with SECIngestor(sec_api_key) as ingestor:
            for ticker in tickers:
                try:
                    self.logger.info(f"Ingesting filings for {ticker}")
                    result = await ingestor.ingest_ticker(ticker, years)
                    results[ticker] = result
                    self.logger.info(f"Completed ingestion for {ticker}: {result}")
                except Exception as e:
                    self.logger.error(f"Failed to ingest {ticker}: {e}")
                    results[ticker] = {"error": str(e)}
        
        return results
    
    async def build_indexes(self, tickers: List[str], force_rebuild: bool = False, 
                          max_concurrent: int = 2) -> Dict[str, str]:
        """Build vector indexes for tickers"""
        self.logger.info(f"Building vector indexes for {len(tickers)} tickers")
        
        import os
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        indexer = EmbeddingIndexer(openai_api_key)
        
        results = await indexer.build_indexes_for_tickers(
            tickers, 
            force_rebuild, 
            max_concurrent
        )
        
        # Log results
        for ticker, result in results.items():
            if result.startswith("ERROR"):
                self.logger.error(f"Index building failed for {ticker}: {result}")
            else:
                self.logger.info(f"Index built for {ticker}: {result}")
        
        return results
    
    async def run_analysis(self, tickers: List[str], 
                          concurrency: int = 1,
                          save_results: bool = True) -> Dict[str, Any]:
        """Run comprehensive analysis for tickers"""
        self.logger.info(f"Starting analysis for {len(tickers)} tickers with concurrency {concurrency}")
        
        semaphore = asyncio.Semaphore(concurrency)
        results = {}
        
        async def analyze_single(ticker: str) -> tuple:
            async with semaphore:
                try:
                    self.logger.info(f"Starting analysis for {ticker}")
                    start_time = time.time()
                    
                    result = await run_stock_analysis(ticker)
                    
                    duration = time.time() - start_time
                    self.logger.info(f"Completed analysis for {ticker} in {duration:.1f} seconds")
                    
                    # Save results if requested
                    if save_results:
                        await self._save_results(ticker, result)
                    
                    return ticker, result
                    
                except Exception as e:
                    self.logger.error(f"Analysis failed for {ticker}: {e}")
                    return ticker, {"error": str(e), "ticker": ticker}
        
        # Run analyses with concurrency control and rate limiting
        tasks = []
        for i, ticker in enumerate(tickers):
            # Add small delay between starting analyses to prevent API overwhelm
            if i > 0:
                await asyncio.sleep(2.0)  # 2 second delay between starts
            tasks.append(analyze_single(ticker))
        
        completed_analyses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in completed_analyses:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed with exception: {result}")
            else:
                ticker, analysis_result = result
                results[ticker] = analysis_result
        
        return results
    
    async def _save_results(self, ticker: str, result: Dict[str, Any]):
        """Save analysis results to files"""
        try:
            # Create output directory
            output_dir = Path("data/outputs") / ticker
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON results
            json_path = output_dir / f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Save standard markdown report (shorter version)
            if "report" in result:
                report_path = output_dir / f"{ticker}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(result["report"])
            
            # Generate and save comprehensive markdown report with full analysis
            try:
                comprehensive_markdown = format_comprehensive_markdown_report(result)
                comprehensive_path = output_dir / f"{ticker}_comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(comprehensive_path, 'w', encoding='utf-8') as f:
                    f.write(comprehensive_markdown)
                
                self.logger.info(f"Saved comprehensive markdown analysis to {comprehensive_path}")
            except Exception as e:
                self.logger.error(f"Failed to generate comprehensive markdown for {ticker}: {e}")
            
            # Save AI-generated comprehensive report (if available)
            if "comprehensive_report" in result and result["comprehensive_report"]:
                ai_comprehensive_path = output_dir / f"{ticker}_ai_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(ai_comprehensive_path, 'w', encoding='utf-8') as f:
                    f.write(result["comprehensive_report"])
                
                self.logger.info(f"Saved AI-generated comprehensive report to {ai_comprehensive_path}")
            
            self.logger.info(f"Saved results for {ticker} to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results for {ticker}: {e}")
    
    def clear_cache(self):
        """Clear the system cache"""
        cache = get_cache()
        cleared = cache.clear_all()
        self.logger.info(f"Cleared {cleared} cache entries")
        return cleared
    
    def get_cache_stats(self):
        """Get cache statistics"""
        cache = get_cache()
        stats = cache.get_stats()
        
        print("\n📊 Cache Statistics:")
        print("-" * 30)
        print(f"Total Entries: {stats['total_entries']}")
        print(f"Valid Entries: {stats['valid_entries']}")
        print(f"Expired Entries: {stats['expired_entries']}")
        print(f"Total Size: {stats['total_size_mb']:.1f} MB")
        print(f"Cache Directory: {stats['cache_dir']}")
        
        return stats
    
    async def run_pipeline(self, tickers: List[str], 
                          ingest: bool = False,
                          build_index: bool = False,
                          analyze: bool = True,
                          **kwargs):
        """Run the complete pipeline"""
        start_time = time.time()
        
        self.logger.info(f"Starting pipeline for tickers: {', '.join(tickers)}")
        
        results = {
            "tickers": tickers,
            "pipeline_config": {
                "ingest": ingest,
                "build_index": build_index,
                "analyze": analyze,
                **kwargs
            },
            "start_time": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Ingest SEC filings
            if ingest:
                print("🔄 Ingesting SEC filings...")
                ingest_results = await self.ingest_filings(tickers, kwargs.get("years", 3))
                results["ingestion"] = ingest_results
                
                # Check for failures
                failed_ingestions = [t for t, r in ingest_results.items() if "error" in r]
                if failed_ingestions:
                    self.logger.warning(f"Ingestion failed for: {', '.join(failed_ingestions)}")
            
            # Step 2: Build vector indexes
            if build_index:
                print("🔄 Building vector indexes...")
                index_results = await self.build_indexes(
                    tickers, 
                    kwargs.get("force_rebuild", False),
                    kwargs.get("max_concurrent", 2)
                )
                results["indexing"] = index_results
                
                # Check for failures
                failed_indexes = [t for t, r in index_results.items() if r.startswith("ERROR")]
                if failed_indexes:
                    self.logger.warning(f"Index building failed for: {', '.join(failed_indexes)}")
            
            # Step 3: Run analysis
            if analyze:
                print("🔄 Running comprehensive analysis...")
                analysis_results = await self.run_analysis(
                    tickers,
                    kwargs.get("concurrency", 3),
                    kwargs.get("save_results", True)
                )
                results["analysis"] = analysis_results
                
                # Display summary
                self._display_analysis_summary(analysis_results)
            
            # Pipeline completion
            total_time = time.time() - start_time
            results["completion_time"] = datetime.now().isoformat()
            results["total_duration_seconds"] = total_time
            
            self.logger.info(f"Pipeline completed in {total_time:.1f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            results["error"] = str(e)
            return results
    
    def _display_analysis_summary(self, analysis_results: Dict[str, Any]):
        """Display analysis summary"""
        print("\n📈 Analysis Summary:")
        print("=" * 60)
        
        successful_analyses = {k: v for k, v in analysis_results.items() if "error" not in v}
        failed_analyses = {k: v for k, v in analysis_results.items() if "error" in v}
        
        if successful_analyses:
            print(f"✅ Successful Analyses: {len(successful_analyses)}")
            print("-" * 40)
            
            for ticker, result in successful_analyses.items():
                score = result.get("investment_score", "N/A")
                fair_value = result.get("fair_value", 0)
                current_price = result.get("current_price", 0)
                
                print(f"{ticker:6} | Score: {score:4.1f}/10 | "
                      f"FV: ${fair_value:7.2f} | Price: ${current_price:7.2f}")
        
        if failed_analyses:
            print(f"\n❌ Failed Analyses: {len(failed_analyses)}")
            for ticker, result in failed_analyses.items():
                print(f"{ticker}: {result.get('error', 'Unknown error')}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Stock Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis for AAPL
  python run.py --tickers AAPL
  
  # Ingest filings and build indexes, then analyze
  python run.py --tickers AAPL,MSFT --ingest --build-index
  
  # Analyze multiple stocks with custom concurrency
  python run.py --tickers AAPL,MSFT,GOOGL --concurrency 2
  
  # Clear cache and get statistics
  python run.py --clear-cache --cache-stats
        """
    )
    
    # Main arguments
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of stock tickers to analyze"
    )
    
    # Pipeline steps
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest SEC filings before analysis"
    )
    
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build vector indexes before analysis"
    )
    
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of existing indexes"
    )
    
    # Configuration options
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Years of SEC filings to ingest (default: 3)"
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent analyses (default: 1, max recommended: 2)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Max concurrent index builds (default: 2)"
    )
    
    # Utility options
    parser.add_argument(
        "--cache-stats",
        action="store_true",
        help="Show cache statistics"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear system cache"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save analysis results to files"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create CLI instance
    cli = StockResearchCLI()
    
    async def run_cli():
        try:
            # Handle utility commands
            if args.clear_cache:
                cleared = cli.clear_cache()
                print(f"✅ Cleared {cleared} cache entries")
            
            if args.cache_stats:
                cli.get_cache_stats()
            
            # Handle analysis pipeline
            if args.tickers:
                tickers = [t.strip().upper() for t in args.tickers.split(",")]
                
                print(f"🚀 Starting stock research for: {', '.join(tickers)}")
                
                # Run pipeline
                results = await cli.run_pipeline(
                    tickers=tickers,
                    ingest=args.ingest,
                    build_index=args.build_index,
                    analyze=True,
                    years=args.years,
                    concurrency=args.concurrency,
                    max_concurrent=args.max_concurrent,
                    force_rebuild=args.force_rebuild,
                    save_results=not args.no_save
                )
                
                # Save pipeline results
                if not args.no_save:
                    pipeline_path = Path("data/outputs") / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(pipeline_path, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    
                    print(f"\n💾 Pipeline results saved to: {pipeline_path}")
            
            elif not args.cache_stats and not args.clear_cache:
                parser.print_help()
                print("\n❌ Error: No tickers specified and no utility commands given")
                sys.exit(1)
        
        except KeyboardInterrupt:
            print("\n🛑 Analysis interrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"\n❌ CLI failed: {e}")
            cli.logger.exception("CLI execution failed")
            sys.exit(1)
    
    # Run the CLI
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
