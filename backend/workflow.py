"""
Deep Stock Research Workflow

LlamaIndex workflow that orchestrates the multi-agent stock research system.
Coordinates question generation, research, valuation, scoring, and reporting.
"""
import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import dotenv

# Load environment variables
env_path = os.getenv('ENV_FILE_PATH', '.env')
if os.path.exists(env_path):
    dotenv.load_dotenv(env_path)

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    InputRequiredEvent,
    HumanResponseEvent,
    step
)
# Optional workflow visualization (disabled for Railway deployment compatibility)
try:
    from llama_index.utils.workflow import draw_all_possible_flows
    WORKFLOW_VISUALIZATION_AVAILABLE = True
except ImportError:
    WORKFLOW_VISUALIZATION_AVAILABLE = False
    draw_all_possible_flows = None

# Import our agents and tools
from agents.questions_fixed import get_question_agent
from agents.answer_fixed import get_answer_agent, answer_question_with_agent
from agents.report_fixed import get_report_agent, generate_report_with_agent
from agents.report_comprehensive import get_comprehensive_report_agent, generate_comprehensive_report
from tools.fmp import get_financials_fmp
from tools.alpha import get_prices_alpha
from tools.polygon import get_prices_polygon
from tools.technicals import compute_indicators
from synthesis.valuation import ensemble_valuation
from synthesis.scoring import score_1_to_10

logger = logging.getLogger(__name__)


# Workflow Events
class GenerateQuestionsEvent(Event):
    ticker: str
    company_name: str

class QuestionGeneratedEvent(Event):
    question: str
    question_id: int

class ResearchCompleteEvent(Event):
    question: str
    question_id: int
    answer: Dict[str, Any]

class ValuationEvent(Event):
    financial_data: Dict[str, Any]
    current_price: float

class ScoringEvent(Event):
    valuation_data: Dict[str, Any]
    financial_data: Dict[str, Any]
    technical_data: Dict[str, Any]
    sentiment_data: Dict[str, Any]

class ReportEvent(Event):
    research_answers: List[Dict[str, Any]]
    valuation_data: Dict[str, Any]
    scoring_data: Dict[str, Any]

class ProgressEvent(Event):
    message: str
    progress: float  # 0.0 to 1.0

class ErrorEvent(Event):
    error_message: str
    component: str

class HumanApprovalEvent(Event):
    draft_results: Dict[str, Any]


class DeepStockResearchWorkflow(Workflow):
    """
    Main workflow for comprehensive stock research
    
    Steps:
    1. Setup and validation
    2. Generate research questions
    3. Research answers in parallel
    4. Perform valuation analysis
    5. Calculate investment scoring
    6. Generate final report
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.research_answers = []
        self.total_questions = 0
    
    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> GenerateQuestionsEvent:
        """Initialize workflow and validate inputs"""
        try:
            # Extract parameters from StartEvent (following reference pattern)
            ticker = ev.ticker
            company_name = getattr(ev, 'company_name', ticker)
            
            # Store agents if passed, otherwise create them
            if hasattr(ev, 'question_agent'):
                self.question_agent = ev.question_agent
                self.answer_agent = ev.answer_agent
                self.report_agent = ev.report_agent
            else:
                # Create agents (fallback)
                self.question_agent = get_question_agent()
                self.answer_agent = get_answer_agent()
                self.report_agent = get_report_agent()
            
            # Try to create comprehensive report agent (optional)
            try:
                self.comprehensive_report_agent = get_comprehensive_report_agent()
                logger.info("Comprehensive reporting enabled")
            except Exception as e:
                logger.warning(f"Comprehensive reporting disabled: {e}")
                self.comprehensive_report_agent = None
            
            # Validate required environment variables
            required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
            for key in required_keys:
                if not os.getenv(key):
                    raise ValueError(f"Missing required environment variable: {key}")
            
            # Store workflow context
            await ctx.set("ticker", ticker)
            await ctx.set("company_name", company_name)
            await ctx.set("start_time", datetime.now())
            
            # Initialize progress tracking
            ctx.write_event_to_stream(ProgressEvent(
                message=f"Starting comprehensive research for {ticker}",
                progress=0.0
            ))
            
            logger.info(f"Initialized workflow for {ticker}")
            return GenerateQuestionsEvent(ticker=ticker, company_name=company_name)
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            ctx.write_event_to_stream(ErrorEvent(
                error_message=str(e),
                component="setup"
            ))
            return StopEvent(result={"error": str(e), "component": "setup"})
    
    @step  
    async def generate_questions(self, ctx: Context, ev: GenerateQuestionsEvent) -> QuestionGeneratedEvent:
        """Generate sector-aware research questions"""
        try:
            ctx.write_event_to_stream(ProgressEvent(
                message=f"Generating research questions for {ev.ticker}",
                progress=0.1
            ))
            
            # Generate questions using the question agent (following reference pattern)
            from agents.questions_fixed import generate_questions_with_agent
            questions = await generate_questions_with_agent(self.question_agent, ev.ticker, ev.company_name)
            
            if not questions:
                raise ValueError("Failed to generate research questions")
            
            # Store questions in context
            await ctx.set("questions", questions)
            await ctx.set("total_questions", len(questions))
            self.total_questions = len(questions)
            
            logger.info(f"Generated {len(questions)} questions for {ev.ticker}")
            
            ctx.write_event_to_stream(ProgressEvent(
                message=f"Generated {len(questions)} research questions",
                progress=0.2
            ))
            
            # Fire off multiple question events (following reference pattern)
            for i, question in enumerate(questions):
                ctx.send_event(QuestionGeneratedEvent(
                    question=question,
                    question_id=i
                ))
            
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            ctx.write_event_to_stream(ErrorEvent(
                error_message=str(e),
                component="question_generation"
            ))
            return [StopEvent(result={"error": str(e), "component": "question_generation"})]
    
    @step
    async def research_question(self, ctx: Context, ev: QuestionGeneratedEvent) -> ResearchCompleteEvent:
        """Research individual questions using the answer agent"""
        try:
            ticker = await ctx.get("ticker")
            company_name = await ctx.get("company_name")
            
            ctx.write_event_to_stream(ProgressEvent(
                message=f"🔍 Researching: {ev.question[:60]}... (Using SEC documents + market data)",
                progress=0.3 + (ev.question_id / self.total_questions) * 0.3
            ))
            
            # Research the question using answer agent (following reference pattern)
            answer = await answer_question_with_agent(self.answer_agent, ticker, ev.question, company_name)
            
            logger.info(f"Completed research for question {ev.question_id}: {ev.question[:50]}...")
            
            return ResearchCompleteEvent(
                question=ev.question,
                question_id=ev.question_id,
                answer=answer
            )
            
        except Exception as e:
            logger.error(f"Research failed for question {ev.question_id}: {e}")
            # Return error answer instead of stopping workflow
            error_answer = {
                "question": ev.question,
                "findings": [],
                "metrics": {},
                "summary": f"Research failed: {str(e)}",
                "confidence": 0.0
            }
            return ResearchCompleteEvent(
                question=ev.question,
                question_id=ev.question_id,
                answer=error_answer
            )
    
    @step
    async def collect_research(self, ctx: Context, ev: ResearchCompleteEvent) -> Optional[ValuationEvent]:
        """Collect research answers and trigger valuation when complete"""
        try:
            # Collect this answer
            self.research_answers.append(ev.answer)
            
            total_questions = await ctx.get("total_questions")
            
            # Check if we have all answers
            if len(self.research_answers) < total_questions:
                ctx.write_event_to_stream(ProgressEvent(
                    message=f"📊 Collected {len(self.research_answers)}/{total_questions} research answers (SEC documents + market data)",
                    progress=0.6
                ))
                return None  # Wait for more answers
            
            # All research complete, get financial data for valuation
            ticker = await ctx.get("ticker")
            
            ctx.write_event_to_stream(ProgressEvent(
                message="Research complete, starting valuation analysis",
                progress=0.7
            ))
            
            # Get financial data and current price
            financial_data = await get_financials_fmp(ticker)
            
            # Get current price from multiple sources
            current_price = 0.0
            try:
                # Try Polygon first
                prices = await get_prices_polygon(ticker, days=1)
                if prices:
                    current_price = prices[0].get("close", 0.0)
                
                # Fallback to Alpha Vantage
                if current_price == 0.0:
                    prices = await get_prices_alpha(ticker, "compact")
                    if prices:
                        current_price = prices[0].get("close", 0.0)
                
                # Fallback to financial data
                if current_price == 0.0 and financial_data.get("profile"):
                    current_price = financial_data["profile"].get("price", 100.0)
                    
            except Exception as e:
                logger.warning(f"Failed to get current price for {ticker}: {e}")
                current_price = 100.0  # Default fallback
            
            # Store research results
            await ctx.set("research_answers", self.research_answers)
            await ctx.set("financial_data", financial_data)
            await ctx.set("current_price", current_price)
            
            logger.info(f"Collected all {len(self.research_answers)} research answers")
            
            return ValuationEvent(
                financial_data=financial_data,
                current_price=current_price
            )
            
        except Exception as e:
            logger.error(f"Research collection failed: {e}")
            ctx.write_event_to_stream(ErrorEvent(
                error_message=str(e),
                component="research_collection"
            ))
            return StopEvent(result={"error": str(e), "component": "research_collection"})
    
    @step
    async def perform_valuation(self, ctx: Context, ev: ValuationEvent) -> ScoringEvent:
        """Perform comprehensive valuation analysis"""
        try:
            ticker = await ctx.get("ticker")
            
            ctx.write_event_to_stream(ProgressEvent(
                message="Performing valuation analysis",
                progress=0.75
            ))
            
            # Perform ensemble valuation
            valuation_data = ensemble_valuation(
                ev.financial_data,
                ticker,
                ev.current_price
            )
            
            # Get technical analysis data
            technical_data = {}
            try:
                prices = await get_prices_polygon(ticker, days=252)  # 1 year of data
                if prices:
                    technical_data = {"indicators": compute_indicators(prices)}
            except Exception as e:
                logger.warning(f"Technical analysis failed: {e}")
                technical_data = {"indicators": {}}
            
            # Extract sentiment data from research answers
            sentiment_data = self._extract_sentiment_from_research(await ctx.get("research_answers"))
            
            # Store valuation results
            await ctx.set("valuation_data", valuation_data)
            await ctx.set("technical_data", technical_data)
            await ctx.set("sentiment_data", sentiment_data)
            
            logger.info(f"Completed valuation analysis for {ticker}")
            
            return ScoringEvent(
                valuation_data=valuation_data,
                financial_data=ev.financial_data,
                technical_data=technical_data,
                sentiment_data=sentiment_data
            )
            
        except Exception as e:
            logger.error(f"Valuation failed: {e}")
            ctx.write_event_to_stream(ErrorEvent(
                error_message=str(e),
                component="valuation"
            ))
            return StopEvent(result={"error": str(e), "component": "valuation"})
    
    @step
    async def calculate_score(self, ctx: Context, ev: ScoringEvent) -> ReportEvent:
        """Calculate comprehensive investment score"""
        try:
            ticker = await ctx.get("ticker")
            
            ctx.write_event_to_stream(ProgressEvent(
                message="Calculating investment score",
                progress=0.85
            ))
            
            # Calculate comprehensive score
            scoring_data = score_1_to_10(
                ev.valuation_data,
                ev.financial_data,
                ev.sentiment_data,
                ev.technical_data
            )
            
            # Store scoring results
            await ctx.set("scoring_data", scoring_data)
            
            logger.info(f"Calculated investment score: {scoring_data.get('final_score', 'N/A')}/10 for {ticker}")
            
            return ReportEvent(
                research_answers=await ctx.get("research_answers"),
                valuation_data=ev.valuation_data,
                scoring_data=scoring_data
            )
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            ctx.write_event_to_stream(ErrorEvent(
                error_message=str(e),
                component="scoring"
            ))
            return StopEvent(result={"error": str(e), "component": "scoring"})
    
    @step
    async def prepare_draft_report(self, ctx: Context, ev: ReportEvent) -> HumanApprovalEvent:
        """Generate comprehensive investment report"""
        try:
            ticker = await ctx.get("ticker")
            company_name = await ctx.get("company_name")
            
            ctx.write_event_to_stream(ProgressEvent(
                message="Generating final investment report",
                progress=0.95
            ))
            
            # Generate comprehensive report with full justification and citations (if available)
            comprehensive_report = None
            if self.comprehensive_report_agent:
                try:
                    comprehensive_report = await generate_comprehensive_report(
                        self.comprehensive_report_agent,
                        ticker,
                        company_name,
                        ev.research_answers,
                        ev.valuation_data,
                        ev.scoring_data,
                        financial_data=await ctx.get("financial_data"),
                        technical_data=await ctx.get("technical_data"),
                        sentiment_data=await ctx.get("sentiment_data")
                    )
                    logger.info(f"Generated comprehensive report for {ticker}")
                except Exception as e:
                    logger.warning(f"Failed to generate comprehensive report for {ticker}: {e}")
                    comprehensive_report = None
            
            # Also generate standard report for backwards compatibility
            report = await generate_report_with_agent(
                self.report_agent,
                ticker,
                company_name,
                ev.research_answers,
                ev.valuation_data,
                ev.scoring_data
            )
            
            # Calculate completion time
            start_time = await ctx.get("start_time")
            completion_time = datetime.now()
            duration = (completion_time - start_time).total_seconds()
            
            # Prepare final results with comprehensive report
            results = {
                "ticker": ticker,
                "company_name": company_name,
                "investment_score": ev.scoring_data.get("final_score", 5.0),
                "fair_value": ev.valuation_data.get("median_fv", 0.0),
                "current_price": await ctx.get("current_price"),
                "probability_undervalued": ev.valuation_data.get("p_underv", 0.5),
                "report": report,  # Standard report
                "comprehensive_report": comprehensive_report,  # Detailed report with full justification (if available)
                "research_answers": ev.research_answers,  # Full research data
                "valuation_data": ev.valuation_data,  # Complete valuation analysis
                "scoring_data": ev.scoring_data,  # Detailed scoring breakdown
                "financial_data": await ctx.get("financial_data"),
                "technical_data": await ctx.get("technical_data"),
                "sentiment_data": await ctx.get("sentiment_data"),
                "research_summary": {
                    "questions_answered": len(ev.research_answers),
                    "average_confidence": sum(a.get("confidence", 0.5) for a in ev.research_answers) / len(ev.research_answers),
                    "valuation_methods": len(ev.valuation_data.get("valuation_methods", [])),
                    "analysis_duration_seconds": duration,
                    "total_citations": self._count_citations(ev.research_answers),
                    "data_quality_score": self._calculate_data_quality(ev.research_answers)
                },
                "component_scores": ev.scoring_data.get("component_scores", {}),
                "analysis_metadata": {
                    "completion_time": completion_time.isoformat(),
                    "analysis_duration": duration,
                    "data_sources_used": self._get_data_sources_used(ev.research_answers),
                    "workflow_version": "2.0_comprehensive",
                    "report_type": "comprehensive_with_full_justification"
                }
            }
            
            ctx.write_event_to_stream(ProgressEvent(
                message=f"Analysis complete! Score: {results['investment_score']:.1f}/10",
                progress=1.0
            ))
            
            logger.info(f"Completed analysis for {ticker} in {duration:.1f} seconds")
            
            # Create draft results for human approval
            draft_results = {
                "ticker": ticker,
                "company_name": company_name,
                "investment_score": ev.scoring_data.get("final_score", 5.0),
                "fair_value": ev.valuation_data.get("median_fv", 0.0),
                "current_price": await ctx.get("current_price"),
                "probability_undervalued": ev.valuation_data.get("p_underv", 0.5),
                "component_scores": ev.scoring_data.get("component_scores", {}),
                "research_summary": {
                    "questions_answered": len(ev.research_answers),
                    "average_confidence": sum(a.get("confidence", 0.5) for a in ev.research_answers) / len(ev.research_answers),
                    "valuation_methods": len(ev.valuation_data.get("valuation_methods", [])),
                },
                "draft_report": report,
                "analysis_metadata": {
                    "completion_time": completion_time.isoformat(),
                    "analysis_duration": duration,
                    "data_sources_used": self._get_data_sources_used(ev.research_answers),
                    "workflow_version": "1.0"
                }
            }
            
            # Store draft for potential finalization
            await ctx.set("draft_results", draft_results)
            
            ctx.write_event_to_stream(ProgressEvent(
                message="Draft analysis complete, requesting human approval",
                progress=0.95
            ))
            
            logger.info(f"Prepared draft analysis for {ticker}")
            
            return HumanApprovalEvent(draft_results=draft_results)
            
        except Exception as e:
            logger.error(f"Draft report generation failed: {e}")
            ctx.write_event_to_stream(ErrorEvent(
                error_message=str(e),
                component="draft_report_generation"
            ))
            return StopEvent(result={"error": str(e), "component": "draft_report_generation"})
    
    @step
    async def human_approval_checkpoint(self, ctx: Context, ev: HumanApprovalEvent) -> StopEvent:
        """Human-in-the-Loop checkpoint before final report - auto-approve for CLI"""
        try:
            # Check if auto-approval is enabled (default for CLI)
            auto_approve = await ctx.get("auto_approve", default=True)
            
            if auto_approve:
                logger.info(f"Auto-approving analysis for {ev.draft_results['ticker']} (CLI mode)")
                ctx.write_event_to_stream(ProgressEvent(
                    message="Auto-approving analysis (CLI mode)",
                    progress=1.0
                ))
                
                # Store for final output
                await ctx.set("draft_results", ev.draft_results)
                
                # Auto-approve and finalize
                ev.draft_results["auto_approved"] = True
                ev.draft_results["approval_method"] = "auto_cli"
                
                return StopEvent(result=ev.draft_results)
            
            # Interactive mode (for future frontend integration)
            draft_json = json.dumps({
                "ticker": ev.draft_results["ticker"],
                "investment_score": ev.draft_results["investment_score"],
                "fair_value": ev.draft_results["fair_value"],
                "current_price": ev.draft_results["current_price"],
                "probability_undervalued": ev.draft_results["probability_undervalued"],
                "component_scores": ev.draft_results["component_scores"],
                "research_summary": ev.draft_results["research_summary"]
            }, indent=2)
            
            # Store for interactive response
            await ctx.set("draft_results", ev.draft_results)
            
            # For now, print the analysis and auto-approve after 5 seconds
            logger.info(f"""
🎯 **Analysis Complete for {ev.draft_results['company_name']} ({ev.draft_results['ticker']})**

**Key Results:**
- Investment Score: {ev.draft_results['investment_score']:.1f}/10
- Fair Value: ${ev.draft_results['fair_value']:.2f}
- Current Price: ${ev.draft_results['current_price']:.2f}
- Probability Undervalued: {ev.draft_results['probability_undervalued']:.1%}

Auto-approving in 5 seconds...
""")
            
            # Wait 5 seconds then auto-approve
            await asyncio.sleep(5)
            
            ctx.write_event_to_stream(ProgressEvent(
                message="Auto-approving after timeout",
                progress=1.0
            ))
            
            ev.draft_results["auto_approved"] = True
            ev.draft_results["approval_method"] = "timeout"
            
            return StopEvent(result=ev.draft_results)
            
        except Exception as e:
            logger.error(f"Human approval checkpoint failed: {e}")
            # Fallback to auto-approval
            ev.draft_results["auto_approved"] = True
            ev.draft_results["approval_error"] = str(e)
            
            return StopEvent(result=ev.draft_results)
    

    
    def _extract_sentiment_from_research(self, research_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract sentiment data from research answers"""
        try:
            # Look for sentiment-related findings in research answers
            sentiment_findings = []
            confidence_scores = []
            
            for answer in research_answers:
                # Extract confidence scores
                confidence_scores.append(answer.get("confidence", 0.5))
                
                # Look for sentiment-related content
                question = answer.get("question", "").lower()
                if any(word in question for word in ["sentiment", "earnings", "management", "outlook"]):
                    sentiment_findings.extend(answer.get("findings", []))
            
            # Calculate average sentiment (simplified)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            # Convert confidence to sentiment score (0.5 neutral, higher = more positive)
            sentiment_score = (avg_confidence - 0.5) * 2  # Scale to -1 to 1
            
            return {
                "average_sentiment": sentiment_score,
                "transcript_analysis": {
                    "overall_sentiment": sentiment_score * 0.8,  # Slightly more conservative
                    "uncertainty_index": 1 - avg_confidence  # Inverse of confidence
                },
                "research_confidence": avg_confidence,
                "sentiment_findings_count": len(sentiment_findings)
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract sentiment from research: {e}")
            return {
                "average_sentiment": 0.0,
                "transcript_analysis": {
                    "overall_sentiment": 0.0,
                    "uncertainty_index": 0.5
                }
            }
    
    def _get_data_sources_used(self, research_answers: List[Dict[str, Any]]) -> List[str]:
        """Extract unique data sources used in research"""
        sources = set()
        
        for answer in research_answers:
            for finding in answer.get("findings", []):
                for evidence in finding.get("evidence", []):
                    source_type = evidence.get("source_type", "unknown")
                    if source_type != "system":
                        sources.add(source_type)
        
        return list(sources)
    
    def _count_citations(self, research_answers: List[Dict[str, Any]]) -> int:
        """Count total number of citations with valid URLs"""
        citation_count = 0
        for answer in research_answers:
            findings = answer.get("findings", [])
            for finding in findings:
                evidence = finding.get("evidence", [])
                for cite in evidence:
                    if cite.get("url") and cite["url"] != "internal://json-error":
                        citation_count += 1
        return citation_count
    
    def _calculate_data_quality(self, research_answers: List[Dict[str, Any]]) -> float:
        """Calculate data quality score based on confidence and citations"""
        if not research_answers:
            return 0.0
        
        total_confidence = sum(a.get("confidence", 0.5) for a in research_answers)
        avg_confidence = total_confidence / len(research_answers)
        
        citation_count = self._count_citations(research_answers)
        citation_score = min(1.0, citation_count / (len(research_answers) * 2))  # Target: 2 citations per question
        
        # Weighted average: 70% confidence, 30% citation coverage
        data_quality = (avg_confidence * 0.7) + (citation_score * 0.3)
        return min(1.0, data_quality)


async def run_stock_analysis(ticker: str, company_name: str = None) -> Dict[str, Any]:
    """
    Convenience function to run complete stock analysis
    
    Args:
        ticker: Stock ticker symbol
        company_name: Optional company name
    
    Returns:
        Complete analysis results
    """
    company_name = company_name or ticker
    
    # Create and run workflow (following reference pattern)
    workflow = DeepStockResearchWorkflow(timeout=600)  # 10 minute timeout
    
    # Run workflow with direct parameters (like the reference)
    handler = workflow.run(
        ticker=ticker,
        company_name=company_name
    )
    
    # Process streaming events
    async for event in handler.stream_events():
        if isinstance(event, ProgressEvent):
            logger.info(f"Progress: {event.message} ({event.progress:.1%})")
        elif isinstance(event, ErrorEvent):
            logger.error(f"Error in {event.component}: {event.error_message}")
    
    # Get final result
    result = await handler
    return result


async def main():
    """CLI entry point for workflow testing"""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run comprehensive stock analysis")
    parser.add_argument("--ticker", required=True, help="Stock ticker to analyze")
    parser.add_argument("--company", help="Company name (optional)")
    parser.add_argument("--save-output", action="store_true", help="Save results to files")
    parser.add_argument("--draw-workflow", action="store_true", help="Generate workflow diagram")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.draw_workflow:
        # Generate workflow diagram
        if WORKFLOW_VISUALIZATION_AVAILABLE:
            draw_all_possible_flows(DeepStockResearchWorkflow, filename="stock_research_workflow.html")
            print("Workflow diagram saved to stock_research_workflow.html")
        else:
            print("❌ Workflow visualization not available - llama_index.utils.workflow not found")
            print("This feature is disabled for Railway deployment compatibility")
        return
    
    # Run analysis
    print(f"\n🔍 Starting comprehensive analysis for {args.ticker}")
    print("=" * 60)
    
    try:
        result = await run_stock_analysis(args.ticker, args.company)
        
        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            return
        
        # Display results
        print(f"\n📊 Analysis Results for {result['company_name']} ({result['ticker']}):")
        print("-" * 60)
        print(f"Investment Score: {result['investment_score']:.1f}/10")
        print(f"Fair Value: ${result['fair_value']:.2f}")
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"Probability Undervalued: {result['probability_undervalued']:.1%}")
        
        # Component scores
        components = result.get('component_scores', {})
        if components:
            print(f"\nComponent Scores:")
            for component, score in components.items():
                print(f"  {component.title()}: {score:.1f}/10")
        
        # Research summary
        summary = result.get('research_summary', {})
        print(f"\nResearch Summary:")
        print(f"  Questions Answered: {summary.get('questions_answered', 0)}")
        print(f"  Average Confidence: {summary.get('average_confidence', 0):.1%}")
        print(f"  Analysis Duration: {summary.get('analysis_duration_seconds', 0):.1f} seconds")
        
        # Save output if requested
        if args.save_output:
            output_dir = Path("data/outputs") / args.ticker
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON results
            json_path = output_dir / f"{args.ticker}_analysis.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Save markdown report
            report_path = output_dir / f"{args.ticker}_report.md"
            with open(report_path, 'w') as f:
                f.write(result.get('report', 'No report generated'))
            
            print(f"\n💾 Results saved to {output_dir}")
            print(f"  JSON: {json_path}")
            print(f"  Report: {report_path}")
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        logger.exception("Analysis failed")


if __name__ == "__main__":
    asyncio.run(main())
