import { useState } from "react";
import { TickerInput } from "@/components/TickerInput";
import { ProgressTracker } from "@/components/ProgressTracker";
import { ResearchReport } from "@/components/ResearchReport";
import { Navigation } from "@/components/Navigation";
import { Component as AnimatedTestimonials } from "@/components/ui/testimonial";
import { HeroSection } from "@/components/HeroSection";
type AppState = "input" | "analysis" | "report";
interface ReportData {
  ticker: string;
  recommendation: "Buy" | "Sell" | "Hold";
  intrinsicValue: number;
  currentPrice: number;
  upside: number;
  confidence: string;
}
const Index = () => {
  const [appState, setAppState] = useState<AppState>("input");
  const [currentTicker, setCurrentTicker] = useState<string>("");
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const handleStartAnalysis = (ticker: string) => {
    setCurrentTicker(ticker);
    setAppState("analysis");
  };
  const handleAnalysisComplete = (data: ReportData) => {
    setReportData(data);
    setAppState("report");
  };
  const handleBackToInput = () => {
    setAppState("input");
    setCurrentTicker("");
    setReportData(null);
  };
  return <div className="min-h-screen bg-gradient-subtle">
      <Navigation />
      <div className="container mx-auto px-4 py-8 pt-24">
        {appState === "input" && <>
            <HeroSection />
            
            <div className="flex items-center justify-center min-h-[60vh] py-16">
              <TickerInput onStartAnalysis={handleStartAnalysis} />
            </div>
            
            {/* How It Works */}
            <section className="py-16 bg-card">
              <div className="container mx-auto px-4">
                <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
                <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
                  <div className="text-center space-y-4">
                    <div className="w-16 h-16 bg-gradient-primary rounded-full flex items-center justify-center mx-auto">
                      <span className="text-2xl font-bold text-primary-foreground">1</span>
                    </div>
                    <h3 className="font-semibold text-lg">Enter Ticker</h3>
                    <p className="text-muted-foreground">Enter a stock ticker (e.g., TSLA) and we'll validate the company.</p>
                  </div>
                  <div className="text-center space-y-4">
                    <div className="w-16 h-16 bg-gradient-primary rounded-full flex items-center justify-center mx-auto">
                      <span className="text-2xl font-bold text-primary-foreground">2</span>
                    </div>
                    <h3 className="font-semibold text-lg">Deep Analysis</h3>
                    <p className="text-muted-foreground">We analyze financials, filings, sentiment, and valuation using the Damodaran framework.</p>
                  </div>
                  <div className="text-center space-y-4">
                    <div className="w-16 h-16 bg-gradient-primary rounded-full flex items-center justify-center mx-auto">
                      <span className="text-2xl font-bold text-primary-foreground">3</span>
                    </div>
                    <h3 className="font-semibold text-lg">Professional Report</h3>
                    <p className="text-muted-foreground">Receive a full markdown report with Buy/Sell/Hold recommendation in 15 minutes.</p>
                  </div>
                </div>
              </div>
            </section>

            {/* Example Report Preview */}
            <section className="py-20 bg-muted/30">
              <div className="container mx-auto px-4">
                <h2 className="text-3xl font-bold text-center mb-4">Example Report Preview</h2>
                <p className="text-center text-muted-foreground mb-16">See what you'll receive after our 15-minute analysis</p>
                
                <div className="max-w-5xl mx-auto">
                  <div className="bg-background rounded-xl border shadow-elegant p-12 space-y-8">
                    {/* Header */}
                    <div className="border-b pb-6">
                      <div className="flex items-start justify-between">
                        <div className="space-y-2">
                          <div className="flex items-center gap-3">
                            <h1 className="text-4xl font-serif font-bold tracking-tight">Tesla, Inc.</h1>
                            <span className="text-2xl font-mono font-bold text-muted-foreground">(TSLA)</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="px-3 py-1 bg-accent text-accent-foreground text-sm font-medium rounded-full">
                              Electric Vehicle & Clean Energy
                            </span>
                            <span className="text-sm text-muted-foreground">• NASDAQ • Large Cap</span>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="inline-flex items-center gap-2 bg-success text-success-foreground px-6 py-3 rounded-xl font-bold text-xl shadow-sm">
                            <div className="w-3 h-3 bg-success-foreground rounded-full"></div>
                            BUY
                          </div>
                          <p className="text-sm text-muted-foreground mt-2">Strong Buy Rating</p>
                        </div>
                      </div>
                    </div>

                    {/* Key Metrics */}
                    <div className="grid grid-cols-4 gap-8">
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-muted-foreground tracking-wide uppercase">Current Price</p>
                        <p className="text-3xl font-mono font-bold">$248.42</p>
                        <p className="text-sm text-muted-foreground">As of Dec 20, 2024</p>
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-muted-foreground tracking-wide uppercase">Intrinsic Value</p>
                        <p className="text-3xl font-mono font-bold text-success">$295.00</p>
                        <p className="text-sm text-success">+18.7% upside</p>
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-muted-foreground tracking-wide uppercase">Confidence</p>
                        <p className="text-2xl font-semibold">High</p>
                        <p className="text-sm text-muted-foreground">85% confidence</p>
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-muted-foreground tracking-wide uppercase">Time Horizon</p>
                        <p className="text-2xl font-semibold">12-18 months</p>
                        <p className="text-sm text-muted-foreground">Target period</p>
                      </div>
                    </div>

                    {/* Executive Summary */}
                    <div className="space-y-4 pt-6 border-t">
                      <h2 className="text-2xl font-serif font-bold">Executive Summary</h2>
                      <div className="prose prose-gray max-w-none">
                        <p className="text-foreground leading-relaxed text-lg">
                          Based on Damodaran valuation methodology, Tesla shows strong fundamentals in the growing EV market with solid execution on production targets. The company's vertical integration strategy and expanding energy business provide multiple value drivers beyond traditional automotive metrics.
                        </p>
                        <div className="flex items-center gap-4 mt-4 pt-4 border-t border-muted">
                          <span className="text-sm text-muted-foreground">Analysis completed:</span>
                          <span className="text-sm font-mono">15:42 minutes</span>
                          <span className="text-sm text-muted-foreground">•</span>
                          <span className="text-sm text-muted-foreground">Methodology:</span>
                          <span className="text-sm font-medium">Damodaran DCF + Scenario Analysis</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* Why DeepResearch */}
            <section className="py-16 bg-card">
              <div className="container mx-auto px-4">
                <h2 className="text-3xl font-bold text-center mb-12">Why Valuation100x?</h2>
                <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-primary rounded-full flex-shrink-0 mt-1"></div>
                    <div>
                      <h3 className="font-semibold mb-2">Institutional-grade insights in 15 minutes</h3>
                      <p className="text-muted-foreground">Get professional-quality research without waiting days or weeks.</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-primary rounded-full flex-shrink-0 mt-1"></div>
                    <div>
                      <h3 className="font-semibold mb-2">Built on trusted Damodaran methodology</h3>
                      <p className="text-muted-foreground">Uses the proven valuation framework from NYU Stern's Aswath Damodaran.</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-primary rounded-full flex-shrink-0 mt-1"></div>
                    <div>
                      <h3 className="font-semibold mb-2">Clear recommendations</h3>
                      <p className="text-muted-foreground">Simple Buy / Sell / Hold recommendations backed by transparent analysis.</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-primary rounded-full flex-shrink-0 mt-1"></div>
                    <div>
                      <h3 className="font-semibold mb-2">Shareable, professional reports</h3>
                      <p className="text-muted-foreground">Export to PDF or markdown for easy sharing and reference.</p>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* Testimonials */}
            <section className="bg-background py-16">
              <div className="container mx-auto px-4">
                <h2 className="text-3xl font-bold text-center mb-12">What Users Say</h2>
                <AnimatedTestimonials />
              </div>
            </section>

            {/* Built For */}
            <section className="py-16 bg-card">
              <div className="container mx-auto px-4">
                <h2 className="text-3xl font-bold text-center mb-8">Built For</h2>
                
                <div className="flex justify-center gap-4 flex-wrap">
                  <span className="bg-accent text-accent-foreground px-4 py-2 rounded-full font-medium">Individual Investors</span>
                  <span className="bg-accent text-accent-foreground px-4 py-2 rounded-full font-medium">DIY Analysts</span>
                  <span className="bg-accent text-accent-foreground px-4 py-2 rounded-full font-medium">Finance Nerds</span>
                </div>
              </div>
            </section>

            {/* Footer */}
            <footer className="py-12 border-t">
              <div className="container mx-auto px-4">
                <div className="flex justify-center space-x-8 mb-6">
                  <a href="#" className="text-muted-foreground hover:text-foreground transition-smooth">About</a>
                  <a href="#" className="text-muted-foreground hover:text-foreground transition-smooth">Security</a>
                  <a href="#" className="text-muted-foreground hover:text-foreground transition-smooth">Contact</a>
                </div>
                <p className="text-center text-sm text-muted-foreground">
                  For demo purposes only — not financial advice.
                </p>
              </div>
            </footer>
          </>}
        
        {appState === "analysis" && <div className="py-8">
            <ProgressTracker ticker={currentTicker} onComplete={handleAnalysisComplete} />
          </div>}
        
        {appState === "report" && reportData && <div className="py-8">
            <ResearchReport reportData={reportData} onBack={handleBackToInput} />
          </div>}
      </div>
    </div>;
};
export default Index;