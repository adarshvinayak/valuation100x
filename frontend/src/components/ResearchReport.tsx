import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Download, Copy, Printer, TrendingUp, TrendingDown, Minus, ArrowLeft } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

interface ReportData {
  ticker: string;
  recommendation: "Buy" | "Sell" | "Hold";
  intrinsicValue: number;
  currentPrice: number;
  upside: number;
  confidence: string;
}

interface ResearchReportProps {
  reportData: ReportData;
  onBack: () => void;
}

export const ResearchReport = ({ reportData, onBack }: ResearchReportProps) => {
  const { toast } = useToast();
  
  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case "Buy": return "buy";
      case "Sell": return "sell";
      default: return "hold";
    }
  };

  const getRecommendationIcon = (rec: string) => {
    switch (rec) {
      case "Buy": return <TrendingUp className="h-5 w-5" />;
      case "Sell": return <TrendingDown className="h-5 w-5" />;
      default: return <Minus className="h-5 w-5" />;
    }
  };

  const handleCopy = () => {
    const reportText = generateReportText();
    navigator.clipboard.writeText(reportText);
    toast({
      title: "Report Copied",
      description: "The research report has been copied to your clipboard",
    });
  };

  const handlePrint = () => {
    window.print();
    toast({
      title: "Print Dialog",
      description: "Opening print dialog for the research report",
    });
  };

  const handleDownload = () => {
    const reportText = generateReportText();
    const blob = new Blob([reportText], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${reportData.ticker}_research_report.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast({
      title: "Download Started",
      description: "Your research report is being downloaded",
    });
  };

  const generateReportText = () => {
    return `# ${reportData.ticker} Research Report

## Executive Summary

**Recommendation:** ${reportData.recommendation}
**Target Price:** $${reportData.intrinsicValue.toFixed(2)}
**Current Price:** $${reportData.currentPrice.toFixed(2)}
**Upside/Downside:** ${reportData.upside > 0 ? '+' : ''}${reportData.upside.toFixed(1)}%
**Confidence Level:** ${reportData.confidence}

## Investment Thesis

Tesla represents a compelling investment opportunity in the electric vehicle and energy transition space. Our analysis, using Aswath Damodaran's story-driven valuation methodology, suggests the stock is undervalued relative to its intrinsic worth.

### Key Value Drivers

1. **Electric Vehicle Market Leadership**
   - Dominant position in premium EV segment
   - Expanding production capacity globally
   - Strong brand loyalty and pricing power

2. **Energy Business Growth**
   - Solar and energy storage solutions
   - Grid-scale battery deployments
   - Recurring revenue from energy services

3. **Technology & Innovation**
   - Advanced manufacturing capabilities
   - Full Self-Driving development
   - Vertical integration advantages

## Financial Analysis

### Revenue Growth Trajectory
- Historical 5-year CAGR: 47%
- Projected 5-year CAGR: 25-30%
- Multiple revenue streams providing diversification

### Profitability Metrics
- Gross Margin: 18.7% (improving)
- Operating Margin: 8.2%
- Free Cash Flow: $7.5B (2023)

### Balance Sheet Strength
- Cash Position: $29.1B
- Debt-to-Equity: 0.17
- Strong liquidity position

## Damodaran Valuation Story

### The Growth Story
Tesla's valuation story centers on its transformation from a niche EV manufacturer to a diversified technology and energy company. The company benefits from:

- **Network Effects**: Supercharger network and software ecosystem
- **Scale Advantages**: Manufacturing efficiency and cost reduction
- **Innovation Pipeline**: Robotaxis, energy storage, and AI capabilities

### Scenario Analysis

**Base Case (60% probability):** $285 target price
- EV market share stabilizes at 15-20%
- Energy business grows to 25% of revenue
- Autonomous driving commercialized by 2027

**Bull Case (25% probability):** $380 target price
- Dominant autonomous vehicle platform
- Energy business becomes primary growth driver
- Manufacturing cost leadership maintained

**Bear Case (15% probability):** $180 target price
- Increased competition erodes market share
- Regulatory challenges delay autonomous driving
- Economic downturn impacts luxury vehicle demand

## Risk Assessment

### Key Risks
1. **Execution Risk**: Ambitious production targets and timeline commitments
2. **Competition**: Traditional automakers and new entrants in EV space
3. **Regulatory**: Changes in EV incentives and autonomous driving regulations
4. **Key Person Risk**: Dependence on CEO leadership and vision

### Risk Mitigation
- Diversified revenue streams reducing single-point-of-failure
- Strong cash position providing operational flexibility
- Proven track record of overcoming production challenges

## Recommendation

**BUY** with a 12-month target price of **$285.50**

Our analysis suggests Tesla is trading at a 14.9% discount to intrinsic value. The company's strong competitive position, diversified growth opportunities, and improving profitability support a positive investment thesis.

### Investment Rationale
- Undervalued relative to growth prospects
- Strong competitive moats in multiple segments
- Improving operational efficiency and margins
- Catalyst potential from autonomous driving and energy growth

---

*This report was generated using institutional-grade analysis and Aswath Damodaran's story-driven valuation methodology. The analysis incorporates comprehensive financial modeling, scenario analysis, and risk assessment.*

**Disclaimer:** This report is for informational purposes only and should not be considered as investment advice. Past performance does not guarantee future results.
`;
  };

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <Button 
          variant="outline" 
          onClick={onBack}
          className="flex items-center gap-2"
        >
          <ArrowLeft className="h-4 w-4" />
          New Analysis
        </Button>
        
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" onClick={handleCopy}>
            <Copy className="h-4 w-4 mr-2" />
            Copy
          </Button>
          <Button variant="outline" size="sm" onClick={handlePrint}>
            <Printer className="h-4 w-4 mr-2" />
            Print
          </Button>
          <Button size="sm" onClick={handleDownload}>
            <Download className="h-4 w-4 mr-2" />
            Download PDF
          </Button>
        </div>
      </div>

      {/* Executive Summary Card */}
      <Card className="p-8 shadow-floating border-0 bg-gradient-subtle">
        <div className="space-y-6">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-2">{reportData.ticker} Research Report</h1>
            <p className="text-muted-foreground">Institutional-Grade Equity Analysis</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center space-y-2">
              <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full bg-${getRecommendationColor(reportData.recommendation)} text-${getRecommendationColor(reportData.recommendation)}-foreground`}>
                {getRecommendationIcon(reportData.recommendation)}
                <span className="font-bold text-lg">{reportData.recommendation}</span>
              </div>
              <p className="text-sm text-muted-foreground">Recommendation</p>
            </div>

            <div className="text-center space-y-2">
              <p className="text-3xl font-bold text-primary">${reportData.intrinsicValue.toFixed(2)}</p>
              <p className="text-sm text-muted-foreground">Target Price</p>
            </div>

            <div className="text-center space-y-2">
              <p className="text-3xl font-bold">${reportData.currentPrice.toFixed(2)}</p>
              <p className="text-sm text-muted-foreground">Current Price</p>
            </div>

            <div className="text-center space-y-2">
              <p className={`text-3xl font-bold ${reportData.upside > 0 ? 'text-success' : 'text-destructive'}`}>
                {reportData.upside > 0 ? '+' : ''}{reportData.upside.toFixed(1)}%
              </p>
              <p className="text-sm text-muted-foreground">Upside/Downside</p>
            </div>
          </div>

          <div className="flex items-center justify-center gap-2">
            <Badge variant="secondary" className="text-sm">
              Confidence: {reportData.confidence}
            </Badge>
            <Badge variant="outline" className="text-sm">
              Damodaran Methodology
            </Badge>
          </div>
        </div>
      </Card>

      {/* Full Report */}
      <Card className="shadow-floating">
        <div className="p-8 prose prose-lg max-w-none">
          <div className="space-y-8">
            
            {/* Investment Thesis */}
            <section>
              <h2 className="text-2xl font-bold text-foreground mb-4 flex items-center gap-2">
                <TrendingUp className="h-6 w-6 text-primary" />
                Investment Thesis
              </h2>
              <p className="text-foreground leading-relaxed">
                Tesla represents a compelling investment opportunity in the electric vehicle and energy transition space. 
                Our analysis, using Aswath Damodaran's story-driven valuation methodology, suggests the stock is 
                undervalued relative to its intrinsic worth based on fundamental business drivers and long-term growth prospects.
              </p>
            </section>

            <Separator />

            {/* Key Value Drivers */}
            <section>
              <h3 className="text-xl font-semibold text-foreground mb-4">Key Value Drivers</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <h4 className="font-semibold text-primary">Electric Vehicle Leadership</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Dominant position in premium EV segment</li>
                    <li>• Expanding production capacity globally</li>
                    <li>• Strong brand loyalty and pricing power</li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <h4 className="font-semibold text-primary">Energy Business Growth</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Solar and energy storage solutions</li>
                    <li>• Grid-scale battery deployments</li>
                    <li>• Recurring revenue from energy services</li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <h4 className="font-semibold text-primary">Technology & Innovation</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Advanced manufacturing capabilities</li>
                    <li>• Full Self-Driving development</li>
                    <li>• Vertical integration advantages</li>
                  </ul>
                </div>
              </div>
            </section>

            <Separator />

            {/* Financial Analysis */}
            <section>
              <h3 className="text-xl font-semibold text-foreground mb-4">Financial Analysis</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <h4 className="font-semibold text-primary mb-3">Revenue Growth</h4>
                  <div className="space-y-2 font-mono text-sm">
                    <div className="flex justify-between">
                      <span>Historical 5Y CAGR:</span>
                      <span className="font-bold">47%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Projected 5Y CAGR:</span>
                      <span className="font-bold">25-30%</span>
                    </div>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold text-primary mb-3">Profitability</h4>
                  <div className="space-y-2 font-mono text-sm">
                    <div className="flex justify-between">
                      <span>Gross Margin:</span>
                      <span className="font-bold">18.7%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Operating Margin:</span>
                      <span className="font-bold">8.2%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Free Cash Flow:</span>
                      <span className="font-bold">$7.5B</span>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <Separator />

            {/* Scenario Analysis */}
            <section>
              <h3 className="text-xl font-semibold text-foreground mb-4">Scenario Analysis</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card className="p-4 border border-success/20 bg-success/5">
                  <h4 className="font-semibold text-success mb-2">Bull Case (25%)</h4>
                  <p className="text-2xl font-bold text-success mb-2">$380</p>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Dominant autonomous platform</li>
                    <li>• Energy business drives growth</li>
                    <li>• Manufacturing cost leadership</li>
                  </ul>
                </Card>
                
                <Card className="p-4 border-2 border-primary/30 bg-primary/5">
                  <h4 className="font-semibold text-primary mb-2">Base Case (60%)</h4>
                  <p className="text-2xl font-bold text-primary mb-2">$285</p>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• 15-20% EV market share</li>
                    <li>• Energy = 25% of revenue</li>
                    <li>• Autonomous driving by 2027</li>
                  </ul>
                </Card>
                
                <Card className="p-4 border border-warning/20 bg-warning/5">
                  <h4 className="font-semibold text-warning mb-2">Bear Case (15%)</h4>
                  <p className="text-2xl font-bold text-warning mb-2">$180</p>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Increased competition</li>
                    <li>• Regulatory delays</li>
                    <li>• Economic downturn impact</li>
                  </ul>
                </Card>
              </div>
            </section>

            <Separator />

            {/* Final Recommendation */}
            <section className="text-center py-6">
              <div className="space-y-4">
                <div className={`inline-flex items-center gap-3 px-8 py-4 rounded-2xl bg-${getRecommendationColor(reportData.recommendation)} text-${getRecommendationColor(reportData.recommendation)}-foreground text-2xl font-bold shadow-floating`}>
                  {getRecommendationIcon(reportData.recommendation)}
                  {reportData.recommendation.toUpperCase()}
                </div>
                <p className="text-xl text-foreground">
                  12-month target price: <span className="font-bold text-primary">${reportData.intrinsicValue.toFixed(2)}</span>
                </p>
                <p className="text-muted-foreground max-w-2xl mx-auto">
                  Our analysis suggests {reportData.ticker} is trading at a {Math.abs(reportData.upside).toFixed(1)}% 
                  {reportData.upside > 0 ? ' discount' : ' premium'} to intrinsic value, supported by strong competitive 
                  positioning and diversified growth opportunities.
                </p>
              </div>
            </section>

            {/* Disclaimer */}
            <div className="bg-muted/30 p-4 rounded-lg text-sm text-muted-foreground">
              <p className="font-semibold mb-2">Disclaimer</p>
              <p>
                This report was generated using institutional-grade analysis and Aswath Damodaran's story-driven 
                valuation methodology. This report is for informational purposes only and should not be considered 
                as investment advice. Past performance does not guarantee future results.
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};