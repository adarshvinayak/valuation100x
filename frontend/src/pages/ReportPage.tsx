import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ArrowLeft, Download, Copy, Printer, FileText } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import ReactMarkdown from "react-markdown";

interface ReportData {
  markdown_content: string;
  ticker?: string;
  company_name?: string;
}

const ReportPage = () => {
  const { analysisId } = useParams<{ analysisId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchReportData = async () => {
    if (!analysisId) return;

    try {
      setIsLoading(true);
      const response = await fetch(
        `https://valuation100x-production.up.railway.app/api/reports/${analysisId}/markdown`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch report: ${response.status}`);
      }

      const markdownContent = await response.text();
      
      // Get ticker from URL params
      const urlParams = new URLSearchParams(window.location.search);
      const ticker = urlParams.get('ticker');
      
      setReportData({
        markdown_content: markdownContent,
        ticker: ticker || undefined,
        company_name: undefined // Will be extracted from markdown content if needed
      });
      setError(null);
    } catch (err) {
      console.error('Error fetching report:', err);
      setError(err instanceof Error ? err.message : 'Failed to load report');
      
      // Mock data for demo purposes
      setReportData({
        markdown_content: `# Apple Inc. (AAPL) - Investment Analysis Report

## Executive Summary

Apple Inc. represents a compelling investment opportunity with significant upside potential. Our comprehensive analysis indicates a **BUY** recommendation with high confidence based on strong fundamentals, market position, and growth prospects.

### Key Metrics
- **Investment Score**: 8.5/10
- **Recommendation**: BUY
- **Fair Value**: $185.00
- **Current Price**: $150.25
- **Upside Potential**: +23.1%
- **Confidence Level**: 87%

## Investment Thesis

Apple continues to demonstrate exceptional financial performance and strategic positioning. The company's ecosystem approach, brand loyalty, and innovation pipeline support our bullish outlook.

### Key Strengths

1. **Dominant Market Position**
   - Leading smartphone market share in premium segment
   - Strong ecosystem lock-in effect
   - Brand loyalty unmatched in tech sector

2. **Financial Excellence**
   - Consistent revenue growth of 8-12% annually
   - Operating margins above 25%
   - Strong cash generation and balance sheet

3. **Innovation Pipeline**
   - Vision Pro represents new product category
   - AI integration across product lineup
   - Services segment showing strong growth

4. **Capital Allocation**
   - Shareholder-friendly dividend policy
   - Aggressive share buyback program
   - Strategic acquisitions and R&D investment

### Key Risks

1. **Market Saturation**
   - Smartphone market maturity in developed countries
   - Longer upgrade cycles for devices

2. **Geopolitical Risks**
   - China supply chain dependencies
   - Regulatory scrutiny in multiple markets

3. **Competition**
   - Android ecosystem competition
   - Emerging AI competitors

## Financial Analysis

### Revenue Streams
- iPhone: 52% of total revenue
- Services: 22% and growing
- Mac: 11%
- iPad: 8%
- Wearables: 7%

### Profitability Metrics
- Gross Margin: 44.1%
- Operating Margin: 25.3%
- Net Margin: 23.5%
- ROE: 63.2%

## Valuation Analysis

### DCF Model
Our discounted cash flow model suggests a fair value of $185 per share, representing 23.1% upside from current levels.

**Key Assumptions:**
- Revenue CAGR: 6.5% (2024-2028)
- Terminal Growth Rate: 2.5%
- Discount Rate: 9.2%

### Peer Comparison
Apple trades at premium multiples justified by superior:
- Profit margins
- Return on capital
- Brand strength
- Ecosystem effects

## Scenario Analysis

### Bull Case ($210 target)
- Successful Vision Pro adoption
- AI services monetization
- Market share gains in emerging markets
- Multiple expansion to historical levels

### Base Case ($185 target)
- Steady iPhone replacement cycles
- Services growth continuation
- Margin stability
- Current market conditions

### Bear Case ($135 target)
- iPhone demand deterioration
- Increased competition pressure
- Regulatory headwinds
- Economic downturn impact

## Conclusion

Apple Inc. represents a high-quality investment with attractive risk-adjusted returns. The company's strong competitive position, financial performance, and growth prospects support our BUY recommendation.

### Investment Recommendation: **BUY**
### Price Target: **$185.00**
### Risk Rating: **Moderate**

---

*This analysis is for informational purposes only and does not constitute financial advice. Please consult with a qualified financial advisor before making investment decisions.*

*Report generated on ${new Date().toLocaleDateString()}*`,
        ticker: "DEMO",
        company_name: "Demo Company"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const generatePdf = async () => {
    if (!analysisId) return;

    try {
      setIsGeneratingPdf(true);
      const response = await fetch(
        `https://valuation100x-production.up.railway.app/api/reports/${analysisId}/pdf`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to generate PDF: ${response.status}`);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${reportData?.ticker || 'report'}_analysis_${new Date().toISOString().split('T')[0]}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast({
        title: "PDF Generated",
        description: "Report downloaded as PDF",
      });
    } catch (err) {
      console.error('Error generating PDF:', err);
      toast({
        title: "PDF Generation Failed",
        description: err instanceof Error ? err.message : 'Failed to generate PDF',
        variant: "destructive",
      });
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  const handleCopy = () => {
    if (!reportData?.markdown_content) return;
    
    navigator.clipboard.writeText(reportData.markdown_content);
    toast({
      title: "Report Copied",
      description: "Full report copied to clipboard",
    });
  };

  const handlePrint = () => {
    window.print();
  };

  const handleDownload = () => {
    if (!reportData?.markdown_content) return;
    
    const blob = new Blob([reportData.markdown_content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${reportData.ticker || 'report'}_analysis_${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Report Downloaded",
      description: "Report saved as Markdown file",
    });
  };

  useEffect(() => {
    if (!analysisId) {
      navigate('/');
      return;
    }
    fetchReportData();
  }, [analysisId]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-subtle">
        <div className="container mx-auto px-4 py-8">
          <div className="max-w-5xl mx-auto space-y-6">
            <Skeleton className="h-8 w-48" />
            <Card className="p-8">
              <Skeleton className="h-64 w-full" />
            </Card>
            <Card className="p-8">
              <Skeleton className="h-96 w-full" />
            </Card>
          </div>
        </div>
      </div>
    );
  }

  if (error || !reportData) {
    return (
      <div className="min-h-screen bg-gradient-subtle">
        <div className="container mx-auto px-4 py-8">
          <div className="max-w-5xl mx-auto">
            <Card className="p-8 text-center">
              <h2 className="text-2xl font-semibold mb-4">Report Not Found</h2>
              <p className="text-muted-foreground mb-6">
                {error || "The requested report could not be found."}
              </p>
              <Button onClick={() => navigate('/')}>
                Back to Home
              </Button>
            </Card>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Button 
            onClick={() => navigate('/')}
            variant="outline" 
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </Button>
          
          <div className="flex items-center gap-2">
            <Button onClick={handleCopy} variant="outline" size="sm">
              <Copy className="h-4 w-4 mr-2" />
              Copy
            </Button>
            <Button onClick={handlePrint} variant="outline" size="sm">
              <Printer className="h-4 w-4 mr-2" />
              Print
            </Button>
            <Button onClick={handleDownload} variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Download
            </Button>
            <Button 
              onClick={generatePdf} 
              variant="default" 
              size="sm"
              disabled={isGeneratingPdf}
            >
              <FileText className="h-4 w-4 mr-2" />
              {isGeneratingPdf ? 'Generating...' : 'PDF Report'}
            </Button>
          </div>
        </div>

        <div className="max-w-5xl mx-auto">
          {/* Title */}
          {reportData?.company_name && reportData?.ticker && (
            <div className="text-center space-y-2 mb-8">
              <h1 className="text-4xl font-bold">{reportData.company_name}</h1>
              <div className="flex items-center justify-center gap-2">
                <span className="text-2xl font-mono text-muted-foreground">({reportData.ticker})</span>
                <Badge variant="outline">Investment Report</Badge>
              </div>
            </div>
          )}

          {/* Full Report Content */}
          <Card className="p-8 shadow-floating">
            <div className="prose prose-lg max-w-none dark:prose-invert markdown-content">
              <ReactMarkdown 
                components={{
                  h1: ({ children, ...props }) => (
                    <h1 className="text-4xl font-bold mb-6 text-foreground border-b pb-4" {...props}>
                      {children}
                    </h1>
                  ),
                  h2: ({ children, ...props }) => (
                    <h2 className="text-3xl font-semibold mt-8 mb-4 text-foreground" {...props}>
                      {children}
                    </h2>
                  ),
                  h3: ({ children, ...props }) => (
                    <h3 className="text-2xl font-semibold mt-6 mb-3 text-foreground" {...props}>
                      {children}
                    </h3>
                  ),
                  p: ({ children, ...props }) => (
                    <p className="text-foreground leading-relaxed mb-4" {...props}>
                      {children}
                    </p>
                  ),
                  ul: ({ children, ...props }) => (
                    <ul className="list-disc list-inside space-y-2 mb-4 text-foreground" {...props}>
                      {children}
                    </ul>
                  ),
                  ol: ({ children, ...props }) => (
                    <ol className="list-decimal list-inside space-y-2 mb-4 text-foreground" {...props}>
                      {children}
                    </ol>
                  ),
                  li: ({ children, ...props }) => (
                    <li className="text-foreground" {...props}>
                      {children}
                    </li>
                  ),
                  strong: ({ children, ...props }) => (
                    <strong className="font-bold text-foreground" {...props}>
                      {children}
                    </strong>
                  ),
                  blockquote: ({ children, ...props }) => (
                    <blockquote className="border-l-4 border-primary pl-4 italic text-muted-foreground" {...props}>
                      {children}
                    </blockquote>
                  ),
                  table: ({ children, ...props }) => (
                    <div className="overflow-x-auto mb-4">
                      <table className="min-w-full border-collapse border border-border" {...props}>
                        {children}
                      </table>
                    </div>
                  ),
                  th: ({ children, ...props }) => (
                    <th className="border border-border px-4 py-2 bg-muted font-semibold text-left" {...props}>
                      {children}
                    </th>
                  ),
                  td: ({ children, ...props }) => (
                    <td className="border border-border px-4 py-2" {...props}>
                      {children}
                    </td>
                  ),
                }}
              >
                {reportData?.markdown_content || ''}
              </ReactMarkdown>
            </div>
          </Card>

          {/* Disclaimer */}
          <Card className="p-6 mt-8 bg-muted/50">
            <p className="text-sm text-muted-foreground text-center">
              <strong>Disclaimer:</strong> This report is for informational purposes only and does not constitute financial advice. 
              Please consult with a qualified financial advisor before making investment decisions.
            </p>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ReportPage;