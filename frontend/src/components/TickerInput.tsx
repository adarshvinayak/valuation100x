import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Search, AlertCircle, CheckCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { API_ENDPOINTS } from "@/config/api";

interface ValidationResult {
  symbol: string;
  isValid: boolean;
  companyName?: string;
  sector?: string;
  error?: string;
}

interface TickerInputProps {
  onStartAnalysis: (ticker: string) => void;
}

export const TickerInput = ({
  onStartAnalysis
}: TickerInputProps) => {
  const [ticker, setTicker] = useState("");
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [isStartingAnalysis, setIsStartingAnalysis] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();

  const handleTickerChange = (value: string) => {
    const upperValue = value.toUpperCase();
    setTicker(upperValue);
    // Clear any previous validation results when user types
    setValidationResult(null);
  };
  const handleSubmit = async () => {
    if (!ticker || ticker.trim() === "" || isStartingAnalysis) return;
    
    setIsStartingAnalysis(true);
    setIsValidating(true);
    
    try {
      // First validate the ticker
      console.log('Validating ticker:', ticker);
      const response = await fetch(API_ENDPOINTS.VALIDATE_TICKER(ticker));
      console.log('Validation response status:', response.status);
      
      const data = await response.json();
      console.log('Validation data:', data);
      
      if (data.is_valid && data.company_name) {
        // Valid ticker - show success card and start analysis
        setValidationResult({
          symbol: ticker,
          isValid: true,
          companyName: data.company_name,
          sector: data.sector || "Unknown"
        });
        
        setIsValidating(false);
        
        // Start the analysis
        console.log('Starting analysis for:', ticker, data.company_name);
        
        const analysisResponse = await fetch(
          API_ENDPOINTS.START_ANALYSIS,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
              ticker,
              company_name: data.company_name 
            }),
          }
        );

        console.log('Analysis response status:', analysisResponse.status);

        if (!analysisResponse.ok) {
          const errorData = await analysisResponse.json().catch(() => ({}));
          console.error('Analysis start failed:', errorData);
          throw new Error(errorData.details || `Request failed with status ${analysisResponse.status}`);
        }

        const analysisData = await analysisResponse.json();
        console.log('Analysis started:', analysisData);
        
        toast({
          title: "Analysis Started",
          description: `Starting comprehensive analysis for ${ticker}`,
        });
        
        // Navigate to analysis progress page
        navigate(`/analysis/${analysisData.analysis_id}?ticker=${ticker}`);
        
      } else {
        // Invalid ticker - show error card
        setValidationResult({
          symbol: ticker,
          isValid: false,
          error: data.error || "Ticker not found in any data source"
        });
        setIsValidating(false);
        setIsStartingAnalysis(false);
      }
      
    } catch (error) {
      console.error('Error during validation/analysis:', error);
      setValidationResult({
        symbol: ticker,
        isValid: false,
        error: "Failed to validate ticker. Please try again."
      });
      setIsValidating(false);
      setIsStartingAnalysis(false);
    }
  };
  return <div className="w-full max-w-2xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-3">
          
          <h1 className="text-3xl font-bold text-center">Enter Stock Ticker</h1>
        </div>
        <p className="text-lg text-muted-foreground">Get institutional-grade valuation reports for any US stock in 15 minutes</p>
      </div>

      {/* Input Section */}
      <Card className="p-8 shadow-floating border-0 bg-gradient-subtle">
        <div className="space-y-6">
          <div className="space-y-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-muted-foreground" />
              <Input id="ticker" value={ticker} onChange={e => handleTickerChange(e.target.value)} placeholder="Enter any US stock symbol (e.g., TSLA, MSFT, NVDA, etc.)" className="pl-10 h-14 text-lg border-2 transition-smooth focus:border-primary" autoComplete="off" />
            </div>
          </div>

          {/* Validation Loading */}
          {isValidating && <div className="animate-pulse">
              <div className="h-4 bg-muted rounded w-3/4 mb-2"></div>
              <div className="h-3 bg-muted rounded w-1/2"></div>
            </div>}

          {/* Validation Result Cards */}
          {validationResult && !isValidating && (
            <div className="space-y-4">
              {validationResult.isValid ? (
                // Valid Ticker Card
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="h-6 w-6 text-green-600" />
                    <div className="flex-1">
                      <h3 className="font-semibold text-green-900">{validationResult.companyName}</h3>
                      <p className="text-sm text-green-700">{validationResult.sector}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-lg font-semibold text-green-600">✓ Valid</p>
                      <p className="text-sm text-green-600">Starting analysis...</p>
                    </div>
                  </div>
                </div>
              ) : (
                // Invalid Ticker Card
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-center gap-3">
                    <AlertCircle className="h-6 w-6 text-red-600" />
                    <div className="flex-1">
                      <h3 className="font-semibold text-red-900">Invalid ticker symbol</h3>
                      <p className="text-sm text-red-700">{validationResult.error}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-lg font-semibold text-red-600">✗ Invalid</p>
                      <p className="text-sm text-red-600">Please try another ticker</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Start Analysis Button */}
          <Button 
            onClick={handleSubmit} 
            disabled={ticker.trim() === "" || isValidating || isStartingAnalysis} 
            className="w-full h-14 text-lg bg-gradient-primary hover:opacity-90 transition-smooth shadow-floating disabled:opacity-50"
          >
            {isStartingAnalysis ? "Starting Analysis..." : "Start Deep Analysis"}
          </Button>
        </div>
      </Card>

      {/* Example Stocks */}
      <div className="text-center space-y-3">
        <p className="text-sm text-muted-foreground">Try any US stock symbol (e.g., TSLA, MSFT, NVDA, GOOGL, AMZN, META, etc.)</p>
        <div className="flex flex-wrap justify-center gap-2">
          {["TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "JNJ", "UNH", "DIS"].map(symbol => <Button key={symbol} variant="outline" size="sm" onClick={() => handleTickerChange(symbol)} className="transition-smooth hover:bg-accent">
              {symbol}
            </Button>)}
        </div>
        <p className="text-xs text-muted-foreground mt-2">Supports 5,000+ US stocks across all sectors</p>
      </div>
    </div>;
};