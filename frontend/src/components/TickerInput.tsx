import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Search, TrendingUp } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { API_ENDPOINTS } from "@/config/api";
interface CompanyPreview {
  symbol: string;
  name: string;
  sector: string;
  price: string;
}
interface TickerInputProps {
  onStartAnalysis: (ticker: string) => void;
}
export const TickerInput = ({
  onStartAnalysis
}: TickerInputProps) => {
  const [ticker, setTicker] = useState("");
  const [preview, setPreview] = useState<CompanyPreview | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [isStartingAnalysis, setIsStartingAnalysis] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();

  const handleTickerChange = async (value: string) => {
    const upperValue = value.toUpperCase();
    setTicker(upperValue);
    
    if (upperValue.length >= 1) {
      setIsValidating(true);
      
      try {
        // Get real data from Lambda API for any US stock
        const response = await fetch(API_ENDPOINTS.VALIDATE_TICKER(upperValue));
        const data = await response.json();
        
        if (data.is_valid && data.company_name) {
          setPreview({
            symbol: upperValue,
            name: data.company_name,
            sector: data.sector || "Unknown",
            price: data.current_price ? `$${data.current_price.toFixed(2)}` : 
                   data.market_cap ? `$${(data.market_cap / 1000000000).toFixed(1)}B MC` : "N/A"
          });
        } else {
          // Clear preview if ticker is not valid
          setPreview(null);
        }
      } catch (error) {
        console.error('Error fetching ticker data:', error);
        // Clear preview on error
        setPreview(null);
      } finally {
        setIsValidating(false);
      }
    } else {
      setPreview(null);
    }
  };
  const validateTicker = async (tickerSymbol: string) => {
    try {
      const response = await fetch(API_ENDPOINTS.VALIDATE_TICKER(tickerSymbol));
      const data = await response.json();
      return data.is_valid;
    } catch (error) {
      console.error('Validation API error:', error);
      return false;
    }
  };


  const handleSubmit = async () => {
    if (!ticker || !preview || isStartingAnalysis) return;
    
    setIsStartingAnalysis(true);
    
    try {
      // First validate the ticker
      const isValid = await validateTicker(ticker);
      
      if (!isValid) {
        toast({
          title: "Invalid ticker symbol",
          description: "Please enter a valid stock ticker symbol.",
          variant: "destructive",
        });
        setIsStartingAnalysis(false);
        return;
      }

      // Start the analysis via API
      const response = await fetch(
        API_ENDPOINTS.START_ANALYSIS,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            ticker,
            company_name: preview.name 
          }),
        }
      );

      if (!response.ok) {
        throw new Error('Failed to start analysis');
      }

      const analysisData = await response.json();
      
      toast({
        title: "Analysis Started",
        description: `Starting comprehensive analysis for ${ticker}`,
      });
      
      // Navigate to analysis progress page with ticker parameter
      navigate(`/analysis/${analysisData.analysis_id}?ticker=${ticker}`);
      
    } catch (error) {
      toast({
        title: "Analysis Failed",
        description: "Unable to start analysis. Please try again.",
        variant: "destructive",
      });
    } finally {
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
              <Input id="ticker" value={ticker} onChange={e => handleTickerChange(e.target.value)} placeholder="Enter any US stock symbol (e.g., AAPL, GOOGL, AMD, etc.)" className="pl-10 h-14 text-lg border-2 transition-smooth focus:border-primary" autoComplete="off" />
            </div>
          </div>

          {/* Company Preview */}
          {isValidating && <div className="animate-pulse">
              <div className="h-4 bg-muted rounded w-3/4 mb-2"></div>
              <div className="h-3 bg-muted rounded w-1/2"></div>
            </div>}

          {preview && !isValidating && <div className="p-4 bg-card rounded-lg border shadow-card">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-lg">{preview.name}</h3>
                  <p className="text-sm text-muted-foreground">{preview.sector}</p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-primary">{preview.price}</p>
                  <p className="text-sm text-muted-foreground">Current Price</p>
                </div>
              </div>
            </div>}

          {/* Start Analysis Button */}
          <Button 
            onClick={handleSubmit} 
            disabled={!preview || isValidating || isStartingAnalysis} 
            className="w-full h-14 text-lg bg-gradient-primary hover:opacity-90 transition-smooth shadow-floating disabled:opacity-100 disabled:bg-gradient-primary"
          >
            {isStartingAnalysis ? "Starting Analysis..." : "Start Deep Analysis"}
          </Button>
        </div>
      </Card>

      {/* Example Stocks */}
      <div className="text-center space-y-3">
        <p className="text-sm text-muted-foreground">Try any US stock symbol (e.g., AAPL, GOOGL, TSLA, AMD, NVDA, AMZN, META, etc.)</p>
        <div className="flex flex-wrap justify-center gap-2">
          {["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA", "AMZN", "META", "JPM", "JNJ", "UNH"].map(symbol => <Button key={symbol} variant="outline" size="sm" onClick={() => handleTickerChange(symbol)} className="transition-smooth hover:bg-accent">
              {symbol}
            </Button>)}
        </div>
        <p className="text-xs text-muted-foreground mt-2">Supports 5,000+ US stocks across all sectors</p>
      </div>
    </div>;
};