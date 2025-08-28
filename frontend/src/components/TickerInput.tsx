import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Search, TrendingUp } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
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
  const mockPreview = (symbol: string): CompanyPreview => {
    const mockData: Record<string, CompanyPreview> = {
      "TSLA": {
        symbol: "TSLA",
        name: "Tesla, Inc.",
        sector: "Consumer Discretionary",
        price: "$248.42"
      },
      "AAPL": {
        symbol: "AAPL",
        name: "Apple Inc.",
        sector: "Technology",
        price: "$192.53"
      },
      "MSFT": {
        symbol: "MSFT",
        name: "Microsoft Corporation",
        sector: "Technology",
        price: "$421.78"
      },
      "GOOGL": {
        symbol: "GOOGL",
        name: "Alphabet Inc.",
        sector: "Communication Services",
        price: "$141.52"
      },
      "NVDA": {
        symbol: "NVDA",
        name: "NVIDIA Corporation",
        sector: "Technology",
        price: "$462.89"
      }
    };
    return mockData[symbol] || {
      symbol,
      name: `${symbol} Corporation`,
      sector: "Unknown",
      price: "$0.00"
    };
  };
  const handleTickerChange = async (value: string) => {
    const upperValue = value.toUpperCase();
    setTicker(upperValue);
    if (upperValue.length >= 2) {
      setIsValidating(true);
      // Simulate API call delay
      setTimeout(() => {
        setPreview(mockPreview(upperValue));
        setIsValidating(false);
      }, 500);
    } else {
      setPreview(null);
    }
  };
  const validateTicker = async (tickerSymbol: string) => {
    try {
      const response = await fetch(`https://valuation100x-production.up.railway.app/api/validate/ticker/${tickerSymbol}`);
      const data = await response.json();
      return data.is_valid;
    } catch (error) {
      console.error('Validation API error:', error);
      // Fallback to mock validation for known tickers
      const knownTickers = ["TSLA", "AAPL", "MSFT", "GOOGL", "NVDA"];
      return knownTickers.includes(tickerSymbol);
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
        'https://valuation100x-production.up.railway.app/api/analysis/comprehensive/start',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ ticker }),
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
        <p className="text-lg text-muted-foreground">Get institutional-grade stock valuation reports in 15 minutes</p>
      </div>

      {/* Input Section */}
      <Card className="p-8 shadow-floating border-0 bg-gradient-subtle">
        <div className="space-y-6">
          <div className="space-y-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-muted-foreground" />
              <Input id="ticker" value={ticker} onChange={e => handleTickerChange(e.target.value)} placeholder="e.g., TSLA, AAPL, MSFT" className="pl-10 h-14 text-lg border-2 transition-smooth focus:border-primary" autoComplete="off" />
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

      {/* Popular Stocks */}
      <div className="text-center space-y-3">
        <p className="text-sm text-muted-foreground">Popular stocks to research:</p>
        <div className="flex flex-wrap justify-center gap-2">
          {["TSLA", "AAPL", "MSFT", "GOOGL", "NVDA"].map(symbol => <Button key={symbol} variant="outline" size="sm" onClick={() => handleTickerChange(symbol)} className="transition-smooth hover:bg-accent">
              {symbol}
            </Button>)}
        </div>
      </div>
    </div>;
};