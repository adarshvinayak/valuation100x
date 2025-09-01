import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ArrowLeft, X } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { API_ENDPOINTS } from "@/config/api";

interface ProgressMessage {
  timestamp: string;
  message: string;
  type: 'info' | 'progress' | 'success' | 'error' | 'warning';
}

const AnalysisProgress = () => {
  const { analysisId } = useParams<{ analysisId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  
  const [progress, setProgress] = useState(0);
  const [progressMessages, setProgressMessages] = useState<ProgressMessage[]>([]);
  const [connectionStatus, setConnectionStatus] = useState('Connecting');
  const [webSocket, setWebSocket] = useState<WebSocket | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const [tickerDetails, setTickerDetails] = useState<any>(null);
  
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight;
      }
    }
  }, [progressMessages]);

  const handleStatusUpdate = (status: any) => {
    console.log('Status update received:', status);
    
    // Update progress
    if (status.progress !== undefined) {
      setProgress(status.progress);
    }
    
    // Store ticker details for display
    if (status.company_name || status.ticker) {
      setTickerDetails({
        ticker: status.ticker,
        company_name: status.company_name,
        current_price: status.current_price,
        market_cap: status.market_cap,
        exchange: status.exchange,
        sector: status.sector
      });
    }
    
    // Add progress message with user-friendly text
    const userMessage = status.user_message || status.current_step_description || `Analysis progress: ${status.progress}%`;
    setProgressMessages(prev => [...prev, {
      timestamp: new Date().toLocaleTimeString(),
      message: userMessage,
      type: status.status === 'error' ? 'error' : 
             status.progress >= 100 ? 'success' : 'progress'
    }]);
    
    // Handle completion
    if (status.status === 'completed' || status.progress >= 100) {
      setConnectionStatus('Completed');
      setProgressMessages(prev => [...prev, {
        timestamp: new Date().toLocaleTimeString(),
        message: '🎉 Analysis completed! Generating report...',
        type: 'success'
      }]);
      setTimeout(() => {
        navigate(`/report/${status.analysis_id}`);
      }, 2000);
    } else if (status.status === 'error') {
      setConnectionStatus('Error');
      setProgressMessages(prev => [...prev, {
        timestamp: new Date().toLocaleTimeString(),
        message: `❌ Error: ${status.error || 'An error occurred during analysis'}`,
        type: 'error'
      }]);
    }
  };

  const fetchTickerDetails = async (ticker: string) => {
    try {
      console.log('Fetching ticker details for:', ticker);
      const response = await fetch(API_ENDPOINTS.VALIDATE_TICKER(ticker));
      if (response.ok) {
        const data = await response.json();
        if (data.is_valid) {
          setTickerDetails({
            ticker: ticker,
            company_name: data.company_name,
            current_price: data.current_price,
            market_cap: data.market_cap,
            exchange: data.exchange,
            sector: data.sector
          });
        }
      }
    } catch (error) {
      console.error('Failed to fetch ticker details:', error);
    }
  };

  const connectToExistingAnalysis = async (existingAnalysisId: string, ticker: string) => {
    try {
      // Analysis already exists - start polling for progress updates
      console.log(`Starting polling for analysis: ${existingAnalysisId} for ${ticker}`);
      
      // Fetch ticker details if we have the ticker from URL
      if (ticker) {
        await fetchTickerDetails(ticker);
      }
      
      setProgressMessages(prev => [...prev, {
        timestamp: new Date().toLocaleTimeString(),
        message: `Connected to analysis ${existingAnalysisId}`,
        type: 'info'
      }]);

      // Start polling for status updates
      const pollStatus = async () => {
        try {
          const response = await fetch(API_ENDPOINTS.ANALYSIS_STATUS(existingAnalysisId));
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
          
          const status = await response.json();
          handleStatusUpdate(status);
          
          // Stop polling if analysis is complete or error
          if (status.status === 'completed' || status.status === 'error' || status.status === 'cancelled') {
            if (pingIntervalRef.current) {
              clearInterval(pingIntervalRef.current);
              pingIntervalRef.current = null;
            }
          }
        } catch (error) {
          console.error('Failed to fetch analysis status:', error);
          setConnectionStatus('Error');
        }
      };

      // Start polling immediately, then every 2 seconds
      setConnectionStatus('Connected');
      pollStatus();
      const pollingInterval = setInterval(pollStatus, 2000);
      pingIntervalRef.current = pollingInterval;


      
    } catch (error) {
      console.error('Failed to connect to existing analysis:', error);
      setProgressMessages(prev => [...prev, {
        timestamp: new Date().toLocaleTimeString(),
        message: `❌ Failed to connect to analysis: ${error.message}`,
        type: 'error'
      }]);
      
      // Fallback: redirect back to home
      setTimeout(() => {
        navigate('/');
      }, 3000);
    }
  };

  // Cancel analysis function
  const cancelAnalysis = async () => {
    if (analysisId && webSocket) {
      setIsCancelling(true);
      try {
        // Close WebSocket first
        webSocket.close();
        
        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }
        
        // Call cancel API
        await fetch(API_ENDPOINTS.CANCEL_ANALYSIS(analysisId), {
          method: 'DELETE'
        });
        
        setProgressMessages(prev => [...prev, {
          timestamp: new Date().toLocaleTimeString(),
          message: '⚠️ Analysis cancelled by user',
          type: 'warning'
        }]);
        
        toast({
          title: "Analysis Cancelled",
          description: "The analysis has been stopped successfully.",
        });
        
        // Navigate back after a short delay
        setTimeout(() => {
          navigate('/');
        }, 1500);
        
      } catch (error) {
        console.error('Error cancelling analysis:', error);
        setProgressMessages(prev => [...prev, {
          timestamp: new Date().toLocaleTimeString(),
          message: `❌ Failed to cancel analysis: ${error.message}`,
          type: 'error'
        }]);
      } finally {
        setIsCancelling(false);
      }
    }
  };

  const handleBack = () => {
    if (webSocket) {
      webSocket.close(1000, "User navigated away");
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }
    navigate('/');
  };

  useEffect(() => {
    if (!analysisId) {
      navigate('/');
      return;
    }

    // Get ticker from URL params or location state
    const urlParams = new URLSearchParams(window.location.search);
    const ticker = urlParams.get('ticker');
    
    if (!ticker) {
      toast({
        title: "Missing ticker symbol",
        description: "No ticker symbol provided for analysis.",
        variant: "destructive",
      });
      navigate('/');
      return;
    }
    
    setProgressMessages([{
      timestamp: new Date().toLocaleTimeString(),
      message: `Connecting to analysis for ${ticker}...`,
      type: 'info'
    }]);
    
    // Analysis already started by TickerInput - just connect to WebSocket
    // Don't start duplicate analysis here!
    connectToExistingAnalysis(analysisId, ticker);

    return () => {
      if (webSocket) {
        webSocket.close();
      }
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
    };
  }, [analysisId]);

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
          <div className="flex items-center justify-between mb-8">
          <Button 
            onClick={handleBack}
            variant="outline" 
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </Button>
          
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${connectionStatus === 'Connected' ? 'bg-success animate-pulse' : 'bg-muted'}`} />
            <span className="text-sm text-muted-foreground">
              {connectionStatus}
            </span>
          </div>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Title */}
          <div className="text-center space-y-2">
            <h1 className="text-3xl font-bold">Analysis in Progress</h1>
            <p className="text-muted-foreground">
              Analysis ID: <span className="font-mono">{analysisId}</span>
            </p>
          </div>

          {/* Ticker Details Section */}
          {tickerDetails && (
            <Card className="p-6 shadow-floating bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="bg-blue-600 text-white px-3 py-1 rounded-md font-mono font-bold text-lg">
                    {tickerDetails.ticker}
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-blue-900">{tickerDetails.company_name}</h3>
                    <p className="text-blue-700 text-sm">{tickerDetails.exchange} • {tickerDetails.sector}</p>
                  </div>
                </div>
                <div className="text-right">
                  {tickerDetails.current_price && (
                    <div className="text-2xl font-bold text-blue-600">
                      ${typeof tickerDetails.current_price === 'number' 
                        ? tickerDetails.current_price.toFixed(2) 
                        : tickerDetails.current_price}
                    </div>
                  )}
                  {tickerDetails.market_cap && (
                    <div className="text-sm text-blue-700">
                      Market Cap: ${(tickerDetails.market_cap / 1e9).toFixed(1)}B
                    </div>
                  )}
                </div>
              </div>
            </Card>
          )}

          {/* Progress Section */}
          <Card className="p-6 shadow-floating">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">Progress</h2>
                <Button 
                  onClick={cancelAnalysis}
                  disabled={isCancelling}
                  variant="destructive"
                  size="sm"
                  className="flex items-center gap-2"
                >
                  <X className="h-4 w-4" />
                  {isCancelling ? 'Cancelling...' : 'Cancel Analysis'}
                </Button>
              </div>
              
              <Progress value={progress} className="h-3" />
              
              <div className="text-center">
                <span className="text-2xl font-bold text-primary">{Math.round(progress)}%</span>
                <span className="text-muted-foreground ml-2">Complete</span>
              </div>
            </div>
          </Card>

          {/* Messages Feed */}
          <Card className="p-6 shadow-floating">
            <div className="space-y-4">
              <h2 className="text-xl font-semibold">Live Updates</h2>
              
              <ScrollArea ref={scrollAreaRef} className="h-96 w-full border rounded-md p-4 bg-card">
                <div className="space-y-3">
                  {progressMessages.map((msg, index) => (
                    <div key={index} className="flex items-start gap-3 text-sm">
                      <span className="text-muted-foreground font-mono text-xs shrink-0 mt-0.5">
                        {msg.timestamp}
                      </span>
                      <div className="flex-1">
                        <span className={`text-foreground ${
                          msg.type === 'error' ? 'text-destructive' : 
                          msg.type === 'success' ? 'text-success' : 
                          msg.type === 'warning' ? 'text-warning' : 
                          'text-foreground'
                        }`}>
                          {msg.message}
                        </span>
                      </div>
                    </div>
                  ))}
                  
                  {progressMessages.length === 0 && (
                    <div className="text-center text-muted-foreground py-8">
                      Waiting for updates...
                    </div>
                  )}
                </div>
              </ScrollArea>
            </div>
          </Card>

          {/* Error State */}
          {connectionStatus === 'Error' && (
            <Card className="p-6 border-destructive bg-destructive/5">
              <div className="text-center space-y-4">
                <h3 className="text-lg font-semibold text-destructive">Connection Issues</h3>
                <p className="text-muted-foreground">
                  There was a problem connecting to the analysis stream. 
                  You can try refreshing the page or starting a new analysis.
                </p>
                <div className="flex gap-4 justify-center">
                  <Button onClick={() => window.location.reload()} variant="outline">
                    Refresh Page
                  </Button>
                  <Button onClick={handleBack}>
                    Start New Analysis
                  </Button>
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisProgress;