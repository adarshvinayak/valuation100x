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

  const startAnalysis = async (ticker: string) => {
    try {
      // 1. Start the analysis
      const startResponse = await fetch(API_ENDPOINTS.START_ANALYSIS, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: ticker.toUpperCase(),
          company_name: `${ticker.toUpperCase()} Corporation`
        }),
      });

      if (!startResponse.ok) {
        throw new Error(`Failed to start analysis: ${startResponse.statusText}`);
      }

      const startData = await startResponse.json();
      const newAnalysisId = startData.analysis_id;

      // 2. Connect to WebSocket with proper error handling
      const wsUrl = API_ENDPOINTS.WEBSOCKET_ANALYSIS(newAnalysisId);
      const ws = new WebSocket(wsUrl);

      // Connection opened
      ws.onopen = () => {
        console.log('WebSocket connected successfully');
        setConnectionStatus('Connected');
        
        // Send ping to keep connection alive
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          } else {
            clearInterval(pingInterval);
          }
        }, 30000); // Ping every 30 seconds
        
        pingIntervalRef.current = pingInterval;
      };

      // Handle incoming messages
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('WebSocket message:', message);

          switch (message.type) {
            case 'connection_established':
              setProgressMessages(prev => [...prev, {
                timestamp: new Date().toLocaleTimeString(),
                message: `Connected to analysis stream for ${ticker}`,
                type: 'info'
              }]);
              break;

            case 'progress_update':
              setProgress(message.data.progress);
              setProgressMessages(prev => [...prev, {
                timestamp: new Date().toLocaleTimeString(),
                message: message.data.step_description || `Progress: ${message.data.progress}%`,
                type: 'progress'
              }]);
              break;

            case 'analysis_log':
              setProgressMessages(prev => [...prev, {
                timestamp: new Date().toLocaleTimeString(),
                message: message.data.message,
                type: message.data.level.toLowerCase()
              }]);
              break;

            case 'step_completed':
              setProgressMessages(prev => [...prev, {
                timestamp: new Date().toLocaleTimeString(),
                message: `✅ Completed: ${message.data.completed_step}`,
                type: 'success'
              }]);
              break;

            case 'analysis_completed':
              setProgress(100);
              setProgressMessages(prev => [...prev, {
                timestamp: new Date().toLocaleTimeString(),
                message: `🎉 Analysis completed! Score: ${message.data.investment_score}/10`,
                type: 'success'
              }]);
              
              // Close WebSocket and redirect to results
              ws.close();
              setTimeout(() => {
                window.location.href = `/report/${newAnalysisId}?ticker=${ticker}`;
              }, 2000);
              break;

            case 'analysis_error':
              setProgressMessages(prev => [...prev, {
                timestamp: new Date().toLocaleTimeString(),
                message: `❌ Error: ${message.data.error_message}`,
                type: 'error'
              }]);
              break;

            case 'pong':
              // Keep-alive response
              console.log('Received pong');
              break;

            case 'heartbeat':
              // Heartbeat from server
              console.log('Received heartbeat');
              break;

            default:
              console.log('Unknown message type:', message.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      // Connection closed
      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setConnectionStatus('Disconnected');
        
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }
        
        if (event.code !== 1000) { // Not a normal closure
          setProgressMessages(prev => [...prev, {
            timestamp: new Date().toLocaleTimeString(),
            message: `Connection closed unexpectedly (${event.code}). Checking results...`,
            type: 'warning'
          }]);
          
          // Try to fetch results after a delay
          setTimeout(async () => {
            try {
              const resultsResponse = await fetch(API_ENDPOINTS.ANALYSIS_RESULTS(newAnalysisId));
              if (resultsResponse.ok) {
                const urlParams = new URLSearchParams(window.location.search);
                const ticker = urlParams.get('ticker');
                window.location.href = `/report/${newAnalysisId}${ticker ? `?ticker=${ticker}` : ''}`;
              }
            } catch (error) {
              console.error('Error checking results:', error);
            }
          }, 3000);
        }
      };

      // Connection error
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('Error');
        setProgressMessages(prev => [...prev, {
          timestamp: new Date().toLocaleTimeString(),
          message: '❌ Connection error. Retrying...',
          type: 'error'
        }]);
      };

      // Store WebSocket reference for cleanup
      setWebSocket(ws);
      
    } catch (error) {
      console.error('Error starting analysis:', error);
      setProgressMessages(prev => [...prev, {
        timestamp: new Date().toLocaleTimeString(),
        message: `❌ Failed to start analysis: ${error.message}`,
        type: 'error'
      }]);
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
      message: `Initializing analysis for ${ticker}...`,
      type: 'info'
    }]);
    
    startAnalysis(ticker);

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