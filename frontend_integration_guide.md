# 🔗 Frontend Integration Guide

## 🎯 Overview

This guide shows how to integrate your React frontend with the DeepResearch FastAPI backend for comprehensive stock analysis.

---

## 🚀 Quick Start

### **1. Start the Backend Server**

```bash
# Install dependencies
pip install -r api_requirements.txt

# Start Redis (if not using Docker)
redis-server

# Start the FastAPI server
python api_server.py

# Or using uvicorn directly
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### **2. Using Docker Compose (Recommended)**

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

---

## 🔌 Frontend Integration Examples

### **1. React Hook for Analysis**

```typescript
// hooks/useAnalysis.ts
import { useState, useEffect, useRef } from 'react';

interface AnalysisState {
  analysisId: string | null;
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  currentStep: string;
  results: any | null;
  error: string | null;
}

export const useAnalysis = () => {
  const [state, setState] = useState<AnalysisState>({
    analysisId: null,
    status: 'idle',
    progress: 0,
    currentStep: '',
    results: null,
    error: null
  });
  
  const [logs, setLogs] = useState<any[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const startAnalysis = async (ticker: string, companyName?: string) => {
    try {
      setState(prev => ({ ...prev, status: 'running', error: null }));
      
      const response = await fetch('/api/analysis/comprehensive/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, company_name: companyName })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      setState(prev => ({
        ...prev,
        analysisId: data.analysis_id,
        status: 'running'
      }));
      
      // Connect to WebSocket
      connectWebSocket(data.analysis_id);
      
      return data;
    } catch (error) {
      setState(prev => ({
        ...prev,
        status: 'failed',
        error: error.message
      }));
      throw error;
    }
  };

  const connectWebSocket = (analysisId: string) => {
    const wsUrl = `ws://localhost:8000/ws/analysis/${analysisId}`;
    wsRef.current = new WebSocket(wsUrl);
    
    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
    };
    
    wsRef.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };
    
    wsRef.current.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  };

  const handleWebSocketMessage = (message: any) => {
    switch (message.type) {
      case 'progress_update':
        setState(prev => ({
          ...prev,
          progress: message.data.progress,
          currentStep: message.data.current_step
        }));
        break;
        
      case 'analysis_log':
        setLogs(prev => [...prev, message.data]);
        break;
        
      case 'analysis_completed':
        setState(prev => ({
          ...prev,
          status: 'completed',
          progress: 100
        }));
        fetchResults();
        break;
        
      case 'analysis_error':
        setState(prev => ({
          ...prev,
          status: 'failed',
          error: message.data.error_message
        }));
        break;
    }
  };

  const fetchResults = async () => {
    if (!state.analysisId) return;
    
    try {
      const response = await fetch(`/api/analysis/${state.analysisId}/results`);
      const results = await response.json();
      
      setState(prev => ({
        ...prev,
        results: results
      }));
    } catch (error) {
      console.error('Failed to fetch results:', error);
    }
  };

  const cancelAnalysis = async () => {
    if (!state.analysisId) return;
    
    try {
      await fetch(`/api/analysis/${state.analysisId}/cancel`, {
        method: 'DELETE'
      });
      
      setState(prev => ({
        ...prev,
        status: 'cancelled'
      }));
    } catch (error) {
      console.error('Failed to cancel analysis:', error);
    }
  };

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    ...state,
    logs,
    startAnalysis,
    cancelAnalysis,
    fetchResults
  };
};
```

### **2. Landing Page Component**

```typescript
// components/LandingPage.tsx
import React, { useState } from 'react';
import { useAnalysis } from '../hooks/useAnalysis';

export const LandingPage: React.FC = () => {
  const [ticker, setTicker] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [validation, setValidation] = useState<any>(null);
  const { startAnalysis, status } = useAnalysis();

  const validateTicker = async (tickerValue: string) => {
    if (!tickerValue.trim()) {
      setValidation(null);
      return;
    }
    
    setIsValidating(true);
    try {
      const response = await fetch(`/api/validate/ticker/${tickerValue.toUpperCase()}`);
      const data = await response.json();
      setValidation(data);
    } catch (error) {
      console.error('Validation failed:', error);
      setValidation({ is_valid: false });
    } finally {
      setIsValidating(false);
    }
  };

  const handleTickerChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.toUpperCase();
    setTicker(value);
    
    // Debounced validation
    setTimeout(() => validateTicker(value), 500);
  };

  const handleStartAnalysis = async () => {
    if (!validation?.is_valid) return;
    
    try {
      await startAnalysis(ticker, validation.company_name);
      // Navigate to progress page
    } catch (error) {
      console.error('Failed to start analysis:', error);
    }
  };

  return (
    <div className="landing-page">
      <div className="container">
        <h1>Comprehensive Stock Analysis</h1>
        <p>Get institutional-grade analysis in ~15 minutes</p>
        
        <div className="ticker-input-section">
          <div className="ticker-input-container">
            <input
              type="text"
              value={ticker}
              onChange={handleTickerChange}
              placeholder="Enter stock ticker (e.g., AAPL)"
              className="ticker-input"
              maxLength={10}
            />
            
            {isValidating && (
              <div className="validation-spinner">Validating...</div>
            )}
            
            {validation && (
              <div className={`validation-result ${validation.is_valid ? 'valid' : 'invalid'}`}>
                {validation.is_valid ? (
                  <div className="company-preview">
                    <span className="company-name">{validation.company_name}</span>
                    <span className="exchange">{validation.exchange}</span>
                  </div>
                ) : (
                  <span className="error">Invalid ticker symbol</span>
                )}
              </div>
            )}
          </div>
          
          <button
            onClick={handleStartAnalysis}
            disabled={!validation?.is_valid || status === 'running'}
            className="start-analysis-btn"
          >
            {status === 'running' ? 'Analysis Running...' : 'Start Comprehensive Analysis'}
          </button>
        </div>
        
        <div className="analysis-info">
          <h3>What's included in comprehensive analysis:</h3>
          <ul>
            <li>Damodaran Story-Driven Framework</li>
            <li>SEC Filing Analysis</li>
            <li>Financial Statement Analysis</li>
            <li>Sentiment Analysis</li>
            <li>Technical Analysis</li>
            <li>DCF Valuation with Scenarios</li>
            <li>Risk Assessment</li>
            <li>Investment Recommendation</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
```

### **3. Progress Page Component**

```typescript
// components/ProgressPage.tsx
import React from 'react';
import { useAnalysis } from '../hooks/useAnalysis';

export const ProgressPage: React.FC = () => {
  const { progress, currentStep, logs, status, cancelAnalysis } = useAnalysis();

  const stepLabels = {
    'question_generation': 'Question Generation',
    'sec_filing_analysis': 'SEC Filing Analysis',
    'financial_data_collection': 'Financial Data Collection',
    'comprehensive_research': 'Comprehensive Research',
    'damodaran_story_development': 'Damodaran Story Development',
    'valuation_scenario_analysis': 'Valuation & Scenario Analysis',
    'report_generation': 'Report Generation'
  };

  return (
    <div className="progress-page">
      <div className="progress-container">
        {/* Circular Progress Indicator */}
        <div className="progress-circle">
          <svg viewBox="0 0 100 100" className="progress-ring">
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="#e5e7eb"
              strokeWidth="8"
            />
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="#3b82f6"
              strokeWidth="8"
              strokeDasharray={`${progress * 2.83} 283`}
              strokeLinecap="round"
              transform="rotate(-90 50 50)"
              className="progress-bar"
            />
          </svg>
          <div className="progress-text">
            <span className="progress-percentage">{progress}%</span>
            <span className="progress-label">Complete</span>
          </div>
        </div>
        
        {/* Current Step */}
        <div className="current-step">
          <h2>{stepLabels[currentStep] || 'Processing...'}</h2>
          <p>Running comprehensive analysis...</p>
        </div>
        
        {/* Step Timeline */}
        <div className="step-timeline">
          {Object.entries(stepLabels).map(([stepId, label], index) => (
            <div
              key={stepId}
              className={`timeline-step ${
                currentStep === stepId ? 'active' : 
                progress > (index / Object.keys(stepLabels).length) * 100 ? 'completed' : 'pending'
              }`}
            >
              <div className="step-indicator">
                {progress > (index / Object.keys(stepLabels).length) * 100 ? '✓' : index + 1}
              </div>
              <span className="step-label">{label}</span>
            </div>
          ))}
        </div>
        
        {/* Live Logs */}
        <div className="logs-section">
          <button className="logs-toggle">Show Analysis Logs</button>
          <div className="logs-container">
            {logs.slice(-10).map((log, index) => (
              <div key={index} className={`log-entry ${log.level.toLowerCase()}`}>
                <span className="log-time">{new Date(log.timestamp).toLocaleTimeString()}</span>
                <span className="log-component">{log.component}:</span>
                <span className="log-message">{log.message}</span>
              </div>
            ))}
          </div>
        </div>
        
        {/* Cancel Button */}
        <button
          onClick={cancelAnalysis}
          className="cancel-btn"
          disabled={status !== 'running'}
        >
          Cancel Analysis
        </button>
      </div>
    </div>
  );
};
```

### **4. Results Page Component**

```typescript
// components/ResultsPage.tsx
import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ResultsPageProps {
  analysisId: string;
}

export const ResultsPage: React.FC<ResultsPageProps> = ({ analysisId }) => {
  const [markdownContent, setMarkdownContent] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMarkdownReport();
  }, [analysisId]);

  const fetchMarkdownReport = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`/api/reports/${analysisId}/markdown`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch report: ${response.statusText}`);
      }
      
      const markdown = await response.text();
      setMarkdownContent(markdown);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadPDF = async () => {
    try {
      const response = await fetch(`/api/reports/${analysisId}/pdf`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error('PDF generation failed');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `analysis_report_${analysisId}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('PDF download failed:', err);
    }
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(markdownContent);
      // Show success notification
    } catch (err) {
      console.error('Copy failed:', err);
    }
  };

  const printReport = () => {
    window.print();
  };

  if (isLoading) {
    return (
      <div className="results-loading">
        <div className="loading-spinner">Loading report...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="results-error">
        <h2>Error Loading Report</h2>
        <p>{error}</p>
        <button onClick={fetchMarkdownReport}>Retry</button>
      </div>
    );
  }

  return (
    <div className="results-page">
      {/* Action Buttons */}
      <div className="results-actions">
        <button onClick={downloadPDF} className="action-btn download-btn">
          📄 Download PDF
        </button>
        <button onClick={copyToClipboard} className="action-btn copy-btn">
          📋 Copy to Clipboard
        </button>
        <button onClick={printReport} className="action-btn print-btn">
          🖨️ Print Report
        </button>
      </div>
      
      {/* Markdown Report Display */}
      <div className="markdown-container">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            table: ({ children, ...props }) => (
              <div className="table-container">
                <table className="markdown-table" {...props}>
                  {children}
                </table>
              </div>
            ),
            h1: ({ children, ...props }) => (
              <h1 className="markdown-h1" {...props}>
                {children}
              </h1>
            ),
            h2: ({ children, ...props }) => (
              <h2 className="markdown-h2" {...props}>
                {children}
              </h2>
            ),
            code: ({ inline, className, children, ...props }) => {
              if (inline) {
                return <code className="inline-code" {...props}>{children}</code>;
              }
              return (
                <pre className="code-block">
                  <code className={className} {...props}>
                    {children}
                  </code>
                </pre>
              );
            }
          }}
        >
          {markdownContent}
        </ReactMarkdown>
      </div>
    </div>
  );
};
```

---

## 🎨 CSS Styling Example

```css
/* styles/components.css */

/* Landing Page */
.landing-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.container {
  max-width: 600px;
  padding: 2rem;
  background: white;
  border-radius: 12px;
  box-shadow: 0 20px 40px rgba(0,0,0,0.1);
  text-align: center;
}

.ticker-input {
  width: 100%;
  padding: 1rem;
  font-size: 1.2rem;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  text-align: center;
  text-transform: uppercase;
  margin-bottom: 1rem;
}

.start-analysis-btn {
  width: 100%;
  padding: 1rem 2rem;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.start-analysis-btn:hover:not(:disabled) {
  background: #2563eb;
}

.start-analysis-btn:disabled {
  background: #9ca3af;
  cursor: not-allowed;
}

/* Progress Page */
.progress-page {
  min-height: 100vh;
  padding: 2rem;
  background: #f8fafc;
  display: flex;
  align-items: center;
  justify-content: center;
}

.progress-circle {
  position: relative;
  width: 200px;
  height: 200px;
  margin: 0 auto 2rem;
}

.progress-ring {
  width: 100%;
  height: 100%;
  transform: rotate(-90deg);
}

.progress-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}

.progress-percentage {
  display: block;
  font-size: 2rem;
  font-weight: bold;
  color: #1f2937;
}

.progress-label {
  color: #6b7280;
  font-size: 0.9rem;
}

.step-timeline {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin: 2rem 0;
}

.timeline-step {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem;
  border-radius: 8px;
  transition: background 0.2s;
}

.timeline-step.active {
  background: #dbeafe;
}

.timeline-step.completed {
  background: #dcfce7;
}

.step-indicator {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 0.9rem;
}

.timeline-step.completed .step-indicator {
  background: #22c55e;
  color: white;
}

.timeline-step.active .step-indicator {
  background: #3b82f6;
  color: white;
}

.timeline-step.pending .step-indicator {
  background: #e5e7eb;
  color: #6b7280;
}

/* Results Page */
.results-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

.results-actions {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
  justify-content: center;
}

.action-btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 6px;
  background: #374151;
  color: white;
  cursor: pointer;
  font-weight: 500;
  transition: background 0.2s;
}

.action-btn:hover {
  background: #1f2937;
}

.markdown-container {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
  max-height: 80vh;
  overflow-y: auto;
  font-family: 'Times New Roman', serif;
  line-height: 1.6;
}

.markdown-h1 {
  color: #1f2937;
  border-bottom: 2px solid #e5e7eb;
  padding-bottom: 0.5rem;
  margin-bottom: 1.5rem;
}

.markdown-h2 {
  color: #374151;
  margin-top: 2rem;
  margin-bottom: 1rem;
}

.markdown-table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  font-family: 'Courier New', monospace;
}

.markdown-table th {
  background: #f9fafb;
  padding: 0.75rem;
  text-align: left;
  border: 1px solid #d1d5db;
  font-weight: 600;
}

.markdown-table td {
  padding: 0.75rem;
  border: 1px solid #d1d5db;
}

.table-container {
  overflow-x: auto;
  margin: 1rem 0;
}

/* Print styles */
@media print {
  .results-actions {
    display: none;
  }
  
  .markdown-container {
    box-shadow: none;
    max-height: none;
    overflow: visible;
  }
}
```

---

## 📋 Environment Configuration

Create a `.env.example` file:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Redis Configuration
REDIS_URL=redis://localhost:6379

# External API Keys (from your existing .env)
OPENAI_API_KEY=your_openai_key
FMP_API_KEY=your_fmp_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
SEC_API_KEY=your_sec_api_key
TAVILY_API_KEY=your_tavily_key

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Frontend URLs (for CORS)
FRONTEND_URLS=http://localhost:3000,http://localhost:5173
```

---

## 🚀 Deployment Instructions

### **Development Setup**

1. **Backend:**
   ```bash
   pip install -r api_requirements.txt
   python api_server.py
   ```

2. **Frontend:**
   ```bash
   npm install
   npm start
   ```

### **Production with Docker**

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Scale API if needed
docker-compose up -d --scale api=2
```

---

## 🔧 Testing the Integration

### **1. Test API Endpoints**

```bash
# Health check
curl http://localhost:8000/api/health

# Validate ticker
curl http://localhost:8000/api/validate/ticker/AAPL

# Start analysis
curl -X POST http://localhost:8000/api/analysis/comprehensive/start \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

### **2. Test WebSocket Connection**

```javascript
// Browser console test
const ws = new WebSocket('ws://localhost:8000/ws/analysis/test-id');
ws.onmessage = (event) => console.log(JSON.parse(event.data));
ws.onopen = () => ws.send(JSON.stringify({type: 'ping'}));
```

This comprehensive integration provides everything needed to connect your React frontend to the DeepResearch backend with real-time progress tracking and professional report display.

