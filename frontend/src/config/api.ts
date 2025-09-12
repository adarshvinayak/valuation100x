// API Configuration
// Configure the backend URL for AWS Lambda

const config = {
  // Railway deployment URL - Live production backend
  // Note: CORS is properly configured in the FastAPI backend
  API_BASE_URL: 'https://valuation100x-production.up.railway.app',
  
  // WebSocket URL - Points to Railway backend (if WebSocket is supported)
  WS_BASE_URL: 'wss://valuation100x-production.up.railway.app',
  
  // Railway URL (for reference/debugging)
  RAILWAY_BASE_URL: 'https://valuation100x-production.up.railway.app',
}

// API endpoints - Direct to Railway backend for Netlify deployment
export const API_ENDPOINTS = {
  // Ticker validation (direct to Railway)
  VALIDATE_TICKER: (ticker: string) => `${config.API_BASE_URL}/api/validate/ticker/${ticker}`,
  
  // Analysis endpoints (direct to Railway)
  START_ANALYSIS: `${config.API_BASE_URL}/api/analysis/comprehensive/start`,
  ANALYSIS_STATUS: (analysisId: string) => `${config.API_BASE_URL}/api/analysis/${analysisId}/status`,
  ANALYSIS_RESULTS: (analysisId: string) => `${config.API_BASE_URL}/api/analysis/${analysisId}/results`,
  CANCEL_ANALYSIS: (analysisId: string) => `${config.API_BASE_URL}/api/analysis/${analysisId}/cancel`,
  
  // WebSocket endpoint
  WEBSOCKET_ANALYSIS: (analysisId: string) => `${config.WS_BASE_URL}/ws/analysis/${analysisId}`,
  
  // Report endpoints
  REPORT_MARKDOWN: (analysisId: string) => `${config.API_BASE_URL}/api/reports/${analysisId}/markdown`,
  REPORT_PDF: (analysisId: string) => `${config.API_BASE_URL}/api/reports/${analysisId}/pdf`,
  
  // Health check (direct to Railway)
  HEALTH: `${config.API_BASE_URL}/health`,
  API_HEALTH: `${config.API_BASE_URL}/api/health`,
}

export default config;
