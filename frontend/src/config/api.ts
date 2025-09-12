// API Configuration
// Configure the backend URL for AWS Lambda

const config = {
  // Direct Lambda URL for Netlify deployment
  // Note: CORS must be properly configured on the Lambda backend
  API_BASE_URL: 'https://fi1vdvikyl.execute-api.us-east-1.amazonaws.com',
  
  // WebSocket URL - Points to current Lambda
  WS_BASE_URL: 'wss://fi1vdvikyl.execute-api.us-east-1.amazonaws.com',
  
  // Direct Lambda URL (for reference/debugging)
  LAMBDA_BASE_URL: 'https://fi1vdvikyl.execute-api.us-east-1.amazonaws.com',
}

// API endpoints - Direct to Lambda backend for Netlify deployment
export const API_ENDPOINTS = {
  // Ticker validation (direct to Lambda)
  VALIDATE_TICKER: (ticker: string) => `${config.API_BASE_URL}/api/validate/${ticker}`,
  
  // Analysis endpoints (direct to Lambda)
  START_ANALYSIS: `${config.API_BASE_URL}/api/analysis/comprehensive/start`,
  ANALYSIS_STATUS: (analysisId: string) => `${config.API_BASE_URL}/api/analysis/${analysisId}/status`,
  ANALYSIS_RESULTS: (analysisId: string) => `${config.API_BASE_URL}/api/analysis/${analysisId}/results`,
  CANCEL_ANALYSIS: (analysisId: string) => `${config.API_BASE_URL}/api/analysis/${analysisId}/cancel`,
  
  // WebSocket endpoint
  WEBSOCKET_ANALYSIS: (analysisId: string) => `${config.WS_BASE_URL}/ws/analysis/${analysisId}`,
  
  // Report endpoints
  REPORT_MARKDOWN: (analysisId: string) => `${config.API_BASE_URL}/api/reports/${analysisId}/markdown`,
  REPORT_PDF: (analysisId: string) => `${config.API_BASE_URL}/api/reports/${analysisId}/pdf`,
  
  // Health check (direct to Lambda)
  HEALTH: `${config.API_BASE_URL}/health`,
  API_HEALTH: `${config.API_BASE_URL}/api/health`,
}

export default config;
