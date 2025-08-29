// API Configuration
// Configure the backend URL for AWS Lambda

const config = {
  // AWS Lambda Backend (Production)
  API_BASE_URL: 'https://ppi7ci4lyhypmox7p4kp73bmsi0ydcon.lambda-url.us-east-1.on.aws',
  
  // WebSocket URL (Note: Lambda Function URLs don't support WebSocket directly)
  // For WebSocket, we'll implement polling or use API Gateway WebSocket
  WS_BASE_URL: 'wss://ppi7ci4lyhypmox7p4kp73bmsi0ydcon.lambda-url.us-east-1.on.aws',
}

// API endpoints
export const API_ENDPOINTS = {
  // Ticker validation
  VALIDATE_TICKER: (ticker: string) => `${config.API_BASE_URL}/api/validate/ticker/${ticker}`,
  
  // Analysis endpoints
  START_ANALYSIS: `${config.API_BASE_URL}/api/analysis/comprehensive/start`,
  ANALYSIS_STATUS: (analysisId: string) => `${config.API_BASE_URL}/api/analysis/${analysisId}/status`,
  ANALYSIS_RESULTS: (analysisId: string) => `${config.API_BASE_URL}/api/analysis/${analysisId}/results`,
  CANCEL_ANALYSIS: (analysisId: string) => `${config.API_BASE_URL}/api/analysis/${analysisId}/cancel`,
  
  // WebSocket endpoint (polling-based for now)
  WEBSOCKET_ANALYSIS: (analysisId: string) => `${config.WS_BASE_URL}/ws/analysis/${analysisId}`,
  
  // Report endpoints
  REPORT_MARKDOWN: (analysisId: string) => `${config.API_BASE_URL}/api/reports/${analysisId}/markdown`,
  REPORT_PDF: (analysisId: string) => `${config.API_BASE_URL}/api/reports/${analysisId}/pdf`,
  
  // Health check
  HEALTH: `${config.API_BASE_URL}/health`,
  API_HEALTH: `${config.API_BASE_URL}/api/health`,
}

export default config;
