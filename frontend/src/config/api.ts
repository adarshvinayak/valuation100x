// API Configuration
// Configure the backend URL for different environments

const config = {
  // AWS Lambda Backend (Production)
  API_BASE_URL: 'https://ppi7ci4lyhypmox7p4kp73bmsi0ydcon.lambda-url.us-east-1.on.aws',
  
  // WebSocket URL (Note: Lambda Function URLs don't support WebSocket directly)
  // For WebSocket, we'll need to implement polling or use API Gateway WebSocket later
  WS_BASE_URL: 'wss://ppi7ci4lyhypmox7p4kp73bmsi0ydcon.lambda-url.us-east-1.on.aws',
  
  // Fallback to Railway for WebSocket (until we implement API Gateway WebSocket)
  WS_FALLBACK_URL: 'wss://valuation100x-production.up.railway.app',
  
  // Enable fallback for development
  USE_WEBSOCKET_FALLBACK: true,
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
  
  // WebSocket endpoint (with fallback)
  WEBSOCKET_ANALYSIS: (analysisId: string) => {
    if (config.USE_WEBSOCKET_FALLBACK) {
      return `${config.WS_FALLBACK_URL}/ws/analysis/${analysisId}`;
    }
    return `${config.WS_BASE_URL}/ws/analysis/${analysisId}`;
  },
  
  // Health check
  HEALTH: `${config.API_BASE_URL}/health`,
  API_HEALTH: `${config.API_BASE_URL}/api/health`,
}

export default config;
