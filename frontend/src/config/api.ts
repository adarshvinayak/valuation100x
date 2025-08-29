// API Configuration
// Configure the backend URL for AWS Lambda

const config = {
  // CORS Solution: Use Vercel API routes as proxy to Lambda backend
  // This completely bypasses CORS issues by making server-side requests
  API_BASE_URL: '',  // Use relative URLs for Vercel API routes
  
  // WebSocket URL - Still points to Lambda (will implement WebSocket proxy if needed)
  WS_BASE_URL: 'wss://qkw44e47tsqq7ol6k6bf6n6iem0vjqzh.lambda-url.us-east-1.on.aws',
  
  // Direct Lambda URL (for reference/debugging)
  LAMBDA_BASE_URL: 'https://qkw44e47tsqq7ol6k6bf6n6iem0vjqzh.lambda-url.us-east-1.on.aws',
}

// API endpoints - Using Vercel API routes to bypass CORS
export const API_ENDPOINTS = {
  // Ticker validation (Vercel API route)
  VALIDATE_TICKER: (ticker: string) => `/api/validate/${ticker}`,
  
  // Analysis endpoints (Vercel API routes)
  START_ANALYSIS: `/api/analysis/start`,
  ANALYSIS_STATUS: (analysisId: string) => `${config.LAMBDA_BASE_URL}/api/analysis/${analysisId}/status`,
  ANALYSIS_RESULTS: (analysisId: string) => `${config.LAMBDA_BASE_URL}/api/analysis/${analysisId}/results`,
  CANCEL_ANALYSIS: (analysisId: string) => `${config.LAMBDA_BASE_URL}/api/analysis/${analysisId}/cancel`,
  
  // WebSocket endpoint (still direct to Lambda - will proxy if needed)
  WEBSOCKET_ANALYSIS: (analysisId: string) => `${config.WS_BASE_URL}/ws/analysis/${analysisId}`,
  
  // Report endpoints (direct to Lambda for now)
  REPORT_MARKDOWN: (analysisId: string) => `${config.LAMBDA_BASE_URL}/api/reports/${analysisId}/markdown`,
  REPORT_PDF: (analysisId: string) => `${config.LAMBDA_BASE_URL}/api/reports/${analysisId}/pdf`,
  
  // Health check (Vercel API route)
  HEALTH: `/api/health`,
  API_HEALTH: `/api/health`,
}

export default config;
