// Vercel API route to proxy analysis start requests

export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }
  
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }
  
  try {
    console.log('Received analysis start request:', req.body);
    
    // Validate request body
    if (!req.body || !req.body.ticker) {
      return res.status(400).json({ 
        error: 'Invalid request',
        details: 'ticker field is required'
      });
    }
    
    const lambdaUrl = 'https://qkw44e47tsqq7ol6k6bf6n6iem0vjqzh.lambda-url.us-east-1.on.aws';
    console.log('Proxying to Lambda:', `${lambdaUrl}/api/analysis/comprehensive/start`);
    
    const requestBody = {
      ticker: req.body.ticker,
      company_name: req.body.company_name || req.body.ticker
    };
    
    console.log('Request body:', requestBody);
    
    const response = await fetch(`${lambdaUrl}/api/analysis/comprehensive/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    
    console.log('Lambda response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Lambda error response:', errorText);
      
      // Return more specific error info
      return res.status(response.status).json({
        error: 'Backend analysis service error',
        details: `Lambda returned ${response.status}: ${errorText}`,
        lambdaStatus: response.status
      });
    }
    
    const data = await response.json();
    console.log('Lambda response data:', data);
    res.status(200).json(data);
    
  } catch (error) {
    console.error('Proxy error:', error);
    res.status(500).json({ 
      error: 'Failed to start analysis',
      details: error.message,
      timestamp: new Date().toISOString(),
      type: error.name
    });
  }
}
