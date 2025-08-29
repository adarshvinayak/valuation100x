// Vercel API route to proxy Lambda requests and add CORS headers
// This bypasses CORS issues by making server-side requests

export default async function handler(req, res) {
  // Enable CORS for all origins
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }
  
  // Only allow GET requests
  if (req.method !== 'GET') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }
  
  const { ticker } = req.query;
  
  if (!ticker) {
    res.status(400).json({ error: 'Ticker parameter required' });
    return;
  }
  
  try {
    // Proxy request to Lambda function
    const lambdaUrl = 'https://qkw44e47tsqq7ol6k6bf6n6iem0vjqzh.lambda-url.us-east-1.on.aws';
    const response = await fetch(`${lambdaUrl}/api/validate/ticker/${ticker}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`Lambda responded with ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    // Return the Lambda response with CORS headers
    res.status(200).json(data);
    
  } catch (error) {
    console.error('Proxy error:', error);
    res.status(500).json({ 
      error: 'Failed to fetch ticker data',
      details: error.message 
    });
  }
}
