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
    console.log('Validating ticker:', ticker);
    
    // Use correct API Gateway URL
    const apiGatewayUrl = 'https://i5xlj4nhie.execute-api.us-east-1.amazonaws.com';
    const targetUrl = `${apiGatewayUrl}/api/validate/ticker/${ticker}`;
    
    console.log('Calling API Gateway:', targetUrl);
    
    const response = await fetch(targetUrl, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'Vercel-Proxy/1.0',
      },
    });
    
    if (!response.ok) {
      throw new Error(`API Gateway responded with ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('✅ Validation successful:', data.company_name);
    
    // Return the API Gateway response with CORS headers
    res.status(200).json(data);
    
  } catch (error) {
    console.error('Validation error:', error);
    res.status(500).json({ 
      error: 'Failed to validate ticker',
      details: error.message,
      ticker: ticker,
      timestamp: new Date().toISOString()
    });
  }
}
