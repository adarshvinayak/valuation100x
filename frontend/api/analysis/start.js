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
    
    // Try multiple backend URLs
    const backendUrls = [
      'https://i5xlj4nhie.execute-api.us-east-1.amazonaws.com',
    ];
    
    const requestBody = {
      ticker: req.body.ticker,
      company_name: req.body.company_name || req.body.ticker
    };
    
    console.log('Request body:', requestBody);
    
    let lastError = null;
    
    for (const lambdaUrl of backendUrls) {
      try {
        const targetUrl = `${lambdaUrl}/api/analysis/comprehensive/start`;
        console.log('Trying backend:', targetUrl);
        
        const response = await fetch(targetUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'Vercel-Proxy/1.0',
          },
          body: JSON.stringify(requestBody),
        });
        
        console.log(`Backend ${lambdaUrl} response status:`, response.status);
        
        if (response.ok) {
          const data = await response.json();
          console.log('Analysis start success:', data);
          return res.status(200).json(data);
        } else {
          const errorText = await response.text();
          console.error(`Backend ${lambdaUrl} error:`, errorText);
          lastError = `${response.status}: ${errorText}`;
        }
      } catch (err) {
        console.error(`Backend ${lambdaUrl} failed:`, err.message);
        lastError = err.message;
      }
    }
    
    // All backends failed - return a helpful error response
    console.error('All analysis backends failed:', lastError);
    return res.status(503).json({ 
      error: 'Analysis service temporarily unavailable',
      details: `Backend services are experiencing issues: ${lastError}`,
      suggestion: 'Please try again in a few minutes',
      timestamp: new Date().toISOString()
    });
    
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
