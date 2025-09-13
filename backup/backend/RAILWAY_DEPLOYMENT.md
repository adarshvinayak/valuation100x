# 🚀 Railway Deployment Guide

## Quick Deploy to Railway

### 1. Prerequisites
- Git repository with this backend code
- Railway account (railway.app)
- Required API keys (see Environment Variables section)

### 2. Deploy Steps

#### Option A: GitHub Integration (Recommended)
1. Push this code to GitHub repository
2. Connect Railway to your GitHub account
3. Create new Railway project from GitHub repo
4. Set environment variables in Railway dashboard
5. Deploy automatically triggers

#### Option B: Railway CLI
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

### 3. Required Environment Variables

Set these in Railway dashboard under Variables:

```
OPENAI_API_KEY=sk-your-openai-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here
FMP_API_KEY=your-fmp-api-key-here
SEC_API_KEY=your-sec-api-key-here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key-here
```

### 4. Optional Add-ons
- **Redis**: Add Railway Redis service for caching
- **PostgreSQL**: If you need additional database storage

### 5. Verification
After deployment, check:
- Health endpoint: `https://your-app.railway.app/health`
- API docs: `https://your-app.railway.app/docs`
- Test endpoint: `https://your-app.railway.app/api/validate/ticker/AAPL`

## Configuration Files

This backend is pre-configured for Railway with:
- ✅ `railway.json` - Railway build configuration
- ✅ `Dockerfile` - Optimized multi-stage build
- ✅ `requirements.railway.txt` - Streamlined dependencies
- ✅ Port binding to Railway's PORT environment variable
- ✅ Production settings detection via RAILWAY_ENVIRONMENT

## Architecture

- **FastAPI** server with async/await
- **Redis** caching (optional Railway add-on)
- **Supabase** database integration
- **Vector search** with FAISS
- **Comprehensive financial analysis** pipeline

## API Endpoints

- `GET /health` - Health check
- `POST /api/analysis/comprehensive/start` - Start analysis
- `GET /api/analysis/{id}/status` - Check progress
- `GET /api/analysis/{id}/results` - Get results
- `GET /docs` - Interactive API documentation

## Support

For deployment issues:
1. Check Railway logs in dashboard
2. Verify all environment variables are set
3. Ensure API keys are valid and have sufficient quotas
4. Monitor memory usage (may need to upgrade plan for ML models)
