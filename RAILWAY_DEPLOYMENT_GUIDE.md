# 🚀 Railway Deployment Guide for DeepResearch

## ✅ Setup Complete!

Your repository is now configured for Railway deployment with:
- ✅ `backend/railway.json` - Railway configuration
- ✅ `backend/Dockerfile.api` - Railway-compatible Docker setup
- ✅ `backend/api_server.py` - Railway environment variables
- ✅ `backend/deploy_railway.py` - Automated deployment helper
- ✅ Railway CLI installed and ready

---

## 🎯 What You Need to Do Now

### Step 1: Login to Railway
```bash
railway login
```
This will open your browser for authentication.

### Step 2: Navigate to Backend Directory
```bash
cd backend
```

### Step 3: Create Railway Project and Link Repository

**Option A: Use Railway Dashboard (Recommended)**
1. Go to [railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose: `nocommitmentsyet/valuation100x`
5. Set **Root Directory**: `backend/`
6. Railway will auto-detect `Dockerfile.api`
7. Click "Deploy"

**Option B: Use Railway CLI**
```bash
# Link to your GitHub repository
railway link https://github.com/nocommitmentsyet/valuation100x

# Deploy
railway up
```

### Step 4: Configure Environment Variables

**Method 1: Railway CLI (Faster)**
```bash
# Required API Keys
railway variables set OPENAI_API_KEY=your_openai_key_here
railway variables set FMP_API_KEY=your_fmp_key_here
railway variables set TAVILY_API_KEY=your_tavily_key_here
railway variables set SEC_API_KEY=your_sec_api_key_here

# Optional API Keys (if you have them)
railway variables set ALPHAVANTAGE_API_KEY=your_alpha_key_here
railway variables set POLYGON_API_KEY=your_polygon_key_here

# Database (if using Supabase)
railway variables set SUPABASE_URL=your_supabase_url
railway variables set SUPABASE_ANON_KEY=your_supabase_key
```

**Method 2: Railway Dashboard**
1. Go to your project in Railway dashboard
2. Click "Variables" tab
3. Add each environment variable one by one

### Step 5: Get Your Live URL
```bash
railway domain
```

---

## 📋 Environment Variables Checklist

### ✅ Required Variables:
- [ ] `OPENAI_API_KEY` - Your OpenAI API key
- [ ] `FMP_API_KEY` - Financial Modeling Prep API key  
- [ ] `TAVILY_API_KEY` - Tavily search API key
- [ ] `SEC_API_KEY` - SEC API key

### 🔧 Optional Variables:
- [ ] `ALPHAVANTAGE_API_KEY` - Alpha Vantage API key
- [ ] `POLYGON_API_KEY` - Polygon.io API key
- [ ] `SUPABASE_URL` - Supabase project URL
- [ ] `SUPABASE_ANON_KEY` - Supabase anonymous key

### 🤖 Auto-Set by Railway:
- ✅ `PORT` - Application port (auto-assigned)
- ✅ `RAILWAY_ENVIRONMENT` - Environment (production/development)

---

## 🔍 Verification Steps

### 1. Check Deployment Status
```bash
railway status
```

### 2. View Deployment Logs
```bash
railway logs
```

### 3. Test Your API
Once deployed, your API will be available at:
```
https://your-app-name.up.railway.app/
```

Test endpoints:
- **Health Check**: `GET /api/health`
- **API Docs**: `GET /docs`
- **Analysis Start**: `POST /api/v1/analysis/start`

### 4. Example API Test
```bash
# Replace YOUR_RAILWAY_URL with your actual Railway URL
curl https://your-app-name.up.railway.app/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-23T10:30:00.000Z",
  "version": "1.0.0",
  "environment": "production"
}
```

---

## 🚨 Troubleshooting

### Common Issues:

**1. Build Failures**
```bash
# Check build logs
railway logs --deployment
```

**2. Environment Variable Issues**
```bash
# List all variables
railway variables

# Check specific variable
railway variables get OPENAI_API_KEY
```

**3. Port Issues**
- Railway automatically assigns `$PORT`
- Your app should use `os.getenv("PORT", 8000)`
- ✅ Already configured in your `api_server.py`

**4. Docker Build Issues**
- Check `backend/Dockerfile.api` syntax
- Ensure all requirements are in `requirements.txt`
- ✅ Already configured correctly

### Getting Help:
```bash
# Railway CLI help
railway help

# Get project info
railway status

# View real-time logs
railway logs --follow
```

---

## 🎯 Quick Deployment Commands

If you want to use the automated helper script:
```bash
cd backend
python deploy_railway.py
```

Or manual deployment:
```bash
cd backend
railway login
railway link https://github.com/nocommitmentsyet/valuation100x
railway up
railway domain
```

---

## 🔐 Security Notes

- ✅ Your `.env` file is in `.gitignore` (secrets safe)
- ✅ Environment variables are encrypted in Railway
- ✅ HTTPS is automatically provided by Railway
- ✅ Only necessary ports are exposed

---

## 📊 What Happens Next

1. **Railway builds your Docker container** using `Dockerfile.api`
2. **Installs all dependencies** from `requirements.txt`
3. **Starts your FastAPI server** on Railway's assigned port
4. **Provides HTTPS URL** for your API
5. **Auto-redeploys** on every git push to main branch

Your API will be live at: `https://your-app-name.up.railway.app`

---

## 🎉 Success Indicators

You'll know it's working when:
- ✅ Railway dashboard shows "Active" status
- ✅ Health endpoint returns 200 OK
- ✅ API documentation loads at `/docs`
- ✅ No errors in Railway logs

---

## 🔄 Future Updates

Every time you push to GitHub:
1. Railway automatically detects changes
2. Rebuilds and redeploys your app
3. Zero-downtime deployment
4. Rollback available if needed

---

Ready to deploy? Run: `railway login` to get started! 🚀
