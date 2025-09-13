# PowerShell deployment script for Lambda
Write-Host "🚀 Deploying Lambda with Price Fix" -ForegroundColor Green
Write-Host "=================================="

# Clean previous build
Write-Host "1. Cleaning previous build..."
if (Test-Path ".aws-sam") {
    Remove-Item -Recurse -Force .aws-sam
    Write-Host "   ✅ Cleaned .aws-sam directory"
}

# Build
Write-Host "2. Building Lambda container..."
$buildResult = sam build --use-container
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Build successful"
} else {
    Write-Host "   ❌ Build failed" -ForegroundColor Red
    exit 1
}

# Deploy
Write-Host "3. Deploying to AWS..."
$deployResult = sam deploy --disable-rollback
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Deploy successful"
} else {
    Write-Host "   ❌ Deploy failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🧪 Testing deployment..."
try {
    $response = Invoke-RestMethod -Uri "https://ppi7ci4lyhypmox7p4kp73bmsi0ydcon.lambda-url.us-east-1.on.aws/api/validate/ticker/AAPL" -Method GET
    
    if ($response.PSObject.Properties['current_price']) {
        Write-Host "✅ SUCCESS - current_price field is now present!" -ForegroundColor Green
        Write-Host "   Price: $($response.current_price)"
    } else {
        Write-Host "❌ FAILED - current_price field still missing" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Error testing API: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "Deployment complete!"
