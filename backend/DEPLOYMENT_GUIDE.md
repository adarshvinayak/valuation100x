# Complete Lambda Rebuild and Deployment Guide

## 🎯 Goal
Rebuild the Lambda container with current_price field and deploy to a fresh stack.

## 📋 Step-by-Step Instructions

### PHASE 1: Clean Up Old Resources
```powershell
# 1. Navigate to backend directory
cd backend

# 2. List existing CloudFormation stacks
aws cloudformation list-stacks --query 'StackSummaries[?contains(StackName, `deepresearch`)].{Name:StackName,Status:StackStatus}' --output table

# 3. Delete old stacks (run for each stack found)
aws cloudformation delete-stack --stack-name deepresearch-backend
aws cloudformation delete-stack --stack-name deepresearch100xvaluation-backend

# 4. Wait for deletion to complete (check status)
aws cloudformation describe-stacks --stack-name deepresearch-backend --query 'Stacks[0].StackStatus' --output text
# Should return "DELETE_COMPLETE" or stack not found

# 5. Clean local build artifacts
Remove-Item -Recurse -Force .aws-sam -ErrorAction SilentlyContinue
Remove-Item lambda-deployment.zip -ErrorAction SilentlyContinue
```

### PHASE 2: Update Configuration for New Stack
```powershell
# 6. Update samconfig.toml with new stack name
# Edit backend/samconfig.toml and change:
# stack_name = "deepresearch-backend-v2"
# s3_prefix = "deepresearch-backend-v2"
```

### PHASE 3: Rebuild Container with New Code
```powershell
# 7. Force rebuild without cache
sam build --use-container --no-cached --debug

# 8. Verify the build includes our changes
Get-Content .aws-sam\build\DeepResearchAPI\api_server.py | Select-String "current_price.*Optional"
# Should show: current_price: Optional[float]
```

### PHASE 4: Deploy to New Stack
```powershell
# 9. Deploy with guided setup (first time)
sam deploy --guided

# Answer prompts:
# Stack Name: deepresearch-backend-v2
# AWS Region: us-east-1
# Parameter Environment: prod
# Confirm changes before deploy: Y
# Allow SAM CLI IAM role creation: Y
# Disable rollback: Y
# Save parameters to samconfig.toml: Y

# 10. If guided deploy fails, try force deploy
sam deploy --force-upload --no-confirm-changeset --disable-rollback
```

### PHASE 5: Update Function URL (if needed)
```powershell
# 11. Get the new Lambda Function URL
aws lambda get-function-url-config --function-name deepresearch-api-v2-prod --query 'FunctionUrl' --output text

# 12. If no Function URL exists, create one
aws lambda create-function-url-config --function-name deepresearch-api-v2-prod --auth-type NONE --cors AllowCredentials=false,AllowHeaders=["*"],AllowMethods=["*"],AllowOrigins=["*"]
```

### PHASE 6: Test Deployment
```powershell
# 13. Test the new Lambda function
# Replace with your actual Function URL from step 11
$NEW_URL = "https://[NEW-LAMBDA-URL]/api/validate/ticker/AAPL"
Invoke-RestMethod -Uri $NEW_URL -Method GET

# 14. Verify current_price field is present
$response = Invoke-RestMethod -Uri $NEW_URL -Method GET
if ($response.PSObject.Properties['current_price']) {
    Write-Host "✅ SUCCESS - current_price field is present!" -ForegroundColor Green
    Write-Host "Value: $($response.current_price)"
} else {
    Write-Host "❌ FAILED - current_price field still missing" -ForegroundColor Red
}
```

### PHASE 7: Update Frontend Configuration
```powershell
# 15. Update frontend API configuration
# Edit frontend/src/config/api.ts
# Change API_BASE_URL to the new Lambda Function URL
```

### PHASE 8: Clean Up Old Lambda Function (Optional)
```powershell
# 16. After confirming new deployment works, delete old function
aws lambda delete-function --function-name deepresearch-api-prod
```

## 🎯 Expected Results

### Success Indicators:
- ✅ New CloudFormation stack created successfully
- ✅ Lambda function responds with current_price field
- ✅ API response includes: current_price, day_low, day_high, volume
- ✅ Frontend displays real-time prices for all US stocks

### Sample Successful Response:
```json
{
  "ticker": "AAPL",
  "is_valid": true,
  "company_name": "Apple Inc.",
  "exchange": "NASDAQ",
  "sector": "Unknown",
  "market_cap": 3451283394232.0,
  "current_price": 227.52,
  "day_low": 225.30,
  "day_high": 229.87,
  "volume": 45000000,
  "last_updated": "2025-08-29T12:30:00.000000"
}
```

## 🚨 Troubleshooting

### If deployment still fails:
1. Check Docker is running: `docker ps`
2. Check AWS credentials: `aws sts get-caller-identity`
3. Try different region: `--region us-west-2`
4. Use manual ECR push (see advanced section)

### If current_price is still missing:
1. Verify code changes: `Get-Content api_server.py | Select-String "current_price"`
2. Check FMP API: test `/api/debug/fmp-test/AAPL` endpoint
3. Check Lambda logs: `sam logs --stack-name deepresearch-backend-v2`

## 🎉 Final Step
Once current_price field appears in the API response, the real-time price display will work for ALL US stocks in your frontend!
