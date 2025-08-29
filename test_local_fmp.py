#!/usr/bin/env python3
"""
Test local FMP validation to diagnose root cause
"""
import sys
import os
import asyncio
sys.path.append('.')

from tools.fmp import FMPClient

async def test_local_validation():
    print('🔍 Testing local FMP validation...')
    
    # Check API key
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        print('❌ FMP_API_KEY not found in environment')
        return None
    
    print(f'✅ FMP_API_KEY found (length: {len(api_key)})')
    
    # Test FMP client
    try:
        async with FMPClient(api_key) as client:
            print('📡 Making FMP API call for AAPL...')
            profile = await client.get_company_profile('AAPL')
            
            if profile:
                print(f'✅ Local FMP call successful!')
                print(f'   Company: {profile.get("company_name", "No name")}')
                print(f'   Price: {profile.get("current_price", "No price")}')
                print(f'   Valid: {profile.get("company_name") is not None}')
                return profile
            else:
                print('❌ FMP returned empty response')
                return None
                
    except Exception as e:
        print(f'❌ Local FMP call failed: {e}')
        print(f'   Error type: {type(e).__name__}')
        return None

if __name__ == "__main__":
    result = asyncio.run(test_local_validation())
    if result:
        print('\n✅ LOCAL CODE WORKING - Issue is in Lambda deployment')
    else:
        print('\n❌ LOCAL CODE BROKEN - Need to fix FMP integration')
