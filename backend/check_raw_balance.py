#!/usr/bin/env python3
import asyncio
import os
from dotenv import load_dotenv
from tools.fmp import FMPClient

load_dotenv()

async def check_raw_balance():
    api_key = os.getenv("FMP_API_KEY")
    async with FMPClient(api_key) as client:
        # Get raw API response
        raw_data = await client._make_request("balance-sheet-statement/MSFT", {"limit": 1})
        
        if raw_data and len(raw_data) > 0:
            print("RAW EQUITY/DEBT FIELDS:")
            for key, value in raw_data[0].items():
                if ('equity' in key.lower() or 'debt' in key.lower() or 
                    'stockholder' in key.lower()):
                    print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(check_raw_balance())
