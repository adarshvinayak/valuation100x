#!/usr/bin/env python3
import asyncio
import os
from dotenv import load_dotenv
from tools.fmp import FMPClient

load_dotenv()

async def check_equity():
    api_key = os.getenv("FMP_API_KEY")
    async with FMPClient(api_key) as client:
        balance = await client.get_balance_sheet("MSFT", "annual", 1)
        
        if balance and len(balance) > 0:
            print("EQUITY FIELDS:")
            for key in balance[0].keys():
                if 'equity' in key.lower() or 'stockholder' in key.lower():
                    print(f"  {key}: {balance[0][key]}")
            
            # Show all debt related values
            print("\nDEBT VALUES:")
            data = balance[0]
            print(f"  total_debt: {data.get('total_debt', 'NOT FOUND')}")
            print(f"  total_stockholders_equity: {data.get('total_stockholders_equity', 'NOT FOUND')}")
            print(f"  shareholders_equity: {data.get('shareholders_equity', 'NOT FOUND')}")

if __name__ == "__main__":
    asyncio.run(check_equity())
