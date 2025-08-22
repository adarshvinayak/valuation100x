#!/usr/bin/env python3
import asyncio
import os
from dotenv import load_dotenv
from tools.fmp import FMPClient

load_dotenv()

async def check_fields():
    api_key = os.getenv("FMP_API_KEY")
    async with FMPClient(api_key) as client:
        balance = await client.get_balance_sheet("MSFT", "annual", 1)
        cash_flow = await client.get_cash_flow("MSFT", "annual", 1)
        
        if balance and len(balance) > 0:
            print("BALANCE SHEET DEBT/CASH FIELDS:")
            for key in balance[0].keys():
                if 'debt' in key.lower() or 'cash' in key.lower():
                    print(f"  {key}: {balance[0][key]}")
        
        if cash_flow and len(cash_flow) > 0:
            print("\nCASH FLOW FREE CASH FLOW FIELDS:")
            for key in cash_flow[0].keys():
                if 'free' in key.lower() or 'cash' in key.lower():
                    print(f"  {key}: {cash_flow[0][key]}")

if __name__ == "__main__":
    asyncio.run(check_fields())
