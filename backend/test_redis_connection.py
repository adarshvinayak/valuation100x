#!/usr/bin/env python3
"""
Simple Redis connection test for Railway deployment
"""
import asyncio
import os
import sys

try:
    import redis.asyncio as redis
    print("✅ Redis library available")
except ImportError as e:
    print(f"❌ Redis library not available: {e}")
    sys.exit(1)

async def test_redis_connection():
    """Test Redis connection with Railway configuration"""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    print(f"🔧 Testing Redis connection to: {redis_url.split('@')[0] if '@' in redis_url else redis_url}")
    
    try:
        # Create connection with same settings as main app
        redis_client = await asyncio.wait_for(
            redis.from_url(
                redis_url, 
                decode_responses=True, 
                socket_timeout=5.0, 
                socket_connect_timeout=10.0,
                retry_on_timeout=True,
                health_check_interval=30
            ), 
            timeout=15.0
        )
        
        # Test ping
        print("🔄 Testing Redis ping...")
        pong = await asyncio.wait_for(redis_client.ping(), timeout=5.0)
        print(f"✅ Redis ping successful: {pong}")
        
        # Test basic operations
        print("🔄 Testing Redis operations...")
        await redis_client.set("test_key", "test_value")
        value = await redis_client.get("test_key")
        print(f"✅ Redis set/get successful: {value}")
        
        # Cleanup
        await redis_client.delete("test_key")
        await redis_client.close()
        
        print("✅ Redis connection test passed!")
        return True
        
    except asyncio.TimeoutError:
        print("❌ Redis connection timed out")
        return False
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Redis connection test...")
    result = asyncio.run(test_redis_connection())
    sys.exit(0 if result else 1)
