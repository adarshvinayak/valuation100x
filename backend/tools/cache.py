"""
Filesystem JSON Cache with TTL

Provides caching functionality for API responses with time-to-live support.
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Dict, Callable
import hashlib
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class FileCache:
    """Filesystem-based cache with TTL support"""
    
    def __init__(self, cache_dir: str = "data/cache", default_ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_hours = default_ttl_hours
    
    def _get_cache_key(self, key: str) -> str:
        """Generate a safe filename from cache key"""
        # Hash the key to ensure safe filename
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the full path for a cache key"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_expired(self, cache_data: Dict) -> bool:
        """Check if cache entry is expired"""
        if "expires_at" not in cache_data:
            return True
        
        expires_at = datetime.fromisoformat(cache_data["expires_at"])
        return datetime.now() > expires_at
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if self._is_expired(cache_data):
                # Clean up expired cache
                cache_path.unlink(missing_ok=True)
                return None
            
            logger.debug(f"Cache hit for key: {key}")
            return cache_data["value"]
            
        except Exception as e:
            logger.error(f"Failed to read cache for key {key}: {e}")
            # Clean up corrupted cache
            cache_path.unlink(missing_ok=True)
            return None
    
    def set(self, key: str, value: Any, ttl_hours: Optional[int] = None) -> bool:
        """Set value in cache with TTL"""
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)
        
        ttl = ttl_hours or self.default_ttl_hours
        expires_at = datetime.now() + timedelta(hours=ttl)
        
        cache_data = {
            "value": value,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
            "original_key": key
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            logger.debug(f"Cache set for key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write cache for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_path.unlink(missing_ok=True)
            logger.debug(f"Cache deleted for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache for key {key}: {e}")
            return False
    
    def clear_expired(self) -> int:
        """Clear all expired cache entries"""
        cleared = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                if self._is_expired(cache_data):
                    cache_file.unlink()
                    cleared += 1
                    
            except Exception as e:
                logger.error(f"Failed to check cache file {cache_file}: {e}")
                # Remove corrupted files
                cache_file.unlink(missing_ok=True)
                cleared += 1
        
        logger.info(f"Cleared {cleared} expired cache entries")
        return cleared
    
    def clear_all(self) -> int:
        """Clear all cache entries"""
        cleared = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                cleared += 1
            except Exception as e:
                logger.error(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {cleared} cache entries")
        return cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_files = 0
        expired_files = 0
        total_size = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            total_files += 1
            total_size += cache_file.stat().st_size
            
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                if self._is_expired(cache_data):
                    expired_files += 1
                    
            except Exception:
                expired_files += 1  # Count corrupted as expired
        
        return {
            "total_entries": total_files,
            "expired_entries": expired_files,
            "valid_entries": total_files - expired_files,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }


# Global cache instance
_cache = None

def get_cache(cache_dir: str = "data/cache", default_ttl_hours: int = 24) -> FileCache:
    """Get global cache instance"""
    global _cache
    if _cache is None:
        _cache = FileCache(cache_dir, default_ttl_hours)
    return _cache


def cached(ttl_hours: Optional[int] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl_hours)
            
            return result
        
        return wrapper
    return decorator


async def cached_async(ttl_hours: Optional[int] = None, key_func: Optional[Callable] = None):
    """Decorator for caching async function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl_hours)
            
            return result
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test the cache
    cache = FileCache("test_cache")
    
    # Test basic operations
    cache.set("test_key", {"data": "test_value"})
    result = cache.get("test_key")
    print(f"Retrieved: {result}")
    
    # Test stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Clean up
    cache.clear_all()
