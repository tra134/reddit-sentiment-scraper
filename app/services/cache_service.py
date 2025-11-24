import pickle
import hashlib
import time
from typing import Any, Optional, Dict
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class CacheService:
    """Simple cache service for storing analysis results"""
    
    def __init__(self, cache_dir: str = "data/cache", default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key from string"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{key}.pkl"
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cache value"""
        try:
            cache_key = self._get_cache_key(key)
            cache_data = {
                'value': value,
                'expires_at': time.time() + (ttl or self.default_ttl),
                'created_at': time.time()
            }
            
            # Store in memory cache
            self.memory_cache[cache_key] = cache_data
            
            # Store in file cache
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            cache_key = self._get_cache_key(key)
            
            # Check memory cache first
            if cache_key in self.memory_cache:
                cache_data = self.memory_cache[cache_key]
                if time.time() < cache_data['expires_at']:
                    return cache_data['value']
                else:
                    del self.memory_cache[cache_key]
            
            # Check file cache
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Check if expired
                if time.time() < cache_data['expires_at']:
                    # Update memory cache
                    self.memory_cache[cache_key] = cache_data
                    return cache_data['value']
                else:
                    # Remove expired cache file
                    cache_path.unlink()
            
            return None
        except Exception as e:
            logger.error(f"Error getting cache: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        try:
            cache_key = self._get_cache_key(key)
            
            # Remove from memory cache
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
            
            # Remove from file cache
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
            
            return True
        except Exception as e:
            logger.error(f"Error deleting cache: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache"""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear file cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            logger.info("🧹 Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        memory_items = len(self.memory_cache)
        file_items = len(list(self.cache_dir.glob("*.pkl")))
        
        memory_size = sum(
            len(pickle.dumps(value)) 
            for value in self.memory_cache.values()
        )
        
        file_size = sum(
            cache_file.stat().st_size 
            for cache_file in self.cache_dir.glob("*.pkl")
        )
        
        return {
            'memory_items': memory_items,
            'file_items': file_items,
            'memory_size_kb': memory_size / 1024,
            'file_size_kb': file_size / 1024,
            'total_size_kb': (memory_size + file_size) / 1024
        }
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries"""
        cleaned_count = 0
        current_time = time.time()
        
        # Clean memory cache
        expired_keys = [
            key for key, data in self.memory_cache.items()
            if current_time >= data['expires_at']
        ]
        for key in expired_keys:
            del self.memory_cache[key]
            cleaned_count += 1
        
        # Clean file cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if current_time >= cache_data['expires_at']:
                    cache_file.unlink()
                    cleaned_count += 1
            except:
                # If file is corrupted, delete it
                cache_file.unlink()
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} expired cache entries")
        
        return cleaned_count

# Global cache instance
cache_service = CacheService()