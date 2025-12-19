import json
import time
import hashlib
import logging
import threading
from typing import Any, Optional, Dict
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CacheService:
    """
    Cache service for Reddit fetch & analysis
    - Memory LRU + TTL
    - Disk JSON cache + TTL
    - Thread-safe
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        default_ttl: int = 3600,
        max_memory_items: int = 500
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.default_ttl = default_ttl
        self.max_memory_items = max_memory_items

        # LRU memory cache
        self.memory_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Thread safety
        self._lock = threading.RLock()

    # -------------------------
    # Internal helpers
    # -------------------------

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _cache_path(self, hashed_key: str) -> Path:
        return self.cache_dir / f"{hashed_key}.json"

    def _is_expired(self, expires_at: float) -> bool:
        return time.time() >= expires_at

    def _evict_if_needed(self):
        while len(self.memory_cache) > self.max_memory_items:
            self.memory_cache.popitem(last=False)  # remove LRU

    # -------------------------
    # Public API
    # -------------------------

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            with self._lock:
                hashed_key = self._hash_key(key)
                expires_at = time.time() + (ttl or self.default_ttl)

                cache_data = {
                    "value": value,
                    "expires_at": expires_at,
                    "created_at": time.time()
                }

                # Memory cache (LRU)
                self.memory_cache[hashed_key] = cache_data
                self.memory_cache.move_to_end(hashed_key)
                self._evict_if_needed()

                # Disk cache
                with open(self._cache_path(hashed_key), "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, ensure_ascii=False)

                return True
        except Exception as e:
            logger.error(f"[Cache] set failed: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        try:
            with self._lock:
                hashed_key = self._hash_key(key)

                # 1️⃣ Memory cache
                if hashed_key in self.memory_cache:
                    data = self.memory_cache[hashed_key]
                    if not self._is_expired(data["expires_at"]):
                        self.memory_cache.move_to_end(hashed_key)
                        return data["value"]
                    else:
                        del self.memory_cache[hashed_key]

                # 2️⃣ Disk cache
                cache_file = self._cache_path(hashed_key)
                if cache_file.exists():
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if not self._is_expired(data["expires_at"]):
                        self.memory_cache[hashed_key] = data
                        self.memory_cache.move_to_end(hashed_key)
                        self._evict_if_needed()
                        return data["value"]
                    else:
                        cache_file.unlink(missing_ok=True)

                return None
        except Exception as e:
            logger.error(f"[Cache] get failed: {e}")
            return None

    def delete(self, key: str) -> bool:
        try:
            with self._lock:
                hashed_key = self._hash_key(key)
                self.memory_cache.pop(hashed_key, None)

                cache_file = self._cache_path(hashed_key)
                if cache_file.exists():
                    cache_file.unlink()

                return True
        except Exception as e:
            logger.error(f"[Cache] delete failed: {e}")
            return False

    def clear(self) -> bool:
        try:
            with self._lock:
                self.memory_cache.clear()
                for f in self.cache_dir.glob("*.json"):
                    f.unlink()
            logger.info("[Cache] cleared")
            return True
        except Exception as e:
            logger.error(f"[Cache] clear failed: {e}")
            return False

    def cleanup_expired(self) -> int:
        cleaned = 0
        now = time.time()

        with self._lock:
            # Memory
            expired_keys = [
                k for k, v in self.memory_cache.items()
                if now >= v["expires_at"]
            ]
            for k in expired_keys:
                del self.memory_cache[k]
                cleaned += 1

            # Disk
            for f in self.cache_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if now >= data["expires_at"]:
                        f.unlink()
                        cleaned += 1
                except Exception:
                    f.unlink(missing_ok=True)
                    cleaned += 1

        if cleaned:
            logger.info(f"[Cache] cleaned {cleaned} expired entries")
        return cleaned

    def get_stats(self) -> Dict[str, Any]:
        memory_items = len(self.memory_cache)
        file_items = len(list(self.cache_dir.glob("*.json")))

        return {
            "memory_items": memory_items,
            "file_items": file_items,
            "max_memory_items": self.max_memory_items
        }

