"""
Query caching for improved performance
"""

from functools import lru_cache
from typing import Optional
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class QueryCache:
    """LRU cache for query embeddings and results"""

    def __init__(self, max_size: int = 10000, ttl: int = 300):
        """
        Initialize query cache

        Args:
            max_size: Maximum number of cached entries
            ttl: Time to live in seconds (for future implementation)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._embedding_cache = {}  # Simple dict cache (can upgrade to TTL cache)

    def cache_key(self, query: str, **kwargs) -> str:
        """
        Generate cache key from query and params

        Args:
            query: Query string
            **kwargs: Additional parameters

        Returns:
            Cache key hash
        """
        params = json.dumps(kwargs, sort_keys=True)
        cache_string = f"{query}:{params}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def get_embedding(self, query: str) -> Optional:
        """Get cached embedding for query"""
        key = self.cache_key(query)
        return self._embedding_cache.get(key)

    def set_embedding(self, query: str, embedding):
        """Cache embedding for query"""
        key = self.cache_key(query)

        # Simple size limit (LRU would be better, but this works)
        if len(self._embedding_cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            first_key = next(iter(self._embedding_cache))
            del self._embedding_cache[first_key]

        self._embedding_cache[key] = embedding
        logger.debug(f"Cached embedding for query (cache size: {len(self._embedding_cache)})")

    def clear(self):
        """Clear all cached embeddings"""
        self._embedding_cache.clear()
        logger.info("Query cache cleared")

    def get_stats(self):
        """Get cache statistics"""
        return {
            'size': len(self._embedding_cache),
            'max_size': self.max_size,
            'ttl': self.ttl
        }
