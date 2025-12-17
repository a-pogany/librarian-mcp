"""
Semantic Query Cache for RAG systems

Unlike exact-match caching, semantic caching finds cached results for
queries that are semantically similar, even if worded differently.
This provides cache hits for paraphrased queries like:
- "how to search" vs "searching documentation"
- "configure auth" vs "authentication setup"
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached query and its results"""
    query: str
    embedding: np.ndarray
    results: List[Dict[str, Any]]
    timestamp: float
    hit_count: int = 0
    search_mode: str = "hybrid"


class SemanticQueryCache:
    """
    LRU cache with semantic similarity matching

    Features:
    - Exact match lookup (fast, O(1))
    - Semantic similarity lookup (finds paraphrased queries)
    - TTL-based expiration
    - LRU eviction policy
    """

    def __init__(
        self,
        embedding_generator=None,
        max_size: int = 1000,
        ttl_seconds: int = 300,
        similarity_threshold: float = 0.92,
        enable_semantic: bool = True
    ):
        """
        Initialize semantic cache

        Args:
            embedding_generator: EmbeddingGenerator for query encoding
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries
            similarity_threshold: Minimum similarity for semantic match (0.92 = very similar)
            enable_semantic: Enable semantic similarity matching
        """
        self.embedding_generator = embedding_generator
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.enable_semantic = enable_semantic

        # Ordered dict for LRU eviction
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Stats tracking
        self._stats = {
            'exact_hits': 0,
            'semantic_hits': 0,
            'misses': 0,
            'evictions': 0
        }

        logger.info(
            f"Semantic cache initialized: max_size={max_size}, "
            f"ttl={ttl_seconds}s, threshold={similarity_threshold}"
        )

    def get(
        self,
        query: str,
        search_mode: str = None,
        filters: Dict = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached results for a query

        Tries exact match first, then semantic similarity.

        Args:
            query: Search query
            search_mode: Search mode (for cache key differentiation)
            filters: Search filters (for cache key differentiation)

        Returns:
            Cached results if found and valid, None otherwise
        """
        # Generate cache key
        cache_key = self._make_cache_key(query, search_mode, filters)

        # Try exact match first (fast)
        entry = self._cache.get(cache_key)
        if entry:
            if self._is_valid(entry):
                self._stats['exact_hits'] += 1
                entry.hit_count += 1
                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                logger.debug(f"Cache exact hit for: {query[:50]}...")
                return entry.results
            else:
                # Expired entry
                del self._cache[cache_key]

        # Try semantic similarity match
        if self.enable_semantic and self.embedding_generator:
            semantic_result = self._semantic_lookup(query, search_mode, filters)
            if semantic_result:
                self._stats['semantic_hits'] += 1
                logger.debug(f"Cache semantic hit for: {query[:50]}...")
                return semantic_result

        self._stats['misses'] += 1
        return None

    def set(
        self,
        query: str,
        results: List[Dict[str, Any]],
        search_mode: str = None,
        filters: Dict = None
    ):
        """
        Cache results for a query

        Args:
            query: Search query
            results: Search results to cache
            search_mode: Search mode used
            filters: Search filters used
        """
        # Generate cache key
        cache_key = self._make_cache_key(query, search_mode, filters)

        # Generate embedding for semantic matching
        embedding = None
        if self.enable_semantic and self.embedding_generator:
            try:
                embedding = self.embedding_generator.encode_query(query)
            except Exception as e:
                logger.warning(f"Failed to generate embedding for cache: {e}")

        # Create cache entry
        entry = CacheEntry(
            query=query,
            embedding=embedding,
            results=results,
            timestamp=time.time(),
            search_mode=search_mode or "hybrid"
        )

        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            self._evict_oldest()

        # Add to cache
        self._cache[cache_key] = entry
        self._cache.move_to_end(cache_key)

        logger.debug(f"Cached results for: {query[:50]}... (cache size: {len(self._cache)})")

    def _semantic_lookup(
        self,
        query: str,
        search_mode: str,
        filters: Dict
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find semantically similar cached query

        Args:
            query: Search query
            search_mode: Search mode
            filters: Search filters

        Returns:
            Cached results if similar query found, None otherwise
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_query(query)

            best_match = None
            best_similarity = 0.0

            # Search through cache for similar queries
            for cache_key, entry in self._cache.items():
                # Skip if no embedding or expired
                if entry.embedding is None or not self._is_valid(entry):
                    continue

                # Skip if different search mode (optional strictness)
                if search_mode and entry.search_mode != search_mode:
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, entry.embedding)

                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = entry

            if best_match:
                best_match.hit_count += 1
                logger.debug(
                    f"Semantic match found: '{query[:30]}...' ~ '{best_match.query[:30]}...' "
                    f"(similarity: {best_similarity:.3f})"
                )
                return best_match.results

        except Exception as e:
            logger.warning(f"Semantic lookup failed: {e}")

        return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _make_cache_key(
        self,
        query: str,
        search_mode: str = None,
        filters: Dict = None
    ) -> str:
        """Generate cache key from query and parameters"""
        import hashlib
        import json

        key_parts = [query.lower().strip()]

        if search_mode:
            key_parts.append(f"mode:{search_mode}")

        if filters:
            # Sort filters for consistent keys
            filter_str = json.dumps(filters, sort_keys=True)
            key_parts.append(f"filters:{filter_str}")

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _is_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid (not expired)"""
        age = time.time() - entry.timestamp
        return age < self.ttl_seconds

    def _evict_oldest(self):
        """Evict the oldest (least recently used) entry"""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats['evictions'] += 1

    def invalidate(self, query: str = None, pattern: str = None):
        """
        Invalidate cache entries

        Args:
            query: Specific query to invalidate
            pattern: Pattern to match for invalidation (substring match)
        """
        if query:
            # Invalidate specific query (all modes/filters)
            keys_to_delete = [
                k for k, v in self._cache.items()
                if v.query.lower() == query.lower()
            ]
        elif pattern:
            # Invalidate by pattern
            keys_to_delete = [
                k for k, v in self._cache.items()
                if pattern.lower() in v.query.lower()
            ]
        else:
            # Invalidate all
            keys_to_delete = list(self._cache.keys())

        for key in keys_to_delete:
            del self._cache[key]

        logger.info(f"Invalidated {len(keys_to_delete)} cache entries")

    def clear(self):
        """Clear all cached entries"""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = (
            self._stats['exact_hits'] +
            self._stats['semantic_hits'] +
            self._stats['misses']
        )

        hit_rate = 0.0
        if total_requests > 0:
            total_hits = self._stats['exact_hits'] + self._stats['semantic_hits']
            hit_rate = total_hits / total_requests

        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'similarity_threshold': self.similarity_threshold,
            'exact_hits': self._stats['exact_hits'],
            'semantic_hits': self._stats['semantic_hits'],
            'misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'hit_rate': round(hit_rate, 3),
            'semantic_enabled': self.enable_semantic
        }

    def warm_up(self, common_queries: List[str], search_func):
        """
        Pre-populate cache with common queries

        Args:
            common_queries: List of frequently used queries
            search_func: Function to execute search and get results
        """
        for query in common_queries:
            try:
                results = search_func(query)
                self.set(query, results)
            except Exception as e:
                logger.warning(f"Failed to warm cache for '{query}': {e}")

        logger.info(f"Cache warmed with {len(common_queries)} queries")
