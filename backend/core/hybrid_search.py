"""
Hybrid search engine combining keyword and semantic search
"""

import logging
from typing import List, Dict, Optional, Any
from collections import defaultdict
from .search import SearchEngine
from .semantic_search import SemanticSearchEngine

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Hybrid search combining keyword and semantic search

    Supports three modes:
    - keyword: Pure keyword search
    - semantic: Pure vector similarity search
    - hybrid: Combined search with configurable weighting
    """

    def __init__(
        self,
        keyword_engine: SearchEngine,
        semantic_engine: Optional[SemanticSearchEngine] = None,
        default_mode: str = "hybrid",
        semantic_weight: float = 0.5,
        use_rrf: bool = True
    ):
        """
        Initialize hybrid search engine

        Args:
            keyword_engine: Keyword search engine
            semantic_engine: Semantic search engine (optional for keyword-only mode)
            default_mode: Default search mode (keyword, semantic, or hybrid)
            semantic_weight: Weight for semantic scores in hybrid mode (0-1)
            use_rrf: Use Reciprocal Rank Fusion instead of weighted average
        """
        self.keyword_engine = keyword_engine
        self.semantic_engine = semantic_engine
        self.default_mode = default_mode
        self.semantic_weight = semantic_weight
        self.use_rrf = use_rrf

        # Validate mode
        if default_mode not in ['keyword', 'semantic', 'hybrid']:
            logger.warning(f"Invalid mode '{default_mode}', defaulting to 'keyword'")
            self.default_mode = 'keyword'

        # Validate semantic mode availability
        if default_mode in ['semantic', 'hybrid'] and not semantic_engine:
            logger.warning(f"Semantic engine not available, falling back to keyword mode")
            self.default_mode = 'keyword'

        fusion_method = "RRF" if use_rrf else "weighted average"
        logger.info(f"Initialized HybridSearchEngine in '{self.default_mode}' mode ({fusion_method})")

    def search(
        self,
        query: str,
        product: Optional[str] = None,
        component: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        max_results: int = 10,
        mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents using configured mode

        Args:
            query: Search query
            product: Filter by product
            component: Filter by component
            file_types: Filter by file extensions
            max_results: Maximum results to return
            mode: Override default mode (keyword, semantic, or hybrid)

        Returns:
            List of matching documents with relevance scores
        """
        # Use provided mode or default
        search_mode = mode or self.default_mode

        # Route to appropriate search method
        if search_mode == 'keyword':
            return self._keyword_search(query, product, component, file_types, max_results)
        elif search_mode == 'semantic':
            return self._semantic_search(query, product, component, file_types, max_results)
        elif search_mode == 'hybrid':
            return self._hybrid_search(query, product, component, file_types, max_results)
        else:
            logger.warning(f"Unknown mode '{search_mode}', using keyword search")
            return self._keyword_search(query, product, component, file_types, max_results)

    def _keyword_search(
        self,
        query: str,
        product: Optional[str],
        component: Optional[str],
        file_types: Optional[List[str]],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Execute keyword-only search"""
        logger.debug(f"Executing keyword search: {query}")
        results = self.keyword_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=max_results
        )

        # Add search mode metadata
        for result in results:
            result['search_mode'] = 'keyword'

        return results

    def _semantic_search(
        self,
        query: str,
        product: Optional[str],
        component: Optional[str],
        file_types: Optional[List[str]],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Execute semantic-only search"""
        if not self.semantic_engine:
            logger.warning("Semantic engine not available, falling back to keyword")
            return self._keyword_search(query, product, component, file_types, max_results)

        logger.debug(f"Executing semantic search: {query}")
        results = self.semantic_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=max_results
        )

        # Add search mode metadata
        for result in results:
            result['search_mode'] = 'semantic'

        return results

    def _hybrid_search(
        self,
        query: str,
        product: Optional[str],
        component: Optional[str],
        file_types: Optional[List[str]],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search combining keyword and semantic results

        Uses either RRF or weighted average based on configuration
        """
        if not self.semantic_engine:
            logger.warning("Semantic engine not available, falling back to keyword")
            return self._keyword_search(query, product, component, file_types, max_results)

        logger.debug(f"Executing hybrid search: {query}")

        # Get results from both engines
        # Fetch proportionally more results for better fusion
        candidate_multiplier = 3 if self.use_rrf else 2

        keyword_results = self.keyword_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=max_results * candidate_multiplier
        )

        semantic_results = self.semantic_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=max_results * candidate_multiplier
        )

        # Use RRF or weighted average
        if self.use_rrf:
            combined_results = self._reciprocal_rank_fusion(keyword_results, semantic_results)
            return combined_results[:max_results]

        # Create score maps for efficient lookup
        keyword_scores = {r['id']: r['relevance_score'] for r in keyword_results}
        semantic_scores = {r['id']: r['similarity_score'] for r in semantic_results}

        # Combine results
        all_doc_ids = set(keyword_scores.keys()) | set(semantic_scores.keys())
        combined_results = []

        for doc_id in all_doc_ids:
            # Get scores (0 if not present in that search mode)
            keyword_score = keyword_scores.get(doc_id, 0.0)
            semantic_score = semantic_scores.get(doc_id, 0.0)

            # Calculate hybrid score
            hybrid_score = (
                (1 - self.semantic_weight) * keyword_score +
                self.semantic_weight * semantic_score
            )

            # Get document metadata (prefer keyword results, fall back to semantic)
            doc_data = None
            for result in keyword_results:
                if result['id'] == doc_id:
                    doc_data = result
                    break

            if not doc_data:
                for result in semantic_results:
                    if result['id'] == doc_id:
                        doc_data = result
                        break

            if doc_data:
                # Create combined result
                combined_results.append({
                    'id': doc_id,
                    'file_path': doc_data['file_path'],
                    'product': doc_data['product'],
                    'component': doc_data['component'],
                    'file_name': doc_data['file_name'],
                    'file_type': doc_data['file_type'],
                    'snippet': doc_data['snippet'],
                    'relevance_score': round(hybrid_score, 2),
                    'keyword_score': round(keyword_score, 2),
                    'semantic_score': round(semantic_score, 2),
                    'search_mode': 'hybrid',
                    'last_modified': doc_data['last_modified']
                })

        # Sort by hybrid score
        combined_results.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Return top N results
        return combined_results[:max_results]

    def _reciprocal_rank_fusion(
        self,
        keyword_results: List[Dict],
        semantic_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        RRF: Better than weighted average for combining rankings

        Formula: score = 1/(k + rank)

        Args:
            keyword_results: Results from keyword search
            semantic_results: Results from semantic search
            k: Constant for RRF (default 60)

        Returns:
            Fused results sorted by RRF score
        """
        scores = defaultdict(float)
        doc_data = {}

        # Add keyword scores
        for rank, result in enumerate(keyword_results):
            doc_id = result['id']
            scores[doc_id] += 1.0 / (k + rank)
            doc_data[doc_id] = result

        # Add semantic scores
        for rank, result in enumerate(semantic_results):
            doc_id = result['id']
            scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in doc_data:
                doc_data[doc_id] = result

        # Sort by RRF score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Format results
        results = []
        for doc_id, rrf_score in sorted_docs:
            result = doc_data[doc_id].copy()
            result['relevance_score'] = round(rrf_score, 4)
            result['search_mode'] = 'hybrid_rrf'
            results.append(result)

        return results

    def get_document(self, path: str, section: Optional[str] = None) -> Optional[Dict]:
        """
        Get full document content (delegates to keyword engine)

        Args:
            path: Relative path to document
            section: Optional section heading to extract

        Returns:
            Document content and metadata
        """
        return self.keyword_engine.get_document(path, section)

    def get_mode(self) -> str:
        """Get current default search mode"""
        return self.default_mode

    def set_mode(self, mode: str):
        """
        Set default search mode

        Args:
            mode: Search mode (keyword, semantic, or hybrid)
        """
        if mode not in ['keyword', 'semantic', 'hybrid']:
            raise ValueError(f"Invalid mode: {mode}")

        if mode in ['semantic', 'hybrid'] and not self.semantic_engine:
            raise ValueError(f"Cannot set mode to '{mode}': semantic engine not available")

        self.default_mode = mode
        logger.info(f"Search mode changed to: {mode}")

    def set_semantic_weight(self, weight: float):
        """
        Set semantic weight for hybrid mode

        Args:
            weight: Semantic weight (0-1)
        """
        if not 0 <= weight <= 1:
            raise ValueError("Semantic weight must be between 0 and 1")

        self.semantic_weight = weight
        logger.info(f"Semantic weight set to: {weight}")

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        stats = {
            'default_mode': self.default_mode,
            'semantic_weight': self.semantic_weight,
            'keyword_engine': 'available',
            'semantic_engine': 'available' if self.semantic_engine else 'not available'
        }

        if self.semantic_engine:
            stats['semantic_stats'] = self.semantic_engine.get_stats()

        return stats
