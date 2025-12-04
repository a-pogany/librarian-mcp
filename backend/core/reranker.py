"""
Cross-encoder reranking for improved precision
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """
    Rerank search results using cross-encoder for better precision
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker

        Args:
            model_name: Cross-encoder model from sentence-transformers
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Reranker model loaded successfully")

        except ImportError:
            logger.warning("sentence-transformers not installed, reranking disabled")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading reranker: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results

        Args:
            query: Search query
            results: Search results from retrieval
            top_k: Number of top results to return

        Returns:
            Reranked results with cross-encoder scores
        """
        if not self.model or not results:
            return results[:top_k]

        try:
            # Prepare query-document pairs
            pairs = [(query, result['snippet'] if 'snippet' in result else result.get('content', ''))
                     for result in results]

            # Get cross-encoder scores
            scores = self.model.predict(pairs)

            # Add scores to results
            for result, score in zip(results, scores):
                result['rerank_score'] = float(score)
                result['original_score'] = result.get('relevance_score', 0.0)

            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)

            logger.debug(f"Reranked {len(results)} results to top {top_k}")
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return results[:top_k]

    def is_available(self) -> bool:
        """Check if reranker is available"""
        return self.model is not None
