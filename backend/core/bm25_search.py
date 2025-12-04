"""
BM25-based keyword search (better than TF-IDF)
"""

from typing import List, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BM25Search:
    """BM25-based keyword search (better than simple keyword matching)"""

    def __init__(self, documents: List[Dict] = None):
        """
        Initialize BM25 search

        Args:
            documents: List of documents with 'content' field
        """
        self.documents = documents or []
        self.bm25 = None
        self.tokenized_corpus = []

        if documents:
            self._build_index(documents)

    def _build_index(self, documents: List[Dict]):
        """Build BM25 index from documents"""
        try:
            from rank_bm25 import BM25Okapi

            # Tokenize documents
            self.tokenized_corpus = [
                self._tokenize(doc.get('content', ''))
                for doc in documents
            ]

            # Build BM25 index
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self.documents = documents

            logger.info(f"BM25 index built with {len(documents)} documents")

        except ImportError:
            logger.warning("rank-bm25 not installed, BM25 search disabled")
            logger.warning("Install with: pip install rank-bm25")
            self.bm25 = None
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (can be enhanced with stemming/lemmatization)

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and split on whitespace
        return text.lower().split()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        BM25 search

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of search results with BM25 scores
        """
        if not self.bm25 or not self.documents:
            return []

        try:
            # Tokenize query
            tokenized_query = self._tokenize(query)

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            # Build results
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include docs with non-zero scores
                    result = self.documents[idx].copy()
                    result['bm25_score'] = float(scores[idx])
                    results.append(result)

            logger.debug(f"BM25 search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error during BM25 search: {e}")
            return []

    def is_available(self) -> bool:
        """Check if BM25 search is available"""
        return self.bm25 is not None

    def update_index(self, documents: List[Dict]):
        """Update BM25 index with new documents"""
        self._build_index(documents)
