"""
Semantic search engine using vector embeddings
"""

import logging
from typing import List, Dict, Optional, Any
from .embeddings import EmbeddingGenerator
from .vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """Semantic search using vector embeddings and similarity matching"""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_db: VectorDatabase,
        indexer
    ):
        """
        Initialize semantic search engine

        Args:
            embedding_generator: Embedding generator instance
            vector_db: Vector database instance
            indexer: Document indexer for metadata lookup
        """
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.indexer = indexer

    def search(
        self,
        query: str,
        product: Optional[str] = None,
        component: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        max_results: int = 10,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search documents using semantic similarity

        Args:
            query: Search query
            product: Filter by product
            component: Filter by component
            file_types: Filter by file extensions
            max_results: Maximum results to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of matching documents with similarity scores
        """
        if not query.strip():
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_query(query)

            # Build metadata filter
            where_filter = self._build_filter(product, component, file_types)

            # Search vector database
            vector_results = self.vector_db.search(
                query_embedding=query_embedding,
                n_results=max_results * 2,  # Get more candidates for filtering
                where=where_filter
            )

            # Filter by minimum similarity
            filtered_results = [
                r for r in vector_results
                if r['similarity'] >= min_similarity
            ]

            # Enrich with document metadata from indexer
            enriched_results = []
            for result in filtered_results[:max_results]:
                doc_id = result['id']
                doc = self.indexer.index.documents.get(doc_id)

                if doc:
                    # Extract snippet from content
                    snippet = self._extract_snippet(doc['content'], query)

                    enriched_results.append({
                        'id': doc_id,
                        'file_path': doc_id,
                        'product': doc['product'],
                        'component': doc['component'],
                        'file_name': doc['file_name'],
                        'file_type': doc['file_type'],
                        'snippet': snippet,
                        'relevance_score': round(result['similarity'], 2),
                        'similarity_score': round(result['similarity'], 2),
                        'last_modified': doc['last_modified']
                    })

            return enriched_results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise

    def _build_filter(
        self,
        product: Optional[str],
        component: Optional[str],
        file_types: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Build metadata filter for ChromaDB"""
        where_filter = {}

        if product:
            where_filter['product'] = product

        if component:
            where_filter['component'] = component

        if file_types:
            # ChromaDB supports $in operator for multiple values
            where_filter['file_type'] = {"$in": file_types}

        return where_filter if where_filter else None

    def _extract_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """
        Extract relevant snippet from content based on query

        Args:
            content: Document content
            query: Search query
            max_length: Maximum snippet length

        Returns:
            Relevant snippet
        """
        lines = content.split('\n')
        query_lower = query.lower()

        # Find line most similar to query (simple keyword matching)
        best_line = ""
        max_matches = 0

        query_words = query_lower.split()

        for line in lines:
            line_lower = line.lower()
            matches = sum(1 for word in query_words if word in line_lower)

            if matches > max_matches:
                max_matches = matches
                best_line = line

        # If no matches, return first non-empty line
        if not best_line:
            for line in lines:
                if line.strip():
                    best_line = line
                    break

        # Truncate if too long
        snippet = best_line.strip()
        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "..."

        return snippet

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            'type': 'semantic',
            'model': self.embedding_generator.get_model_name(),
            'embedding_dimension': self.embedding_generator.get_dimension(),
            'vector_db_stats': self.vector_db.get_stats()
        }
