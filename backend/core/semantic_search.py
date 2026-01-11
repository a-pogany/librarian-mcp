"""
Semantic search engine using vector embeddings
"""

import logging
from typing import List, Dict, Optional, Any
from .embeddings import EmbeddingGenerator
from .vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """Semantic search using vector embeddings and similarity matching with reranking"""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_db: VectorDatabase,
        indexer,
        use_reranking: bool = True
    ):
        """
        Initialize semantic search engine

        Args:
            embedding_generator: Embedding generator instance
            vector_db: Vector database instance
            indexer: Document indexer for metadata lookup
            use_reranking: Enable two-stage reranking for better precision
        """
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.indexer = indexer
        self.use_reranking = use_reranking
        self.reranker = None

        # Initialize reranker if enabled
        if use_reranking:
            try:
                from .reranker import Reranker
                self.reranker = Reranker()
                if not self.reranker.is_available():
                    logger.warning("Reranker not available, disabling reranking")
                    self.reranker = None
            except ImportError:
                logger.warning("Reranker module not found, disabling reranking")
                self.reranker = None

    def search(
        self,
        query: str,
        product: Optional[str] = None,
        component: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        max_results: int = 10,
        min_similarity: float = 0.0,
        # Email-specific filters (pre-filter via ChromaDB WHERE clause)
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        cc: Optional[str] = None,
        folder: Optional[str] = None,
        subject_contains: Optional[str] = None,
        has_attachments: Optional[bool] = None,
        date_after: Optional[str] = None,
        date_before: Optional[str] = None,
        thread_id: Optional[str] = None
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
            sender: Filter emails by sender (partial match)
            recipient: Filter emails by recipient (partial match)
            cc: Filter emails by CC recipient (partial match)
            folder: Filter emails by folder (case-insensitive)
            subject_contains: Filter emails by subject (partial match)
            has_attachments: Filter emails with/without attachments
            date_after: Filter emails after this date (ISO 8601)
            date_before: Filter emails before this date (ISO 8601)
            thread_id: Filter emails by thread ID (exact match)

        Returns:
            List of matching documents with similarity scores
        """
        if not query.strip():
            return []

        # Build email filters dict for partial match post-filtering
        email_filters = {
            'sender': sender,
            'recipient': recipient,
            'cc': cc,
            'folder': folder,
            'subject_contains': subject_contains,
            'has_attachments': has_attachments,
            'date_after': date_after,
            'date_before': date_before,
            'thread_id': thread_id
        }
        # Remove None values
        email_filters = {k: v for k, v in email_filters.items() if v is not None}

        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_query(query)

            # Build metadata filter (includes email filters that ChromaDB can handle)
            where_filter = self._build_filter(
                product, component, file_types,
                folder=folder,
                has_attachments=has_attachments,
                thread_id=thread_id,
                date_after=date_after,
                date_before=date_before
            )

            # Search vector database
            # If using reranking, fetch more candidates for stage 2
            candidate_multiplier = 5 if self.reranker and self.reranker.is_available() else 2
            vector_results = self.vector_db.search(
                query_embedding=query_embedding,
                n_results=max_results * candidate_multiplier,
                where=where_filter
            )

            # Filter by minimum similarity
            filtered_results = [
                r for r in vector_results
                if r['similarity'] >= min_similarity
            ]

            # Enrich with document metadata from indexer
            enriched_results = []
            for result in filtered_results:
                doc_id = result['id']
                # Support chunked document IDs (e.g., 'path#chunk0') by using base doc id
                base_doc_id = doc_id.split('#', 1)[0]
                doc = self.indexer.index.documents.get(base_doc_id)

                # Fallback: try full doc_id if base lookup failed
                if not doc:
                    doc = self.indexer.index.documents.get(doc_id)

                if doc:
                    # Apply partial-match email filters (not handled by ChromaDB WHERE)
                    if email_filters and not self._matches_partial_email_filters(doc, email_filters):
                        continue

                    # Extract snippet from content
                    snippet = self._extract_snippet(doc['content'], query)

                    enriched_results.append({
                        'id': base_doc_id,
                        'file_path': base_doc_id,
                        'product': doc['product'],
                        'component': doc['component'],
                        'file_name': doc['file_name'],
                        'file_type': doc['file_type'],
                        'snippet': snippet,
                        'content': snippet,  # For reranker
                        'content_preview': doc.get('content', '')[:500],
                        'relevance_score': round(result['similarity'], 2),
                        'similarity_score': round(result['similarity'], 2),
                        'last_modified': doc['last_modified'],
                        'metadata': doc.get('metadata', {}),
                        'headings': doc.get('headings', []),
                        'doc_type': doc.get('doc_type'),
                        'tags': doc.get('tags', [])
                    })

            # Stage 2: Rerank results for better precision
            if self.reranker and self.reranker.is_available() and len(enriched_results) > max_results:
                logger.debug(f"Reranking {len(enriched_results)} candidates to top {max_results}")
                enriched_results = self.reranker.rerank(query, enriched_results, top_k=max_results)

            return enriched_results[:max_results]

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise

    def _build_filter(
        self,
        product: Optional[str],
        component: Optional[str],
        file_types: Optional[List[str]],
        # Email filters (exact match - can use ChromaDB WHERE)
        folder: Optional[str] = None,
        has_attachments: Optional[bool] = None,
        thread_id: Optional[str] = None,
        date_after: Optional[str] = None,
        date_before: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Build metadata filter for ChromaDB

        ChromaDB requires $and operator when multiple conditions are present.
        Single condition: {"field": "value"}
        Multiple conditions: {"$and": [{"field1": "value1"}, {"field2": "value2"}]}

        Email filters use the email_* metadata fields stored during indexing:
        - email_folder, email_has_attachments, email_thread_id, email_date
        """
        conditions = []

        if product:
            conditions.append({'product': product})

        if component:
            conditions.append({'component': component})

        if file_types:
            # ChromaDB supports $in operator for multiple values
            conditions.append({'file_type': {"$in": file_types}})

        # Email-specific filters (use email_* fields from chunk metadata)
        if folder:
            # Case-insensitive folder match - we store lowercase in metadata
            conditions.append({'email_folder': folder.lower()})

        if has_attachments is not None:
            conditions.append({'email_has_attachments': has_attachments})

        if thread_id:
            conditions.append({'email_thread_id': thread_id})

        if date_after:
            # ChromaDB supports $gte operator
            conditions.append({'email_date': {'$gte': date_after}})

        if date_before:
            # ChromaDB supports $lte operator
            conditions.append({'email_date': {'$lte': date_before}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            # Single condition - return directly
            return conditions[0]
        else:
            # Multiple conditions - wrap in $and
            return {"$and": conditions}

    def _matches_partial_email_filters(self, doc: Dict, filters: Dict) -> bool:
        """Check if document matches partial-match email filters

        These filters require substring matching which ChromaDB doesn't support,
        so we apply them as post-filtering after vector search.

        Args:
            doc: Document dict from indexer
            filters: Dict of filter name -> value (only non-None values)

        Returns:
            True if doc matches all partial-match filters
        """
        metadata = doc.get('metadata', {})
        if not metadata:
            return False

        # Sender filter (partial match, case-insensitive)
        if 'sender' in filters:
            sender_val = metadata.get('from', '')
            if not sender_val or filters['sender'].lower() not in sender_val.lower():
                return False

        # Recipient filter (partial match in 'to' list)
        if 'recipient' in filters:
            to_list = metadata.get('to', [])
            if isinstance(to_list, str):
                to_list = [to_list]
            recipient_lower = filters['recipient'].lower()
            found = any(recipient_lower in addr.lower() for addr in to_list if addr)
            if not found:
                return False

        # CC filter (partial match in 'cc' list)
        if 'cc' in filters:
            cc_list = metadata.get('cc', [])
            if isinstance(cc_list, str):
                cc_list = [cc_list]
            cc_lower = filters['cc'].lower()
            found = any(cc_lower in addr.lower() for addr in cc_list if addr)
            if not found:
                return False

        # Subject filter (partial match, case-insensitive)
        if 'subject_contains' in filters:
            subject_val = metadata.get('subject', '')
            if not subject_val or filters['subject_contains'].lower() not in subject_val.lower():
                return False

        return True

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
