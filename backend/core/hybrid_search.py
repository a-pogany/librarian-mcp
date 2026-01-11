"""
Hybrid search engine combining keyword and semantic search

Enhanced with:
- HyDE (Hypothetical Document Embeddings) for conceptual queries
- Semantic query caching for paraphrased queries
- Intelligent query routing based on query analysis
- Parent document context enrichment
"""

import logging
from typing import List, Dict, Optional, Any
from collections import defaultdict
from .search import SearchEngine
from .semantic_search import SemanticSearchEngine
from .result_enhancer import ResultEnhancer

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Hybrid search combining keyword and semantic search

    Supports five modes:
    - keyword: Pure keyword search
    - semantic: Pure vector similarity search
    - hybrid: Combined search with configurable weighting (RRF or weighted average)
    - rerank: Two-stage search (semantic candidates + keyword filtering)
    - hyde: Hypothetical Document Embeddings for conceptual queries

    Enhanced features:
    - Semantic query caching (finds similar queries)
    - Intelligent query routing (auto-selects optimal mode)
    - Parent document context enrichment
    """

    def __init__(
        self,
        keyword_engine: SearchEngine,
        semantic_engine: Optional[SemanticSearchEngine] = None,
        default_mode: str = "hybrid",
        semantic_weight: float = 0.5,
        use_rrf: bool = True,
        rerank_candidates: int = 50,
        rerank_keyword_threshold: float = 0.1,
        enable_hyde: bool = True,
        enable_semantic_cache: bool = True,
        enable_query_routing: bool = True,
        enable_parent_context: bool = True,
        cache_ttl: int = 300,
        cache_similarity_threshold: float = 0.92,
        enable_document_limiting: bool = True,
        max_per_document: int = 3
    ):
        """
        Initialize hybrid search engine

        Args:
            keyword_engine: Keyword search engine
            semantic_engine: Semantic search engine (optional for keyword-only mode)
            default_mode: Default search mode (keyword, semantic, hybrid, rerank, or hyde)
            semantic_weight: Weight for semantic scores in hybrid mode (0-1)
            use_rrf: Use Reciprocal Rank Fusion instead of weighted average
            rerank_candidates: Number of candidates for reranking mode
            rerank_keyword_threshold: Minimum keyword score threshold for reranking
            enable_hyde: Enable HyDE search mode
            enable_semantic_cache: Enable semantic query caching
            enable_query_routing: Enable automatic query routing
            enable_parent_context: Enable parent document context enrichment
            cache_ttl: Cache time-to-live in seconds
            cache_similarity_threshold: Minimum similarity for semantic cache hit
            enable_document_limiting: Enable document-level result limiting
            max_per_document: Maximum chunks per document (default: 3)
        """
        self.keyword_engine = keyword_engine
        self.semantic_engine = semantic_engine
        self.default_mode = default_mode
        self.semantic_weight = semantic_weight
        self.use_rrf = use_rrf
        self.rerank_candidates = rerank_candidates
        self.rerank_keyword_threshold = rerank_keyword_threshold
        self.enable_hyde = enable_hyde
        self.enable_semantic_cache = enable_semantic_cache
        self.enable_query_routing = enable_query_routing
        self.enable_parent_context = enable_parent_context
        self.enable_document_limiting = enable_document_limiting
        self.max_per_document = max_per_document

        self.result_enhancer = ResultEnhancer(summary_length=150)

        # Validate mode
        valid_modes = ['keyword', 'semantic', 'hybrid', 'rerank', 'hyde', 'auto']
        if default_mode not in valid_modes:
            logger.warning(f"Invalid mode '{default_mode}', defaulting to 'hybrid'")
            self.default_mode = 'hybrid'

        # Validate semantic mode availability
        if default_mode in ['semantic', 'hybrid', 'rerank', 'hyde'] and not semantic_engine:
            logger.warning(f"Semantic engine not available, falling back to keyword mode")
            self.default_mode = 'keyword'

        # Initialize HyDE generator
        self.hyde_generator = None
        if enable_hyde and semantic_engine and hasattr(semantic_engine, 'embedding_generator') and semantic_engine.embedding_generator:
            try:
                from .hyde import HyDEGenerator
                self.hyde_generator = HyDEGenerator(
                    embedding_generator=semantic_engine.embedding_generator
                )
                logger.info("HyDE generator initialized")
            except ImportError as e:
                logger.warning(f"Failed to initialize HyDE: {e}")
        elif enable_hyde and semantic_engine:
            logger.warning("HyDE requested but semantic engine has no embedding_generator; HyDE disabled")

        # Initialize semantic cache
        self.semantic_cache = None
        if enable_semantic_cache and semantic_engine and hasattr(semantic_engine, 'embedding_generator') and semantic_engine.embedding_generator:
            try:
                from .semantic_cache import SemanticQueryCache
                self.semantic_cache = SemanticQueryCache(
                    embedding_generator=semantic_engine.embedding_generator,
                    max_size=1000,
                    ttl_seconds=cache_ttl,
                    similarity_threshold=cache_similarity_threshold
                )
                logger.info("Semantic query cache initialized")
            except ImportError as e:
                logger.warning(f"Failed to initialize semantic cache: {e}")
        elif enable_semantic_cache and semantic_engine:
            logger.warning("Semantic cache requested but semantic engine has no embedding_generator; cache disabled")

        # Initialize query router
        self.query_router = None
        if enable_query_routing:
            try:
                from .query_router import QueryRouter
                self.query_router = QueryRouter(
                    default_mode=default_mode if default_mode != 'auto' else 'hybrid',
                    enable_hyde=enable_hyde and self.hyde_generator is not None
                )
                logger.info("Query router initialized")
            except ImportError as e:
                logger.warning(f"Failed to initialize query router: {e}")

        # Extract underlying vector_db and indexer from semantic engine
        # Handles both SemanticSearchEngine and HierarchicalSearchEngine
        self._vector_db = None
        self._semantic_indexer = None
        if semantic_engine:
            # Check if it's a HierarchicalSearchEngine (has semantic_search attribute)
            if hasattr(semantic_engine, 'semantic_search') and semantic_engine.semantic_search:
                self._vector_db = getattr(semantic_engine.semantic_search, 'vector_db', None)
                self._semantic_indexer = getattr(semantic_engine.semantic_search, 'indexer', None)
            else:
                # Standard SemanticSearchEngine
                self._vector_db = getattr(semantic_engine, 'vector_db', None)
                self._semantic_indexer = getattr(semantic_engine, 'indexer', None)

        # Initialize parent context enricher
        self.parent_context_enricher = None
        if enable_parent_context and self._semantic_indexer:
            try:
                from .parent_context import ParentContextEnricher
                self.parent_context_enricher = ParentContextEnricher(
                    indexer=self._semantic_indexer
                )
                logger.info("Parent context enricher initialized")
            except ImportError as e:
                logger.warning(f"Failed to initialize parent context enricher: {e}")

        fusion_method = "RRF" if use_rrf else "weighted average"
        logger.info(f"Initialized HybridSearchEngine in '{self.default_mode}' mode ({fusion_method})")

    def search(
        self,
        query: str,
        product: Optional[str] = None,
        component: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        max_results: int = 10,
        mode: Optional[str] = None,
        include_parent_context: Optional[bool] = None,
        use_cache: bool = True,
        enhance_results: bool = True,
        include_full_metadata: bool = False,
        max_per_document: Optional[int] = None,
        # Email-specific filters (pre-filter before scoring)
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
        Search documents using configured mode

        Args:
            query: Search query
            product: Filter by product
            component: Filter by component
            file_types: Filter by file extensions
            max_results: Maximum results to return
            mode: Override default mode (keyword, semantic, hybrid, rerank, hyde, auto)
            include_parent_context: Override parent context setting
            use_cache: Whether to use semantic cache
            enhance_results: Whether to enhance results with metadata
            include_full_metadata: Whether to include full metadata in results
            max_per_document: Override max chunks per document (None = use default)
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
            List of matching documents with relevance scores
        """
        # Build filter dict for cache key
        filters = {}
        if product:
            filters['product'] = product
        if component:
            filters['component'] = component
        if file_types:
            filters['file_types'] = file_types
        # Include email filters in cache key
        if sender:
            filters['sender'] = sender
        if recipient:
            filters['recipient'] = recipient
        if cc:
            filters['cc'] = cc
        if folder:
            filters['folder'] = folder
        if subject_contains:
            filters['subject_contains'] = subject_contains
        if has_attachments is not None:
            filters['has_attachments'] = has_attachments
        if date_after:
            filters['date_after'] = date_after
        if date_before:
            filters['date_before'] = date_before
        if thread_id:
            filters['thread_id'] = thread_id

        # Build email filters dict for passing to internal methods
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

        # Determine search mode
        search_mode = mode or self.default_mode

        # Use query router for 'auto' mode or when enabled
        if search_mode == 'auto' and self.query_router:
            search_mode = self.query_router.route(query)
            logger.debug(f"Query router selected mode: {search_mode}")

        # Check semantic cache first
        if use_cache and self.semantic_cache and search_mode != 'keyword':
            cached_results = self.semantic_cache.get(query, search_mode, filters)
            if cached_results:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                # Still apply parent context if needed (cached results may not have it)
                if self._should_include_parent_context(include_parent_context):
                    cached_results = self._enrich_with_parent_context(cached_results)
                results = cached_results[:max_results]
                if enhance_results:
                    results = self.result_enhancer.enhance(
                        results,
                        include_full_metadata=include_full_metadata
                    )
                return results

        # Determine document limiting settings
        doc_limit = max_per_document if max_per_document is not None else self.max_per_document
        apply_limiting = self.enable_document_limiting and doc_limit > 0

        # Calculate how many candidates to fetch (need more if limiting will be applied)
        candidate_multiplier = 2 if apply_limiting else 1
        candidate_count = max_results * candidate_multiplier

        # Route to appropriate search method
        if search_mode == 'keyword':
            results = self._keyword_search(query, product, component, file_types, candidate_count, email_filters)
        elif search_mode == 'semantic':
            results = self._semantic_search(query, product, component, file_types, candidate_count, email_filters)
        elif search_mode == 'hybrid':
            results = self._hybrid_search(query, product, component, file_types, candidate_count, email_filters)
        elif search_mode == 'rerank':
            results = self._rerank_search(query, product, component, file_types, candidate_count, email_filters)
        elif search_mode == 'hyde':
            results = self._hyde_search(query, product, component, file_types, candidate_count, email_filters)
        else:
            logger.warning(f"Unknown mode '{search_mode}', using hybrid search")
            results = self._hybrid_search(query, product, component, file_types, candidate_count, email_filters)

        # Apply document-level limiting if enabled
        if apply_limiting:
            results = self._limit_results_per_document(results, doc_limit)

        # Truncate to max_results after limiting
        results = results[:max_results]

        # Cache results
        if use_cache and self.semantic_cache and search_mode != 'keyword' and results:
            self.semantic_cache.set(query, results, search_mode, filters)

        # Enrich with parent context
        if self._should_include_parent_context(include_parent_context):
            results = self._enrich_with_parent_context(results)

        if enhance_results:
            results = self.result_enhancer.enhance(
                results,
                include_full_metadata=include_full_metadata
            )

        return results

    def _should_include_parent_context(self, override: Optional[bool]) -> bool:
        """Determine whether to include parent context"""
        if override is not None:
            return override
        return self.enable_parent_context and self.parent_context_enricher is not None

    def _enrich_with_parent_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add parent document context to results"""
        if not self.parent_context_enricher:
            return results

        try:
            return self.parent_context_enricher.enrich_results(results)
        except Exception as e:
            logger.warning(f"Failed to enrich with parent context: {e}")
            return results

    def _keyword_search(
        self,
        query: str,
        product: Optional[str],
        component: Optional[str],
        file_types: Optional[List[str]],
        max_results: int,
        email_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute keyword-only search"""
        logger.debug(f"Executing keyword search: {query}")
        
        # Extract email filter params (default to None if not provided)
        ef = email_filters or {}
        
        results = self.keyword_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=max_results,
            sender=ef.get('sender'),
            recipient=ef.get('recipient'),
            cc=ef.get('cc'),
            folder=ef.get('folder'),
            subject_contains=ef.get('subject_contains'),
            has_attachments=ef.get('has_attachments'),
            date_after=ef.get('date_after'),
            date_before=ef.get('date_before'),
            thread_id=ef.get('thread_id')
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
        max_results: int,
        email_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute semantic-only search"""
        if not self.semantic_engine:
            logger.warning("Semantic engine not available, falling back to keyword")
            return self._keyword_search(query, product, component, file_types, max_results, email_filters)

        logger.debug(f"Executing semantic search: {query}")
        
        # Extract email filter params (default to None if not provided)
        ef = email_filters or {}
        
        results = self.semantic_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=max_results,
            sender=ef.get('sender'),
            recipient=ef.get('recipient'),
            cc=ef.get('cc'),
            folder=ef.get('folder'),
            subject_contains=ef.get('subject_contains'),
            has_attachments=ef.get('has_attachments'),
            date_after=ef.get('date_after'),
            date_before=ef.get('date_before'),
            thread_id=ef.get('thread_id')
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
        max_results: int,
        email_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search combining keyword and semantic results

        Uses either RRF or weighted average based on configuration
        """
        if not self.semantic_engine:
            logger.warning("Semantic engine not available, falling back to keyword")
            return self._keyword_search(query, product, component, file_types, max_results, email_filters)

        logger.debug(f"Executing hybrid search: {query}")

        # Extract email filter params (default to None if not provided)
        ef = email_filters or {}

        # Get results from both engines
        # Fetch proportionally more results for better fusion
        candidate_multiplier = 3 if self.use_rrf else 2

        keyword_results = self.keyword_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=max_results * candidate_multiplier,
            sender=ef.get('sender'),
            recipient=ef.get('recipient'),
            cc=ef.get('cc'),
            folder=ef.get('folder'),
            subject_contains=ef.get('subject_contains'),
            has_attachments=ef.get('has_attachments'),
            date_after=ef.get('date_after'),
            date_before=ef.get('date_before'),
            thread_id=ef.get('thread_id')
        )

        semantic_results = self.semantic_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=max_results * candidate_multiplier,
            sender=ef.get('sender'),
            recipient=ef.get('recipient'),
            cc=ef.get('cc'),
            folder=ef.get('folder'),
            subject_contains=ef.get('subject_contains'),
            has_attachments=ef.get('has_attachments'),
            date_after=ef.get('date_after'),
            date_before=ef.get('date_before'),
            thread_id=ef.get('thread_id')
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
                    'content_preview': doc_data.get('content_preview', ''),
                    'relevance_score': round(hybrid_score, 2),
                    'keyword_score': round(keyword_score, 2),
                    'semantic_score': round(semantic_score, 2),
                    'search_mode': 'hybrid',
                    'last_modified': doc_data['last_modified'],
                    'metadata': doc_data.get('metadata', {}),
                    'headings': doc_data.get('headings', []),
                    'doc_type': doc_data.get('doc_type'),
                    'tags': doc_data.get('tags', [])
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

    def _rerank_search(
        self,
        query: str,
        product: Optional[str],
        component: Optional[str],
        file_types: Optional[List[str]],
        max_results: int,
        email_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Two-stage reranking search:
        1. Semantic search for broad recall (N candidates)
        2. Keyword scoring for precision filtering

        This filters out semantically similar but contextually irrelevant documents.
        """
        if not self.semantic_engine:
            logger.warning("Semantic engine not available, falling back to keyword")
            return self._keyword_search(query, product, component, file_types, max_results, email_filters)

        logger.debug(f"Executing rerank search: {query}")

        # Extract email filter params (default to None if not provided)
        ef = email_filters or {}

        # Stage 1: Semantic search (broad recall)
        semantic_results = self.semantic_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=self.rerank_candidates,
            sender=ef.get('sender'),
            recipient=ef.get('recipient'),
            cc=ef.get('cc'),
            folder=ef.get('folder'),
            subject_contains=ef.get('subject_contains'),
            has_attachments=ef.get('has_attachments'),
            date_after=ef.get('date_after'),
            date_before=ef.get('date_before'),
            thread_id=ef.get('thread_id')
        )

        # Stage 2: Keyword reranking
        reranked = []
        keywords = query.lower().split()

        for result in semantic_results:
            # Calculate keyword score for this document
            keyword_score = self._calculate_keyword_score(
                keywords=keywords,
                content=result.get('snippet', ''),
                filename=result.get('file_name', '')
            )

            # Filter out results with no keyword matches
            if keyword_score < self.rerank_keyword_threshold:
                logger.debug(f"Filtered out {result['file_name']} (keyword_score={keyword_score:.2f})")
                continue

            # Combine scores (70% semantic, 30% keyword)
            semantic_score = result.get('similarity_score', 0.0)
            combined_score = 0.7 * semantic_score + 0.3 * keyword_score

            # Create reranked result
            reranked.append({
                'id': result['id'],
                'file_path': result['file_path'],
                'product': result['product'],
                'component': result['component'],
                'file_name': result['file_name'],
                'file_type': result['file_type'],
                'snippet': result['snippet'],
                'content_preview': result.get('content_preview', ''),
                'relevance_score': round(combined_score, 2),
                'semantic_score': round(semantic_score, 2),
                'keyword_score': round(keyword_score, 2),
                'search_mode': 'rerank',
                'last_modified': result['last_modified'],
                'metadata': result.get('metadata', {}),
                'headings': result.get('headings', []),
                'doc_type': result.get('doc_type'),
                'tags': result.get('tags', [])
            })

        # Sort by combined score
        reranked.sort(key=lambda x: x['relevance_score'], reverse=True)

        logger.debug(f"Reranking: {len(semantic_results)} candidates → {len(reranked)} filtered results")

        # Return top N results
        return reranked[:max_results]

    def _hyde_search(
        self,
        query: str,
        product: Optional[str],
        component: Optional[str],
        file_types: Optional[List[str]],
        max_results: int,
        email_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        HyDE (Hypothetical Document Embeddings) search:

        1. Generate a hypothetical document that would answer the query
        2. Use the hypothetical document's embedding to find similar documents
        3. Return documents similar to the hypothetical answer

        This bridges the semantic gap between short queries and document content.
        """
        if not self.hyde_generator or not self.semantic_engine or not self._vector_db:
            logger.warning("HyDE not available, falling back to semantic search")
            return self._semantic_search(query, product, component, file_types, max_results, email_filters)

        logger.debug(f"Executing HyDE search: {query}")

        # Extract email filter params (default to None if not provided)
        ef = email_filters or {}

        try:
            # Generate HyDE embedding (combines query with hypothetical document)
            hyde_embedding = self.hyde_generator.generate_hyde_embedding(query)

            # Build metadata filter (ChromaDB requires $and for multiple conditions)
            conditions = []
            if product:
                conditions.append({'product': product})
            if component:
                conditions.append({'component': component})
            if file_types:
                conditions.append({'file_type': {"$in": file_types}})

            # Add email-specific filters (exact match via ChromaDB WHERE clause)
            if ef.get('folder'):
                conditions.append({'email_folder': ef['folder'].lower()})
            if ef.get('has_attachments') is not None:
                conditions.append({'email_has_attachments': ef['has_attachments']})
            if ef.get('thread_id'):
                conditions.append({'email_thread_id': ef['thread_id']})
            if ef.get('date_after'):
                conditions.append({'email_date': {'$gte': ef['date_after']}})
            if ef.get('date_before'):
                conditions.append({'email_date': {'$lte': ef['date_before']}})

            # Build proper where filter
            if not conditions:
                where_filter = None
            elif len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}

            # Search vector database with HyDE embedding
            vector_results = self._vector_db.search(
                query_embedding=hyde_embedding,
                n_results=max_results * 2,  # Get more candidates
                where=where_filter
            )

            # Enrich results with document metadata
            enriched_results = []
            for result in vector_results:
                doc_id = result['id']
                # Support chunked document IDs (e.g., 'path#chunk0') by using base doc id
                base_doc_id = doc_id.split('#', 1)[0]
                doc = self._semantic_indexer.index.documents.get(base_doc_id) if self._semantic_indexer else None
                
                # Fallback: try full doc_id if base lookup failed
                if not doc and self._semantic_indexer:
                    doc = self._semantic_indexer.index.documents.get(doc_id)

                if doc:
                    # Apply partial-match email filters (not handled by ChromaDB WHERE)
                    if ef and not self._matches_partial_email_filters(doc, ef):
                        continue

                    snippet = self._extract_snippet(doc['content'], query)

                    enriched_results.append({
                        'id': base_doc_id,
                        'file_path': base_doc_id,
                        'product': doc['product'],
                        'component': doc['component'],
                        'file_name': doc['file_name'],
                        'file_type': doc['file_type'],
                        'snippet': snippet,
                        'content_preview': doc.get('content', '')[:500],
                        'relevance_score': round(result['similarity'], 2),
                        'similarity_score': round(result['similarity'], 2),
                        'search_mode': 'hyde',
                        'last_modified': doc['last_modified'],
                        'metadata': doc.get('metadata', {}),
                        'headings': doc.get('headings', []),
                        'doc_type': doc.get('doc_type'),
                        'tags': doc.get('tags', [])
                    })

            logger.debug(f"HyDE search returned {len(enriched_results)} results")
            return enriched_results[:max_results]

        except Exception as e:
            logger.error(f"HyDE search failed: {e}, falling back to semantic")
            return self._semantic_search(query, product, component, file_types, max_results, email_filters)


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
        if filters.get('sender'):
            sender_val = metadata.get('from', '')
            if not sender_val or filters['sender'].lower() not in sender_val.lower():
                return False

        # Recipient filter (partial match in 'to' list)
        if filters.get('recipient'):
            to_list = metadata.get('to', [])
            if isinstance(to_list, str):
                to_list = [to_list]
            recipient_lower = filters['recipient'].lower()
            found = any(recipient_lower in addr.lower() for addr in to_list if addr)
            if not found:
                return False

        # CC filter (partial match in 'cc' list)
        if filters.get('cc'):
            cc_list = metadata.get('cc', [])
            if isinstance(cc_list, str):
                cc_list = [cc_list]
            cc_lower = filters['cc'].lower()
            found = any(cc_lower in addr.lower() for addr in cc_list if addr)
            if not found:
                return False

        # Subject filter (partial match, case-insensitive)
        if filters.get('subject_contains'):
            subject_val = metadata.get('subject', '')
            if not subject_val or filters['subject_contains'].lower() not in subject_val.lower():
                return False

        return True

    def _extract_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """Extract relevant snippet from content"""
        lines = content.split('\n')
        query_lower = query.lower()
        query_words = query_lower.split()

        # Find line most similar to query
        best_line = ""
        max_matches = 0

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

        snippet = best_line.strip()
        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "..."

        return snippet

    def _limit_results_per_document(
        self,
        results: List[Dict[str, Any]],
        max_per_document: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Limit number of results per document to prevent single-source domination

        This provides diversity across documents while preserving precision.
        Unlike MMR, this simple approach doesn't sacrifice relevance for variety.

        Args:
            results: Search results (already sorted by relevance)
            max_per_document: Maximum chunks to return from each document (default: 3)

        Returns:
            Filtered results with diversity across documents
        """
        if max_per_document <= 0:
            # No limiting, return all results
            return results

        doc_counts = {}
        filtered = []

        for result in results:
            # Get parent document ID (handles both chunked and non-chunked results)
            doc_id = result.get('parent_doc', result.get('file_path', ''))

            # If no parent_doc, try to extract from chunk_id
            if not doc_id and 'id' in result:
                chunk_id = result['id']
                if '#chunk_' in chunk_id:
                    doc_id = chunk_id.split('#chunk_')[0]
                else:
                    doc_id = chunk_id

            # Count chunks from this document
            count = doc_counts.get(doc_id, 0)

            if count < max_per_document:
                filtered.append(result)
                doc_counts[doc_id] = count + 1
            else:
                # Skip this result (already have enough from this document)
                logger.debug(
                    f"Skipping {result.get('file_name', 'unknown')} "
                    f"(document limit reached: {max_per_document})"
                )

        # Log summary if filtering occurred
        filtered_count = len(results) - len(filtered)
        if filtered_count > 0:
            logger.info(
                f"Document limiting: {len(filtered)}/{len(results)} results kept "
                f"({filtered_count} filtered, max_per_document={max_per_document})"
            )

        return filtered

    def _calculate_keyword_score(
        self,
        keywords: List[str],
        content: str,
        filename: str
    ) -> float:
        """
        Calculate keyword relevance score for a document

        Args:
            keywords: Query keywords (already lowercased)
            content: Document content/snippet
            filename: Document filename

        Returns:
            Normalized keyword score (0-1)
        """
        content_lower = content.lower()
        filename_lower = filename.lower()

        score = 0
        max_score = 0

        for keyword in keywords:
            if len(keyword) < 2:
                continue

            max_score += 5  # Maximum possible score per keyword

            # Filename match: +3 points
            if keyword in filename_lower:
                score += 3

            # Content match: +1 per occurrence (capped at 5)
            count = min(content_lower.count(keyword), 5)
            score += count

        # Normalize by max possible score
        return score / max_score if max_score > 0 else 0.0

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
            mode: Search mode (keyword, semantic, hybrid, rerank, hyde, or auto)
        """
        valid_modes = ['keyword', 'semantic', 'hybrid', 'rerank', 'hyde', 'auto']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: {valid_modes}")

        if mode in ['semantic', 'hybrid', 'rerank'] and not self.semantic_engine:
            raise ValueError(f"Cannot set mode to '{mode}': semantic engine not available")

        if mode == 'hyde' and not self.hyde_generator:
            raise ValueError("Cannot set mode to 'hyde': HyDE generator not available")

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
            'semantic_engine': 'available' if self.semantic_engine else 'not available',
            'enhanced_features': {
                'hyde': 'available' if self.hyde_generator else 'not available',
                'semantic_cache': 'available' if self.semantic_cache else 'not available',
                'query_routing': 'available' if self.query_router else 'not available',
                'parent_context': 'available' if self.parent_context_enricher else 'not available'
            }
        }

        if self.semantic_engine:
            stats['semantic_stats'] = self.semantic_engine.get_stats()

        if self.semantic_cache:
            stats['cache_stats'] = self.semantic_cache.get_stats()

        return stats

    def clear_cache(self):
        """Clear the semantic query cache"""
        if self.semantic_cache:
            self.semantic_cache.clear()
            logger.info("Query cache cleared")

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query and return routing information

        Args:
            query: Search query to analyze

        Returns:
            Query analysis with recommended mode
        """
        if not self.query_router:
            return {'error': 'Query router not available'}

        analysis = self.query_router.analyze(query)
        return {
            'query': analysis.query,
            'word_count': analysis.word_count,
            'has_exact_terms': analysis.has_exact_terms,
            'has_technical_terms': analysis.has_technical_terms,
            'is_question': analysis.is_question,
            'is_conceptual': analysis.is_conceptual,
            'query_type': analysis.query_type,
            'complexity_score': round(analysis.complexity_score, 2),
            'recommended_mode': analysis.recommended_mode.value,
            'confidence': round(analysis.confidence, 2),
            'reasoning': analysis.reasoning
        }
