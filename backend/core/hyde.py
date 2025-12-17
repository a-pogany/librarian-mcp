"""
HyDE (Hypothetical Document Embeddings) for improved retrieval

HyDE generates a hypothetical answer to the query, then uses that answer's
embedding to search for similar documents. This bridges the semantic gap
between short queries and longer document content.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class HyDEGenerator:
    """
    Generate hypothetical documents for improved retrieval

    Since we don't want to require an external LLM API, this implementation
    uses template-based expansion and query analysis techniques to create
    hypothetical document content.
    """

    def __init__(
        self,
        embedding_generator=None,
        use_templates: bool = True,
        expansion_strategies: List[str] = None
    ):
        """
        Initialize HyDE generator

        Args:
            embedding_generator: EmbeddingGenerator instance for encoding
            use_templates: Use template-based expansion
            expansion_strategies: List of strategies to use
        """
        self.embedding_generator = embedding_generator
        self.use_templates = use_templates
        self.expansion_strategies = expansion_strategies or [
            'question_to_statement',
            'add_context_words',
            'technical_expansion'
        ]

        # Domain-specific templates for documentation
        self.templates = {
            'how_to': (
                "This document explains how to {topic}. "
                "The following steps describe the process for {topic}. "
                "To accomplish {topic}, you need to follow these instructions."
            ),
            'what_is': (
                "This document defines and explains {topic}. "
                "{topic} is a concept that refers to the following. "
                "The definition of {topic} includes these key aspects."
            ),
            'api_reference': (
                "API documentation for {topic}. "
                "This reference describes the {topic} endpoint, method, or function. "
                "Parameters and return values for {topic} are documented here."
            ),
            'troubleshooting': (
                "Troubleshooting guide for {topic}. "
                "Common issues and solutions related to {topic}. "
                "If you encounter problems with {topic}, try these solutions."
            ),
            'configuration': (
                "Configuration guide for {topic}. "
                "Settings and options for {topic} are explained here. "
                "To configure {topic}, use the following parameters."
            ),
            'general': (
                "Documentation about {topic}. "
                "This section covers {topic} in detail. "
                "Information and guidance regarding {topic}."
            )
        }

        # Keywords that indicate query type
        self.query_indicators = {
            'how_to': ['how to', 'how do i', 'how can i', 'steps to', 'guide to', 'tutorial'],
            'what_is': ['what is', 'what are', 'define', 'definition', 'meaning of', 'explain'],
            'api_reference': ['api', 'endpoint', 'method', 'function', 'parameter', 'request', 'response'],
            'troubleshooting': ['error', 'issue', 'problem', 'fix', 'solve', 'debug', 'not working', 'fails'],
            'configuration': ['config', 'configure', 'setting', 'setup', 'install', 'option']
        }

        logger.info("HyDE generator initialized")

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query

        Args:
            query: User's search query

        Returns:
            Hypothetical document content
        """
        # Detect query type
        query_type = self._detect_query_type(query)

        # Extract topic from query
        topic = self._extract_topic(query)

        # Generate hypothetical content using template
        template = self.templates.get(query_type, self.templates['general'])
        hypothetical = template.format(topic=topic)

        # Add expanded terms
        expanded_terms = self._expand_query_terms(query)
        if expanded_terms:
            hypothetical += f" Related terms: {', '.join(expanded_terms)}."

        logger.debug(f"Generated hypothetical document for query type '{query_type}'")
        return hypothetical

    def generate_hyde_embedding(self, query: str) -> np.ndarray:
        """
        Generate HyDE embedding by combining query and hypothetical document

        Args:
            query: User's search query

        Returns:
            Combined embedding vector
        """
        if not self.embedding_generator:
            raise RuntimeError("Embedding generator not configured")

        # Generate hypothetical document
        hypothetical = self.generate_hypothetical_document(query)

        # Encode both query and hypothetical
        query_embedding = self.embedding_generator.encode_query(query)
        hyp_embedding = self.embedding_generator.encode_document(hypothetical)

        # Combine embeddings (weighted average: 40% query, 60% hypothetical)
        # The hypothetical document gets more weight as it contains expanded context
        combined = 0.4 * query_embedding + 0.6 * hyp_embedding

        # Normalize the combined embedding
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined

    def generate_multi_hyde_embeddings(self, query: str, num_hypotheticals: int = 3) -> List[np.ndarray]:
        """
        Generate multiple hypothetical embeddings for diverse retrieval

        Args:
            query: User's search query
            num_hypotheticals: Number of hypothetical documents to generate

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Generate hypotheticals with different templates
        query_types = list(self.templates.keys())
        detected_type = self._detect_query_type(query)

        # Always include the detected type first
        types_to_use = [detected_type]

        # Add other relevant types
        for qt in query_types:
            if qt != detected_type and len(types_to_use) < num_hypotheticals:
                types_to_use.append(qt)

        for query_type in types_to_use[:num_hypotheticals]:
            topic = self._extract_topic(query)
            template = self.templates[query_type]
            hypothetical = template.format(topic=topic)

            if self.embedding_generator:
                hyp_embedding = self.embedding_generator.encode_document(hypothetical)
                embeddings.append(hyp_embedding)

        return embeddings

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query based on keywords"""
        query_lower = query.lower()

        for query_type, indicators in self.query_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    return query_type

        return 'general'

    def _extract_topic(self, query: str) -> str:
        """Extract the main topic from the query"""
        # Remove common question words
        stop_phrases = [
            'how to', 'how do i', 'how can i', 'what is', 'what are',
            'where is', 'where can i find', 'why does', 'when should i',
            'can i', 'could you', 'please', 'help me', 'i want to',
            'i need to', 'tell me about', 'explain', 'describe'
        ]

        topic = query.lower()
        for phrase in stop_phrases:
            topic = topic.replace(phrase, '')

        # Clean up
        topic = ' '.join(topic.split())  # Normalize whitespace
        topic = topic.strip('?.,!')

        # If topic is too short, use original query
        if len(topic) < 3:
            topic = query

        return topic

    def _expand_query_terms(self, query: str) -> List[str]:
        """Expand query with related technical terms"""
        expansions = []
        query_lower = query.lower()

        # Technical term mappings
        term_expansions = {
            'api': ['endpoint', 'REST', 'HTTP', 'request', 'response'],
            'auth': ['authentication', 'authorization', 'login', 'token', 'OAuth'],
            'config': ['configuration', 'settings', 'options', 'parameters'],
            'db': ['database', 'SQL', 'query', 'schema', 'table'],
            'error': ['exception', 'failure', 'bug', 'issue', 'troubleshoot'],
            'install': ['setup', 'deployment', 'installation', 'configure'],
            'test': ['testing', 'unit test', 'integration', 'assertion'],
            'deploy': ['deployment', 'release', 'production', 'CI/CD'],
            'search': ['query', 'find', 'lookup', 'retrieval', 'filter'],
            'cache': ['caching', 'memory', 'performance', 'optimization'],
        }

        for term, related in term_expansions.items():
            if term in query_lower:
                expansions.extend(related[:3])  # Limit expansions per term

        return list(set(expansions))[:5]  # Dedupe and limit total


class HyDERetriever:
    """
    Retriever that uses HyDE for improved search results
    """

    def __init__(
        self,
        hyde_generator: HyDEGenerator,
        vector_db,
        use_multi_hyde: bool = False
    ):
        """
        Initialize HyDE retriever

        Args:
            hyde_generator: HyDEGenerator instance
            vector_db: Vector database for similarity search
            use_multi_hyde: Use multiple hypothetical documents
        """
        self.hyde_generator = hyde_generator
        self.vector_db = vector_db
        self.use_multi_hyde = use_multi_hyde

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using HyDE-enhanced retrieval

        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filter

        Returns:
            Search results with similarity scores
        """
        if self.use_multi_hyde:
            return self._multi_hyde_search(query, n_results, where)
        else:
            return self._single_hyde_search(query, n_results, where)

    def _single_hyde_search(
        self,
        query: str,
        n_results: int,
        where: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Search with single HyDE embedding"""
        # Generate HyDE embedding
        hyde_embedding = self.hyde_generator.generate_hyde_embedding(query)

        # Search vector database
        results = self.vector_db.search(
            query_embedding=hyde_embedding,
            n_results=n_results,
            where=where
        )

        # Mark results as HyDE-enhanced
        for result in results:
            result['search_method'] = 'hyde'

        return results

    def _multi_hyde_search(
        self,
        query: str,
        n_results: int,
        where: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Search with multiple HyDE embeddings and merge results"""
        # Generate multiple hypothetical embeddings
        hyde_embeddings = self.hyde_generator.generate_multi_hyde_embeddings(query, num_hypotheticals=3)

        # Search with each embedding
        all_results = {}
        for i, embedding in enumerate(hyde_embeddings):
            results = self.vector_db.search(
                query_embedding=embedding,
                n_results=n_results * 2,  # Get more candidates
                where=where
            )

            # Accumulate scores using RRF-like fusion
            for rank, result in enumerate(results):
                doc_id = result['id']
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        'result': result,
                        'score': 0.0
                    }
                # RRF score accumulation
                all_results[doc_id]['score'] += 1.0 / (60 + rank)

        # Sort by accumulated score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        # Format results
        final_results = []
        for item in sorted_results[:n_results]:
            result = item['result']
            result['similarity'] = item['score']
            result['search_method'] = 'multi_hyde'
            final_results.append(result)

        return final_results
