"""
Query Router for intelligent search mode selection

Analyzes query characteristics to automatically route to the optimal
search mode, balancing speed and quality.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Available search modes"""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    RERANK = "rerank"
    HYDE = "hyde"


@dataclass
class QueryAnalysis:
    """Results of query analysis"""
    query: str
    word_count: int
    has_exact_terms: bool
    has_technical_terms: bool
    is_question: bool
    is_conceptual: bool
    query_type: str  # factual, conceptual, navigational, troubleshooting
    complexity_score: float  # 0.0 (simple) to 1.0 (complex)
    recommended_mode: SearchMode
    confidence: float
    reasoning: str


class QueryRouter:
    """
    Intelligent query routing based on query analysis

    Routes queries to the optimal search mode:
    - Simple keyword queries -> keyword mode (fast)
    - Technical exact-match queries -> hybrid mode
    - Conceptual/vague queries -> semantic or HyDE mode
    - Complex multi-faceted queries -> rerank mode
    """

    def __init__(
        self,
        default_mode: str = "hybrid",
        enable_hyde: bool = True,
        complexity_threshold_low: float = 0.3,
        complexity_threshold_high: float = 0.7
    ):
        """
        Initialize query router

        Args:
            default_mode: Fallback mode when uncertain
            enable_hyde: Allow routing to HyDE mode
            complexity_threshold_low: Below this -> keyword mode
            complexity_threshold_high: Above this -> rerank/hyde mode
        """
        self.default_mode = SearchMode(default_mode)
        self.enable_hyde = enable_hyde
        self.complexity_threshold_low = complexity_threshold_low
        self.complexity_threshold_high = complexity_threshold_high

        # Technical terms that benefit from exact matching
        self.technical_terms = {
            'api', 'sdk', 'cli', 'http', 'https', 'rest', 'graphql',
            'json', 'xml', 'yaml', 'sql', 'nosql', 'redis', 'mongodb',
            'docker', 'kubernetes', 'k8s', 'aws', 'gcp', 'azure',
            'oauth', 'jwt', 'ssl', 'tls', 'ssh', 'vpc', 'cdn',
            'cpu', 'gpu', 'ram', 'ssd', 'ip', 'dns', 'url', 'uri',
            'get', 'post', 'put', 'delete', 'patch', 'crud',
            'npm', 'pip', 'yarn', 'maven', 'gradle', 'cargo',
            'git', 'github', 'gitlab', 'svn', 'ci', 'cd',
            'async', 'await', 'promise', 'callback', 'mutex', 'thread',
            'config', 'env', 'param', 'arg', 'flag', 'option',
            'v1', 'v2', 'v3', 'beta', 'alpha', 'stable', 'latest'
        }

        # Patterns indicating exact-match needs
        self.exact_match_patterns = [
            r'\b\d+\.\d+\.\d+\b',  # Version numbers (1.2.3)
            r'\b[A-Z]{2,}\b',  # Acronyms (API, SDK)
            r'`[^`]+`',  # Code in backticks
            r'"[^"]+"',  # Quoted terms
            r'\b\w+\.\w+\b',  # File extensions or dotted terms
            r'\b[A-Z][a-z]+[A-Z]\w*\b',  # CamelCase
            r'\b[a-z]+_[a-z]+\b',  # snake_case
            r'\b[a-z]+-[a-z]+\b',  # kebab-case
        ]

        # Question indicators
        self.question_words = {'how', 'what', 'why', 'when', 'where', 'which', 'who', 'can', 'does', 'is', 'are'}

        # Conceptual query indicators
        self.conceptual_indicators = [
            'best practice', 'recommend', 'should i', 'better',
            'difference between', 'compare', 'vs', 'versus',
            'overview', 'introduction', 'getting started',
            'understand', 'explain', 'concept', 'theory',
            'approach', 'strategy', 'pattern', 'architecture'
        ]

        # Troubleshooting indicators
        self.troubleshooting_indicators = [
            'error', 'issue', 'problem', 'bug', 'fix', 'solve',
            'not working', 'fails', 'broken', 'crash', 'exception',
            'timeout', 'slow', 'hang', 'freeze', 'stuck'
        ]

        logger.info(f"Query router initialized with default mode: {default_mode}")

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a query and determine optimal search mode

        Args:
            query: User's search query

        Returns:
            QueryAnalysis with recommended mode and reasoning
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        word_count = len(words)

        # Check for exact-match patterns
        has_exact_terms = self._has_exact_match_needs(query)

        # Check for technical terms
        has_technical_terms = any(word in self.technical_terms for word in words)

        # Check if it's a question
        is_question = (
            query.endswith('?') or
            (words and words[0] in self.question_words)
        )

        # Check if conceptual
        is_conceptual = any(ind in query_lower for ind in self.conceptual_indicators)

        # Determine query type
        query_type = self._determine_query_type(query_lower)

        # Calculate complexity score
        complexity_score = self._calculate_complexity(
            query=query,
            word_count=word_count,
            has_exact_terms=has_exact_terms,
            has_technical_terms=has_technical_terms,
            is_question=is_question,
            is_conceptual=is_conceptual,
            query_type=query_type
        )

        # Determine recommended mode
        recommended_mode, confidence, reasoning = self._determine_mode(
            word_count=word_count,
            has_exact_terms=has_exact_terms,
            has_technical_terms=has_technical_terms,
            is_question=is_question,
            is_conceptual=is_conceptual,
            query_type=query_type,
            complexity_score=complexity_score
        )

        return QueryAnalysis(
            query=query,
            word_count=word_count,
            has_exact_terms=has_exact_terms,
            has_technical_terms=has_technical_terms,
            is_question=is_question,
            is_conceptual=is_conceptual,
            query_type=query_type,
            complexity_score=complexity_score,
            recommended_mode=recommended_mode,
            confidence=confidence,
            reasoning=reasoning
        )

    def route(self, query: str) -> str:
        """
        Get the recommended search mode for a query

        Args:
            query: User's search query

        Returns:
            Search mode string (keyword, semantic, hybrid, rerank, hyde)
        """
        analysis = self.analyze(query)
        logger.debug(
            f"Query routing: '{query[:50]}...' -> {analysis.recommended_mode.value} "
            f"(confidence: {analysis.confidence:.2f}, reason: {analysis.reasoning})"
        )
        return analysis.recommended_mode.value

    def _has_exact_match_needs(self, query: str) -> bool:
        """Check if query needs exact matching"""
        for pattern in self.exact_match_patterns:
            if re.search(pattern, query):
                return True
        return False

    def _determine_query_type(self, query_lower: str) -> str:
        """Determine the type of query"""
        # Check for troubleshooting
        if any(ind in query_lower for ind in self.troubleshooting_indicators):
            return 'troubleshooting'

        # Check for conceptual
        if any(ind in query_lower for ind in self.conceptual_indicators):
            return 'conceptual'

        # Check for navigational (looking for specific doc)
        if any(word in query_lower for word in ['where', 'find', 'locate', 'path', 'file']):
            return 'navigational'

        # Default to factual
        return 'factual'

    def _calculate_complexity(
        self,
        query: str,
        word_count: int,
        has_exact_terms: bool,
        has_technical_terms: bool,
        is_question: bool,
        is_conceptual: bool,
        query_type: str
    ) -> float:
        """
        Calculate query complexity score (0.0 to 1.0)

        Higher score = more complex, needs sophisticated search
        """
        score = 0.0

        # Word count contributes to complexity
        if word_count <= 2:
            score += 0.1
        elif word_count <= 5:
            score += 0.3
        elif word_count <= 10:
            score += 0.5
        else:
            score += 0.7

        # Conceptual queries are more complex
        if is_conceptual:
            score += 0.2

        # Questions often need semantic understanding
        if is_question:
            score += 0.1

        # Technical terms with exact needs reduce complexity (exact match is simpler)
        if has_exact_terms and has_technical_terms:
            score -= 0.2

        # Query type adjustments
        if query_type == 'troubleshooting':
            score += 0.15  # Need broader search
        elif query_type == 'conceptual':
            score += 0.2
        elif query_type == 'navigational':
            score -= 0.1  # Usually straightforward

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _determine_mode(
        self,
        word_count: int,
        has_exact_terms: bool,
        has_technical_terms: bool,
        is_question: bool,
        is_conceptual: bool,
        query_type: str,
        complexity_score: float
    ) -> tuple:
        """
        Determine the optimal search mode

        Returns:
            Tuple of (SearchMode, confidence, reasoning)
        """
        # Very short, specific queries -> keyword
        if word_count <= 2 and has_technical_terms and not is_question:
            return (
                SearchMode.KEYWORD,
                0.85,
                "Short technical query benefits from exact keyword matching"
            )

        # Queries with exact-match needs -> hybrid (combines exact + semantic)
        if has_exact_terms:
            return (
                SearchMode.HYBRID,
                0.80,
                "Query contains terms requiring exact matching (versions, code, etc.)"
            )

        # Low complexity -> keyword or hybrid
        if complexity_score < self.complexity_threshold_low:
            if has_technical_terms:
                return (
                    SearchMode.HYBRID,
                    0.75,
                    "Low complexity with technical terms"
                )
            return (
                SearchMode.KEYWORD,
                0.70,
                "Simple query suitable for keyword search"
            )

        # High complexity -> rerank or HyDE
        if complexity_score > self.complexity_threshold_high:
            if is_conceptual and self.enable_hyde:
                return (
                    SearchMode.HYDE,
                    0.80,
                    "Conceptual query benefits from hypothetical document expansion"
                )
            return (
                SearchMode.RERANK,
                0.75,
                "Complex query benefits from two-stage reranking"
            )

        # Conceptual queries -> semantic or HyDE
        if is_conceptual:
            if self.enable_hyde:
                return (
                    SearchMode.HYDE,
                    0.75,
                    "Conceptual query benefits from HyDE"
                )
            return (
                SearchMode.SEMANTIC,
                0.70,
                "Conceptual query benefits from semantic understanding"
            )

        # Troubleshooting -> hybrid with rerank potential
        if query_type == 'troubleshooting':
            return (
                SearchMode.HYBRID,
                0.75,
                "Troubleshooting query needs both keyword precision and semantic breadth"
            )

        # Default to hybrid for medium complexity
        return (
            SearchMode.HYBRID,
            0.65,
            "Default hybrid mode for balanced search"
        )

    def get_mode_description(self, mode: str) -> str:
        """Get human-readable description of a search mode"""
        descriptions = {
            'keyword': "Fast exact keyword matching, best for specific technical terms",
            'semantic': "Vector similarity search, understands conceptual meaning",
            'hybrid': "Combines keyword and semantic search with rank fusion",
            'rerank': "Two-stage search with cross-encoder reranking for precision",
            'hyde': "Generates hypothetical answer before search for better semantic matching"
        }
        return descriptions.get(mode, "Unknown search mode")


class AdaptiveRouter(QueryRouter):
    """
    Adaptive query router that learns from feedback

    Extends QueryRouter with the ability to adjust routing based on
    search result quality feedback.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Track mode performance per query type
        self._mode_performance: Dict[str, Dict[str, float]] = {
            'factual': {},
            'conceptual': {},
            'troubleshooting': {},
            'navigational': {}
        }

    def record_feedback(
        self,
        query: str,
        mode_used: str,
        quality_score: float
    ):
        """
        Record feedback about search quality for a mode

        Args:
            query: The query that was searched
            mode_used: The search mode that was used
            quality_score: Quality score (0-1, higher is better)
        """
        analysis = self.analyze(query)
        query_type = analysis.query_type

        if query_type not in self._mode_performance:
            self._mode_performance[query_type] = {}

        if mode_used not in self._mode_performance[query_type]:
            self._mode_performance[query_type][mode_used] = []

        # Keep last 100 scores per mode per query type
        scores = self._mode_performance[query_type].get(mode_used, [])
        scores.append(quality_score)
        if len(scores) > 100:
            scores = scores[-100:]
        self._mode_performance[query_type][mode_used] = scores

        logger.debug(
            f"Recorded feedback: query_type={query_type}, mode={mode_used}, "
            f"score={quality_score:.2f}"
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for each mode"""
        stats = {}
        for query_type, mode_scores in self._mode_performance.items():
            stats[query_type] = {}
            for mode, scores in mode_scores.items():
                if scores:
                    stats[query_type][mode] = {
                        'avg_score': sum(scores) / len(scores),
                        'sample_count': len(scores)
                    }
        return stats
