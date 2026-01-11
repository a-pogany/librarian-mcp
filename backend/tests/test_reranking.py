"""Unit tests for reranking search mode"""

import pytest
from backend.core.hybrid_search import HybridSearchEngine
from backend.core.search import SearchEngine
from backend.core.semantic_search import SemanticSearchEngine
from backend.core.indexer import DocumentIndex


class MockSemanticEngine:
    """Mock semantic search engine for testing"""

    def search(self, query, product=None, component=None, file_types=None, max_results=10, **kwargs):
        """Return mock semantic results (accepts additional email filter kwargs)"""
        return [
            {
                'id': 'doc1.md',
                'file_path': 'product1/comp1/doc1.md',
                'product': 'product1',
                'component': 'comp1',
                'file_name': 'doc1.md',
                'file_type': '.md',
                'snippet': 'This document contains important authentication information.',
                'similarity_score': 0.9,
                'last_modified': '2024-01-01T00:00:00'
            },
            {
                'id': 'doc2.md',
                'file_path': 'product1/comp1/doc2.md',
                'product': 'product1',
                'component': 'comp1',
                'file_name': 'doc2.md',
                'file_type': '.md',
                'snippet': 'This is a completely different topic about databases.',
                'similarity_score': 0.7,
                'last_modified': '2024-01-02T00:00:00'
            },
            {
                'id': 'doc3.md',
                'file_path': 'product1/comp1/doc3.md',
                'product': 'product1',
                'component': 'comp1',
                'file_name': 'doc3.md',
                'file_type': '.md',
                'snippet': 'API authentication reference guide.',
                'similarity_score': 0.85,
                'last_modified': '2024-01-03T00:00:00'
            }
        ]


class MockKeywordEngine:
    """Mock keyword search engine for testing"""

    def __init__(self):
        self.index = DocumentIndex()

    def search(self, query, product=None, component=None, file_types=None, max_results=10, **kwargs):
        """Return mock keyword results (accepts additional email filter kwargs)"""
        return []


def test_rerank_mode_filters_irrelevant():
    """Test that reranking filters out semantically similar but irrelevant documents"""
    keyword_engine = MockKeywordEngine()
    semantic_engine = MockSemanticEngine()

    hybrid_engine = HybridSearchEngine(
        keyword_engine=keyword_engine,
        semantic_engine=semantic_engine,
        default_mode='rerank',
        rerank_candidates=50,
        rerank_keyword_threshold=0.1
    )

    # Search for "authentication API" - should filter out doc2 (no matching keywords)
    results = hybrid_engine.search(query="authentication API", max_results=10)

    # Should have results for doc1 and doc3 (both mention authentication or API)
    # doc2 should be filtered out (only mentions databases)
    assert len(results) <= 2, "Reranking should filter out documents without query keywords"

    # All results should contain at least one query keyword
    for result in results:
        content = result.get('snippet', '').lower()
        filename = result.get('file_name', '').lower()
        combined = content + ' ' + filename
        assert 'authentication' in combined or 'api' in combined, \
            f"Result should contain query keywords: {result['file_name']}"


def test_rerank_mode_combines_scores():
    """Test that reranking combines semantic and keyword scores correctly"""
    keyword_engine = MockKeywordEngine()
    semantic_engine = MockSemanticEngine()

    hybrid_engine = HybridSearchEngine(
        keyword_engine=keyword_engine,
        semantic_engine=semantic_engine,
        default_mode='rerank',
        rerank_candidates=50,
        rerank_keyword_threshold=0.1
    )

    results = hybrid_engine.search(query="authentication", max_results=10)

    # Check that results have both scores
    for result in results:
        assert 'relevance_score' in result, "Should have combined relevance score"
        assert 'semantic_score' in result, "Should preserve semantic score"
        assert 'keyword_score' in result, "Should have keyword score"
        assert result['search_mode'] == 'rerank', "Should indicate rerank mode"

        # Relevance should be weighted combination (70% semantic + 30% keyword)
        semantic = result['semantic_score']
        keyword = result['keyword_score']
        expected = round(0.7 * semantic + 0.3 * keyword, 2)
        assert result['relevance_score'] == expected, \
            f"Relevance score should be 0.7*{semantic} + 0.3*{keyword} = {expected}"


def test_rerank_keyword_threshold():
    """Test that keyword threshold filters out low-scoring documents"""
    keyword_engine = MockKeywordEngine()
    semantic_engine = MockSemanticEngine()

    # Set high threshold
    hybrid_engine = HybridSearchEngine(
        keyword_engine=keyword_engine,
        semantic_engine=semantic_engine,
        default_mode='rerank',
        rerank_candidates=50,
        rerank_keyword_threshold=0.5  # High threshold
    )

    results = hybrid_engine.search(query="nonexistent_term", max_results=10)

    # Should have no results if no documents meet keyword threshold
    assert len(results) == 0, "High keyword threshold should filter out all results with no matches"


def test_calculate_keyword_score():
    """Test keyword score calculation"""
    keyword_engine = MockKeywordEngine()
    semantic_engine = MockSemanticEngine()

    hybrid_engine = HybridSearchEngine(
        keyword_engine=keyword_engine,
        semantic_engine=semantic_engine,
        default_mode='rerank'
    )

    # Test with multiple keywords in content
    keywords = ['authentication', 'api']
    content = 'This is an authentication API reference. authentication is important.'
    filename = 'api_auth.md'

    score = hybrid_engine._calculate_keyword_score(keywords, content, filename)

    # Should have positive score for matching keywords
    assert score > 0, "Should have positive score for matching keywords"

    # Score should be normalized (0-1)
    assert 0 <= score <= 1, "Score should be normalized between 0 and 1"

    # Filename matches should boost score
    filename_only_score = hybrid_engine._calculate_keyword_score(keywords, '', filename)
    assert filename_only_score > 0, "Filename matches should contribute to score"


def test_rerank_fallback_to_keyword():
    """Test that rerank mode falls back to keyword when semantic engine is unavailable"""
    keyword_engine = MockKeywordEngine()

    hybrid_engine = HybridSearchEngine(
        keyword_engine=keyword_engine,
        semantic_engine=None,  # No semantic engine
        default_mode='rerank'
    )

    # Should automatically fall back to keyword mode
    assert hybrid_engine.default_mode == 'keyword', \
        "Should fall back to keyword mode when semantic engine not available"


def test_rerank_respects_filters():
    """Test that reranking respects product/component filters"""
    keyword_engine = MockKeywordEngine()
    semantic_engine = MockSemanticEngine()

    hybrid_engine = HybridSearchEngine(
        keyword_engine=keyword_engine,
        semantic_engine=semantic_engine,
        default_mode='rerank'
    )

    # Search with filters should pass them through to semantic engine
    results = hybrid_engine.search(
        query="authentication",
        product="product1",
        component="comp1",
        max_results=10
    )

    # All results should match the filters
    for result in results:
        assert result['product'] == 'product1', "Results should match product filter"
        assert result['component'] == 'comp1', "Results should match component filter"
