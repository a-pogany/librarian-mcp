"""Tests for search engine"""

import pytest
from core.search import SearchEngine


def test_basic_search(search_engine):
    """Test basic keyword search"""
    results = search_engine.search("authentication")

    assert len(results) > 0
    assert any('authentication' in r['snippet'].lower() for r in results)


def test_search_with_product_filter(search_engine):
    """Test search with product filter"""
    results = search_engine.search("test", product="product-a")

    assert all(r['product'] == 'product-a' for r in results)


def test_search_relevance_scoring(search_engine):
    """Test relevance scoring"""
    results = search_engine.search("API")

    # Results should be sorted by relevance
    if len(results) > 1:
        scores = [r['relevance_score'] for r in results]
        assert scores == sorted(scores, reverse=True)


def test_get_document(search_engine):
    """Test document retrieval"""
    doc = search_engine.get_document("product-a/component-1/readme.md")

    assert doc is not None
    assert 'content' in doc
    assert 'Component 1' in doc['content']


def test_get_nonexistent_document(search_engine):
    """Test retrieving non-existent document"""
    doc = search_engine.get_document("nonexistent/path.md")
    assert doc is None


def test_empty_query(search_engine):
    """Test empty query returns no results"""
    results = search_engine.search("")
    assert len(results) == 0


def test_max_results_limit(search_engine):
    """Test max results limit"""
    results = search_engine.search("test", max_results=1)
    assert len(results) <= 1


def test_parse_query(search_engine):
    """Test query parsing"""
    keywords = search_engine._parse_query("OAuth Authentication API")
    assert 'oauth' in keywords
    assert 'authentication' in keywords
    assert 'api' in keywords


def test_snippet_extraction(search_engine):
    """Test snippet extraction"""
    snippet = search_engine._extract_snippet(
        "This is a test document about authentication and OAuth.",
        ["authentication", "oauth"]
    )
    assert len(snippet) <= 200
    assert 'authentication' in snippet.lower()
