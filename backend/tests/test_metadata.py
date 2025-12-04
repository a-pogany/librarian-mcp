"""Unit tests for enhanced metadata extraction and filtering"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from backend.core.indexer import FileIndexer
from backend.mcp_server.tools import _apply_metadata_filters


def test_extract_frontmatter_tags():
    """Test tag extraction from YAML frontmatter"""
    config = {'file_extensions': ['.md'], 'max_file_size_mb': 10}
    indexer = FileIndexer(docs_root="./docs", config=config)

    # Test with list tags
    content = """---
tags: [api, authentication, security]
title: Auth API
---

# Authentication API

This is the content.
"""

    tags = indexer._extract_frontmatter_tags(content)

    assert isinstance(tags, list), "Tags should be a list"
    assert 'api' in tags, "Should extract 'api' tag"
    assert 'authentication' in tags, "Should extract 'authentication' tag"
    assert 'security' in tags, "Should extract 'security' tag"


def test_extract_frontmatter_tags_comma_separated():
    """Test tag extraction with comma-separated string"""
    config = {'file_extensions': ['.md'], 'max_file_size_mb': 10}
    indexer = FileIndexer(docs_root="./docs", config=config)

    content = """---
tags: api, authentication, security
---

Content here.
"""

    tags = indexer._extract_frontmatter_tags(content)

    assert isinstance(tags, list), "Tags should be converted to list"
    assert 'api' in tags, "Should parse comma-separated tags"
    assert 'authentication' in tags, "Should parse comma-separated tags"
    assert 'security' in tags, "Should parse comma-separated tags"


def test_extract_frontmatter_no_tags():
    """Test content without tags"""
    config = {'file_extensions': ['.md'], 'max_file_size_mb': 10}
    indexer = FileIndexer(docs_root="./docs", config=config)

    content = """---
title: Document Title
author: John Doe
---

Content without tags.
"""

    tags = indexer._extract_frontmatter_tags(content)

    assert tags == [], "Should return empty list when no tags"


def test_extract_frontmatter_no_frontmatter():
    """Test content without frontmatter"""
    config = {'file_extensions': ['.md'], 'max_file_size_mb': 10}
    indexer = FileIndexer(docs_root="./docs", config=config)

    content = """# Regular Document

No frontmatter here.
"""

    tags = indexer._extract_frontmatter_tags(content)

    assert tags == [], "Should return empty list when no frontmatter"


def test_infer_doc_type_from_filename():
    """Test document type inference from filename"""
    config = {'file_extensions': ['.md'], 'max_file_size_mb': 10}
    indexer = FileIndexer(docs_root="./docs", config=config)

    # Test various filename patterns
    test_cases = [
        (Path("api-reference.md"), "", "api"),
        (Path("architecture-guide.md"), "", "architecture"),
        (Path("tutorial.md"), "", "guide"),
        (Path("user-guide.md"), "", "guide"),
        (Path("spec.md"), "", "reference"),
        (Path("README.md"), "", "readme"),
    ]

    for file_path, content, expected in test_cases:
        doc_type = indexer._infer_doc_type(file_path, content)
        assert doc_type == expected, f"File {file_path.name} should be inferred as '{expected}', got '{doc_type}'"


def test_infer_doc_type_from_content():
    """Test document type inference from content"""
    config = {'file_extensions': ['.md'], 'max_file_size_mb': 10}
    indexer = FileIndexer(docs_root="./docs", config=config)

    # Test content patterns
    test_cases = [
        ("class AuthService { ... }", "api"),
        ("function authenticate() { ... }", "api"),
        ("# Architecture Overview\n\nThis document describes the system architecture.", "architecture"),
        ("# Tutorial: Getting Started\n\nFollow these steps...", "guide"),
    ]

    for content, expected in test_cases:
        doc_type = indexer._infer_doc_type(Path("document.md"), content)
        assert doc_type == expected, f"Content should be inferred as '{expected}', got '{doc_type}'"


def test_infer_doc_type_default():
    """Test default document type when no patterns match"""
    config = {'file_extensions': ['.md'], 'max_file_size_mb': 10}
    indexer = FileIndexer(docs_root="./docs", config=config)

    doc_type = indexer._infer_doc_type(Path("random-file.md"), "Some random content without patterns.")

    assert doc_type == "documentation", "Should default to 'documentation' when no patterns match"


def test_apply_metadata_filters_doc_type():
    """Test filtering by document type"""
    results = [
        {'id': '1', 'doc_type': 'api', 'last_modified': '2024-01-01T00:00:00'},
        {'id': '2', 'doc_type': 'guide', 'last_modified': '2024-01-01T00:00:00'},
        {'id': '3', 'doc_type': 'api', 'last_modified': '2024-01-01T00:00:00'},
    ]

    filtered = _apply_metadata_filters(results, doc_type='api')

    assert len(filtered) == 2, "Should filter to only API documents"
    assert all(r['doc_type'] == 'api' for r in filtered), "All results should be API type"


def test_apply_metadata_filters_tags():
    """Test filtering by tags"""
    results = [
        {'id': '1', 'tags': ['api', 'security'], 'last_modified': '2024-01-01T00:00:00'},
        {'id': '2', 'tags': ['database'], 'last_modified': '2024-01-01T00:00:00'},
        {'id': '3', 'tags': ['api', 'authentication'], 'last_modified': '2024-01-01T00:00:00'},
    ]

    # Filter for documents with 'api' tag
    filtered = _apply_metadata_filters(results, tags=['api'])

    assert len(filtered) == 2, "Should filter to documents with 'api' tag"
    assert all('api' in r['tags'] for r in filtered), "All results should have 'api' tag"


def test_apply_metadata_filters_tags_multiple():
    """Test filtering by multiple tags (OR logic)"""
    results = [
        {'id': '1', 'tags': ['api', 'security'], 'last_modified': '2024-01-01T00:00:00'},
        {'id': '2', 'tags': ['database'], 'last_modified': '2024-01-01T00:00:00'},
        {'id': '3', 'tags': ['authentication'], 'last_modified': '2024-01-01T00:00:00'},
    ]

    # Filter for documents with 'api' OR 'database' tag
    filtered = _apply_metadata_filters(results, tags=['api', 'database'])

    assert len(filtered) == 2, "Should filter to documents with 'api' OR 'database' tags"


def test_apply_metadata_filters_date_after():
    """Test filtering by modified_after date"""
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)

    results = [
        {'id': '1', 'last_modified': today.isoformat()},
        {'id': '2', 'last_modified': yesterday.isoformat()},
        {'id': '3', 'last_modified': week_ago.isoformat()},
    ]

    # Filter for docs modified in last 3 days
    three_days_ago = (today - timedelta(days=3)).isoformat()
    filtered = _apply_metadata_filters(results, modified_after=three_days_ago)

    assert len(filtered) == 2, "Should filter to docs modified in last 3 days"
    assert results[2] not in filtered, "Week-old document should be filtered out"


def test_apply_metadata_filters_date_before():
    """Test filtering by modified_before date"""
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)

    results = [
        {'id': '1', 'last_modified': today.isoformat()},
        {'id': '2', 'last_modified': yesterday.isoformat()},
        {'id': '3', 'last_modified': week_ago.isoformat()},
    ]

    # Filter for docs modified before yesterday
    filtered = _apply_metadata_filters(results, modified_before=yesterday.isoformat())

    assert len(filtered) == 1, "Should filter to docs modified before yesterday"
    assert filtered[0]['id'] == '3', "Should only include week-old document"


def test_apply_metadata_filters_date_range():
    """Test filtering with both date range filters"""
    today = datetime.now()
    days_ago_2 = today - timedelta(days=2)
    days_ago_5 = today - timedelta(days=5)
    days_ago_10 = today - timedelta(days=10)

    results = [
        {'id': '1', 'last_modified': today.isoformat()},
        {'id': '2', 'last_modified': days_ago_2.isoformat()},
        {'id': '3', 'last_modified': days_ago_5.isoformat()},
        {'id': '4', 'last_modified': days_ago_10.isoformat()},
    ]

    # Filter for docs modified between 7 and 1 days ago
    seven_days_ago = (today - timedelta(days=7)).isoformat()
    one_day_ago = (today - timedelta(days=1)).isoformat()

    filtered = _apply_metadata_filters(
        results,
        modified_after=seven_days_ago,
        modified_before=one_day_ago
    )

    assert len(filtered) == 2, "Should filter to docs in date range"
    filtered_ids = [r['id'] for r in filtered]
    assert '2' in filtered_ids, "Should include doc from 2 days ago"
    assert '3' in filtered_ids, "Should include doc from 5 days ago"


def test_apply_metadata_filters_combined():
    """Test filtering with multiple criteria"""
    today = datetime.now()
    yesterday = today - timedelta(days=1)

    results = [
        {'id': '1', 'doc_type': 'api', 'tags': ['security'], 'last_modified': today.isoformat()},
        {'id': '2', 'doc_type': 'api', 'tags': ['database'], 'last_modified': yesterday.isoformat()},
        {'id': '3', 'doc_type': 'guide', 'tags': ['security'], 'last_modified': today.isoformat()},
    ]

    # Filter for API docs with 'security' tag modified today
    filtered = _apply_metadata_filters(
        results,
        doc_type='api',
        tags=['security'],
        modified_after=yesterday.isoformat()
    )

    assert len(filtered) == 1, "Should apply all filters"
    assert filtered[0]['id'] == '1', "Should only match doc 1"


def test_apply_metadata_filters_missing_metadata():
    """Test handling of documents with missing metadata"""
    results = [
        {'id': '1', 'last_modified': '2024-01-01T00:00:00'},  # No doc_type, no tags
        {'id': '2', 'doc_type': 'api'},  # No last_modified, no tags
        {'id': '3', 'doc_type': 'api', 'tags': ['security'], 'last_modified': '2024-01-01T00:00:00'},
    ]

    # Should handle missing metadata gracefully
    filtered = _apply_metadata_filters(results, doc_type='api')
    assert len(filtered) == 2, "Should filter docs with doc_type field"

    filtered = _apply_metadata_filters(results, tags=['security'])
    assert len(filtered) == 1, "Should only match docs with tags field"


def test_apply_metadata_filters_invalid_dates():
    """Test handling of invalid date formats"""
    results = [
        {'id': '1', 'last_modified': 'invalid-date'},
        {'id': '2', 'last_modified': '2024-01-01T00:00:00'},
    ]

    # Should skip invalid dates
    filtered = _apply_metadata_filters(results, modified_after='2023-01-01T00:00:00')

    assert len(filtered) == 1, "Should skip documents with invalid date format"
    assert filtered[0]['id'] == '2', "Should only include valid date document"
