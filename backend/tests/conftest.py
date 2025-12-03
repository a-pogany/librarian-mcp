"""Pytest configuration and fixtures"""

import pytest
import tempfile
from pathlib import Path
from core.indexer import FileIndexer
from core.search import SearchEngine


@pytest.fixture
def temp_docs(tmp_path):
    """Create temporary documentation structure"""
    docs_root = tmp_path / "docs"
    docs_root.mkdir()

    # Create product/component structure
    (docs_root / "product-a" / "component-1").mkdir(parents=True)
    (docs_root / "product-a" / "component-2").mkdir(parents=True)
    (docs_root / "product-b" / "api").mkdir(parents=True)

    # Create sample files
    (docs_root / "product-a" / "component-1" / "readme.md").write_text(
        "# Component 1\nThis is a test document about authentication."
    )
    (docs_root / "product-a" / "component-2" / "spec.md").write_text(
        "# Spec\nAPI specification for OAuth integration."
    )
    (docs_root / "product-b" / "api" / "endpoints.md").write_text(
        "# Endpoints\nGET /api/v1/users"
    )

    return docs_root


@pytest.fixture
def indexer(temp_docs):
    """Create indexer with test documents"""
    indexer = FileIndexer(str(temp_docs), {'watch_for_changes': False})
    indexer.build_index()
    return indexer


@pytest.fixture
def search_engine(indexer):
    """Create search engine with indexed documents"""
    return SearchEngine(indexer)
