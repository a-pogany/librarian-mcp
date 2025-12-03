"""Tests for file indexer"""

import pytest
from core.indexer import FileIndexer, DocumentIndex


def test_document_index_initialization():
    """Test DocumentIndex initializes correctly"""
    index = DocumentIndex()
    assert len(index.documents) == 0
    assert len(index.products) == 0
    assert len(index.components) == 0
    assert index.last_indexed is None


def test_indexer_initialization(temp_docs):
    """Test indexer initializes correctly"""
    indexer = FileIndexer(str(temp_docs))
    assert indexer.docs_root == temp_docs
    assert isinstance(indexer.index, DocumentIndex)


def test_build_index(temp_docs):
    """Test building complete index"""
    indexer = FileIndexer(str(temp_docs), {'watch_for_changes': False})
    result = indexer.build_index()

    assert result['status'] == 'complete'
    assert result['files_indexed'] == 3
    assert len(indexer.index.documents) == 3


def test_index_products(indexer):
    """Test product indexing"""
    products = indexer.get_products()
    assert len(products) == 2

    product_names = [p['name'] for p in products]
    assert 'product-a' in product_names
    assert 'product-b' in product_names


def test_index_components(indexer):
    """Test component indexing"""
    components = indexer.get_components('product-a')
    assert len(components) == 2

    component_names = [c['name'] for c in components]
    assert 'component-1' in component_names
    assert 'component-2' in component_names


def test_get_status(indexer):
    """Test index status"""
    status = indexer.get_status()
    assert status['status'] == 'ready'
    assert status['total_documents'] == 3
    assert status['products'] == 2


def test_add_remove_document():
    """Test adding and removing documents"""
    index = DocumentIndex()

    doc = {
        'path': 'test/component/file.md',
        'product': 'test',
        'component': 'component',
        'file_name': 'file.md',
        'file_type': '.md',
        'content': 'test content',
        'headings': [],
        'metadata': {},
        'size_bytes': 100,
        'last_modified': '2024-01-01T00:00:00'
    }

    index.add_document(doc)
    assert len(index.documents) == 1
    assert 'test' in index.products

    index.remove_document('test/component/file.md')
    assert len(index.documents) == 0
