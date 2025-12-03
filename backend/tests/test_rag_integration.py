"""
Integration tests for RAG/semantic search functionality
"""

import pytest
from pathlib import Path
import tempfile
import shutil


def test_embedding_generator():
    """Test basic embedding generation"""
    try:
        from backend.core.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator()

        # Test single text encoding
        embedding = generator.encode_single("This is a test document")
        assert embedding.shape == (384,)  # MiniLM dimension

        # Test batch encoding
        texts = ["Document 1", "Document 2", "Document 3"]
        embeddings = generator.encode(texts)
        assert embeddings.shape == (3, 384)

    except ImportError:
        pytest.skip("RAG dependencies not installed")


def test_vector_database():
    """Test ChromaDB vector database operations"""
    try:
        from backend.core.vector_db import VectorDatabase
        from backend.core.embeddings import EmbeddingGenerator
        import numpy as np

        # Use in-memory database for testing
        db = VectorDatabase(persist_directory=None, collection_name="test_docs")
        generator = EmbeddingGenerator()

        # Add documents
        docs = ["Machine learning basics", "Python programming guide", "Data science tutorial"]
        embeddings = generator.encode(docs)

        doc_ids = ["doc1.md", "doc2.md", "doc3.md"]
        metadatas = [
            {"product": "ml", "component": "intro", "file_type": ".md"},
            {"product": "python", "component": "guide", "file_type": ".md"},
            {"product": "ds", "component": "tutorial", "file_type": ".md"}
        ]

        db.add_documents(doc_ids, embeddings, metadatas)

        # Test count
        assert db.get_count() == 3

        # Test search
        query_embedding = generator.encode_query("machine learning")
        results = db.search(query_embedding, n_results=2)

        assert len(results) == 2
        assert results[0]['id'] == 'doc1.md'  # Should match ML document best
        assert 0 <= results[0]['similarity'] <= 1.0

        # Test delete
        db.delete_document("doc1.md")
        assert db.get_count() == 2

        # Test clear
        db.clear()
        assert db.get_count() == 0

    except ImportError:
        pytest.skip("RAG dependencies not installed")


def test_semantic_search_engine():
    """Test semantic search engine"""
    try:
        from backend.core.embeddings import EmbeddingGenerator
        from backend.core.vector_db import VectorDatabase
        from backend.core.semantic_search import SemanticSearchEngine
        from backend.core.indexer import FileIndexer, DocumentIndex

        # Create temporary docs directory
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs" / "product1" / "component1"
            docs_dir.mkdir(parents=True)

            # Create test document
            test_file = docs_dir / "test.md"
            test_file.write_text("# Machine Learning\n\nIntroduction to machine learning concepts")

            # Initialize indexer with embeddings
            indexer = FileIndexer(str(Path(temp_dir) / "docs"), enable_embeddings=True)
            indexer.build_index()

            # Create semantic search engine
            search_engine = SemanticSearchEngine(
                indexer.embedding_generator,
                indexer.vector_db,
                indexer
            )

            # Test search
            results = search_engine.search("machine learning tutorial", max_results=5)

            assert len(results) > 0
            assert results[0]['file_name'] == 'test.md'
            assert results[0]['similarity_score'] > 0.5

    except ImportError:
        pytest.skip("RAG dependencies not installed")


def test_hybrid_search_modes():
    """Test hybrid search engine with different modes"""
    try:
        from backend.core.search import SearchEngine
        from backend.core.hybrid_search import HybridSearchEngine
        from backend.core.indexer import FileIndexer

        # Create temporary docs directory
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs" / "product1" / "component1"
            docs_dir.mkdir(parents=True)

            # Create test document
            test_file = docs_dir / "authentication.md"
            test_file.write_text("# Authentication\n\nUser authentication using OAuth2")

            # Initialize with embeddings disabled for keyword-only test
            indexer_keyword = FileIndexer(str(Path(temp_dir) / "docs"), enable_embeddings=False)
            indexer_keyword.build_index()

            keyword_engine = SearchEngine(indexer_keyword)

            # Create hybrid engine with keyword-only
            hybrid_engine = HybridSearchEngine(
                keyword_engine=keyword_engine,
                semantic_engine=None,
                default_mode='keyword'
            )

            # Test keyword mode
            results = hybrid_engine.search("authentication", mode='keyword')
            assert len(results) > 0
            assert results[0]['search_mode'] == 'keyword'

            # Test mode switching
            assert hybrid_engine.get_mode() == 'keyword'
            hybrid_engine.set_mode('keyword')
            assert hybrid_engine.get_mode() == 'keyword'

    except ImportError:
        pytest.skip("RAG dependencies not installed")


def test_search_mode_configuration():
    """Test that search mode can be configured"""
    from backend.config.settings import load_config
    import os

    # Test default mode from config
    config = load_config()
    search_mode = config.get('search', {}).get('mode', 'keyword')
    assert search_mode in ['keyword', 'semantic', 'hybrid']

    # Test environment variable override
    os.environ['SEARCH_MODE'] = 'keyword'
    config = load_config()
    assert config['search']['mode'] == 'keyword'

    os.environ['SEARCH_MODE'] = 'hybrid'
    config = load_config()
    assert config['search']['mode'] == 'hybrid'

    # Cleanup
    if 'SEARCH_MODE' in os.environ:
        del os.environ['SEARCH_MODE']
