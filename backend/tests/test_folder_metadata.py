"""
Test folder metadata extraction and hierarchical search
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from core.indexer import FileIndexer
from core.folder_metadata import FolderMetadataExtractor
from core.hierarchical_search import HierarchicalSearchEngine
from core.semantic_search import SemanticSearchEngine


@pytest.fixture
def temp_docs_dir():
    """Create temporary docs directory with test structure"""
    temp_dir = tempfile.mkdtemp()
    docs_path = Path(temp_dir) / "docs"
    docs_path.mkdir()

    # Create dge folder with API documentation
    dge_api = docs_path / "dge" / "api"
    dge_api.mkdir(parents=True)

    # API endpoint documentation
    (dge_api / "get-draw-data.md").write_text("""
# GET /api/draw-data

REST API endpoint for fetching draw data.

## Endpoint
`GET /api/v1/draw-data`

## Parameters
- `draw_id`: ID of the draw to fetch
- `format`: Response format (json, xml)

## Response
Returns draw data including numbers, dates, and prize information.

## Example
```
curl https://api.example.com/v1/draw-data?draw_id=123
```
""")

    (dge_api / "database-schema.md").write_text("""
# Database Schema

## Tables

### draws
- id: Primary key
- draw_date: Date of draw
- numbers: Winning numbers
- jackpot: Prize amount

### batches
- id: Primary key
- batch_name: Name of batch process
- status: Processing status
""")

    # Create MIGRATION folder with migration documentation
    migration_docs = docs_path / "migration" / "guides"
    migration_docs.mkdir(parents=True)

    (migration_docs / "api-migration.md").write_text("""
# API Migration Guide

Guide for migrating legacy API endpoints to new REST architecture.

## Overview
This document covers migrating old SOAP endpoints to REST API format.

## Steps
1. Identify legacy endpoints
2. Map to new REST endpoints
3. Update API consumers
4. Test migration

## Affected Endpoints
- `/legacy/get-draw-data` → `/api/v1/draw-data`
- `/legacy/submit-batch` → `/api/v1/batches`

## Testing
Verify all API endpoints work after migration.
""")

    (migration_docs / "database-migration.md").write_text("""
# Database Migration Process

How to migrate data from old system to new database schema.

## Migration Scripts
Run these SQL scripts to migrate database:

1. `migrate_draws.sql` - Migrate draw data
2. `migrate_batches.sql` - Migrate batch data

## Validation
Query API endpoints to verify data integrity after migration.
""")

    yield docs_path

    # Cleanup
    shutil.rmtree(temp_dir)


def test_folder_metadata_extraction(temp_docs_dir):
    """Test that folder metadata is correctly extracted"""
    config = {
        'file_extensions': ['.md'],
        'max_file_size_mb': 10,
        'watch_for_changes': False
    }

    # Initialize indexer without embeddings
    indexer = FileIndexer(
        str(temp_docs_dir),
        config=config,
        enable_embeddings=False
    )

    # Build document index
    indexer.build_index()

    # Extract folder metadata
    extractor = FolderMetadataExtractor(indexer)
    folder_metadata = extractor.build_folder_metadata()

    # Should have metadata for dge/api and migration/guides
    assert len(folder_metadata) >= 2

    # Check dge/api metadata
    dge_api_meta = folder_metadata.get('dge/api')
    assert dge_api_meta is not None
    assert dge_api_meta.doc_count == 2
    assert 'api' in dge_api_meta.description.lower() or 'endpoint' in dge_api_meta.description.lower()

    # Check migration/guides metadata
    migration_meta = folder_metadata.get('migration/guides')
    assert migration_meta is not None
    assert migration_meta.doc_count == 2
    assert 'migration' in migration_meta.description.lower()

    print(f"\n=== Folder Metadata ===")
    for path, meta in folder_metadata.items():
        print(f"\nFolder: {path}")
        print(f"  Description: {meta.description[:100]}...")
        print(f"  Topics: {', '.join(meta.topics[:5])}")
        print(f"  Doc count: {meta.doc_count}")


def test_folder_descriptions_are_distinct(temp_docs_dir):
    """Test that dge and migration folders have clearly distinct descriptions"""
    config = {
        'file_extensions': ['.md'],
        'max_file_size_mb': 10,
        'watch_for_changes': False
    }

    indexer = FileIndexer(str(temp_docs_dir), config=config, enable_embeddings=False)
    indexer.build_index()

    extractor = FolderMetadataExtractor(indexer)
    folder_metadata = extractor.build_folder_metadata()

    dge_meta = folder_metadata.get('dge/api')
    migration_meta = folder_metadata.get('migration/guides')

    assert dge_meta is not None
    assert migration_meta is not None

    # Check that topics/keywords are distinct
    dge_topics_set = set(dge_meta.topics)
    migration_topics_set = set(migration_meta.topics)

    # Should have some API-related topics in dge
    assert any(topic in ['api', 'endpoint', 'rest', 'data', 'database'] for topic in dge_topics_set)

    # Should have migration-related topics in migration
    assert any(topic in ['migration', 'migrate', 'guide'] for topic in migration_topics_set)

    print(f"\n=== Topic Comparison ===")
    print(f"dge/api topics: {dge_meta.topics}")
    print(f"migration/guides topics: {migration_meta.topics}")


@pytest.mark.skipif(
    reason="Requires embeddings - skip by default"
)
def test_hierarchical_search_dge_vs_migration(temp_docs_dir):
    """
    Test that hierarchical search correctly prioritizes dge over migration
    for API endpoint queries
    """
    config = {
        'file_extensions': ['.md'],
        'max_file_size_mb': 10,
        'watch_for_changes': False,
        'embeddings': {
            'enabled': True,
            'model': 'all-MiniLM-L6-v2',
            'chunk_size': 512,
            'chunk_overlap': 128
        },
        'folder_metadata': {
            'enabled': True,
            'max_folders_to_search': 3,
            'folder_similarity_threshold': 0.3
        }
    }

    # Initialize indexer WITH embeddings
    try:
        indexer = FileIndexer(
            str(temp_docs_dir),
            config=config,
            enable_embeddings=True
        )

        # Build index (includes folder metadata)
        result = indexer.build_index()
        print(f"\n=== Index Built ===")
        print(f"Files: {result['files_indexed']}")
        print(f"Folders: {result.get('folders_indexed', 0)}")

        # Initialize hierarchical search
        base_semantic_engine = SemanticSearchEngine(
            indexer.embedding_generator,
            indexer.vector_db,
            indexer
        )

        hierarchical_engine = HierarchicalSearchEngine(
            embedding_generator=indexer.embedding_generator,
            folder_metadata_extractor=indexer.folder_metadata_extractor,
            folder_vector_db=indexer.folder_vector_db,
            semantic_search_engine=base_semantic_engine,
            enable_folder_filtering=True
        )

        # Test query about API endpoint
        query = "fetch information about dge rest api endpoint get-draw-data"
        results = hierarchical_engine.search(
            query=query,
            max_results=5,
            max_folders=3
        )

        print(f"\n=== Search Results for: '{query}' ===")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['file_path']}")
            print(f"   Score: {result.get('relevance_score', 'N/A')}")
            print(f"   Folder matched: {result.get('folder_matched', False)}")
            if result.get('folder_similarity'):
                print(f"   Folder similarity: {result['folder_similarity']:.2f}")

        # Assertions
        assert len(results) > 0, "Should return results"

        # Top result should be from dge/api folder, NOT migration
        top_result = results[0]
        assert 'dge' in top_result['file_path'].lower(), \
            f"Top result should be from dge folder, got: {top_result['file_path']}"
        assert 'migration' not in top_result['file_path'].lower(), \
            f"Top result should NOT be from migration folder, got: {top_result['file_path']}"

        # Specifically should find get-draw-data.md
        assert 'get-draw-data' in top_result['file_path'].lower(), \
            f"Should find get-draw-data document, got: {top_result['file_path']}"

    except ImportError as e:
        pytest.skip(f"Embeddings dependencies not available: {e}")


if __name__ == "__main__":
    # Run basic tests without embeddings
    print("Running folder metadata tests...")

    with tempfile.TemporaryDirectory() as temp_dir:
        docs_path = Path(temp_dir) / "docs"
        docs_path.mkdir()

        # Create test structure
        dge_api = docs_path / "dge" / "api"
        dge_api.mkdir(parents=True)
        (dge_api / "test.md").write_text("# API Documentation\nREST endpoints")

        migration = docs_path / "migration" / "guides"
        migration.mkdir(parents=True)
        (migration / "test.md").write_text("# Migration Guide\nHow to migrate systems")

        config = {'file_extensions': ['.md'], 'max_file_size_mb': 10, 'watch_for_changes': False}
        indexer = FileIndexer(str(docs_path), config=config, enable_embeddings=False)
        indexer.build_index()

        extractor = FolderMetadataExtractor(indexer)
        metadata = extractor.build_folder_metadata()

        print(f"\nExtracted metadata for {len(metadata)} folders:")
        for path, meta in metadata.items():
            print(f"  {path}: {meta.description[:80]}...")

    print("\n✓ Basic tests passed!")
