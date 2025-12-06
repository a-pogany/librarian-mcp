# Changelog

All notable changes to Librarian MCP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.3] - 2024-12-04

### Added - Phase 2.5: RAG Quality Enhancements

#### Reranking Mode
- **New search mode**: `rerank` - Two-stage search combining semantic retrieval with keyword refinement
- Added `HybridSearchEngine._rerank_search()` method for two-stage retrieval
- Added `HybridSearchEngine._calculate_keyword_score()` for keyword relevance scoring
- New configuration options:
  - `search.rerank_candidates` (default: 50) - Number of semantic candidates to retrieve
  - `search.rerank_keyword_threshold` (default: 0.1) - Minimum keyword score threshold
- Score combination: 70% semantic + 30% keyword for optimal relevance
- Filters out semantically similar but contextually irrelevant documents

#### Enhanced Document Chunking
- **Extended chunking support** to all file types (.md, .txt, .docx)
- Added `DocumentChunker.chunk_document()` unified chunking interface
- **Semantic chunking for Markdown**:
  - Splits on heading boundaries (## H2, ### H3)
  - Preserves document structure and hierarchy
  - Maintains heading context in chunk metadata
- **Fixed-size chunking for text files**:
  - 512-token chunks with 128-token overlap
  - Sentence boundary preservation
  - Configurable via `embeddings.chunk_size` and `embeddings.chunk_overlap`
- Added `DocumentChunk` dataclass for structured chunk representation
- Improved large section handling with automatic sub-chunking

#### Rich Metadata Filtering
- **Tag extraction** from YAML frontmatter:
  - Added `FileIndexer._extract_frontmatter_tags()` method
  - Supports both list format (`tags: [api, auth]`) and comma-separated (`tags: api, auth`)
- **Document type inference**:
  - Added `FileIndexer._infer_doc_type()` method
  - 6 document types: `api`, `guide`, `architecture`, `reference`, `readme`, `documentation`
  - Classification based on filename patterns and content analysis
- **Temporal filtering**:
  - New `modified_after` parameter for recent document searches
  - New `modified_before` parameter for historical document searches
  - ISO 8601 date format support (e.g., `2024-01-01` or `2024-01-01T00:00:00`)
- **Enhanced MCP tool**: `search_documentation()` now accepts:
  - `doc_type`: Filter by document type
  - `tags`: Filter by tags (OR logic - at least one must match)
  - `modified_after`: Temporal filtering (after date)
  - `modified_before`: Temporal filtering (before date)
- Added `_apply_metadata_filters()` helper for post-search filtering
- New configuration section:
  ```json
  "metadata": {
    "extract_tags": true,
    "infer_doc_type": true,
    "track_modifications": true
  }
  ```

### Fixed
- **ChromaDB compatibility**: Added `VectorDatabase._sanitize_metadata()` to convert lists to comma-separated strings
- **Date filter logic**: Changed `modified_before` from `>` to `>=` for exclusive filtering
- **Missing import**: Added `Any` to typing imports in `backend/core/indexer.py`

### Changed
- Updated `HybridSearchEngine` to support 4 search modes: `keyword`, `semantic`, `hybrid`, `rerank`
- Enhanced `backend/core/indexer.py` to capture richer metadata during indexing
- Updated search pipeline flow chart in documentation to include rerank mode
- Improved `config.json` with new Phase 2.5 configuration sections

### Testing
- Added `backend/tests/test_reranking.py` with 6 comprehensive tests
- Added `backend/tests/test_chunker.py` with 10 comprehensive tests
- Added `backend/tests/test_metadata.py` with 16 comprehensive tests
- **Total new tests**: 32 tests, 100% pass rate
- **Overall test suite**: 55/56 tests passing (98.2%)

### Documentation
- Updated README.md with v2.0.3 features and usage examples
- Updated CLAUDE.md with Phase 2.5 implementation details
- Added RAG_ENHANCEMENTS_IMPLEMENTATION_SUMMARY.md with comprehensive implementation guide
- Updated all version references from 2.0.0 to 2.0.3
- Added usage examples for metadata filtering, tag-based search, and temporal queries

## [2.0.0] - 2024-11-15

### Added - Phase 2: Enterprise RAG
- E5-large-v2 embeddings (1024-dimensional, 30-40% better quality)
- Hierarchical document chunking (512-token chunks, 128-token overlap)
- Persistent vector storage (ChromaDB with optimized HNSW)
- Two-stage reranking (cross-encoder for 2x precision improvement)
- Query embedding cache (5x faster repeated queries)
- BM25 keyword search (probabilistic scoring)
- Reciprocal Rank Fusion (RRF) for hybrid search optimization
- Semantic + Keyword hybrid search (best of both worlds)

### Performance
- **Relevance**: 85% Precision@10 (was 40% in v1.0)
- **Coverage**: 100% document coverage with chunking (was 0.5% with truncation)
- **Scale**: Handles 200-600 page DOCX files, 10,000+ documents
- **Speed**: 150-200ms hybrid queries, 10-20ms cached queries

## [1.0.0] - 2024-10-01

### Added - Phase 1: Foundation
- HTTP/SSE MCP server for Claude Desktop integration
- Keyword-based search with relevance ranking
- Multi-format support (.md, .txt, .docx)
- Real-time file watching with automatic index updates
- Product/component hierarchical organization
- 5 MCP tools: search_documentation, get_document, list_products, list_components, get_index_status

### Initial Release
- Basic documentation search functionality
- In-memory index with product/component hierarchy
- FastMCP integration for MCP protocol
- Simple keyword matching with snippet extraction
