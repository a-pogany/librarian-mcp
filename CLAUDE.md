# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Librarian MCP** is a documentation search system enabling LLMs to autonomously retrieve technical documentation through MCP (Model Context Protocol).

**Current Status:** Phase 2 complete - Hybrid search with RAG/semantic capabilities
**Features:** Keyword search, semantic search, hybrid mode with configurable weighting
**Planned:** Phase 3 (Web UI)

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r backend/requirements.txt
```

### Running the Server

**HTTP/SSE Transport (default, for Claude Desktop):**
```bash
cd backend
python main.py
# Server starts on http://127.0.0.1:3001
```

**STDIO Transport (alternative, for other MCP clients):**
```bash
cd backend
python stdio_server.py
# Uses stdin/stdout for MCP communication
```

### Testing
```bash
# Run all tests from project root
pytest backend/tests/

# Run specific test file
pytest backend/tests/test_indexer.py

# Run specific test function
pytest backend/tests/test_indexer.py::test_build_index -v

# Run with coverage
pytest backend/tests/ --cov=backend/core --cov=backend/mcp_server --cov-report=html
```

### Management Scripts
```bash
# From project root:
./scripts/setup.sh        # Initial setup (creates docs/, venv, installs deps)
./scripts/start.sh        # Start MCP server
./scripts/status.sh       # Check server health
./scripts/test.sh         # Run test suite
./scripts/check_index.sh  # Detailed index diagnostics
```

## Architecture Overview

### Component Flow (Phase 2 - Hybrid Search)
```
Documentation Files (.md, .txt, .docx)
    ↓
FileIndexer → Dual Index
    ├─ In-Memory Index (keyword search)
    └─ Vector Database (semantic search via ChromaDB)
    ↓
HybridSearchEngine
    ├─ KeywordEngine (Phase 1)
    ├─ SemanticEngine (Phase 2)
    └─ Mode: keyword | semantic | hybrid
    ↓
MCP Server (HTTP/SSE or STDIO)
    ↓
LLM Client (Claude Desktop, Cline, etc.)
```

### Key Components

**backend/core/parsers.py**
- `Parser` abstract base class
- `MarkdownParser`, `TextParser`, `DOCXParser` - Extract content/headings with encoding detection
- Automatic encoding fallback using chardet

**backend/core/indexer.py**
- `DocumentIndex` - In-memory index with product/component hierarchy
- `FileIndexer` - Scans docs folder, builds dual index (keyword + vectors)
- `FileWatcher` - Monitors file changes, auto-updates both indices
- Optional embedding generation during indexing (controlled by config)

**backend/core/search.py**
- `SearchEngine` - Keyword search with weighted relevance scoring
- Scoring weights: phrase match (+5), filename (+3), heading (+2), content (+1)
- Snippet extraction with context lines
- Optional section extraction by heading name

**backend/core/embeddings.py** (Phase 2)
- `EmbeddingGenerator` - Generate vector embeddings using sentence-transformers
- Model: all-MiniLM-L6-v2 (384 dimensions)
- Batch processing support for efficiency

**backend/core/vector_db.py** (Phase 2)
- `VectorDatabase` - ChromaDB wrapper for vector similarity search
- Cosine similarity matching
- Metadata filtering (product, component, file_type)
- Optional persistent storage

**backend/core/semantic_search.py** (Phase 2)
- `SemanticSearchEngine` - Vector similarity-based search
- Uses embeddings + ChromaDB for semantic matching
- Returns documents ranked by cosine similarity

**backend/core/hybrid_search.py** (Phase 2)
- `HybridSearchEngine` - Combines keyword and semantic search
- Three modes: keyword, semantic, hybrid
- Configurable semantic weight (0-1) for hybrid mode
- Graceful fallback if semantic search unavailable

**backend/mcp_server/tools.py**
- Defines 5 MCP tools exposed to LLMs
- Uses FastMCP decorators for tool registration
- Tools: search_documentation, get_document, list_products, list_components, get_index_status
- Works transparently with all search modes

**backend/main.py**
- FastAPI app with HTTP/SSE transport for MCP
- Startup: builds index, initializes embeddings if enabled, creates hybrid search engine
- Graceful fallback to keyword-only if RAG initialization fails
- Health check endpoint: `/health`

**backend/stdio_server.py**
- Alternative STDIO transport for MCP
- Same dual-mode search architecture as HTTP/SSE
- Use when MCP client doesn't support HTTP

**backend/config/settings.py**
- Loads config.json and .env
- Supports SEARCH_MODE and ENABLE_EMBEDDINGS environment variables
- Validates and sets defaults for both keyword and semantic search

## Documentation Structure

The system expects this hierarchy:
```
docs/
├── {product}/              # First directory = product name
│   ├── {component}/        # Second directory = component name
│   │   ├── *.md
│   │   ├── *.txt
│   │   └── *.docx
│   └── architecture/       # Example component
├── meetings/
│   └── {product}/
└── shared/                 # Cross-product docs
```

**Path Extraction Logic:**
- Product: First directory after docs root
- Component: Second directory
- Files without this structure are skipped with warning

## MCP Integration

### Claude Desktop Setup (HTTP/SSE)
Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "doc-search": {
      "url": "http://127.0.0.1:3001/mcp"
    }
  }
}
```

### Alternative MCP Clients (STDIO)
For clients supporting STDIO transport, configure to run:
```bash
python /path/to/backend/stdio_server.py
```

### Available MCP Tools

1. **search_documentation** - Search with filters (product, component, file_types)
   - Uses configured search mode (keyword, semantic, or hybrid)
   - Returns documents with relevance scores
2. **get_document** - Retrieve full content, optionally extract section by heading
3. **list_products** - List all products with component counts
4. **list_components** - List components for specific product with doc counts
5. **get_index_status** - Index statistics and health check (includes RAG status)

## Configuration

### config.json Structure
```json
{
  "docs": {
    "root_path": "./docs",           // Path to documentation
    "file_extensions": [".md", ".txt", ".docx"],
    "max_file_size_mb": 10,          // Skip larger files
    "watch_for_changes": true,       // Auto-update on file changes
    "index_on_startup": true         // Build index at startup
  },
  "search": {
    "max_results": 50,               // Hard limit on results
    "snippet_length": 200,           // Characters in snippets
    "context_lines": 3,              // Lines around matches
    "min_keyword_length": 2,         // Minimum keyword length
    "mode": "hybrid"                 // keyword | semantic | hybrid
  },
  "embeddings": {
    "enabled": true,                 // Enable RAG/semantic search
    "model": "all-MiniLM-L6-v2",     // Sentence transformer model
    "persist_directory": null,       // Optional persistent storage
    "semantic_weight": 0.5           // Weight for hybrid mode (0-1)
  },
  "mcp": {
    "transport": "http-sse",
    "host": "127.0.0.1",
    "port": 3001,
    "endpoint": "/mcp"
  }
}
```

### .env Overrides
```bash
DOCS_ROOT_PATH=/absolute/path/to/docs
MCP_HOST=127.0.0.1
MCP_PORT=3001
LOG_LEVEL=info  # debug, info, warning, error

# Search mode configuration (Phase 2)
SEARCH_MODE=hybrid         # keyword | semantic | hybrid
ENABLE_EMBEDDINGS=true     # Enable/disable RAG functionality
```

### Search Modes

**Keyword Mode** (`SEARCH_MODE=keyword`):
- Pure keyword-based search (Phase 1 behavior)
- Fast, deterministic, exact keyword matching
- Best for specific term searches

**Semantic Mode** (`SEARCH_MODE=semantic`):
- Pure vector similarity search
- Understands conceptual similarity, handles vague queries
- Better for "find documentation about X concept"

**Hybrid Mode** (`SEARCH_MODE=hybrid`, default):
- Combines keyword and semantic results
- Formula: `score = (1-weight) * keyword + weight * semantic`
- Best of both worlds: handles both exact and conceptual searches
- Configurable `semantic_weight` (default: 0.5)

## Critical Implementation Details

### Keyword Search Algorithm
```
Score per document:
- Phrase match (all keywords adjacent): +5 points
- Keyword in filename: +3 points per keyword
- Keyword in heading: +2 points per keyword
- Keyword in content: +1 point per occurrence (capped at 5 per keyword)

Final score = sum / max_possible_score (normalized to [0, 1])
```

### Semantic Search Algorithm (Phase 2)
```
1. Document Indexing:
   - Extract text content from documents
   - Generate 384-dim vector embeddings (sentence-transformers)
   - Store in ChromaDB with metadata (product, component, file_type)

2. Query Processing:
   - Convert query to same 384-dim vector
   - Perform cosine similarity search against all document vectors
   - Return top N most similar documents

3. Similarity Score:
   - Cosine similarity: 1.0 (identical) to 0.0 (unrelated)
   - Converted from distance: similarity = 1 - (distance / 2)
```

### Hybrid Search Algorithm (Phase 2)
```
1. Execute both keyword and semantic searches in parallel
2. Merge results by document ID
3. Calculate hybrid score for each document:
   hybrid_score = (1 - semantic_weight) * keyword_score + semantic_weight * semantic_score

4. Sort by hybrid_score descending
5. Return top N results

Default semantic_weight = 0.5 (equal weighting)
```

### RAG Components Initialization
- **Lazy Loading**: Embeddings only loaded if `embeddings.enabled = true`
- **Graceful Fallback**: If RAG initialization fails, falls back to keyword-only
- **Model Download**: First run downloads ~80MB sentence-transformer model
- **Memory Usage**: ~4KB per document for embeddings (in-memory mode)
- **Persistent Storage**: Optional ChromaDB persistence to disk

### File Watching Behavior
- Uses `watchdog` library for cross-platform monitoring
- Triggers on: file created, modified, deleted
- Debounces: 1 second delay to batch rapid changes
- Can be disabled: set `watch_for_changes: false` in config

### Parser Error Handling
- **Encoding:** Tries UTF-8 first, falls back to chardet detection
- **DOCX tables:** Converted to pipe-separated text format
- **Large files:** Skipped with warning if > max_file_size_mb
- **Missing product/component:** File skipped, warning logged

### Path Handling
- **Always use `pathlib.Path`** for cross-platform compatibility
- **Relative paths:** Calculated relative to docs_root from config
- **Absolute paths:** Config should specify absolute paths to avoid ambiguity

## Development Workflows

### Adding New File Type
1. Create parser class in `backend/core/parsers.py` extending `Parser`
2. Implement `parse(file_path: str) -> Dict[str, Any]` method
3. Register in `FileIndexer.__init__`: `self.parsers['.ext'] = NewParser()`
4. Add extension to config.json `file_extensions` list

### Adding New MCP Tool
1. Add tool function in `backend/mcp_server/tools.py`
2. Use `@mcp.tool()` decorator (FastMCP)
3. Add type hints for parameters (auto-generates schema)
4. Return dictionary (auto-serialized to JSON)
5. Add docstring (visible to LLMs as tool description)

### Debugging Index Issues
```bash
# Check index diagnostics
./scripts/check_index.sh

# Check logs
tail -f mcp_server.log | grep -i index

# Verify server health
curl http://127.0.0.1:3001/health
```

## Testing Strategy

### Test Organization
- **Unit tests:** `backend/tests/test_*.py` - Test components in isolation
- **Fixtures:** `backend/tests/conftest.py` - Shared test data and temp directories
- **Coverage target:** >80% for core/, mcp_server/

### Test File Structure
```
backend/tests/
├── conftest.py           # Pytest fixtures (temp docs, sample data)
├── test_parsers.py       # Parser unit tests (3 tests)
├── test_indexer.py       # Indexer unit tests (6 tests)
└── test_search.py        # Search engine unit tests (9 tests)
```

### Running Specific Tests
```bash
# Run only parser tests
pytest backend/tests/test_parsers.py -v

# Run only one test function
pytest backend/tests/test_search.py::test_keyword_search -v

# Run tests matching pattern
pytest backend/tests/ -k "test_index" -v
```

## Common Pitfalls

### MCP Transport Selection
- **HTTP/SSE:** Required for Claude Desktop (doesn't support STDIO)
- **STDIO:** Simpler for CLI-based MCP clients, but Claude Desktop won't work
- **Port conflicts:** Check port 3001 is available before starting HTTP server
- **CORS:** Not needed for MCP server (unlike REST APIs)

### Index Not Updating
- Check `watch_for_changes: true` in config
- Verify file permissions (files must be readable)
- Check logs for watchdog errors
- Restart server if file watcher crashes

### Files Not Indexed
- Verify product/component directory structure
- Check file extension is in config's `file_extensions` list
- Ensure file size < `max_file_size_mb`
- Check encoding (parser will warn about encoding issues)

### Search Returns No Results
- Keywords must be >= `min_keyword_length` (default: 2 chars)
- Check product/component filters aren't too restrictive
- Verify documents were actually indexed (check logs or get_index_status)
- Try broader search terms (search is keyword-based, not semantic)

## Performance Characteristics

### Indexing
- **Speed:** <10 seconds for 500 documents (single-threaded)
- **Memory:** ~2MB per 100 documents in keyword index
- **Scaling:** In-memory index works well up to ~5000 documents

### Search
- **Keyword search:** <100ms for typical queries
- **Index lookup:** O(1) for product/component filtering
- **Snippet extraction:** O(n) where n = document length

### Future Optimizations (Phase 2+)
- Parallel file parsing during indexing
- Persistent index storage (SQLite/Postgres) for >5000 docs
- Vector embeddings for semantic search
- Query result caching
