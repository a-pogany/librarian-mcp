# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Librarian** is a 3-phase documentation search system that makes technical documentation accessible to LLMs and humans:

- **Phase 1:** HTTP/SSE MCP server with keyword search
- **Phase 2:** RAG-enhanced semantic search with vector embeddings
- **Phase 3:** Web UI for human access

**Core Purpose:** Enable LLMs (Claude, GPT, local models) to autonomously retrieve documentation during architecture/design discussions.

## Development Commands

### Python Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r backend/requirements.txt

# Install dev dependencies
pip install -r backend/requirements-dev.txt  # if exists
```

### Testing
```bash
# Run all tests
pytest backend/tests/

# Run specific test file
pytest backend/tests/test_indexer.py

# Run with coverage
pytest backend/tests/ --cov=backend/core --cov-report=html

# Run specific test function
pytest backend/tests/test_indexer.py::test_build_index -v
```

### Running the System

**Phase 1 (MCP Server):**
```bash
# Start MCP server
cd backend
python main.py

# Or with uvicorn directly
uvicorn main:app --host 127.0.0.1 --port 3001 --reload
```

**Phase 3 (Full System with Web UI):**
```bash
# Start backend API
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Start frontend (separate terminal)
cd frontend
npm run dev

# Use management scripts (once implemented)
./scripts/start.sh    # Start entire system
./scripts/stop.sh     # Stop all services
./scripts/status.sh   # Check health
```

### Linting and Code Quality
```bash
# Format code
black backend/

# Type checking (if using mypy)
mypy backend/

# Lint
flake8 backend/
```

## Architecture Overview

### Phase 1: MCP Server + Basic Search
```
Documentation Files
    ↓
FileIndexer → In-Memory Index (DocumentIndex)
    ↓
SearchEngine (Keyword-based)
    ↓
MCP Server (HTTP/SSE via FastMCP)
    ↓
Claude Desktop / Cline
```

**Key Components:**
- **FileIndexer**: Scans docs folder, builds in-memory index with product/component hierarchy
- **Parsers**: Extract content from .md, .txt, .docx files with encoding detection
- **SearchEngine**: Keyword search with relevance scoring (file name > headings > content)
- **MCP Server**: FastMCP-based HTTP/SSE transport exposing 5 tools to LLMs
- **FileWatcher**: Auto-updates index when files change (using watchdog)

### Phase 2: RAG Enhancement
```
Phase 1 Architecture
    +
EmbeddingGenerator (sentence-transformers)
    +
VectorDatabase (ChromaDB)
    +
HybridSearchEngine (keyword + semantic)
```

**Key Components:**
- **EmbeddingGenerator**: Converts documents to vectors using sentence-transformers
- **VectorDatabase**: ChromaDB wrapper for similarity search with cosine distance
- **HybridSearchEngine**: Combines keyword and semantic search with configurable weighting

### Phase 3: Web UI
```
React Frontend (port 3000)
    ↓ HTTP
REST API Backend (port 8000)
    ↓
Shared Search Engine
```

**Key Components:**
- **FastAPI REST API**: HTTP endpoints for search, document retrieval, product listing
- **React Frontend**: TanStack Query + Tailwind CSS for search UI
- **Dual Server**: Same backend serves both MCP (port 3001) and REST API (port 8000)

## Documentation Folder Structure

The system expects documentation in this hierarchy:
```
/docs/
├── {product-name}/           # e.g., symphony, project-x
│   ├── {component}/          # e.g., PAM, auth, database
│   │   ├── *.md
│   │   ├── *.docx
│   │   └── *.txt
│   └── architecture/
├── meetings/
│   └── {product-name}/
└── shared/                   # Cross-product docs
```

**Path Extraction:** First directory = product, second = component

## MCP Tool Definitions

The MCP server exposes these tools to Claude Desktop/Cline:

1. **search_documentation**: Keyword/semantic search with filters (product, component, file types)
2. **get_document**: Retrieve full document content by path, optionally extract specific section
3. **list_products**: Get all products with component counts
4. **list_components**: Get components for a specific product
5. **get_index_status**: Index statistics and health check

## Configuration Management

### config.json
- **docs.root_path**: Absolute path to documentation folder
- **docs.file_extensions**: Supported file types (default: .md, .txt, .docx)
- **docs.max_file_size_mb**: Skip files larger than this (default: 10MB)
- **search.max_results**: Hard limit on search results (default: 50)
- **mcp.port**: MCP server port (default: 3001)
- **embeddings.model**: Sentence transformer model (Phase 2, default: all-MiniLM-L6-v2)

### .env (for sensitive/local overrides)
```bash
DOCS_ROOT_PATH=/path/to/docs
MCP_HOST=127.0.0.1
MCP_PORT=3001
LOG_LEVEL=info
```

## Critical Implementation Details

### Search Relevance Scoring
Keyword search uses weighted scoring:
- **Phrase match in content:** +5 points
- **Keyword in filename:** +3 points per keyword
- **Keyword in heading:** +2 points per keyword
- **Keyword in content:** +1 point per occurrence (capped at 5)
- **Normalization:** Score divided by max possible, range [0, 1]

### Hybrid Search (Phase 2)
Combines keyword and semantic results:
```python
hybrid_score = (1 - weight) * keyword_score + weight * semantic_score
```
- `weight = 0`: Pure keyword search
- `weight = 1`: Pure semantic search
- `weight = 0.5`: Balanced (default)

### File Watching
- Uses `watchdog` library for cross-platform file monitoring
- Auto-updates index on file create/modify/delete
- Can be disabled via `watch_for_changes: false` in config

### Parser Error Handling
- **Encoding detection:** Uses `chardet` for non-UTF8 files
- **DOCX tables:** Formatted as pipe-separated text
- **Large files:** Skipped with warning (configurable max_file_size_mb)
- **Invalid paths:** Files without product/component structure are skipped

## Development Workflow

### Adding New File Types
1. Create new parser class in `backend/core/parsers.py` extending `Parser`
2. Implement `parse(file_path: str) -> Dict` method
3. Register parser in `FileIndexer.__init__` parsers dictionary
4. Add extension to `config.json` file_extensions list

### Adding New MCP Tools
1. Define tool function in `backend/mcp/tools.py` with `@mcp.tool()` decorator
2. Use type hints for parameters (auto-generates tool schema)
3. Return dictionary with results (auto-serialized to JSON)
4. Add comprehensive docstring (visible to LLMs as tool description)

### Testing Strategy
- **Unit tests:** Test each component in isolation (indexer, parsers, search)
- **Fixtures:** Use `pytest` fixtures for temporary documentation structures
- **Integration tests:** Test MCP tool calls end-to-end
- **Coverage target:** >80% code coverage

## Phase-Specific Guidance

### Phase 1 Development
Focus on getting MCP integration working before optimizing search:
1. Implement basic FileIndexer with .md support only
2. Get MCP server responding to Claude Desktop
3. Add remaining parsers (.txt, .docx)
4. Implement keyword search and relevance scoring
5. Add file watching last

### Phase 2 Development
Build on Phase 1 without breaking existing functionality:
1. Install sentence-transformers and ChromaDB
2. Add embeddings generation to indexer (parallel to keyword index)
3. Implement VectorDatabase wrapper
4. Create HybridSearchEngine that delegates to both engines
5. Update MCP tool to support `search_mode` parameter

### Phase 3 Development
Backend and frontend can be developed in parallel:
1. Add FastAPI REST routes alongside existing MCP server
2. Setup React project with Vite
3. Implement API service layer in frontend
4. Build UI components (SearchBar, ResultsList, DocumentViewer)
5. Add CORS configuration for local development

## Common Pitfalls

### Path Handling
- **Always use Path objects** from pathlib for cross-platform compatibility
- **Relative paths:** Always calculated relative to docs_root
- **Absolute paths:** Config should specify absolute paths to avoid confusion

### MCP Server Transport
- **HTTP/SSE required:** Claude Desktop doesn't support stdio for HTTP-based MCP
- **Port conflicts:** Check port 3001 is available before starting
- **CORS not needed:** MCP server doesn't require CORS (but REST API does)

### Vector Embeddings (Phase 2)
- **Model download:** First run downloads ~80MB model from HuggingFace
- **Memory usage:** Embeddings stored in RAM, ~4KB per document
- **Batch processing:** Use batch_size=32 for indexing to avoid OOM

### React State Management (Phase 3)
- **TanStack Query handles caching:** Don't duplicate caching logic
- **Search debouncing:** Implement debounce to avoid excessive API calls
- **Document viewing:** Fetch full content separately, don't include in search results

## Integration Testing

### Testing with Claude Desktop
1. Configure `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "doc-search": {
      "url": "http://127.0.0.1:3001/mcp"
    }
  }
}
```
2. Start MCP server
3. Restart Claude Desktop
4. Verify tools appear in Claude's available tools
5. Test with query like "Search for authentication documentation"

### Testing with Cline
Similar configuration in Cline's MCP settings with HTTP transport.

## Dependencies Rationale

### Core Dependencies
- **fastapi:** ASGI framework for both MCP and REST API
- **mcp:** Official MCP Python SDK with FastMCP helper
- **uvicorn:** ASGI server for running FastAPI

### File Processing
- **python-docx:** DOCX parsing (pure Python, cross-platform)
- **markdown:** Markdown parsing (not rendering, just structure extraction)
- **chardet:** Encoding detection for non-UTF8 files
- **watchdog:** Cross-platform file system monitoring

### Phase 2: RAG
- **sentence-transformers:** Generate embeddings (wraps PyTorch models)
- **torch:** Required by sentence-transformers
- **chromadb:** Vector database with HNSW index

### Testing
- **pytest:** Test framework with fixtures and parametrization
- **pytest-asyncio:** Async test support for FastAPI
- **httpx:** Async HTTP client for API testing
- **pytest-cov:** Coverage reporting

## Performance Considerations

### Indexing Performance
- **Target:** <10 seconds for 500 documents
- **Optimization:** Parallel file parsing (future enhancement)
- **Memory:** ~2MB per 100 documents in keyword index

### Search Performance
- **Keyword search:** <100ms for typical queries
- **Semantic search (Phase 2):** <500ms including embedding generation
- **Hybrid search:** <600ms total

### Scaling Limits (Phase 1)
- **Documents:** Tested up to 1000 documents
- **Memory:** In-memory index, ~20MB for 1000 documents
- **Future:** Move to persistent index (SQLite, Postgres) if >5000 documents
