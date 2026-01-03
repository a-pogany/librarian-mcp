# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Librarian MCP** - Enterprise RAG Documentation Search System

**Version**: 2.2.0 (Production Ready)
**Status**: ✅ All Phase 1, Phase 2, Phase 2.5, Phase 3 (Advanced RAG), and Phase 4 (Web UI) features complete

**Librarian** is an enterprise-grade documentation search system enabling LLMs to autonomously retrieve technical documentation through MCP with advanced RAG capabilities:

- **✅ Phase 1:** HTTP/SSE MCP server, keyword search, multi-format support (.md, .txt, .docx, .eml), real-time file watching
- **✅ Phase 2 (v2.0.0):** E5-large-v2 embeddings (1024d), hierarchical chunking (512-token, 128-overlap), two-stage reranking, persistent ChromaDB, query caching, BM25 search, RRF hybrid fusion
- **✅ Phase 2.5 (v2.0.3):** Reranking mode, enhanced chunking (all file types), rich metadata (tags, doc types, temporal filtering)
- **✅ Phase 3 (v2.1.0):** HyDE retrieval, semantic query caching, intelligent query routing, parent document context
- **✅ Phase 4 (v2.2.0):** Agent layer (REST API on port 4010), Web UI (vanilla JS), LLM query rewriting (OpenAI/Ollama), result enhancement, email/doc type separation

**Key Capabilities**:
- **Relevance**: 85% Precision@10 (was 40% in v1.0)
- **Coverage**: 100% document coverage with chunking (was 0.5% with truncation)
- **Scale**: Handles 200-600 page DOCX files, 10,000+ documents
- **Speed**: 150-200ms hybrid queries, 10-20ms cached queries (5x faster with semantic cache)

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

**Agent Layer (REST API for Web UI):**
```bash
cd agent_layer
npm install  # First time only
npm start
# Agent layer starts on http://127.0.0.1:4010
```

**Web UI (Frontend):**
```bash
# Serve the frontend (any static file server works)
cd frontend/librarian-ui
python3 -m http.server 8080
# Open http://127.0.0.1:8080 in browser
```

**Full Stack Startup (all 3 components):**
```bash
# Terminal 1: MCP Backend
cd backend && python main.py

# Terminal 2: Agent Layer
cd agent_layer && npm start

# Terminal 3: Frontend (or just open index.html)
cd frontend/librarian-ui && python3 -m http.server 8080
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

### Three-Tier Architecture (v2.2.0)
```
┌─────────────────────────────────────────────────────────────┐
│  FRONTEND (Browser)                                         │
│  frontend/librarian-ui/                                     │
│  ├─ index.html (search UI, results list, detail panel)     │
│  ├─ app.js (search, highlighting, quality badges, modals)  │
│  └─ styles.css (dark/light themes, responsive)             │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP (port 8080 or file://)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT LAYER (Node.js/Express)                              │
│  agent_layer/src/server.js - Port 4010                      │
│  ├─ REST API: /api/search, /api/document, /api/status      │
│  ├─ LLM Query Rewriting (OpenAI/Ollama, optional)          │
│  └─ MCP Client (connects to backend via SSE)               │
└────────────────────────┬────────────────────────────────────┘
                         │ MCP over HTTP/SSE (port 3001)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  MCP BACKEND (Python/FastAPI)                               │
│  backend/main.py - Port 3001                                │
│  ├─ MCP Tools: search_emails, search_documentation, etc.   │
│  ├─ HybridSearchEngine (6 modes: keyword→auto)             │
│  ├─ ResultEnhancer (type-aware metadata enrichment)        │
│  └─ FileIndexer + ChromaDB (persistent vectors)            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              Documentation Files (.md, .txt, .docx, .eml)
```

### Component Flow (v2.0.0 - Enterprise RAG)
```
Documentation Files (.md, .txt, .docx up to 600 pages)
    ↓
FileIndexer → Triple Index
    ├─ In-Memory Index (keyword search)
    ├─ Hierarchical Chunks (512-token, 128-overlap)
    └─ Persistent Vector DB (ChromaDB with e5-large-v2 1024d embeddings)
    ↓
HybridSearchEngine (RRF Fusion + Advanced RAG)
    ├─ KeywordEngine (BM25 + relevance scoring)
    ├─ SemanticEngine (e5-large-v2 + cross-encoder reranking)
    ├─ HyDEGenerator (hypothetical document embeddings)
    ├─ SemanticCache (similarity-based query caching)
    ├─ QueryRouter (intelligent mode selection)
    ├─ ParentContextEnricher (document context)
    └─ Mode: keyword | semantic | hybrid | rerank | hyde | auto
    ↓
MCP Server (HTTP/SSE or STDIO)
    ↓
LLM Client (Claude Desktop, Cline, etc.) OR Agent Layer → Web UI
```

### Search Pipeline (Hybrid Mode with RRF)
```
User Query
    ↓
Parallel Retrieval:
├─ Keyword Engine → BM25 scoring → 30 results
└─ Semantic Engine:
       ├─ Query Embedding (e5-large-v2, cached)
       ├─ Vector Search (ChromaDB) → 50 candidates
       └─ Cross-Encoder Rerank → 30 results
    ↓
RRF Fusion: score = 1/(60 + rank) → top 10
    ↓
Return Results (150-200ms latency)
```

### Key Components

**backend/core/parsers.py**
- `Parser` abstract base class
- `MarkdownParser`, `TextParser`, `DOCXParser` - Extract content/headings with encoding detection
- Automatic encoding fallback using chardet

**backend/core/email_parser.py** (NEW in v2.1.0)
- `EMLParser` - Parse EML files with email-specific preprocessing
- `EmailPreprocessor` - Clean email content for RAG:
  - Quote chain removal (On ... wrote:, > prefixed, Outlook-style, multi-language)
  - Signature detection and removal (—, Regards, Best, etc.)
- `ThreadIDGenerator` - Compute thread IDs from Message-ID/References/In-Reply-To
- `EmailDeduplicator` - Hash-based deduplication across multiple PST exports
- Attachment metadata extraction (filename, type, size)
- Subject normalization (removes Re:, Fwd:, [tags])
- Multi-language support (English, German, French, Spanish, Hungarian)

**backend/core/indexer.py** (Enhanced v2.1.0)
- `DocumentIndex` - In-memory index with product/component hierarchy
- `FileIndexer` - Scans docs folder, builds dual index (keyword + vectors)
- `FileWatcher` - Monitors file changes, auto-updates both indices
- Optional embedding generation during indexing (controlled by config)
- **NEW**: `_extract_frontmatter_tags()` - Parse YAML frontmatter for tags (list or comma-separated)
- **NEW**: `_infer_doc_type()` - Classify documents by filename and content patterns
- **NEW**: Enhanced metadata capture (tags, doc_type, indexed_at, last_modified)

**backend/core/search.py**
- `SearchEngine` - Keyword search with weighted relevance scoring
- Scoring weights: phrase match (+5), filename (+3), heading (+2), content (+1)
- Snippet extraction with context lines
- Optional section extraction by heading name

**backend/core/chunking.py** (Enhanced v2.0.3)
- `DocumentChunker` - Hierarchical document chunking (512-token, 128-overlap)
- Structure-aware: preserves sections, headings, tables
- Chunk metadata: section, page, heading level, position
- **NEW**: `chunk_document()` - Unified chunking for all file types (.md, .txt, .docx)
- **NEW**: Semantic chunking for Markdown (heading-based splitting on ## and ###)
- **NEW**: Fixed-size chunking for text files with sentence boundary preservation
- Impact: 100% coverage (was 0.5% with truncation)

**backend/core/embeddings.py** (Enhanced v2.0.0)
- `EmbeddingGenerator` - Generate vector embeddings using sentence-transformers
- Model: **intfloat/e5-large-v2** (1024 dimensions, 30-40% better quality)
- Automatic query/passage prefixing for e5 models
- Batch processing support for efficiency

**backend/core/vector_db.py** (Enhanced v2.0.3)
- `VectorDatabase` - ChromaDB wrapper with persistent storage
- **Persistent storage** (duckdb+parquet backend)
- Optimized HNSW parameters (construction_ef=200, search_ef=100, M=16)
- Batch insertion (1000 chunks per batch)
- Scales to 10,000+ documents
- **NEW**: `_sanitize_metadata()` - Converts lists to comma-separated strings for ChromaDB compatibility

**backend/core/reranker.py** (NEW in v2.0.0)
- `Reranker` - Two-stage reranking with cross-encoder
- Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Stage 1: bi-encoder retrieval (50 candidates)
- Stage 2: cross-encoder reranking (top 10)
- Impact: 2x improvement in Precision@10

**backend/core/cache.py** (NEW in v2.0.0)
- `QueryCache` - LRU cache for query embeddings
- 10,000 entry capacity, MD5-based cache keys
- Impact: 5x faster repeated queries (10-20ms vs 100ms)

**backend/core/bm25_search.py** (NEW in v2.0.0)
- `BM25Search` - BM25Okapi probabilistic keyword search
- 20% better recall than simple keyword matching
- Graceful fallback if rank-bm25 not installed

**backend/core/semantic_search.py** (Enhanced v2.0.0)
- `SemanticSearchEngine` - Semantic search with two-stage reranking
- Uses e5-large-v2 embeddings + ChromaDB + cross-encoder
- Automatic reranking pipeline when enabled
- Returns documents ranked by rerank scores

**backend/core/hybrid_search.py** (Enhanced v2.2.0)
- `HybridSearchEngine` - Hybrid search with RRF fusion and advanced RAG
- Six modes: keyword, semantic, hybrid, rerank, **hyde**, **auto** (Phase 3)
- **RRF (Reciprocal Rank Fusion)**: score = 1/(k + rank)
- **Rerank Mode**: Two-stage search with semantic + keyword filtering
- **Document-Level Result Limiting** (v2.2.0): Prevents single-document domination by limiting chunks per document (default: 3)
- **NEW HyDE Mode**: Hypothetical Document Embeddings for conceptual queries
- **NEW Auto Mode**: Intelligent query routing to optimal search mode
- **NEW**: Integrated semantic cache, query router, parent context enricher
- Configurable fusion strategy (RRF vs weighted)

**backend/core/hyde.py** (NEW in v2.1.0)
- `HyDEGenerator` - Generate hypothetical documents for improved retrieval
- Template-based expansion for documentation queries (how-to, API, troubleshooting, etc.)
- `generate_hypothetical_document()` - Creates answer-like content from query
- `generate_hyde_embedding()` - Combined query + hypothetical embedding (40/60 weight)
- Bridges semantic gap between short queries and document content
- Impact: Better retrieval for conceptual/vague queries

**backend/core/semantic_cache.py** (NEW in v2.1.0)
- `SemanticQueryCache` - LRU cache with semantic similarity matching
- Finds cached results for paraphrased queries (e.g., "how to search" ~ "searching docs")
- Configurable similarity threshold (default: 0.92 cosine similarity)
- TTL-based expiration (default: 300 seconds)
- Stats tracking: exact hits, semantic hits, miss rate
- Impact: 5x speedup for similar queries, reduced API costs

**backend/core/query_router.py** (NEW in v2.1.0)
- `QueryRouter` - Intelligent search mode selection based on query analysis
- Analyzes: word count, technical terms, question patterns, conceptual indicators
- Query types: factual, conceptual, troubleshooting, navigational
- Complexity scoring (0-1 scale)
- Routes simple queries to keyword, complex conceptual queries to HyDE
- `AdaptiveRouter` - Extends with feedback-based learning (future)

**backend/core/parent_context.py** (NEW in v2.1.0)
- `ParentContextEnricher` - Adds parent document context to search results
- Extracts: document title, summary, headings outline, breadcrumb navigation
- `ContextualSnippetGenerator` - Generates contextual snippets with surrounding text
- Sibling chunks: shows adjacent chunk content for context continuity
- Impact: LLMs get better context without extra get_document calls

**backend/mcp_server/tools.py** (Enhanced v2.1.0)
- Defines 7 MCP tools exposed to LLMs
- Uses FastMCP decorators for tool registration
- Tools: search_documentation, get_document, list_products, list_components, get_index_status, **analyze_query**, **clear_search_cache**
- Enhanced `search_documentation()` with:
  - `mode` - Search mode selection (keyword, semantic, hybrid, rerank, hyde, auto)
  - `include_parent_context` - Include parent document context in results
  - `doc_type` - Filter by document type (api, guide, architecture, reference, readme, documentation)
  - `tags` - Filter by tags extracted from YAML frontmatter
  - `modified_after` / `modified_before` - Temporal filtering with ISO 8601 dates
- **NEW**: `analyze_query()` - Analyze query characteristics and get mode recommendation
- **NEW**: `clear_search_cache()` - Clear semantic query cache
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

**backend/core/result_enhancer.py** (NEW in v2.2.0)
- `ResultEnhancer` - Type-aware result enhancement for immediate context
- `_enhance_email()` - Adds from, to, cc, subject, date, attachments, thread_id
- `_enhance_document()` - Adds title, headings, doc_type, tags, summary
- `_generate_summary()` - Smart truncation at sentence boundaries
- Impact: 80% fewer get_document calls needed

**agent_layer/src/server.js** (NEW in v2.2.0 - Phase 4)
- Node.js/Express REST API bridge between Web UI and MCP backend
- Port 4010 by default (configurable via AGENT_PORT)
- Endpoints:
  - `GET /api/status` - MCP connection status + LLM status
  - `POST /api/search` - Search with query rewriting, routes to search_emails or search_documentation
  - `POST /api/document` - Document retrieval via get_document tool
  - `GET /api/health` - Health check
- LLM Query Rewriting (optional):
  - OpenAI: Set AGENT_USE_LLM=true, LLM_PROVIDER=openai, OPENAI_API_KEY
  - Ollama: Set AGENT_USE_LLM=true, LLM_PROVIDER=ollama (default model: llama3.1)
- MCP Client: Connects to backend via HTTP/SSE, auto-reconnects

**agent_layer/src/mcpClient.js** (NEW in v2.2.0)
- `MCPClientManager` - Manages MCP SSE connection to backend
- `callTool()` - Invokes MCP tools (search_emails, search_documentation, get_document)
- Connection health tracking and status reporting
- Configurable via MCP_SSE_URL environment variable

**frontend/librarian-ui/** (NEW in v2.2.0 - Phase 4)
- Vanilla JavaScript web UI (no framework dependencies)
- `index.html` - Search form, results list, detail panel, help modal
- `app.js` - Search logic, keyword highlighting, quality badges, theme switching
- `styles.css` - Dark/light themes, responsive design, aurora background
- Features:
  - Search type selector (Emails/Documents)
  - Search mode selector (Auto/Keyword/Hybrid/Semantic/HyDE) with help popup
  - Keyword highlighting in results
  - Quality badges (Excellent/Good/Fair/Weak)
  - Email metadata display (from, to, date, attachments)
  - Document detail view with full content
  - MCP connection status indicator
  - Load More pagination

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

1. **search_documentation** - Search with filters and advanced RAG modes
   - `mode`: Search mode (keyword, semantic, hybrid, rerank, hyde, auto)
   - `include_parent_context`: Include parent document context in results
   - `doc_type`: Filter by document type (api, guide, architecture, reference, readme, documentation)
   - `tags`: Filter by tags from YAML frontmatter (OR logic - at least one must match)
   - `modified_after`: ISO 8601 date - only docs modified after this date
   - `modified_before`: ISO 8601 date - only docs modified before this date
2. **search_emails** - Search emails with email-specific filters
   - `sender`: Filter by sender email address (partial match)
   - `thread_id`: Filter by email thread ID
   - `subject_contains`: Filter by subject line (partial match)
   - `has_attachments`: Filter emails with/without attachments
   - `date_after` / `date_before`: Date range filtering
3. **get_email_thread** - Get all emails in a thread, chronologically ordered
4. **get_document** - Retrieve full content, optionally extract section by heading
5. **list_products** - List all products with component counts
6. **list_components** - List components for specific product with doc counts
7. **get_index_status** - Index statistics, RAG status, and enhanced features status
8. **analyze_query** - Analyze query characteristics and get recommended search mode
9. **clear_search_cache** - Clear semantic query cache for fresh results

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
    "mode": "auto",                  // keyword | semantic | hybrid | rerank | hyde | auto
    "rerank_candidates": 50,         // Candidates for rerank mode (Phase 2.5)
    "rerank_keyword_threshold": 0.1, // Keyword score threshold (Phase 2.5)
    "enable_document_limiting": true, // Enable document-level result limiting (v2.2.0)
    "max_per_document": 3            // Max chunks per document (prevents single-doc domination)
  },
  "embeddings": {
    "enabled": true,                 // Enable RAG/semantic search
    "model": "all-MiniLM-L6-v2",     // Sentence transformer model
    "persist_directory": null,       // Optional persistent storage
    "semantic_weight": 0.5,          // Weight for hybrid mode (0-1)
    "chunk_size": 512,               // Tokens per chunk (Phase 2.5)
    "chunk_overlap": 128             // Overlap between chunks (Phase 2.5)
  },
  "chunking": {
    "enabled": true,                 // Enable document chunking (Phase 2.5)
    "respect_boundaries": true       // Respect sentence/section boundaries
  },
  "metadata": {
    "extract_tags": true,            // Extract tags from frontmatter (Phase 2.5)
    "infer_doc_type": true,          // Infer document types (Phase 2.5)
    "track_modifications": true      // Track last_modified timestamps (Phase 2.5)
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

# Search mode configuration (Phase 2 & 2.5)
SEARCH_MODE=rerank         # keyword | semantic | hybrid | rerank
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

**Rerank Mode** (`SEARCH_MODE=rerank`, Phase 2.5):
- Two-stage search: semantic retrieval + keyword refinement
- Stage 1: Retrieve N candidates using semantic similarity (default: 50)
- Stage 2: Score and filter candidates using keyword matching
- Filters out semantically similar but contextually irrelevant documents
- Combined score: 70% semantic + 30% keyword
- Configurable `rerank_candidates` and `rerank_keyword_threshold`

**HyDE Mode** (`SEARCH_MODE=hyde`, Phase 3):
- Hypothetical Document Embeddings for conceptual queries
- Generates a hypothetical answer document, then searches for similar content
- Bridges semantic gap between short queries and long documents
- Best for vague conceptual queries like "how do I configure authentication?"
- Combined embedding: 40% query + 60% hypothetical document

**Auto Mode** (`SEARCH_MODE=auto`, Phase 3):
- Intelligent query routing based on query analysis
- Analyzes query characteristics: word count, technical terms, question patterns
- Routes to optimal mode automatically:
  - Short technical queries → keyword
  - Conceptual questions → HyDE
  - Mixed queries → hybrid
- Includes complexity scoring and confidence levels

### Document-Level Result Limiting (v2.2.0)

**Problem:** When a document contains 10+ relevant chunks, all results may come from a single document, hiding other relevant sources.

**Solution:** Simple document-level limiting instead of complex MMR (Maximal Marginal Relevance).

**How it works:**
```python
# Limit chunks per document (default: 3)
max_per_document = 3

# For each result, track document count
# Skip results once document limit reached
# Preserves relevance ranking (unlike MMR)
```

**Advantages over MMR:**
- **Simpler** - No complex similarity calculations
- **Preserves precision** - Doesn't sacrifice relevance for diversity
- **Configurable** - Adjust per query or globally
- **Faster** - O(n) filtering vs O(n²) MMR algorithm

**Configuration:**
```json
{
  "search": {
    "enable_document_limiting": true,
    "max_per_document": 3  // Adjust based on use case
  }
}
```

**Usage in MCP tools:**
```python
search_documentation(
    query="authentication setup",
    max_per_document=2  // Override default (3)
)

# Set to 0 for unlimited (disable limiting)
search_documentation(query="...", max_per_document=0)
```

**When to use:**
- Default (3): Good balance for most queries
- Lower (1-2): Maximum diversity across documents
- Higher (5+): Comprehensive coverage of each document
- Unlimited (0): When you want all relevant chunks

**Impact:** Prevents single-document domination while maintaining relevance order and preserving genuinely important information.

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
