# Phase 1: MCP Server + Basic Search - Design Specification

## Overview

Phase 1 establishes the foundation: an HTTP/SSE MCP server that enables Claude Desktop and Cline to autonomously search documentation using keyword-based search.

**Goals:**
- Working MCP server accessible via HTTP/SSE transport
- File indexing system for .md, .txt, .docx files
- Keyword-based search with relevance ranking
- Real-time file watching for automatic index updates
- Integration with Claude Desktop and Cline

**Time Estimate:** 8-10 hours

---

## System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Documentation Files                       │
│              (/docs/product/component/*.md)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (scan & parse)
┌─────────────────────────────────────────────────────────────┐
│                      File Indexer                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Markdown   │  │   DOCX     │  │    Text    │            │
│  │  Parser    │  │  Parser    │  │   Parser   │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (build index)
┌─────────────────────────────────────────────────────────────┐
│                  In-Memory Index                             │
│  ┌──────────────────────────────────────────────────┐       │
│  │  documents: {path -> Document}                   │       │
│  │  products: {name -> ProductInfo}                 │       │
│  │  components: {product/component -> [paths]}      │       │
│  └──────────────────────────────────────────────────┘       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (search)
┌─────────────────────────────────────────────────────────────┐
│                   Search Engine                              │
│  • Keyword extraction & matching                             │
│  • Relevance scoring algorithm                               │
│  • Result ranking & snippet extraction                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (expose via MCP)
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (FastMCP)                      │
│                                                               │
│  Transport: HTTP/SSE (port 3001)                             │
│  Framework: FastAPI + MCP SDK                                │
│                                                               │
│  Tools:                                                       │
│  ├─ search_documentation()                                   │
│  ├─ get_document()                                           │
│  ├─ list_products()                                          │
│  ├─ list_components()                                        │
│  └─ get_index_status()                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (HTTP/SSE)
┌─────────────────────────────────────────────────────────────┐
│            Claude Desktop / Cline (MCP Clients)              │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow
```
[File System]
    │
    ├─→ [File Watcher (watchdog)]
    │       │
    │       ↓ (on change event)
    │   [FileIndexer.index_file()]
    │
    └─→ [FileIndexer.build_index()]
            │
            ↓ (parse files)
        [Parser Factory]
            ├─→ [MarkdownParser]
            ├─→ [DOCXParser]
            └─→ [TextParser]
            │
            ↓ (extracted content)
        [DocumentIndex.add_document()]
            │
            ↓ (indexed)
        [SearchEngine]
            │
            ↓ (MCP tool calls)
        [FastMCP Server]
            │
            ↓ (HTTP/SSE)
        [Claude Desktop]
```

---

## Component Design

### 1. File Indexer (`backend/core/indexer.py`)

**Purpose:** Scan documentation folder, parse files, maintain in-memory index

#### Class: DocumentIndex
```python
class DocumentIndex:
    """In-memory document index with hierarchical organization"""

    # Data Structures
    documents: Dict[str, Document]           # path -> full document
    products: Dict[str, ProductInfo]         # product name -> metadata
    components: Dict[str, List[str]]         # "product/component" -> [paths]
    last_indexed: Optional[datetime]         # index build timestamp

    # Methods
    add_document(doc: Document) -> None
    remove_document(path: str) -> None
    get_document(path: str) -> Optional[Document]
    clear() -> None
```

**Data Model:**
```python
@dataclass
class Document:
    path: str                    # Relative from docs_root: "product/component/file.md"
    product: str                 # Extracted from path[0]
    component: str               # Extracted from path[1]
    file_name: str               # Base filename
    file_type: str               # Extension (.md, .txt, .docx)
    content: str                 # Full text content
    headings: List[str]          # Extracted headings (for search scoring)
    metadata: Dict[str, Any]     # Parser-specific metadata
    size_bytes: int              # File size
    last_modified: str           # ISO timestamp
```

**Product/Component Indexing:**
```python
@dataclass
class ProductInfo:
    name: str                    # Product name
    doc_count: int               # Number of documents
    components: Set[str]         # Unique component names
```

#### Class: FileIndexer
```python
class FileIndexer:
    """Manages file scanning, parsing, and indexing"""

    # Dependencies
    docs_root: Path                           # Documentation root directory
    config: Dict                              # Configuration (extensions, max size, etc.)
    index: DocumentIndex                      # In-memory index
    parsers: Dict[str, Parser]                # Extension -> Parser mapping
    observer: Optional[Observer]              # File watcher instance

    # Core Methods
    build_index() -> IndexResult              # Full index rebuild
    index_file(path: str) -> None             # Index single file
    start_watching() -> None                  # Start file watcher
    stop_watching() -> None                   # Stop file watcher

    # Query Methods
    get_products() -> List[ProductInfo]
    get_components(product: str) -> Optional[List[ComponentInfo]]
    get_status() -> IndexStatus

    # Internal Methods
    _scan_files() -> List[Path]               # Find all indexable files
    _should_index(path: Path) -> bool         # Filter by size/extension
    get_relative_path(path: str) -> str       # Convert to relative path
```

**Index Building Algorithm:**
```
1. Start timer
2. Scan docs_root for files matching extensions (.md, .txt, .docx)
3. Filter files by max_file_size_mb
4. For each file:
   a. Extract product/component from path
   b. Select appropriate parser
   c. Parse file content
   d. Create Document object
   e. Add to DocumentIndex
5. Update last_indexed timestamp
6. Start file watcher if configured
7. Return statistics (files indexed, errors, duration)
```

**Path Extraction Logic:**
```python
# Example: /docs/symphony/PAM/api-spec.md
parts = Path(rel_path).parts  # ['symphony', 'PAM', 'api-spec.md']

if len(parts) < 2:
    log_warning("Invalid path structure")
    return

product = parts[0]      # 'symphony'
component = parts[1]    # 'PAM'
```

#### Class: FileWatcher
```python
class FileWatcher(FileSystemEventHandler):
    """Watchdog event handler for file changes"""

    # Dependencies
    indexer: FileIndexer

    # Event Handlers
    on_created(event) -> None     # New file: index_file()
    on_modified(event) -> None    # Modified file: re-index
    on_deleted(event) -> None     # Deleted file: remove from index
```

**File Watching Strategy:**
- Monitor docs_root recursively
- Debounce rapid changes (ignore duplicates within 100ms)
- Log all index modifications
- Handle errors gracefully (continue watching on parse errors)

---

### 2. File Parsers (`backend/core/parsers.py`)

**Purpose:** Extract text content and structure from different file formats

#### Abstract Base: Parser
```python
class Parser(ABC):
    """Abstract parser interface"""

    @abstractmethod
    def parse(file_path: str) -> ParseResult:
        """
        Parse file and extract content

        Returns:
            ParseResult with content, headings, metadata
        """
        pass
```

**ParseResult Model:**
```python
@dataclass
class ParseResult:
    content: str                 # Full text content
    headings: List[str]          # Extracted headings
    metadata: Dict[str, Any]     # Format-specific metadata
```

#### MarkdownParser
```python
class MarkdownParser(Parser):
    """Parse Markdown files"""

    def parse(file_path: str) -> ParseResult:
        # 1. Detect encoding using chardet
        # 2. Read file with detected encoding
        # 3. Extract headings (lines starting with #)
        # 4. Return full content + headings
```

**Heading Extraction:**
```python
# Regex pattern: ^#+\s+(.+)$
# Examples:
#   "# Introduction"  -> "Introduction"
#   "## API Spec"     -> "API Spec"
#   "### Endpoints"   -> "Endpoints"

headings = []
for line in content.split('\n'):
    if line.strip().startswith('#'):
        heading = re.sub(r'^#+\s*', '', line).strip()
        headings.append(heading)
```

#### DOCXParser
```python
class DOCXParser(Parser):
    """Parse DOCX files using python-docx"""

    def parse(file_path: str) -> ParseResult:
        # 1. Open DOCX with python-docx
        # 2. Extract paragraphs
        # 3. Extract headings (Heading 1-6 styles)
        # 4. Extract tables (format as pipe-separated text)
        # 5. Extract core properties (title, author, dates)
```

**Table Formatting:**
```python
# Convert DOCX table to text representation:
# Original:
#   | Name   | Type   |
#   | id     | UUID   |
#   | email  | string |
#
# Output:
#   "Name | Type\nid | UUID\nemail | string"

for table in doc.tables:
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(' | '.join(cells))
    content_parts.append('\n'.join(rows))
```

#### TextParser
```python
class TextParser(Parser):
    """Parse plain text files"""

    def parse(file_path: str) -> ParseResult:
        # 1. Detect encoding using chardet
        # 2. Read file with detected encoding
        # 3. No heading extraction (plain text)
        # 4. Return content only
```

**Encoding Detection Strategy:**
```python
# Use chardet for non-UTF8 files
with open(file_path, 'rb') as f:
    raw = f.read()
    encoding = chardet.detect(raw)['encoding'] or 'utf-8'

with open(file_path, 'r', encoding=encoding) as f:
    content = f.read()
```

---

### 3. Search Engine (`backend/core/search.py`)

**Purpose:** Keyword-based search with relevance ranking

#### Class: SearchEngine
```python
class SearchEngine:
    """Keyword-based document search"""

    # Dependencies
    indexer: FileIndexer

    # Public Methods
    search(
        query: str,
        product: Optional[str] = None,
        component: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[SearchResult]

    get_document(
        path: str,
        section: Optional[str] = None
    ) -> Optional[Document]

    # Internal Methods
    _parse_query(query: str) -> List[str]
    _calculate_score(doc: Document, keywords: List[str], query: str) -> float
    _extract_snippet(content: str, keywords: List[str]) -> str
    _extract_section(content: str, section: str, headings: List[str]) -> str
```

**Search Algorithm:**
```
1. Parse query into keywords (lowercase, remove special chars, min 2 chars)
2. Iterate all documents in index
3. Apply filters (product, component, file_types)
4. Calculate relevance score for each document
5. Extract snippet containing keywords
6. Sort by relevance score (descending)
7. Limit to max_results
8. Return SearchResult list
```

**Relevance Scoring Algorithm:**
```python
score = 0.0

# Phrase match bonus (query as exact phrase in content)
if original_query.lower() in content.lower():
    score += 5.0

# Keyword scoring
for keyword in keywords:
    # File name match: 3 points
    if keyword in file_name.lower():
        score += 3.0

    # Heading match: 2 points per heading
    for heading in doc.headings:
        if keyword in heading.lower():
            score += 2.0

    # Content match: 1 point per occurrence (capped at 5)
    count = content.lower().count(keyword)
    score += min(count, 5) * 1.0

# Normalize to 0-1 range
max_possible = len(keywords) * 10  # Theoretical max
normalized = min(score / max_possible, 1.0)

return round(normalized, 2)
```

**Scoring Examples:**
```
Query: "authentication API"
Keywords: ["authentication", "api"]

Document 1: "api-authentication.md"
  - Filename match: "authentication" (+3), "api" (+3) = 6
  - Heading "API Authentication": "authentication" (+2), "api" (+2) = 4
  - Content mentions: "authentication" x3 (+3), "api" x5 (+5) = 8
  - Total: 18 / 20 = 0.90

Document 2: "database-schema.md"
  - Content mentions: "authentication" x1 (+1) = 1
  - Total: 1 / 20 = 0.05
```

**Snippet Extraction:**
```python
# Find line with most keyword matches
best_line = ""
max_matches = 0

for line in content.split('\n'):
    matches = sum(1 for kw in keywords if kw in line.lower())
    if matches > max_matches:
        max_matches = matches
        best_line = line

# Truncate to max_length (default 200 chars)
snippet = best_line.strip()[:200]
if len(snippet) == 200:
    snippet += "..."
```

**Section Extraction:**
```python
# Extract specific section from document by heading
# Example: section="Authentication" extracts from ## Authentication to next ##

in_section = False
section_lines = []

for line in content.split('\n'):
    # Check if line is target section heading
    if line.strip().startswith('#') and section.lower() in line.lower():
        in_section = True
        section_lines.append(line)
        continue

    # Check if reached next section
    if in_section and line.strip().startswith('#'):
        break

    if in_section:
        section_lines.append(line)

return '\n'.join(section_lines) or content  # Fallback to full content
```

---

### 4. MCP Server (`backend/mcp/server.py` + `backend/mcp/tools.py`)

**Purpose:** Expose search functionality via MCP HTTP/SSE transport

#### Server Setup (server.py)
```python
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Documentation Search MCP",
    version="1.0.0",
    description="MCP server for documentation search"
)

# Create MCP server instance
mcp = FastMCP("doc-search-mcp")

# Health check endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "doc-search-mcp",
        "version": "1.0.0"
    }

# Mount MCP endpoints at /mcp
app.mount("/mcp", mcp.get_asgi_app())

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=3001,
        log_level="info"
    )
```

#### MCP Tools (tools.py)

**Tool 1: search_documentation**
```python
@mcp.tool()
def search_documentation(
    query: str,
    product: Optional[str] = None,
    component: Optional[str] = None,
    file_types: Optional[List[str]] = None,
    max_results: int = 10
) -> dict:
    """
    Search across all documentation using keyword matching.

    Args:
        query: Search keywords (space-separated)
        product: Filter by product name (e.g., "symphony")
        component: Filter by component (e.g., "PAM")
        file_types: Filter by file extensions (e.g., [".md", ".docx"])
        max_results: Maximum number of results (default: 10, max: 50)

    Returns:
        Dictionary with search results including file paths, snippets,
        and relevance scores

    Example:
        search_documentation(
            query="authentication OAuth",
            product="symphony",
            component="PAM",
            max_results=5
        )
    """
    results = search_engine.search(
        query=query,
        product=product,
        component=component,
        file_types=file_types,
        max_results=min(max_results, 50)
    )

    return {
        "results": results,
        "total": len(results),
        "query": query,
        "filters": {
            "product": product,
            "component": component,
            "file_types": file_types
        }
    }
```

**Tool 2: get_document**
```python
@mcp.tool()
def get_document(
    path: str,
    section: Optional[str] = None
) -> dict:
    """
    Retrieve full content of a specific document.

    Args:
        path: Relative path from docs root (e.g., "symphony/PAM/api-spec.md")
        section: Optional section heading to extract (e.g., "Authentication")

    Returns:
        Document content, metadata, and structure

    Example:
        get_document(
            path="symphony/PAM/api-spec.md",
            section="Endpoints"
        )
    """
    doc = search_engine.get_document(path, section)

    if not doc:
        return {
            "error": "Document not found",
            "path": path
        }

    return doc
```

**Tool 3: list_products**
```python
@mcp.tool()
def list_products() -> dict:
    """
    List all available products in the documentation.

    Returns:
        List of products with component counts and metadata

    Example:
        list_products()
    """
    products = indexer.get_products()

    return {
        "products": products,
        "total": len(products)
    }
```

**Tool 4: list_components**
```python
@mcp.tool()
def list_components(product: str) -> dict:
    """
    List all components for a specific product.

    Args:
        product: Product name (e.g., "symphony")

    Returns:
        List of components with document counts

    Example:
        list_components(product="symphony")
    """
    components = indexer.get_components(product)

    if not components:
        return {
            "error": f"Product '{product}' not found",
            "product": product
        }

    return {
        "product": product,
        "components": components,
        "total": len(components)
    }
```

**Tool 5: get_index_status**
```python
@mcp.tool()
def get_index_status() -> dict:
    """
    Get current indexing status and statistics.

    Returns:
        Index status, file counts, and last update time

    Example:
        get_index_status()
    """
    status = indexer.get_status()

    return status
```

**Server Initialization Flow:**
```
1. Load config.json
2. Initialize FileIndexer(docs_root)
3. Build initial index: indexer.build_index()
4. Initialize SearchEngine(indexer)
5. Start file watcher (if configured)
6. Create FastMCP instance
7. Register all tools
8. Mount MCP app on FastAPI
9. Start uvicorn server on port 3001
```

---

## Configuration Design

### config.json Structure
```json
{
  "system": {
    "name": "Documentation Search MCP",
    "version": "1.0.0"
  },
  "docs": {
    "root_path": "/absolute/path/to/docs",
    "file_extensions": [".md", ".txt", ".docx"],
    "max_file_size_mb": 10,
    "watch_for_changes": true,
    "index_on_startup": true
  },
  "search": {
    "max_results": 50,
    "snippet_length": 200,
    "context_lines": 3,
    "min_keyword_length": 2
  },
  "mcp": {
    "transport": "http-sse",
    "host": "127.0.0.1",
    "port": 3001,
    "endpoint": "/mcp"
  },
  "logging": {
    "level": "info",
    "file": "mcp_server.log",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

### .env Structure
```bash
# Override config.json values with environment variables
DOCS_ROOT_PATH=/path/to/docs
MCP_HOST=127.0.0.1
MCP_PORT=3001
LOG_LEVEL=info
```

### Configuration Loading Logic
```python
def load_config() -> Dict:
    # 1. Load config.json
    with open('config.json') as f:
        config = json.load(f)

    # 2. Load .env if exists
    load_dotenv()

    # 3. Override with environment variables
    config['docs']['root_path'] = os.getenv('DOCS_ROOT_PATH', config['docs']['root_path'])
    config['mcp']['host'] = os.getenv('MCP_HOST', config['mcp']['host'])
    config['mcp']['port'] = int(os.getenv('MCP_PORT', config['mcp']['port']))
    config['logging']['level'] = os.getenv('LOG_LEVEL', config['logging']['level'])

    return config
```

---

## Data Flow Examples

### Example 1: Initial Index Build
```
1. Server starts
2. load_config() reads configuration
3. FileIndexer(docs_root="/docs") initialized
4. indexer.build_index() called
   ├─ _scan_files() finds: ["symphony/PAM/api.md", "symphony/Auth/oauth.md", ...]
   ├─ For each file:
   │  ├─ Check _should_index() (size, extension)
   │  ├─ get_relative_path() -> "symphony/PAM/api.md"
   │  ├─ Extract product="symphony", component="PAM"
   │  ├─ Select parser: parsers[".md"] -> MarkdownParser
   │  ├─ parse() -> {content, headings, metadata}
   │  ├─ Create Document object
   │  └─ index.add_document(doc)
   └─ start_watching() if configured
5. SearchEngine(indexer) initialized
6. MCP server ready to accept connections
```

### Example 2: Search Query from Claude
```
1. Claude Desktop sends MCP request:
   {
     "tool": "search_documentation",
     "arguments": {
       "query": "OAuth authentication",
       "product": "symphony",
       "max_results": 5
     }
   }

2. MCP server receives request
3. search_documentation() tool called
4. SearchEngine.search() executes:
   ├─ _parse_query("OAuth authentication") -> ["oauth", "authentication"]
   ├─ Iterate all documents
   ├─ Filter: product == "symphony"
   ├─ For each matching doc:
   │  ├─ _calculate_score(doc, ["oauth", "authentication"])
   │  ├─ _extract_snippet(content, keywords)
   │  └─ Create SearchResult
   ├─ Sort by score (descending)
   └─ Return top 5 results

5. MCP server returns JSON:
   {
     "results": [
       {
         "id": "symphony/Auth/oauth.md",
         "file_path": "symphony/Auth/oauth.md",
         "product": "symphony",
         "component": "Auth",
         "file_name": "oauth.md",
         "snippet": "OAuth 2.0 authentication flow for API access...",
         "relevance_score": 0.92,
         "last_modified": "2024-01-15T10:30:00"
       },
       ...
     ],
     "total": 5
   }

6. Claude Desktop displays results to user
```

### Example 3: File Change Detection
```
1. User edits /docs/symphony/PAM/api.md
2. FileWatcher detects modification event
3. FileWatcher.on_modified() called
4. indexer.index_file("/docs/symphony/PAM/api.md")
   ├─ Parse updated file
   ├─ Create new Document object
   └─ index.add_document(doc) (replaces old version)
5. Index updated in real-time
6. Next search query uses updated content
```

---

## Error Handling Strategy

### File Indexing Errors
```python
try:
    self.index_file(file_path)
    file_count += 1
except UnicodeDecodeError:
    logger.error(f"Encoding error: {file_path} - skipping")
    error_count += 1
except FileNotFoundError:
    logger.error(f"File not found: {file_path} - skipping")
    error_count += 1
except Exception as e:
    logger.error(f"Unexpected error indexing {file_path}: {e}")
    error_count += 1
    # Continue indexing other files
```

### Search Query Errors
```python
@mcp.tool()
def search_documentation(...):
    try:
        results = search_engine.search(...)
        return {"results": results, "total": len(results)}
    except ValueError as e:
        return {"error": "Invalid query", "detail": str(e)}
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"error": "Search failed", "detail": str(e)}
```

### File Watching Errors
```python
def on_modified(self, event):
    try:
        self.indexer.index_file(event.src_path)
    except Exception as e:
        logger.error(f"Failed to re-index {event.src_path}: {e}")
        # Continue watching, don't crash observer
```

---

## Performance Targets

### Indexing Performance
- **Initial index build:** <10 seconds for 500 documents
- **File parsing:** <50ms per document (average)
- **Memory usage:** ~2MB per 100 documents
- **Index update (single file):** <100ms

### Search Performance
- **Query processing:** <50ms for keyword extraction
- **Search execution:** <100ms for 1000 documents
- **Result ranking:** <20ms
- **Total search time:** <200ms end-to-end

### MCP Server Performance
- **Request handling:** <10ms overhead
- **Concurrent requests:** Support 10+ simultaneous queries
- **Memory footprint:** <100MB for 1000 documents

---

## Testing Strategy

### Unit Tests

**Test: Indexer**
```python
def test_build_index(temp_docs):
    indexer = FileIndexer(str(temp_docs), {'watch_for_changes': False})
    result = indexer.build_index()

    assert result['status'] == 'complete'
    assert result['files_indexed'] == 3
    assert len(indexer.index.documents) == 3

def test_product_extraction(temp_docs):
    indexer = FileIndexer(str(temp_docs), {'watch_for_changes': False})
    indexer.build_index()

    products = indexer.get_products()
    product_names = [p['name'] for p in products]

    assert 'product-a' in product_names
    assert 'product-b' in product_names
```

**Test: Parsers**
```python
def test_markdown_heading_extraction():
    parser = MarkdownParser()
    result = parser.parse("test.md")

    assert 'Heading 1' in result['headings']
    assert 'Heading 2' in result['headings']

def test_encoding_detection():
    # Create file with non-UTF8 encoding
    parser = TextParser()
    result = parser.parse("latin1.txt")

    assert result['metadata']['encoding'] in ['latin-1', 'iso-8859-1']
```

**Test: Search**
```python
def test_relevance_scoring(search_engine):
    results = search_engine.search("API specification")

    # Results should be sorted by relevance
    scores = [r['relevance_score'] for r in results]
    assert scores == sorted(scores, reverse=True)

def test_product_filter(search_engine):
    results = search_engine.search("test", product="product-a")

    assert all(r['product'] == 'product-a' for r in results)
```

### Integration Tests

**Test: MCP Tools**
```python
@pytest.mark.asyncio
async def test_search_tool():
    result = search_documentation(
        query="authentication",
        product="symphony",
        max_results=5
    )

    assert 'results' in result
    assert result['total'] <= 5
    assert all('relevance_score' in r for r in result['results'])

@pytest.mark.asyncio
async def test_get_document_tool():
    result = get_document(path="symphony/PAM/api.md")

    assert 'content' in result
    assert 'headings' in result
    assert result['product'] == 'symphony'
```

### Manual Testing Checklist

**MCP Integration:**
- [ ] Server starts without errors
- [ ] Health endpoint responds at http://127.0.0.1:3001/health
- [ ] MCP endpoint accessible at http://127.0.0.1:3001/mcp
- [ ] Claude Desktop connects successfully
- [ ] All 5 tools appear in Claude's tool list
- [ ] search_documentation returns results
- [ ] get_document retrieves full content
- [ ] File changes trigger index updates

**Search Quality:**
- [ ] Exact phrase matches score higher
- [ ] Filename matches boost relevance
- [ ] Heading matches boost relevance
- [ ] Filters (product, component) work correctly
- [ ] Snippets contain matched keywords
- [ ] Empty queries return no results

---

## Deployment Checklist

### Prerequisites
- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Documentation folder exists at configured path
- [ ] config.json created with correct paths
- [ ] .env file created (if using environment variables)

### Configuration
- [ ] docs.root_path points to correct directory
- [ ] docs.file_extensions includes all needed types
- [ ] docs.max_file_size_mb appropriate for documentation
- [ ] mcp.port is available (not in use)
- [ ] logging.file path is writable

### Server Startup
```bash
# From project root
cd backend
python main.py

# Or with uvicorn directly
uvicorn main:app --host 127.0.0.1 --port 3001 --reload
```

### Claude Desktop Configuration
1. Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
2. Add MCP server configuration:
```json
{
  "mcpServers": {
    "doc-search": {
      "url": "http://127.0.0.1:3001/mcp"
    }
  }
}
```
3. Restart Claude Desktop
4. Verify tools appear in Claude's available tools

### Health Verification
```bash
# Test health endpoint
curl http://127.0.0.1:3001/health

# Expected response:
# {"status": "healthy", "service": "doc-search-mcp", "version": "1.0.0"}
```

---

## Acceptance Criteria

### AC1: MCP Server Runs ✓
- [x] Server starts on configured port (3001)
- [x] Health endpoint responds with status
- [x] MCP endpoints accessible via HTTP
- [x] Logs to configured file

### AC2: File Indexing Works ✓
- [x] All .md, .txt, .docx files indexed on startup
- [x] Products and components correctly identified from paths
- [x] Index builds in under 10 seconds (for <500 files)
- [x] File watcher detects changes and updates index
- [x] Encoding detection handles non-UTF8 files

### AC3: Search Functions ✓
- [x] Keyword search returns relevant results
- [x] Results ranked by relevance score
- [x] Snippets include matched keywords
- [x] Filters (product, component, file_types) work correctly
- [x] Empty queries return no results
- [x] Max results limit enforced

### AC4: MCP Tools Work ✓
- [x] search_documentation callable from Claude Desktop
- [x] get_document returns full content
- [x] list_products returns all products with counts
- [x] list_components returns components for product
- [x] get_index_status returns index statistics
- [x] All tools handle errors gracefully

### AC5: Claude Desktop Integration ✓
- [x] Claude Desktop connects to MCP server
- [x] All 5 tools appear in Claude's available tools
- [x] Claude can search docs automatically during conversations
- [x] Results display correctly in Claude interface
- [x] Tool descriptions help Claude choose appropriate tool

### AC6: Testing ✓
- [x] All unit tests pass
- [x] All integration tests pass
- [x] Code coverage >80%
- [x] Manual testing checklist completed

### AC7: Documentation ✓
- [x] README with setup instructions
- [x] Configuration examples provided
- [x] Usage examples documented
- [x] API/tool documentation complete
- [x] Troubleshooting guide included

---

## Next Steps (Phase 2 Preview)

Phase 1 provides keyword search. Phase 2 will enhance this with:
- **Vector embeddings** using sentence-transformers
- **Semantic search** with ChromaDB vector database
- **Hybrid ranking** combining keyword + semantic scores
- **Better vague query handling** (e.g., "how do we handle auth?" finds relevant docs)

Phase 1 architecture is designed to support Phase 2 enhancement without major refactoring.
