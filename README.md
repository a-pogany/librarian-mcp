# Librarian MCP - Documentation Search System

A documentation search system that makes technical documentation accessible to LLMs and humans through an MCP (Model Context Protocol) server.

## Features

**Phase 1 (Current):**
- ✅ HTTP/SSE MCP server for Claude Desktop/Cline integration
- ✅ Keyword-based search with relevance ranking
- ✅ Multi-format support (.md, .txt, .docx)
- ✅ Real-time file watching with automatic index updates
- ✅ Product/component hierarchical organization

**Phase 2 (Planned):**
- 🔜 Vector embeddings with sentence-transformers
- 🔜 Semantic search using ChromaDB
- 🔜 Hybrid search combining keyword + semantic

**Phase 3 (Planned):**
- 🔜 REST API for HTTP access
- 🔜 React web UI for human users

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd librarian-mcp
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
python 
```

4. **Create documentation folder**
```bash
mkdir -p docs/product-name/component-name
```

5. **Configure environment**
```bash
cp .env.example .env
# Edit .env to set your documentation path
```

### Running the Server

```bash
cd backend
python main.py
```

The server will start on `http://127.0.0.1:3001`

### Documentation Structure

Organize your documentation in this hierarchy:

```
docs/
├── product-name/          # e.g., symphony, project-x
│   ├── component-name/    # e.g., PAM, auth, database
│   │   ├── file.md
│   │   ├── spec.docx
│   │   └── notes.txt
│   └── architecture/
├── meetings/
│   └── product-name/
└── shared/               # Cross-product docs
```

## Claude Desktop Integration

1. **Configure Claude Desktop**

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "doc-search": {
      "url": "http://127.0.0.1:3001/mcp"
    }
  }
}
```

2. **Restart Claude Desktop**

3. **Verify integration**

The following tools should appear in Claude's available tools:
- `search_documentation` - Search across all documentation
- `get_document` - Retrieve full document content
- `list_products` - List all products
- `list_components` - List components for a product
- `get_index_status` - Get indexing statistics

## Usage Examples

### Search Documentation
```
You (to Claude): "Search for authentication documentation in the symphony product"

Claude will automatically use the search_documentation tool:
{
  "query": "authentication",
  "product": "symphony"
}
```

### Get Specific Document
```
You: "Show me the OAuth spec from symphony/PAM"

Claude will use get_document:
{
  "path": "symphony/PAM/oauth-spec.md"
}
```

### List Available Products
```
You: "What products do we have documentation for?"

Claude will use list_products
```

## Development

### Running Tests

```bash
cd backend
pytest tests/ -v
```

### With coverage

```bash
pytest tests/ --cov=core --cov=mcp --cov-report=html
```

### Project Structure

```
librarian-mcp/
├── backend/
│   ├── core/                 # Core components
│   │   ├── parsers.py        # File parsers
│   │   ├── indexer.py        # Document indexing
│   │   └── search.py         # Search engine
│   ├── mcp/                  # MCP server
│   │   └── tools.py          # MCP tool definitions
│   ├── config/               # Configuration
│   │   └── settings.py       # Config management
│   ├── tests/                # Unit tests
│   ├── main.py               # Server entry point
│   └── requirements.txt      # Dependencies
├── docs/                     # Documentation files
├── config.json               # Configuration file
└── README.md                 # This file
```

## Configuration

### config.json

```json
{
  "docs": {
    "root_path": "./docs",
    "file_extensions": [".md", ".txt", ".docx"],
    "max_file_size_mb": 10,
    "watch_for_changes": true
  },
  "search": {
    "max_results": 50,
    "snippet_length": 200
  },
  "mcp": {
    "host": "127.0.0.1",
    "port": 3001
  }
}
```

### Environment Variables (.env)

```bash
DOCS_ROOT_PATH=/path/to/docs
MCP_HOST=127.0.0.1
MCP_PORT=3001
LOG_LEVEL=info
```

## MCP Tools Reference

### search_documentation

Search across all documentation using keyword matching.

**Parameters:**
- `query` (str): Search keywords
- `product` (str, optional): Filter by product
- `component` (str, optional): Filter by component
- `file_types` (list, optional): Filter by file extensions
- `max_results` (int, default=10): Maximum results

**Returns:**
- `results`: List of matching documents
- `total`: Number of results
- `query`: Search query used

### get_document

Retrieve full content of a specific document.

**Parameters:**
- `path` (str): Relative path from docs root
- `section` (str, optional): Extract specific section by heading

**Returns:**
- `content`: Full document content
- `headings`: List of headings
- `metadata`: Document metadata

### list_products

List all available products.

**Returns:**
- `products`: List of products with component counts
- `total`: Number of products

### list_components

List components for a specific product.

**Parameters:**
- `product` (str): Product name

**Returns:**
- `components`: List of components with document counts
- `total`: Number of components

### get_index_status

Get current indexing status and statistics.

**Returns:**
- `status`: Index status
- `total_documents`: Number of indexed documents
- `products`: Number of products
- `last_indexed`: Last index update time

## Troubleshooting

### Server won't start

**Check port availability:**
```bash
lsof -i :3001
```

**Check configuration:**
```bash
cat config.json
# Verify docs.root_path exists
```

### No documents indexed

**Check documentation path:**
```bash
ls -la docs/
```

**Check file permissions:**
```bash
# Ensure files are readable
chmod -R 755 docs/
```

### Claude Desktop not connecting

**Verify server is running:**
```bash
curl http://127.0.0.1:3001/health
```

**Check Claude Desktop config:**
```bash
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Restart Claude Desktop**

## Performance

### Indexing Performance
- **Target:** <10 seconds for 500 documents
- **Memory:** ~2MB per 100 documents

### Search Performance
- **Query processing:** <100ms
- **Total search time:** <200ms end-to-end

## Roadmap

### Phase 2: RAG Enhancement (Q1 2024)
- Vector embeddings with sentence-transformers
- Semantic search using ChromaDB
- Hybrid search (keyword + semantic)

### Phase 3: Web UI (Q2 2024)
- REST API for HTTP access
- React web interface
- Advanced filtering and navigation

## License

[Your License]

## Contributing

[Contributing guidelines]

## Support

[Support information]
