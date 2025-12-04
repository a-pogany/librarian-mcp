# Librarian MCP - Quick Start Guide

**Version**: 2.0.0 (Enterprise RAG)

Get up and running with Librarian MCP Enterprise RAG in 5-10 minutes (includes one-time model download).

## Step 1: Setup (2 minutes)

```bash
# Clone and setup
git clone <repository-url>
cd librarian-mcp
./scripts/setup.sh
```

This will:
- Create virtual environment
- Install dependencies (including RAG features: sentence-transformers, chromadb, rank-bm25)
- **Download models** (~1.4GB one-time: e5-large-v2 + cross-encoder)
- Create .env configuration
- Create docs directory

**Note**: First run downloads ~1.4GB of AI models (cached in `~/.cache/torch/`).

## Step 2: Configure (1 minute)

Edit `.env` file:

```bash
# Set your documentation path
DOCS_ROOT_PATH=./docs

# Or use an absolute path
# DOCS_ROOT_PATH=/Users/you/Documents/company-docs
```

## Step 3: Add Documentation (1 minute)

Create sample documentation:

```bash
mkdir -p docs/my-project/api
cat > docs/my-project/api/authentication.md << 'EOF'
# Authentication

## Overview

Our API uses OAuth 2.0 for authentication.

## Getting Started

1. Register your application
2. Obtain client credentials
3. Request access token

## Endpoints

### POST /oauth/token

Request an access token using client credentials.
EOF
```

## Step 4: Start Server (30 seconds)

```bash
./scripts/start.sh
```

You should see (v2.0.0):
```
🚀 Starting Librarian MCP Server...
INFO Embeddings enabled: True
INFO Search mode: hybrid
INFO Loading embedding model: intfloat/e5-large-v2
INFO Model loaded successfully. Embedding dimension: 1024
INFO Reranker model loaded successfully
INFO Hybrid search engine initialized in 'hybrid' mode (RRF)
📚 Starting MCP server on http://127.0.0.1:3001
INFO:     Started server process
```

**First startup** may take 1-2 minutes to download models (one-time only).

## Step 5: Configure Claude Desktop (30 seconds)

Edit Claude Desktop configuration:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "doc-search": {
      "url": "http://127.0.0.1:3001/mcp"
    }
  }
}
```

**Restart Claude Desktop**

## Step 6: Test It! (30 seconds)

Open Claude Desktop and try:

**Natural Language Search** (v2.0.0 Semantic Search):
```
How do I implement OAuth2 authentication?
```

Claude will use **hybrid search** (semantic + keyword) to find relevant docs, understanding your intent even without exact keywords!

**Traditional Search**:
```
Search for authentication documentation
```

Both queries use the `search_documentation` tool with enterprise RAG capabilities:
- E5-large-v2 embeddings (1024d)
- Cross-encoder reranking
- RRF hybrid fusion

## Verify Installation

```bash
# Check server status
./scripts/status.sh

# Run tests
./scripts/test.sh
```

## Next Steps

1. **Add More Documentation**: Place your existing docs in the `docs/` folder (supports large DOCX files 200-600 pages!)
2. **Customize Search**: Edit `config.json` to adjust search mode, chunking, reranking settings
3. **Explore Tools**: Try different MCP tools (list_products, get_document, etc.)
4. **Test Enterprise Features**:
   - Large DOCX files: System chunks and indexes 200-600 page documents
   - Semantic search: Ask conceptual questions, not just keyword matches
   - Hybrid search: Get best of both keyword and semantic search

## Common Issues

### Server won't start
```bash
# Check port is available
lsof -i :3001

# Check logs
tail -f mcp_server.log
```

### No documents indexed
```bash
# Verify docs exist
ls -la docs/

# Check file permissions
chmod -R 755 docs/

# Check vector database (v2.0.0)
ls -la vector_db/
# Should show chroma.sqlite3 and other files
```

### Model download issues (v2.0.0)
```bash
# Manually test model download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-large-v2')"

# Check disk space (need ~2GB)
df -h ~/.cache/torch
```

### Claude Desktop won't connect
```bash
# Verify server is running
curl http://localhost:3001/health

# Restart Claude Desktop completely
```

## Documentation Structure

Best practices for organizing your docs:

```
docs/
├── project-alpha/          # Product/project name
│   ├── api/                # Component
│   │   ├── auth.md         # API authentication
│   │   ├── endpoints.md    # API endpoints
│   │   └── errors.md       # Error handling
│   ├── database/
│   │   ├── schema.md
│   │   └── migrations.md
│   └── deployment/
│       └── guide.md
└── project-beta/
    └── frontend/
        └── components.md
```

## Usage Examples

### Search for specific topics
```
You: "Find documentation about database migrations"
Claude: Uses search_documentation with query="database migrations"
```

### List available products
```
You: "What products do we have docs for?"
Claude: Uses list_products tool
```

### Get full document
```
You: "Show me the API authentication guide"
Claude: Uses search to find it, then get_document to retrieve full content
```

## Support

- Check [README.md](README.md) for detailed documentation
- Review [CLAUDE.md](CLAUDE.md) for development guide
- See design docs in `design/` folder

Happy searching! 🚀
