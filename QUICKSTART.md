# Librarian MCP - Quick Start Guide

Get up and running with Librarian MCP in 5 minutes.

## Step 1: Setup (2 minutes)

```bash
# Clone and setup
git clone <repository-url>
cd librarian-mcp
./scripts/setup.sh
```

This will:
- Create virtual environment
- Install dependencies
- Create .env configuration
- Create docs directory

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

You should see:
```
🚀 Starting Librarian MCP Server...
📚 Starting MCP server on http://127.0.0.1:3001
INFO:     Started server process
```

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

```
Search for authentication documentation
```

Claude will automatically use the `search_documentation` tool and find your document!

## Verify Installation

```bash
# Check server status
./scripts/status.sh

# Run tests
./scripts/test.sh
```

## Next Steps

1. **Add More Documentation**: Place your existing docs in the `docs/` folder
2. **Customize Search**: Edit `config.json` to adjust search settings
3. **Explore Tools**: Try different MCP tools (list_products, get_document, etc.)

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
