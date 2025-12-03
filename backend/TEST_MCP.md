# Testing MCP Server with HTTP/SSE

## Issue: Package Not Found

The error you encountered:
```
npm error 404 Not Found - GET https://registry.npmjs.org/@modelcontextprotocol%2fserver-http-client
```

**Root Cause**: The package `@modelcontextprotocol/server-http-client` does not exist on npm. The correct package for testing MCP servers is `@modelcontextprotocol/inspector`.

## Correct Testing Methods

### Method 1: MCP Inspector (Recommended)

The official MCP Inspector is the correct tool for testing HTTP/SSE MCP servers.

**Install and Run:**
```bash
npx @modelcontextprotocol/inspector http://127.0.0.1:3001/mcp/sse
```

This will:
- Start the MCP Inspector proxy server
- Open a browser UI at `http://localhost:6274` (or similar)
- Allow you to visually interact with your MCP server

**Expected Output:**
```
MCP Inspector
Server URL: http://127.0.0.1:3001/mcp/sse
Inspector UI: http://localhost:6274
```

### Method 2: Claude Desktop Integration

Configure Claude Desktop to use your MCP server:

**macOS:** 

First install: npm install -g mcp-remote


Edit `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "librarian-docs": {
      "command": "npx",
      "args": ["mcp-remote", "http://127.0.0.1:3001/mcp/sse", "--transport", "sse-first"]
    },    
  }
}
```

**Note**: The URL should point to the SSE endpoint, typically `/mcp/sse` for FastMCP servers.

### Method 3: Direct HTTP Testing

For basic connectivity testing only (not full MCP protocol):

**Health Check:**
```bash
curl http://127.0.0.1:3001/health
```

**Expected Response:**
```json
{"status":"healthy","service":"Documentation Search MCP","version":"1.0.0"}
```

### Method 4: Python MCP Client

Create a simple Python test client:

```python
import asyncio
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client

async def test_mcp_server():
    async with httpx.AsyncClient() as client:
        async with sse_client("http://127.0.0.1:3001/mcp/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List available tools
                tools = await session.list_tools()
                print(f"Available tools: {[t.name for t in tools.tools]}")

                # Test search_documentation tool
                result = await session.call_tool(
                    "search_documentation",
                    arguments={"query": "test", "max_results": 5}
                )
                print(f"Search result: {result}")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
```

## Troubleshooting

### Server Not Running
```bash
# Check if server is running
curl http://127.0.0.1:3001/health

# If not, start it
cd /Users/attila.pogany/Code/projects/librarian-mcp
source venv/bin/activate
python ./backend/main.py
```

### Wrong Endpoint URL

Common endpoint patterns:
- ✅ `http://127.0.0.1:3001/mcp/sse` (SSE transport)
- ✅ `http://127.0.0.1:3001/mcp/` (general MCP mount)
- ❌ `http://127.0.0.1:3001/mcp` (missing trailing slash)

### Port Already in Use
```bash
# Check what's using port 3001
lsof -i :3001

# Kill the process if needed
kill -9 <PID>
```

## References

- [MCP Inspector Documentation](https://modelcontextprotocol.io/clients)
- [MCP GitHub Repository](https://github.com/modelcontextprotocol/inspector)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

## Quick Test Commands

```bash
# 1. Ensure server is running
curl http://127.0.0.1:3001/health

# 2. Test with MCP Inspector
npx @modelcontextprotocol/inspector http://127.0.0.1:3001/mcp/sse

# 3. Or configure Claude Desktop and restart it
# Then ask Claude to search documentation
```
