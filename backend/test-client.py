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