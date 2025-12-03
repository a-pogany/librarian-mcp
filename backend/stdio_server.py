"""STDIO entry point for the MCP server.

This mirrors backend/main.py but runs the FastMCP instance over stdio so
clients that require a stdio transport (e.g., older Claude Desktop builds)
can connect without an HTTP/SSE proxy.
"""

import asyncio
import logging
from pathlib import Path

from config.settings import load_config, setup_logging
from core.indexer import FileIndexer
from core.search import SearchEngine
from mcp_server.tools import mcp, initialize_tools


def main():
    """Initialize components and start the MCP stdio server."""
    config = load_config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    docs_path = Path(config["docs"]["root_path"])
    if not docs_path.exists():
        logger.warning("Documentation path does not exist: %s", docs_path)
        logger.info("Creating documentation directory: %s", docs_path)
        docs_path.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing indexer for: %s", docs_path)
    indexer = FileIndexer(str(docs_path), config["docs"])

    if config["docs"].get("index_on_startup", True):
        logger.info("Building initial index...")
        result = indexer.build_index()
        logger.info(
            "Index built: %s files in %ss",
            result["files_indexed"],
            result["duration_seconds"],
        )

    search_engine = SearchEngine(indexer)
    initialize_tools(indexer, search_engine)

    logger.info("Starting stdio MCP server")
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
