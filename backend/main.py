"""Main entry point for MCP server"""

from fastapi import FastAPI
import uvicorn
import logging
from pathlib import Path

from config.settings import load_config, setup_logging
from core.indexer import FileIndexer
from core.search import SearchEngine
from mcp_server.tools import mcp, initialize_tools

# Load configuration
config = load_config()

# Setup logging
setup_logging(config)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=config['system']['name'],
    version=config['system']['version'],
    description="MCP server for documentation search"
)


# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": config['system']['name'],
        "version": config['system']['version']
    }


# Initialize indexer and search engine on startup
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Starting Documentation Search MCP Server")
    logger.info(f"Version: {config['system']['version']}")

    # Check if docs path exists
    docs_path = Path(config['docs']['root_path'])
    if not docs_path.exists():
        logger.warning(f"Documentation path does not exist: {docs_path}")
        logger.info(f"Creating documentation directory: {docs_path}")
        docs_path.mkdir(parents=True, exist_ok=True)

    # Initialize indexer
    logger.info(f"Initializing indexer for: {docs_path}")
    indexer = FileIndexer(str(docs_path), config['docs'])

    # Build index if configured
    if config['docs'].get('index_on_startup', True):
        logger.info("Building initial index...")
        result = indexer.build_index()
        logger.info(f"Index built: {result['files_indexed']} files in {result['duration_seconds']}s")

    # Initialize search engine
    search_engine = SearchEngine(indexer)

    # Initialize MCP tools
    initialize_tools(indexer, search_engine)

    logger.info("MCP server ready")


# Shutdown handler
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down MCP server")


# Mount MCP endpoints
app.mount("/mcp", mcp.sse_app())


if __name__ == "__main__":
    # Run server
    host = config['mcp']['host']
    port = config['mcp']['port']

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=config['logging']['level'].lower()
    )
