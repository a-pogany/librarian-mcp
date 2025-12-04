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
from core.hybrid_search import HybridSearchEngine
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

    # Check if RAG/embeddings should be enabled
    enable_embeddings = config.get('embeddings', {}).get('enabled', True)
    search_mode = config.get('search', {}).get('mode', 'hybrid')

    logger.info("Initializing indexer for: %s", docs_path)
    logger.info("Embeddings enabled: %s", enable_embeddings)
    logger.info("Search mode: %s", search_mode)

    try:
        indexer = FileIndexer(
            str(docs_path),
            config["docs"],
            enable_embeddings=enable_embeddings
        )
    except Exception as e:
        logger.error("Error initializing RAG components: %s", e)
        logger.warning("Falling back to keyword-only search")
        enable_embeddings = False
        indexer = FileIndexer(
            str(docs_path),
            config["docs"],
            enable_embeddings=False
        )

    if config["docs"].get("index_on_startup", True):
        logger.info("Building initial index...")
        result = indexer.build_index()
        logger.info(
            "Index built: %s files in %ss",
            result["files_indexed"],
            result["duration_seconds"],
        )

    # Initialize search engines
    keyword_engine = SearchEngine(indexer)
    semantic_engine = None

    # Initialize semantic search if embeddings are enabled
    if enable_embeddings and indexer.embedding_generator and indexer.vector_db:
        try:
            from core.semantic_search import SemanticSearchEngine
            semantic_engine = SemanticSearchEngine(
                indexer.embedding_generator,
                indexer.vector_db,
                indexer
            )
            logger.info("Semantic search engine initialized")
        except Exception as e:
            logger.error("Error initializing semantic search: %s", e)
            logger.warning("Semantic search disabled, using keyword-only")

    # Create hybrid search engine
    semantic_weight = config.get('embeddings', {}).get('semantic_weight', 0.5)
    rerank_candidates = config.get('search', {}).get('rerank_candidates', 50)
    rerank_keyword_threshold = config.get('search', {}).get('rerank_keyword_threshold', 0.1)

    search_engine = HybridSearchEngine(
        keyword_engine=keyword_engine,
        semantic_engine=semantic_engine,
        default_mode=search_mode,
        semantic_weight=semantic_weight,
        rerank_candidates=rerank_candidates,
        rerank_keyword_threshold=rerank_keyword_threshold
    )

    logger.info("Hybrid search engine initialized in '%s' mode", search_engine.get_mode())
    initialize_tools(indexer, search_engine)

    logger.info("Starting stdio MCP server")
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
