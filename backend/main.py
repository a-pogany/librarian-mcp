"""Main entry point for MCP server"""

from fastapi import FastAPI
import uvicorn
import logging
from pathlib import Path

from config.settings import load_config, setup_logging
from core.indexer import FileIndexer
from core.search import SearchEngine
from core.hybrid_search import HybridSearchEngine
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

    # Check if RAG/embeddings should be enabled
    enable_embeddings = config.get('embeddings', {}).get('enabled', True)
    search_mode = config.get('search', {}).get('mode', 'hybrid')

    # Initialize indexer with optional embeddings
    logger.info(f"Initializing indexer for: {docs_path}")
    logger.info(f"Embeddings enabled: {enable_embeddings}")
    logger.info(f"Search mode: {search_mode}")

    try:
        # Pass full config so FileIndexer can access embeddings, chunking, and folder_metadata sections
        indexer = FileIndexer(
            str(docs_path),
            config,
            enable_embeddings=enable_embeddings
        )
    except Exception as e:
        logger.error(f"Error initializing RAG components: {e}")
        logger.warning("Falling back to keyword-only search")
        enable_embeddings = False
        indexer = FileIndexer(
            str(docs_path),
            config,
            enable_embeddings=False
        )

    # Build index if configured
    if config['docs'].get('index_on_startup', True):
        logger.info("Building initial index...")
        result = indexer.build_index()
        logger.info(f"Index built: {result['files_indexed']} files in {result['duration_seconds']}s")
        if result.get('folders_indexed', 0) > 0:
            logger.info(f"Folder metadata: {result['folders_indexed']} folders indexed")

    # Initialize search engines
    keyword_engine = SearchEngine(indexer)
    semantic_engine = None

    # Initialize semantic search if embeddings are enabled
    if enable_embeddings and indexer.embedding_generator and indexer.vector_db:
        try:
            from core.semantic_search import SemanticSearchEngine

            # Check if folder metadata is enabled for hierarchical search
            enable_folder_metadata = config.get('folder_metadata', {}).get('enabled', True)

            if enable_folder_metadata and indexer.enable_folder_metadata and \
               indexer.folder_metadata_extractor and indexer.folder_vector_db:
                # Use hierarchical search with folder filtering
                logger.info("Initializing hierarchical search with folder metadata")
                from core.hierarchical_search import HierarchicalSearchEngine

                # First initialize base semantic search
                base_semantic_engine = SemanticSearchEngine(
                    indexer.embedding_generator,
                    indexer.vector_db,
                    indexer,
                    use_reranking=config.get('search', {}).get('use_reranking', True)
                )

                # Wrap with hierarchical search
                semantic_engine = HierarchicalSearchEngine(
                    embedding_generator=indexer.embedding_generator,
                    folder_metadata_extractor=indexer.folder_metadata_extractor,
                    folder_vector_db=indexer.folder_vector_db,
                    semantic_search_engine=base_semantic_engine,
                    enable_folder_filtering=True
                )
                logger.info("Hierarchical search engine initialized (folder filtering enabled)")
            else:
                # Use standard semantic search
                semantic_engine = SemanticSearchEngine(
                    indexer.embedding_generator,
                    indexer.vector_db,
                    indexer,
                    use_reranking=config.get('search', {}).get('use_reranking', True)
                )
                logger.info("Semantic search engine initialized (no folder filtering)")

        except Exception as e:
            logger.error(f"Error initializing semantic search: {e}")
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

    logger.info(f"Hybrid search engine initialized in '{search_engine.get_mode()}' mode")

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
app.mount("/mcp-http", mcp.streamable_http_app())


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
