"""MCP tool definitions"""

from mcp.server import FastMCP
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# Global references (initialized in server.py)
indexer = None
search_engine = None

# Create MCP instance
mcp = FastMCP("doc-search-mcp")


@mcp.tool()
def search_documentation(
    query: str,
    product: Optional[str] = None,
    component: Optional[str] = None,
    file_types: Optional[List[str]] = None,
    doc_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    modified_after: Optional[str] = None,
    modified_before: Optional[str] = None,
    max_results: int = 10,
    mode: Optional[str] = None,
    include_parent_context: Optional[bool] = None
) -> dict:
    """
    Search across all documentation with enhanced filtering and advanced RAG modes.

    Args:
        query: Search keywords (space-separated)
        product: Filter by product name (e.g., "symphony")
        component: Filter by component (e.g., "PAM")
        file_types: Filter by file extensions (e.g., [".md", ".docx"])
        doc_type: Filter by document type (api, guide, architecture, reference, readme, documentation)
        tags: Filter by tags (documents with at least one matching tag)
        modified_after: Only docs modified after this date (ISO format: 2024-01-01)
        modified_before: Only docs modified before this date (ISO format: 2024-12-31)
        max_results: Maximum number of results (default: 10, max: 50)
        mode: Search mode - keyword, semantic, hybrid, rerank, hyde, or auto (auto selects best mode)
        include_parent_context: Include parent document context (title, summary, headings)

    Returns:
        Dictionary with search results including file paths, snippets,
        relevance scores, metadata, and optionally parent context

    Example:
        search_documentation(
            query="how to configure authentication",
            product="symphony",
            mode="hyde",  # Use HyDE for conceptual queries
            include_parent_context=True,
            max_results=5
        )
    """
    try:
        results = search_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=min(max_results, 50) * 2,  # Get more for filtering
            mode=mode,
            include_parent_context=include_parent_context
        )

        # Apply enhanced metadata filters
        filtered_results = _apply_metadata_filters(
            results,
            doc_type=doc_type,
            tags=tags,
            modified_after=modified_after,
            modified_before=modified_before
        )

        # Determine the actual mode used
        actual_mode = mode or search_engine.get_mode()
        if mode == 'auto' and hasattr(search_engine, 'query_router') and search_engine.query_router:
            actual_mode = search_engine.query_router.route(query)

        return {
            "results": filtered_results[:max_results],
            "total": len(filtered_results),
            "query": query,
            "search_mode": actual_mode,
            "filters": {
                "product": product,
                "component": component,
                "file_types": file_types,
                "doc_type": doc_type,
                "tags": tags,
                "modified_after": modified_after,
                "modified_before": modified_before
            }
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            "error": "Search failed",
            "detail": str(e)
        }


def _apply_metadata_filters(
    results: List[dict],
    doc_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    modified_after: Optional[str] = None,
    modified_before: Optional[str] = None
) -> List[dict]:
    """Apply metadata-based filters to search results"""
    from datetime import datetime

    filtered = []

    for result in results:
        # Doc type filter
        if doc_type and result.get('doc_type') != doc_type:
            continue

        # Tags filter (at least one tag must match)
        if tags:
            result_tags = result.get('tags', [])
            if not any(tag in result_tags for tag in tags):
                continue

        # Date filters
        modified_str = result.get('last_modified')
        if modified_str:
            try:
                modified_dt = datetime.fromisoformat(modified_str)

                if modified_after:
                    after_dt = datetime.fromisoformat(modified_after)
                    if modified_dt < after_dt:
                        continue

                if modified_before:
                    before_dt = datetime.fromisoformat(modified_before)
                    if modified_dt >= before_dt:
                        continue

            except (ValueError, TypeError) as e:
                logger.debug(f"Error parsing date: {e}")
                continue

        filtered.append(result)

    return filtered


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
    try:
        doc = search_engine.get_document(path, section)

        if not doc:
            return {
                "error": "Document not found",
                "path": path
            }

        return doc
    except Exception as e:
        logger.error(f"Get document error: {e}")
        return {
            "error": "Failed to retrieve document",
            "detail": str(e)
        }


@mcp.tool()
def list_products() -> dict:
    """
    List all available products in the documentation.

    Returns:
        List of products with component counts and metadata

    Example:
        list_products()
    """
    try:
        products = indexer.get_products()

        return {
            "products": products,
            "total": len(products)
        }
    except Exception as e:
        logger.error(f"List products error: {e}")
        return {
            "error": "Failed to list products",
            "detail": str(e)
        }


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
    try:
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
    except Exception as e:
        logger.error(f"List components error: {e}")
        return {
            "error": "Failed to list components",
            "detail": str(e)
        }


@mcp.tool()
def get_index_status() -> dict:
    """
    Get current indexing status, statistics, and advanced RAG capabilities.

    Returns:
        Index status, file counts, last update time, and enhanced features status

    Example:
        get_index_status()
    """
    try:
        status = indexer.get_status()

        # Add search engine stats including enhanced features
        if search_engine:
            search_stats = search_engine.get_stats()
            status['search_engine'] = search_stats

        return status
    except Exception as e:
        logger.error(f"Get status error: {e}")
        return {
            "error": "Failed to get status",
            "detail": str(e)
        }


@mcp.tool()
def analyze_query(query: str) -> dict:
    """
    Analyze a query to understand its characteristics and recommended search mode.

    This tool helps understand how the search system interprets queries and
    which search mode would be most effective.

    Args:
        query: The search query to analyze

    Returns:
        Query analysis including:
        - word_count: Number of words in query
        - query_type: Type of query (factual, conceptual, troubleshooting, navigational)
        - complexity_score: How complex the query is (0-1)
        - recommended_mode: Best search mode for this query
        - reasoning: Explanation of the recommendation

    Example:
        analyze_query(query="how to configure OAuth authentication")
    """
    try:
        if hasattr(search_engine, 'analyze_query'):
            analysis = search_engine.analyze_query(query)
            return analysis
        else:
            return {
                "error": "Query analysis not available",
                "detail": "Query router is not enabled"
            }
    except Exception as e:
        logger.error(f"Query analysis error: {e}")
        return {
            "error": "Failed to analyze query",
            "detail": str(e)
        }


@mcp.tool()
def clear_search_cache() -> dict:
    """
    Clear the semantic query cache.

    Use this if you need to force fresh search results or after
    making significant changes to the documentation.

    Returns:
        Confirmation of cache clear

    Example:
        clear_search_cache()
    """
    try:
        if hasattr(search_engine, 'clear_cache'):
            search_engine.clear_cache()
            return {
                "success": True,
                "message": "Search cache cleared successfully"
            }
        else:
            return {
                "success": False,
                "message": "Cache not available"
            }
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        return {
            "error": "Failed to clear cache",
            "detail": str(e)
        }


def initialize_tools(indexer_instance, search_engine_instance):
    """Initialize global references for tools"""
    global indexer, search_engine
    indexer = indexer_instance
    search_engine = search_engine_instance
    logger.info("MCP tools initialized")
