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
    max_results: int = 10
) -> dict:
    """
    Search across all documentation using keyword matching.

    Args:
        query: Search keywords (space-separated)
        product: Filter by product name (e.g., "symphony")
        component: Filter by component (e.g., "PAM")
        file_types: Filter by file extensions (e.g., [".md", ".docx"])
        max_results: Maximum number of results (default: 10, max: 50)

    Returns:
        Dictionary with search results including file paths, snippets,
        and relevance scores

    Example:
        search_documentation(
            query="authentication OAuth",
            product="symphony",
            component="PAM",
            max_results=5
        )
    """
    try:
        results = search_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=min(max_results, 50)
        )

        return {
            "results": results,
            "total": len(results),
            "query": query,
            "filters": {
                "product": product,
                "component": component,
                "file_types": file_types
            }
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            "error": "Search failed",
            "detail": str(e)
        }


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
    Get current indexing status and statistics.

    Returns:
        Index status, file counts, and last update time

    Example:
        get_index_status()
    """
    try:
        status = indexer.get_status()
        return status
    except Exception as e:
        logger.error(f"Get status error: {e}")
        return {
            "error": "Failed to get status",
            "detail": str(e)
        }


def initialize_tools(indexer_instance, search_engine_instance):
    """Initialize global references for tools"""
    global indexer, search_engine
    indexer = indexer_instance
    search_engine = search_engine_instance
    logger.info("MCP tools initialized")
