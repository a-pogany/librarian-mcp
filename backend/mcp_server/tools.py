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
    max_results: int = 10
) -> dict:
    """
    Search across all documentation with enhanced filtering.

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

    Returns:
        Dictionary with search results including file paths, snippets,
        relevance scores, and metadata

    Example:
        search_documentation(
            query="authentication OAuth",
            product="symphony",
            doc_type="api",
            tags=["security"],
            modified_after="2024-01-01",
            max_results=5
        )
    """
    try:
        results = search_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=min(max_results, 50) * 2  # Get more for filtering
        )

        # Apply enhanced metadata filters
        filtered_results = _apply_metadata_filters(
            results,
            doc_type=doc_type,
            tags=tags,
            modified_after=modified_after,
            modified_before=modified_before
        )

        return {
            "results": filtered_results[:max_results],
            "total": len(filtered_results),
            "query": query,
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
