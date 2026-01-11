"""MCP tool definitions"""

from mcp.server import FastMCP
from typing import Optional, List
import logging

try:
    from core.query_parser import QueryOperatorParser
except ImportError:
    from backend.core.query_parser import QueryOperatorParser

logger = logging.getLogger(__name__)

# Query operator parser for email search
query_parser = QueryOperatorParser()

# Global references (initialized in server.py)
indexer = None
search_engine = None

# Create MCP instance
mcp = FastMCP(
    "doc-search-mcp",
    stateless_http=True,
    streamable_http_path="/"
)


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
    include_parent_context: Optional[bool] = None,
    enhance_results: bool = True,
    include_full_metadata: bool = False,
    max_per_document: Optional[int] = None
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
        enhance_results: Include rich metadata and summaries in results
        include_full_metadata: Include full metadata payload in enhanced results
        max_per_document: Maximum chunks per document (default: 3, 0=unlimited)

    Returns:
        Dictionary with search results including file paths, snippets,
        relevance scores, metadata, and optionally parent context

    Example:
        search_documentation(
            query="how to configure authentication",
            product="symphony",
            mode="hyde",  # Use HyDE for conceptual queries
            include_parent_context=True,
            max_results=5,
            max_per_document=2  # Limit to 2 chunks per document for diversity
        )
    """
    try:
        # Exclude .eml files from documentation search (use search_emails for emails)
        effective_file_types = file_types
        if not file_types:
            effective_file_types = ['.md', '.txt', '.docx']  # Default: docs only, no emails
        elif '.eml' in file_types:
            effective_file_types = [ft for ft in file_types if ft != '.eml']

        results = search_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=effective_file_types,
            max_results=min(max_results, 50) * 2,  # Get more for filtering
            mode=mode,
            include_parent_context=include_parent_context,
            enhance_results=False,
            max_per_document=max_per_document
        )

        # Apply enhanced metadata filters
        filtered_results = _apply_metadata_filters(
            results,
            doc_type=doc_type,
            tags=tags,
            modified_after=modified_after,
            modified_before=modified_before
        )

        if enhance_results and hasattr(search_engine, 'result_enhancer'):
            filtered_results = search_engine.result_enhancer.enhance(
                filtered_results,
                include_full_metadata=include_full_metadata
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
def search_emails(
    query: str,
    sender: Optional[str] = None,
    recipient: Optional[str] = None,
    cc: Optional[str] = None,
    thread_id: Optional[str] = None,
    subject_contains: Optional[str] = None,
    folder: Optional[str] = None,
    has_attachments: Optional[bool] = None,
    date_after: Optional[str] = None,
    date_before: Optional[str] = None,
    max_results: int = 10,
    mode: Optional[str] = None,
    include_parent_context: Optional[bool] = None,
    enhance_results: bool = True,
    include_full_metadata: bool = False,
    collapse_threads: bool = True,
    max_per_document: Optional[int] = None,
    parse_operators: bool = True
) -> dict:
    """
    Search emails with email-specific filters.

    This tool is optimized for searching EML files with preprocessing
    (quote removal, signature removal, thread grouping).

    Supports inline search operators (Outlook/Gmail style):
        from:sender       - Emails from sender (partial match)
        to:recipient      - Emails to recipient (partial match)
        cc:recipient      - Emails with CC recipient (partial match)
        subject:text      - Subject contains text
        in:folder         - Emails in folder (inbox, sent, important, etc.)
        has:attachment    - Emails with attachments
        after:YYYY-MM-DD  - Emails after date
        before:YYYY-MM-DD - Emails before date
        thread:id         - Emails in specific thread

    Args:
        query: Search text with optional inline operators (e.g., "from:kiraly project deadline")
        sender: Filter by sender email address (partial match)
        recipient: Filter by recipient email address (partial match)
        cc: Filter by CC email address (partial match)
        thread_id: Filter by email thread ID (groups related emails)
        subject_contains: Filter by subject line (partial match)
        folder: Filter by email folder (inbox, sent, important, etc.)
        has_attachments: Filter emails with/without attachments
        date_after: Only emails after this date (ISO format: 2024-01-01)
        date_before: Only emails before this date (ISO format: 2024-12-31)
        max_results: Maximum number of results (default: 10, max: 50)
        mode: Search mode - keyword, semantic, hybrid, rerank, hyde, or auto
        include_parent_context: Include parent document context
        enhance_results: Include rich metadata and summaries in results
        include_full_metadata: Include full metadata payload in enhanced results
        collapse_threads: Collapse results by thread, showing only the best match per thread (default: True).
            When enabled, adds thread_count metadata showing total emails in that thread.
        max_per_document: Maximum chunks per document (default: 3, 0=unlimited)
        parse_operators: Whether to parse inline operators from query (default: True)

    Returns:
        Dictionary with email search results including:
        - from, to, cc, subject, date
        - thread_id for grouping related emails
        - attachment metadata
        - cleaned content (quotes and signature removed)
        - parsed_operators (if parse_operators=True)

    Examples:
        # With inline operators
        search_emails(query="from:kiraly project deadline")
        search_emails(query="in:important has:attachment budget")
        search_emails(query='subject:"Q4 Report" from:finance after:2024-01-01')

        # With explicit parameters
        search_emails(
            query="project deadline",
            sender="john@example.com",
            has_attachments=True,
            date_after="2024-01-01",
            max_results=10
        )
    """
    try:
        # Parse operators from query if enabled
        parsed_operators = {}
        search_query = query
        parse_warnings = []

        if parse_operators and query:
            parsed = query_parser.parse(query)
            # Use free text as search query, fallback to original if no text
            search_query = parsed.free_text if parsed.free_text else query
            parsed_operators = parsed.get_filter_params()
            parse_warnings = parsed.parse_warnings

            if parsed.has_operators():
                logger.debug(f"Parsed operators: {parsed_operators} from query: {query}")

        # Merge parsed operators with explicit parameters
        # Explicit parameters take precedence over parsed operators
        effective_sender = sender or parsed_operators.get('sender')
        effective_recipient = recipient or parsed_operators.get('recipient')
        effective_cc = cc or parsed_operators.get('cc')
        effective_subject = subject_contains or parsed_operators.get('subject_contains')
        effective_folder = folder or parsed_operators.get('folder')
        effective_thread = thread_id or parsed_operators.get('thread_id')
        effective_date_after = date_after or parsed_operators.get('date_after')
        effective_date_before = date_before or parsed_operators.get('date_before')

        # has:attachment handling - explicit parameter takes precedence
        effective_has_attachments = has_attachments
        if effective_has_attachments is None and 'has_attachments' in parsed_operators:
            effective_has_attachments = parsed_operators['has_attachments']

        # Search only .eml files
        # Fetch more candidates when filters are present so post-filtering doesn't drop everything
        filters_present = any([
            effective_sender,
            effective_recipient,
            effective_cc,
            effective_thread,
            effective_subject,
            effective_folder,
            effective_has_attachments is not None,
            effective_date_after,
            effective_date_before
        ])

        # Base multipliers (collapse_threads needs more to pick best per thread)
        candidate_multiplier = 3 if collapse_threads else 2
        if filters_present:
            candidate_multiplier += 2  # widen recall before filtering

        def _run_search(multiplier: int):
            return search_engine.search(
                query=search_query,
                file_types=['.eml'],
                max_results=min(max_results, 50) * multiplier,
                mode=mode,
                include_parent_context=include_parent_context,
                enhance_results=False,
                max_per_document=max_per_document,
                # Pass email filters for pre-filtering at search layer
                sender=effective_sender,
                recipient=effective_recipient,
                cc=effective_cc,
                folder=effective_folder,
                subject_contains=effective_subject,
                has_attachments=effective_has_attachments,
                date_after=effective_date_after,
                date_before=effective_date_before,
                thread_id=effective_thread
            )

        results = _run_search(candidate_multiplier)

        # Apply email-specific filters (including new ones)
        filtered_results = _apply_email_filters(
            results,
            sender=effective_sender,
            recipient=effective_recipient,
            cc=effective_cc,
            thread_id=effective_thread,
            subject_contains=effective_subject,
            folder=effective_folder,
            has_attachments=effective_has_attachments,
            date_after=effective_date_after,
            date_before=effective_date_before
        )

        # If filters are present and nothing matched, retry with a larger candidate pool
        if filters_present and not filtered_results:
            retry_multiplier = max(candidate_multiplier, 8)
            logger.debug(
                f"No results after filtering; retrying search with multiplier={retry_multiplier}"
            )
            results = _run_search(retry_multiplier)
            filtered_results = _apply_email_filters(
                results,
                sender=effective_sender,
                recipient=effective_recipient,
                cc=effective_cc,
                thread_id=effective_thread,
                subject_contains=effective_subject,
                folder=effective_folder,
                has_attachments=effective_has_attachments,
                date_after=effective_date_after,
                date_before=effective_date_before
            )

        # Collapse by thread if enabled (keeps best match per thread)
        if collapse_threads:
            filtered_results = _collapse_by_thread(filtered_results)

        if enhance_results and hasattr(search_engine, 'result_enhancer'):
            filtered_results = search_engine.result_enhancer.enhance(
                filtered_results,
                include_full_metadata=include_full_metadata
            )

        # Determine the actual mode used
        actual_mode = mode or search_engine.get_mode()
        if mode == 'auto' and hasattr(search_engine, 'query_router') and search_engine.query_router:
            actual_mode = search_engine.query_router.route(search_query)

        return {
            "results": filtered_results[:max_results],
            "total": len(filtered_results),
            "query": search_query,
            "original_query": query if search_query != query else None,
            "parsed_operators": parsed_operators if parse_operators else None,
            "parse_warnings": parse_warnings if parse_warnings else None,
            "search_mode": actual_mode,
            "filters": {
                "sender": effective_sender,
                "recipient": effective_recipient,
                "cc": effective_cc,
                "thread_id": effective_thread,
                "subject_contains": effective_subject,
                "folder": effective_folder,
                "has_attachments": effective_has_attachments,
                "date_after": effective_date_after,
                "date_before": effective_date_before,
                "collapse_threads": collapse_threads
            }
        }
    except Exception as e:
        logger.error(f"Email search error: {e}")
        return {
            "error": "Email search failed",
            "detail": str(e)
        }


def _apply_email_filters(
    results: List[dict],
    sender: Optional[str] = None,
    recipient: Optional[str] = None,
    cc: Optional[str] = None,
    thread_id: Optional[str] = None,
    subject_contains: Optional[str] = None,
    folder: Optional[str] = None,
    has_attachments: Optional[bool] = None,
    date_after: Optional[str] = None,
    date_before: Optional[str] = None
) -> List[dict]:
    """
    Apply email-specific filters to search results.

    Args:
        results: List of search results to filter
        sender: Filter by sender (from field, partial match)
        recipient: Filter by recipient (to field, partial match)
        cc: Filter by CC field (partial match)
        thread_id: Filter by thread ID (exact match)
        subject_contains: Filter by subject (partial match)
        folder: Filter by folder/label (exact match, case-insensitive)
        has_attachments: Filter by attachment presence
        date_after: Filter by date (after, ISO format)
        date_before: Filter by date (before, ISO format)

    Returns:
        Filtered list of results matching all specified criteria
    """
    from datetime import datetime

    filtered = []

    for result in results:
        metadata = result.get('metadata', {})

        # Sender filter (partial match)
        if sender:
            result_sender = metadata.get('from', '')
            if sender.lower() not in result_sender.lower():
                continue

        # Recipient filter (partial match on 'to' field)
        if recipient:
            result_to = metadata.get('to', '')
            # Handle both string and list formats
            if isinstance(result_to, list):
                result_to = ', '.join(result_to)
            if recipient.lower() not in result_to.lower():
                continue

        # CC filter (partial match)
        if cc:
            result_cc = metadata.get('cc', '')
            # Handle both string and list formats
            if isinstance(result_cc, list):
                result_cc = ', '.join(result_cc)
            if cc.lower() not in result_cc.lower():
                continue

        # Thread ID filter
        if thread_id:
            result_thread = metadata.get('thread_id', '')
            if thread_id != result_thread:
                continue

        # Subject filter (partial match)
        if subject_contains:
            result_subject = metadata.get('subject', '')
            if subject_contains.lower() not in result_subject.lower():
                continue

        # Folder filter (exact match, case-insensitive)
        if folder:
            result_folder = metadata.get('folder', '')
            if folder.lower() != result_folder.lower():
                continue

        # Attachments filter
        if has_attachments is not None:
            result_has_attachments = metadata.get('has_attachments', False)
            if has_attachments != result_has_attachments:
                continue

        # Date filters
        date_str = metadata.get('date_iso') or metadata.get('date')
        if date_str:
            try:
                # Handle both ISO format and email date format
                if 'T' in str(date_str):
                    email_dt = datetime.fromisoformat(str(date_str).replace('Z', '+00:00'))
                else:
                    from email.utils import parsedate_to_datetime
                    email_dt = parsedate_to_datetime(str(date_str))

                if date_after:
                    after_dt = datetime.fromisoformat(date_after)
                    if email_dt.replace(tzinfo=None) < after_dt:
                        continue

                if date_before:
                    before_dt = datetime.fromisoformat(date_before)
                    if email_dt.replace(tzinfo=None) >= before_dt:
                        continue

            except (ValueError, TypeError) as e:
                logger.debug(f"Error parsing email date: {e}")

        filtered.append(result)

    return filtered


def _collapse_by_thread(results: List[dict]) -> List[dict]:
    """
    Collapse email results by thread, keeping only the highest-scoring email per thread.
    
    For each thread:
    - Keeps the email with the highest relevance_score
    - Adds thread_count metadata (total emails in thread)
    - Adds is_thread_representative flag
    
    Args:
        results: List of email search results with metadata containing thread_id
        
    Returns:
        Collapsed results with one email per thread, sorted by relevance_score
    """
    from collections import defaultdict
    
    # Group by thread_id
    threads: dict = defaultdict(list)
    no_thread: List[dict] = []
    
    for result in results:
        thread_id = result.get('metadata', {}).get('thread_id')
        if thread_id:
            threads[thread_id].append(result)
        else:
            # Emails without thread_id are kept as-is
            no_thread.append(result)
    
    collapsed = []
    
    # For each thread, pick the highest-scoring email
    for thread_id, thread_emails in threads.items():
        # Sort by relevance_score descending
        thread_emails.sort(
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )
        
        # Take the best one
        best_email = thread_emails[0].copy()
        
        # Add thread metadata
        best_email['thread_count'] = len(thread_emails)
        best_email['is_thread_representative'] = True
        
        # Also add to nested metadata for consistency
        if 'metadata' not in best_email:
            best_email['metadata'] = {}
        best_email['metadata']['thread_count'] = len(thread_emails)
        best_email['metadata']['is_thread_representative'] = True
        
        collapsed.append(best_email)
    
    # Add emails without thread_id
    for email in no_thread:
        email_copy = email.copy()
        email_copy['thread_count'] = 1
        email_copy['is_thread_representative'] = True
        collapsed.append(email_copy)
    
    # Sort by relevance_score descending
    collapsed.sort(
        key=lambda x: x.get('relevance_score', 0),
        reverse=True
    )
    
    return collapsed


@mcp.tool()
def get_email_thread(thread_id: str, max_results: int = 50) -> dict:
    """
    Get all emails in a thread, ordered by date.

    Args:
        thread_id: Thread ID to retrieve (from search_emails results)
        max_results: Maximum emails to return (default: 50)

    Returns:
        List of emails in the thread, chronologically ordered

    Example:
        get_email_thread(thread_id="<original-message-id@example.com>")
    """
    try:
        # Search for all emails with this thread ID using pre-filtering
        results = search_engine.search(
            query="*",  # Match all content
            file_types=['.eml'],
            max_results=max_results * 2,
            thread_id=thread_id  # Pre-filter by thread_id at search layer
        )

        # Results are already filtered by thread_id, just sort by date
        thread_emails = list(results)  # Make a copy to sort

        # Sort by date (oldest first)
        thread_emails.sort(
            key=lambda x: x.get('metadata', {}).get('date_iso', ''),
            reverse=False
        )

        return {
            "thread_id": thread_id,
            "emails": thread_emails[:max_results],
            "total": len(thread_emails)
        }
    except Exception as e:
        logger.error(f"Get email thread error: {e}")
        return {
            "error": "Failed to retrieve email thread",
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
