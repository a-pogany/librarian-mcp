"""
Parent Document Context for enhanced RAG results

When returning chunks, also provides context from the parent document
to help LLMs better understand the retrieved information.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParentContext:
    """Context from a parent document"""
    document_id: str
    title: str
    summary: str  # First paragraph or extracted summary
    headings: List[str]  # Document structure
    total_chunks: int
    sibling_chunks: List[str]  # Adjacent chunk snippets
    breadcrumb: str  # e.g., "Product > Component > Section"


class ParentContextEnricher:
    """
    Enriches search results with parent document context

    When a chunk is retrieved, adds:
    - Document title/summary
    - Surrounding context (previous/next chunks)
    - Document structure (headings outline)
    - Breadcrumb navigation
    """

    def __init__(
        self,
        indexer,
        include_siblings: bool = True,
        sibling_count: int = 1,
        summary_max_length: int = 300,
        include_headings: bool = True,
        max_headings: int = 10
    ):
        """
        Initialize parent context enricher

        Args:
            indexer: FileIndexer instance for document access
            include_siblings: Include adjacent chunks
            sibling_count: Number of siblings on each side
            summary_max_length: Maximum length for document summary
            include_headings: Include document headings outline
            max_headings: Maximum number of headings to include
        """
        self.indexer = indexer
        self.include_siblings = include_siblings
        self.sibling_count = sibling_count
        self.summary_max_length = summary_max_length
        self.include_headings = include_headings
        self.max_headings = max_headings

        logger.info("Parent context enricher initialized")

    def enrich_results(
        self,
        results: List[Dict[str, Any]],
        include_full_context: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Enrich search results with parent document context

        Args:
            results: Search results to enrich
            include_full_context: Include extensive context (slower)

        Returns:
            Enriched results with parent context
        """
        enriched = []

        for result in results:
            try:
                enriched_result = self._enrich_single_result(result, include_full_context)
                enriched.append(enriched_result)
            except Exception as e:
                logger.warning(f"Failed to enrich result {result.get('id')}: {e}")
                enriched.append(result)  # Return original on failure

        return enriched

    def _enrich_single_result(
        self,
        result: Dict[str, Any],
        include_full_context: bool
    ) -> Dict[str, Any]:
        """Enrich a single search result"""
        enriched = result.copy()

        # Get parent document info
        doc_id = result.get('id', result.get('file_path', ''))

        # Handle chunk IDs (e.g., "path/to/doc.md#chunk_0")
        parent_doc_id = doc_id.split('#')[0] if '#' in doc_id else doc_id

        # Get document from indexer
        doc = self.indexer.index.documents.get(parent_doc_id)

        if not doc:
            logger.debug(f"Parent document not found: {parent_doc_id}")
            return enriched

        # Extract parent context
        parent_context = self._build_parent_context(
            doc=doc,
            doc_id=parent_doc_id,
            chunk_id=doc_id,
            include_full_context=include_full_context
        )

        # Add context to result
        enriched['parent_context'] = {
            'document_id': parent_context.document_id,
            'title': parent_context.title,
            'summary': parent_context.summary,
            'breadcrumb': parent_context.breadcrumb,
            'total_chunks': parent_context.total_chunks
        }

        if self.include_headings and parent_context.headings:
            enriched['parent_context']['headings'] = parent_context.headings

        if self.include_siblings and parent_context.sibling_chunks:
            enriched['parent_context']['surrounding_context'] = parent_context.sibling_chunks

        return enriched

    def _build_parent_context(
        self,
        doc: Dict[str, Any],
        doc_id: str,
        chunk_id: str,
        include_full_context: bool
    ) -> ParentContext:
        """Build parent context for a document"""
        content = doc.get('content', '')

        # Extract title (first heading or filename)
        title = self._extract_title(content, doc.get('file_name', ''))

        # Extract summary (first meaningful paragraph)
        summary = self._extract_summary(content)

        # Extract headings outline
        headings = []
        if self.include_headings:
            headings = self._extract_headings(content)

        # Build breadcrumb
        breadcrumb = self._build_breadcrumb(doc)

        # Get sibling chunks if this is a chunk
        sibling_chunks = []
        if self.include_siblings and '#chunk_' in chunk_id:
            sibling_chunks = self._get_sibling_chunks(doc_id, chunk_id, content)

        # Estimate total chunks (rough approximation)
        total_chunks = self._estimate_chunk_count(content)

        return ParentContext(
            document_id=doc_id,
            title=title,
            summary=summary,
            headings=headings,
            total_chunks=total_chunks,
            sibling_chunks=sibling_chunks,
            breadcrumb=breadcrumb
        )

    def _extract_title(self, content: str, filename: str) -> str:
        """Extract document title from content or filename"""
        # Try to find first heading
        lines = content.split('\n')

        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()

            # Markdown heading
            if line.startswith('# '):
                return line[2:].strip()

            # Title in YAML frontmatter
            if line.startswith('title:'):
                return line[6:].strip().strip('"\'')

        # Fall back to filename
        if filename:
            # Remove extension and clean up
            title = filename.rsplit('.', 1)[0]
            title = title.replace('-', ' ').replace('_', ' ')
            return title.title()

        return "Untitled Document"

    def _extract_summary(self, content: str) -> str:
        """Extract document summary (first meaningful paragraph)"""
        lines = content.split('\n')
        summary_lines = []
        in_frontmatter = False
        found_content = False

        for line in lines:
            stripped = line.strip()

            # Skip YAML frontmatter
            if stripped == '---':
                in_frontmatter = not in_frontmatter
                continue

            if in_frontmatter:
                continue

            # Skip headings and empty lines at start
            if not found_content:
                if not stripped or stripped.startswith('#'):
                    continue
                found_content = True

            # Collect paragraph
            if stripped:
                summary_lines.append(stripped)
            elif summary_lines:
                # End of paragraph
                break

        summary = ' '.join(summary_lines)

        # Truncate if too long
        if len(summary) > self.summary_max_length:
            summary = summary[:self.summary_max_length].rsplit(' ', 1)[0] + '...'

        return summary

    def _extract_headings(self, content: str) -> List[str]:
        """Extract document headings as outline"""
        headings = []

        # Match markdown headings
        heading_pattern = r'^(#{1,4})\s+(.+)$'

        for line in content.split('\n'):
            match = re.match(heading_pattern, line.strip())
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()

                # Format with indentation based on level
                indent = '  ' * (level - 1)
                headings.append(f"{indent}{text}")

                if len(headings) >= self.max_headings:
                    break

        return headings

    def _build_breadcrumb(self, doc: Dict[str, Any]) -> str:
        """Build navigation breadcrumb"""
        parts = []

        if doc.get('product'):
            parts.append(doc['product'])

        if doc.get('component'):
            parts.append(doc['component'])

        # Add first heading as section indicator
        content = doc.get('content', '')
        for line in content.split('\n')[:20]:
            if line.strip().startswith('# '):
                parts.append(line.strip()[2:])
                break

        return ' > '.join(parts) if parts else ''

    def _get_sibling_chunks(
        self,
        doc_id: str,
        chunk_id: str,
        content: str
    ) -> List[str]:
        """Get surrounding chunk snippets"""
        siblings = []

        # Parse chunk index from chunk_id
        # Format: "path/to/doc.md#chunk_0" or "path/to/doc.md#chunk_0_1"
        try:
            chunk_part = chunk_id.split('#chunk_')[1]
            if '_' in chunk_part:
                # Sub-chunk format
                chunk_idx = int(chunk_part.split('_')[0])
            else:
                chunk_idx = int(chunk_part)
        except (IndexError, ValueError):
            return siblings

        # Get content chunks (approximate)
        lines = content.split('\n')
        lines_per_chunk = 20  # Approximate

        # Get previous chunk snippet
        if chunk_idx > 0:
            prev_start = max(0, (chunk_idx - 1) * lines_per_chunk)
            prev_end = min(len(lines), chunk_idx * lines_per_chunk)
            prev_lines = lines[prev_start:prev_end]
            if prev_lines:
                prev_snippet = ' '.join(prev_lines[-3:])  # Last 3 lines
                if prev_snippet.strip():
                    siblings.append(f"[Previous]: {prev_snippet[:150]}...")

        # Get next chunk snippet
        next_start = (chunk_idx + 1) * lines_per_chunk
        next_end = min(len(lines), (chunk_idx + 2) * lines_per_chunk)
        if next_start < len(lines):
            next_lines = lines[next_start:next_end]
            if next_lines:
                next_snippet = ' '.join(next_lines[:3])  # First 3 lines
                if next_snippet.strip():
                    siblings.append(f"[Next]: {next_snippet[:150]}...")

        return siblings

    def _estimate_chunk_count(self, content: str) -> int:
        """Estimate number of chunks for this document"""
        # Rough estimate: 512 tokens per chunk, ~4 chars per token
        chars_per_chunk = 512 * 4
        return max(1, len(content) // chars_per_chunk)

    def get_document_outline(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full document outline for navigation

        Args:
            doc_id: Document identifier

        Returns:
            Document outline with structure information
        """
        doc = self.indexer.index.documents.get(doc_id)
        if not doc:
            return None

        content = doc.get('content', '')

        return {
            'document_id': doc_id,
            'title': self._extract_title(content, doc.get('file_name', '')),
            'product': doc.get('product'),
            'component': doc.get('component'),
            'headings': self._extract_headings(content),
            'word_count': len(content.split()),
            'estimated_chunks': self._estimate_chunk_count(content)
        }


class ContextualSnippetGenerator:
    """
    Generate contextual snippets that include surrounding text
    """

    def __init__(
        self,
        context_chars_before: int = 150,
        context_chars_after: int = 150,
        highlight_matches: bool = True
    ):
        """
        Initialize snippet generator

        Args:
            context_chars_before: Characters of context before match
            context_chars_after: Characters of context after match
            highlight_matches: Add markers around matched terms
        """
        self.context_chars_before = context_chars_before
        self.context_chars_after = context_chars_after
        self.highlight_matches = highlight_matches

    def generate_snippet(
        self,
        content: str,
        query: str,
        max_length: int = 400
    ) -> str:
        """
        Generate a contextual snippet around query matches

        Args:
            content: Full document content
            query: Search query
            max_length: Maximum snippet length

        Returns:
            Contextual snippet with surrounding text
        """
        query_terms = query.lower().split()
        content_lower = content.lower()

        # Find best match position
        best_pos = -1
        best_score = 0

        for i in range(len(content_lower)):
            score = sum(1 for term in query_terms if content_lower[i:i+100].count(term) > 0)
            if score > best_score:
                best_score = score
                best_pos = i

        if best_pos < 0:
            # No match, return beginning
            return content[:max_length] + ('...' if len(content) > max_length else '')

        # Extract snippet with context
        start = max(0, best_pos - self.context_chars_before)
        end = min(len(content), best_pos + self.context_chars_after)

        # Adjust to word boundaries
        if start > 0:
            space_pos = content.find(' ', start)
            if space_pos > 0 and space_pos < start + 20:
                start = space_pos + 1

        if end < len(content):
            space_pos = content.rfind(' ', end - 20, end)
            if space_pos > 0:
                end = space_pos

        snippet = content[start:end]

        # Add ellipsis
        if start > 0:
            snippet = '...' + snippet
        if end < len(content):
            snippet = snippet + '...'

        # Highlight matches
        if self.highlight_matches:
            for term in query_terms:
                if len(term) >= 3:  # Only highlight meaningful terms
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    snippet = pattern.sub(f'**{term}**', snippet)

        return snippet[:max_length]
