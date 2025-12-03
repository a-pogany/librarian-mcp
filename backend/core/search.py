"""Keyword-based document search engine"""

import re
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SearchEngine:
    """Keyword-based document search"""

    def __init__(self, indexer):
        self.indexer = indexer

    def search(
        self,
        query: str,
        product: Optional[str] = None,
        component: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search documents using keyword matching

        Args:
            query: Search keywords
            product: Filter by product
            component: Filter by component
            file_types: Filter by file extensions
            max_results: Maximum results to return

        Returns:
            List of matching documents with snippets
        """
        # Parse query into keywords
        keywords = self._parse_query(query)

        if not keywords:
            return []

        # Search across all documents
        results = []

        for path, doc in self.indexer.index.documents.items():
            # Apply filters
            if product and doc['product'] != product:
                continue

            if component and doc['component'] != component:
                continue

            if file_types and doc['file_type'] not in file_types:
                continue

            # Calculate relevance score
            score = self._calculate_score(doc, keywords, query)

            if score > 0:
                # Extract snippet
                snippet = self._extract_snippet(doc['content'], keywords)

                results.append({
                    'id': path,
                    'file_path': path,
                    'product': doc['product'],
                    'component': doc['component'],
                    'file_name': doc['file_name'],
                    'file_type': doc['file_type'],
                    'snippet': snippet,
                    'relevance_score': score,
                    'last_modified': doc['last_modified']
                })

        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Limit results
        return results[:max_results]

    def get_document(
        self,
        path: str,
        section: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get full document content

        Args:
            path: Relative path to document
            section: Optional section heading to extract

        Returns:
            Document content and metadata
        """
        doc = self.indexer.index.documents.get(path)

        if not doc:
            return None

        content = doc['content']

        # Extract specific section if requested
        if section and doc.get('headings'):
            content = self._extract_section(doc['content'], section, doc['headings'])

        return {
            'file_path': path,
            'product': doc['product'],
            'component': doc['component'],
            'file_name': doc['file_name'],
            'file_type': doc['file_type'],
            'content': content,
            'headings': doc.get('headings', []),
            'metadata': doc.get('metadata', {}),
            'size_bytes': doc['size_bytes'],
            'last_modified': doc['last_modified']
        }

    def _parse_query(self, query: str) -> List[str]:
        """Parse query into keywords"""
        # Remove special characters
        cleaned = re.sub(r'[^\w\s]', ' ', query.lower())

        # Split into words
        words = cleaned.split()

        # Filter short words
        keywords = [w for w in words if len(w) >= 2]

        return keywords

    def _calculate_score(self, doc: Dict, keywords: List[str], original_query: str) -> float:
        """
        Calculate relevance score for document

        Scoring:
        - File name match: 3 points per keyword
        - Heading match: 2 points per keyword
        - Content match: 1 point per keyword
        - Phrase match bonus: 5 points
        """
        score = 0.0
        content_lower = doc['content'].lower()
        file_name_lower = doc['file_name'].lower()

        # Check for phrase match
        if original_query.lower() in content_lower:
            score += 5.0

        # Score keywords
        for keyword in keywords:
            # File name matches
            if keyword in file_name_lower:
                score += 3.0

            # Heading matches
            for heading in doc.get('headings', []):
                if keyword in heading.lower():
                    score += 2.0

            # Content matches (count occurrences, but cap contribution)
            count = content_lower.count(keyword)
            score += min(count, 5) * 1.0

        # Normalize score (0-1 range)
        max_possible = len(keywords) * 10
        normalized = min(score / max_possible, 1.0) if max_possible > 0 else 0.0

        return round(normalized, 2)

    def _extract_snippet(self, content: str, keywords: List[str], max_length: int = 200) -> str:
        """Extract relevant snippet containing keywords"""
        lines = content.split('\n')

        # Find line with most keyword matches
        best_line = ""
        max_matches = 0

        for line in lines:
            line_lower = line.lower()
            matches = sum(1 for kw in keywords if kw in line_lower)

            if matches > max_matches:
                max_matches = matches
                best_line = line

        # If no matches, return first non-empty line
        if not best_line:
            for line in lines:
                if line.strip():
                    best_line = line
                    break

        # Truncate if too long
        snippet = best_line.strip()
        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "..."

        return snippet

    def _extract_section(self, content: str, section: str, headings: List[str]) -> str:
        """Extract specific section from document"""
        lines = content.split('\n')
        section_lower = section.lower()

        # Find section heading
        in_section = False
        section_lines = []

        for line in lines:
            # Check if this is the target section
            if line.strip().lower().startswith('#') and section_lower in line.lower():
                in_section = True
                section_lines.append(line)
                continue

            # Check if we've reached next section
            if in_section and line.strip().startswith('#'):
                break

            if in_section:
                section_lines.append(line)

        return '\n'.join(section_lines) if section_lines else content
