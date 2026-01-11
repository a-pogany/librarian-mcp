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
        max_results: int = 10,
        # Email-specific filters (pre-filter before scoring)
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        cc: Optional[str] = None,
        folder: Optional[str] = None,
        subject_contains: Optional[str] = None,
        has_attachments: Optional[bool] = None,
        date_after: Optional[str] = None,
        date_before: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search documents using keyword matching

        Args:
            query: Search keywords
            product: Filter by product
            component: Filter by component
            file_types: Filter by file extensions
            max_results: Maximum results to return
            sender: Filter emails by sender (partial match)
            recipient: Filter emails by recipient (partial match in to field)
            cc: Filter emails by CC recipient (partial match)
            folder: Filter emails by folder (exact match, case-insensitive)
            subject_contains: Filter emails by subject (partial match)
            has_attachments: Filter emails with/without attachments
            date_after: Filter emails after this date (ISO 8601)
            date_before: Filter emails before this date (ISO 8601)
            thread_id: Filter emails by thread ID (exact match)

        Returns:
            List of matching documents with snippets
        """
        # Parse query into keywords
        keywords = self._parse_query(query)

        if not keywords:
            return []

        # Build email filters dict
        email_filters = {
            'sender': sender,
            'recipient': recipient,
            'cc': cc,
            'folder': folder,
            'subject_contains': subject_contains,
            'has_attachments': has_attachments,
            'date_after': date_after,
            'date_before': date_before,
            'thread_id': thread_id
        }
        # Remove None values
        email_filters = {k: v for k, v in email_filters.items() if v is not None}

        # Search across all documents
        results = []

        for path, doc in self.indexer.index.documents.items():
            # Apply basic filters
            if product and doc['product'] != product:
                continue

            if component and doc['component'] != component:
                continue

            if file_types and doc['file_type'] not in file_types:
                continue

            # Apply email-specific filters BEFORE scoring (pre-filter)
            if email_filters and not self._matches_email_filters(doc, email_filters):
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
                    'last_modified': doc['last_modified'],
                    'metadata': doc.get('metadata', {}),
                    'headings': doc.get('headings', []),
                    'content_preview': doc.get('content', '')[:500],
                    'doc_type': doc.get('doc_type'),
                    'tags': doc.get('tags', [])
                })

        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Limit results
        return results[:max_results]

    def _matches_email_filters(self, doc: Dict, filters: Dict) -> bool:
        """Check if document matches email-specific filters (pre-filter logic)

        Args:
            doc: Document dict from index
            filters: Dict of filter name -> value (only non-None values)

        Returns:
            True if doc matches all filters, False otherwise
        """
        # Get email metadata from document
        metadata = doc.get('metadata', {})

        # Only emails have email metadata
        if not metadata and any(filters.values()):
            # If filtering for email fields but doc has no metadata, skip
            return False

        # Sender filter (partial match, case-insensitive)
        if 'sender' in filters:
            sender_val = metadata.get('from', '')
            if not sender_val or filters['sender'].lower() not in sender_val.lower():
                return False

        # Recipient filter (partial match in 'to' list)
        if 'recipient' in filters:
            to_list = metadata.get('to', [])
            if isinstance(to_list, str):
                to_list = [to_list]
            recipient_lower = filters['recipient'].lower()
            found = any(recipient_lower in addr.lower() for addr in to_list if addr)
            if not found:
                return False

        # CC filter (partial match in 'cc' list)
        if 'cc' in filters:
            cc_list = metadata.get('cc', [])
            if isinstance(cc_list, str):
                cc_list = [cc_list]
            cc_lower = filters['cc'].lower()
            found = any(cc_lower in addr.lower() for addr in cc_list if addr)
            if not found:
                return False

        # Folder filter (case-insensitive exact match)
        if 'folder' in filters:
            folder_val = metadata.get('folder', '')
            if not folder_val or folder_val.lower() != filters['folder'].lower():
                return False

        # Subject filter (partial match, case-insensitive)
        if 'subject_contains' in filters:
            subject_val = metadata.get('subject', '')
            if not subject_val or filters['subject_contains'].lower() not in subject_val.lower():
                return False

        # Has attachments filter (boolean)
        if 'has_attachments' in filters:
            has_attach = metadata.get('has_attachments', False)
            if has_attach != filters['has_attachments']:
                return False

        # Date range filters (ISO 8601 string comparison)
        if 'date_after' in filters:
            date_val = metadata.get('date', '')
            if not date_val or date_val < filters['date_after']:
                return False

        if 'date_before' in filters:
            date_val = metadata.get('date', '')
            if not date_val or date_val > filters['date_before']:
                return False

        # Thread ID filter (exact match)
        if 'thread_id' in filters:
            thread_val = metadata.get('thread_id', '')
            if thread_val != filters['thread_id']:
                return False

        return True

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
