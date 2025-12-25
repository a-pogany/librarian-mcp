"""
Result enhancement for type-aware search result presentation.

Transforms raw search results into rich, type-specific formats
that provide immediate context without additional API calls.
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ResultEnhancer:
    """
    Enhances search results with type-specific metadata and summaries.

    Detects content type (email vs document) and extracts relevant
    fields for immediate presentation to users.
    """

    def __init__(self, summary_length: int = 150):
        """
        Args:
            summary_length: Maximum characters for content summary
        """
        self.summary_length = summary_length

    def enhance(self, results: List[Dict], include_full_metadata: bool = False) -> List[Dict]:
        """
        Enhance a list of search results.

        Args:
            results: Raw search results from search engine
            include_full_metadata: If True, include complete metadata dict

        Returns:
            Enhanced results with type-specific fields
        """
        enhanced = []
        for result in results:
            try:
                enhanced_result = self._enhance_single(result, include_full_metadata)
                enhanced.append(enhanced_result)
            except Exception as e:
                logger.warning(f"Failed to enhance result {result.get('id')}: {e}")
                enhanced.append(result)
        return enhanced

    def _enhance_single(self, result: Dict, include_full: bool) -> Dict:
        """Enhance a single result based on its type."""
        file_type = result.get('file_type', '')
        metadata = result.get('metadata', {})

        if file_type == '.eml' or metadata.get('doc_type') == 'email':
            enhanced = self._enhance_email(result)
        else:
            enhanced = self._enhance_document(result)

        if not include_full:
            enhanced.pop('metadata', None)
            enhanced.pop('content_preview', None)

        return enhanced

    def _enhance_email(self, result: Dict) -> Dict:
        """Enhance email result with sender, recipients, subject, etc."""
        metadata = result.get('metadata', {})
        content = result.get('content_preview') or result.get('content', '')

        enhanced = dict(result)
        enhanced.update({
            'type': 'email',
            'title': metadata.get('subject') or 'No Subject',
            'summary': self._generate_summary(content),
            'date': metadata.get('date_iso') or metadata.get('date'),
            'from': metadata.get('from'),
            'to': metadata.get('to', []),
            'cc': metadata.get('cc', []),
            'has_attachments': metadata.get('has_attachments', False),
            'attachment_count': metadata.get('attachment_count', 0),
            'attachments': metadata.get('attachments', []),
            'thread_id': metadata.get('thread_id')
        })

        return enhanced

    def _enhance_document(self, result: Dict) -> Dict:
        """Enhance document result with title, headings, tags, etc."""
        metadata = result.get('metadata', {})
        headings = result.get('headings', [])
        content = result.get('content_preview') or result.get('content', '')

        title = headings[0] if headings else result.get('file_name', 'Untitled')

        enhanced = dict(result)
        enhanced.update({
            'type': 'document',
            'title': title,
            'summary': self._generate_summary(content),
            'date': result.get('last_modified'),
            'headings': headings[:5],
            'doc_type': result.get('doc_type') or metadata.get('doc_type'),
            'tags': result.get('tags', []),
            'product': result.get('product'),
            'component': result.get('component')
        })

        return enhanced

    def _generate_summary(self, content: str) -> str:
        """Generate a short summary from content."""
        if not content:
            return ''

        clean = ' '.join(content.split())

        if len(clean) <= self.summary_length:
            return clean

        truncated = clean[:self.summary_length]

        for end in ['. ', '! ', '? ']:
            last_end = truncated.rfind(end)
            if last_end > self.summary_length * 0.5:
                return truncated[:last_end + 1]

        last_space = truncated.rfind(' ')
        if last_space > self.summary_length * 0.7:
            return truncated[:last_space] + '...'

        return truncated + '...'
