# Librarian MCP Enhancement Proposal - 2024-12-21

## Executive Summary

This document captures planned enhancements for the Librarian MCP search result presentation. The goal is to provide users with immediate, actionable information in search results without requiring additional `get_document` calls.

---

## Problem Statement

### Current User Experience

When a user searches for documents or emails, they receive results like:

```json
{
    "file_path": "emails/2024/message.eml",
    "snippet": "...the project deadline...",
    "relevance_score": 0.85
}
```

**Issues:**
1. **No context at a glance** - User cannot tell WHO sent an email or WHAT a document is about
2. **Extra API calls required** - Must call `get_document` for each result to see basic info
3. **No type differentiation** - Emails and documents look identical in results
4. **Wasted tokens** - LLM must make multiple round-trips to gather basic information

### Why This Matters

| Issue | Impact |
|-------|--------|
| Missing sender/recipient info | User can't prioritize emails by sender importance |
| No subject/title | Can't quickly scan results for relevance |
| No date visible | Can't filter mentally by recency |
| No attachment indicator | Misses important documents attached to emails |
| Same format for all types | Cognitive overhead to distinguish content types |

---

## Current System Analysis

### Data Available But Not Exposed

The system already captures rich metadata during indexing, but search results don't include it.

**Email Metadata** (stored in `doc['metadata']` from `EMLParser.parse()`):
```python
{
    'doc_type': 'email',
    'message_id': '<unique-id@domain.com>',
    'thread_id': 'computed-thread-hash',
    'from': 'sender@example.com',
    'to': ['recipient1@example.com', 'recipient2@example.com'],
    'cc': ['cc@example.com'],
    'date': '2024-01-15T10:30:00',
    'subject': 'Re: Project Timeline Discussion',
    'subject_normalized': 'Project Timeline Discussion',
    'has_attachments': True,
    'attachment_count': 2,
    'attachments': [
        {'filename': 'report.pdf', 'type': 'application/pdf', 'size': '2.3 MB'}
    ],
    'preprocessing': {'quotes_removed': 3, 'signature_removed': True}
}
```

**Document Metadata** (stored in `doc` fields):
```python
{
    'headings': ['Getting Started', 'Installation', 'Configuration'],
    'doc_type': 'guide',  # api, guide, architecture, reference, readme
    'tags': ['authentication', 'api', 'security'],
    'last_modified': '2024-01-10T09:00:00',
    'indexed_at': '2024-01-20T14:00:00'
}
```

### Where Metadata Gets Lost

In `backend/core/search.py:SearchEngine.search()` (lines 58-68):
```python
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
    # NOTE: doc['metadata'] is NOT included!
})
```

---

## Proposed Solution

### Design: Type-Aware Result Enhancement

Create a `ResultEnhancer` component that:
1. Detects content type (email vs document) from file extension or metadata
2. Extracts relevant fields based on type
3. Generates summary from content
4. Returns unified but type-specific result format

### Architecture

```
Search Query
     │
     ▼
┌─────────────────┐
│ HybridSearch    │
│ Engine          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ResultEnhancer  │  ◄── NEW COMPONENT
│                 │
│ ┌─────────────┐ │
│ │EmailEnhancer│ │  - Extracts from, to, subject, date
│ └─────────────┘ │  - Formats attachments
│                 │
│ ┌─────────────┐ │
│ │ DocEnhancer │ │  - Extracts title from headings
│ └─────────────┘ │  - Includes doc_type, tags
└────────┬────────┘
         │
         ▼
   Enhanced Results
```

### Enhanced Result Schema

#### Common Fields (All Types)
```python
{
    # Identification
    'id': str,                      # Unique identifier
    'file_path': str,               # Path to file
    'type': 'email' | 'document',   # Content type indicator

    # Relevance
    'relevance_score': float,       # Search relevance (0-1)
    'snippet': str,                 # Context around keyword match

    # Quick Preview
    'title': str,                   # Subject (email) or first heading (doc)
    'summary': str,                 # First ~150 chars of content
    'date': str,                    # Email date or last_modified
}
```

#### Email-Specific Fields
```python
{
    # Participants
    'from': str,                    # Sender email address
    'to': List[str],                # Recipient list
    'cc': List[str],                # CC list (if any)

    # Attachments
    'has_attachments': bool,        # Quick check
    'attachment_count': int,        # Number of attachments
    'attachments': List[dict],      # [{filename, type, size}]

    # Threading
    'thread_id': str,               # For grouping conversations
}
```

#### Document-Specific Fields
```python
{
    # Structure
    'headings': List[str],          # Document outline (first 5)

    # Classification
    'doc_type': str,                # api, guide, architecture, etc.
    'tags': List[str],              # Frontmatter tags

    # Location
    'product': str,                 # Product category
    'component': str,               # Component category
}
```

---

## Implementation Plan

### Files to Create

#### 1. `backend/core/result_enhancer.py` (NEW)

```python
"""
Result enhancement for type-aware search result presentation.

Transforms raw search results into rich, type-specific formats
that provide immediate context without additional API calls.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
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
                enhanced.append(result)  # Return original on error
        return enhanced

    def _enhance_single(self, result: Dict, include_full: bool) -> Dict:
        """Enhance a single result based on its type."""
        file_type = result.get('file_type', '')
        metadata = result.get('metadata', {})

        if file_type == '.eml' or metadata.get('doc_type') == 'email':
            return self._enhance_email(result, include_full)
        else:
            return self._enhance_document(result, include_full)

    def _enhance_email(self, result: Dict, include_full: bool) -> Dict:
        """Enhance email result with sender, recipients, subject, etc."""
        metadata = result.get('metadata', {})
        content = result.get('content', '')

        enhanced = {
            # Common fields
            'id': result.get('id'),
            'file_path': result.get('file_path'),
            'type': 'email',
            'relevance_score': result.get('relevance_score'),
            'snippet': result.get('snippet'),

            # Title and summary
            'title': metadata.get('subject', 'No Subject'),
            'summary': self._generate_summary(content),
            'date': metadata.get('date'),

            # Email-specific
            'from': metadata.get('from'),
            'to': metadata.get('to', []),
            'cc': metadata.get('cc', []),
            'has_attachments': metadata.get('has_attachments', False),
            'attachment_count': metadata.get('attachment_count', 0),
            'attachments': metadata.get('attachments', []),
            'thread_id': metadata.get('thread_id'),
        }

        if include_full:
            enhanced['metadata'] = metadata

        return enhanced

    def _enhance_document(self, result: Dict, include_full: bool) -> Dict:
        """Enhance document result with title, headings, tags, etc."""
        metadata = result.get('metadata', {})
        headings = result.get('headings', [])
        content = result.get('content', '')

        # Determine title: first heading, or filename
        title = headings[0] if headings else result.get('file_name', 'Untitled')

        enhanced = {
            # Common fields
            'id': result.get('id'),
            'file_path': result.get('file_path'),
            'type': 'document',
            'relevance_score': result.get('relevance_score'),
            'snippet': result.get('snippet'),

            # Title and summary
            'title': title,
            'summary': self._generate_summary(content),
            'date': result.get('last_modified'),

            # Document-specific
            'headings': headings[:5],  # First 5 headings as outline
            'doc_type': result.get('doc_type') or metadata.get('doc_type'),
            'tags': result.get('tags', []),
            'product': result.get('product'),
            'component': result.get('component'),
        }

        if include_full:
            enhanced['metadata'] = metadata

        return enhanced

    def _generate_summary(self, content: str) -> str:
        """Generate a short summary from content."""
        if not content:
            return ''

        # Clean content: remove extra whitespace
        clean = ' '.join(content.split())

        # Truncate to summary length
        if len(clean) <= self.summary_length:
            return clean

        # Find a good break point (end of sentence or word)
        truncated = clean[:self.summary_length]

        # Try to break at sentence end
        for end in ['. ', '! ', '? ']:
            last_end = truncated.rfind(end)
            if last_end > self.summary_length * 0.5:
                return truncated[:last_end + 1]

        # Break at word boundary
        last_space = truncated.rfind(' ')
        if last_space > self.summary_length * 0.7:
            return truncated[:last_space] + '...'

        return truncated + '...'
```

### Files to Modify

#### 2. `backend/core/search.py` - Include metadata in results

**Change in `SearchEngine.search()` method (around line 58):**

```python
# BEFORE:
results.append({
    'id': path,
    'file_path': path,
    'product': doc['product'],
    # ... other fields
    'last_modified': doc['last_modified']
})

# AFTER:
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
    # NEW: Include metadata for enhancement
    'metadata': doc.get('metadata', {}),
    'headings': doc.get('headings', []),
    'content': doc.get('content', '')[:500],  # First 500 chars for summary
    'doc_type': doc.get('doc_type'),
    'tags': doc.get('tags', [])
})
```

#### 3. `backend/core/hybrid_search.py` - Integrate ResultEnhancer

**Add import and initialization:**
```python
from .result_enhancer import ResultEnhancer

class HybridSearchEngine:
    def __init__(self, ...):
        # ... existing code ...
        self.result_enhancer = ResultEnhancer(summary_length=150)
```

**Modify search method to use enhancer:**
```python
def search(self, ..., enhance_results: bool = True) -> List[Dict]:
    # ... existing search logic ...

    # NEW: Enhance results before returning
    if enhance_results:
        results = self.result_enhancer.enhance(results)

    return results
```

#### 4. `backend/mcp_server/tools.py` - Add enhancement parameter

**Modify `search_documentation` and `search_emails`:**
```python
@mcp.tool()
def search_documentation(
    query: str,
    # ... existing params ...
    enhance_results: bool = True  # NEW: Enable enhanced formatting
) -> dict:
    """
    Args:
        # ... existing docs ...
        enhance_results: Include rich metadata (title, summary, email fields)
    """
```

---

## Example Output Comparison

### Before Enhancement

```json
{
    "results": [
        {
            "file_path": "emails/2024-01/project-update.eml",
            "snippet": "...the deadline has been moved to...",
            "relevance_score": 0.92
        },
        {
            "file_path": "docs/symphony/api/authentication.md",
            "snippet": "...configure OAuth tokens...",
            "relevance_score": 0.87
        }
    ]
}
```

### After Enhancement

```json
{
    "results": [
        {
            "type": "email",
            "title": "Re: Project Timeline Update",
            "from": "john.smith@company.com",
            "to": ["team@company.com", "stakeholders@company.com"],
            "date": "2024-01-15T14:30:00",
            "summary": "Hi team, I wanted to let you know that the deadline has been moved to February 15th due to the additional requirements...",
            "has_attachments": true,
            "attachment_count": 1,
            "attachments": [{"filename": "new_timeline.xlsx", "size": "45 KB"}],
            "thread_id": "abc123...",
            "snippet": "...the deadline has been moved to...",
            "relevance_score": 0.92
        },
        {
            "type": "document",
            "title": "Authentication Configuration Guide",
            "headings": ["Overview", "OAuth Setup", "JWT Tokens", "Troubleshooting"],
            "summary": "This guide explains how to configure authentication for the Symphony API. It covers OAuth 2.0 setup, JWT token management...",
            "doc_type": "guide",
            "tags": ["authentication", "oauth", "security"],
            "product": "symphony",
            "component": "api",
            "date": "2024-01-10T09:00:00",
            "snippet": "...configure OAuth tokens...",
            "relevance_score": 0.87
        }
    ]
}
```

---

## Benefits Summary

| Enhancement | Benefit | Impact |
|-------------|---------|--------|
| **Type indicator** | Instantly know if result is email or document | Faster mental filtering |
| **Title/Subject** | Understand content without opening | 80% fewer get_document calls |
| **From/To fields** | Identify sender importance | Prioritize by person |
| **Date visible** | Temporal context at a glance | Filter by recency |
| **Summary** | Quick content preview | Decide relevance faster |
| **Attachments** | See if files are attached | Find documents in emails |
| **Thread ID** | Group related emails | Conversation context |
| **Doc headings** | Document structure preview | Navigate to right section |
| **Tags** | Classification visible | Topic-based filtering |

---

## Testing Strategy

1. **Unit tests** for `ResultEnhancer`:
   - Test email enhancement with various metadata combinations
   - Test document enhancement with/without headings
   - Test summary generation edge cases

2. **Integration tests**:
   - Search returns enhanced results by default
   - `enhance_results=False` returns original format
   - Both `search_documentation` and `search_emails` support enhancement

3. **Backward compatibility**:
   - Existing clients work without changes
   - Enhancement is additive (new fields, existing fields unchanged)

---

## Implementation Order

1. Create `backend/core/result_enhancer.py`
2. Add unit tests for ResultEnhancer
3. Modify `backend/core/search.py` to include metadata
4. Modify `backend/core/hybrid_search.py` to use enhancer
5. Update MCP tools with `enhance_results` parameter
6. Run full test suite
7. Manual testing with Claude Desktop

---

## Session Notes - 2024-12-21

### Completed Today
1. Fixed persistence bug - config chain was broken (`main.py` passed `config['docs']` instead of full `config`)
2. Fixed in both `main.py` and `stdio_server.py`
3. Fixed `watch_for_changes` config access in `indexer.py`
4. All 48 core tests passing (5 pre-existing failures in test_reranking.py unrelated)

### Ready for Next Session
- Implement ResultEnhancer as designed above
- This document contains complete implementation specs
