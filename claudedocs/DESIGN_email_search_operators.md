# Design: Email Search Operators

**Feature:** Outlook/Gmail-style search operators for email search
**Status:** Design Draft
**Version:** 1.0
**Date:** 2025-01-11

---

## 1. Executive Summary

### The Request
Add support for inline search operators (e.g., `from:kiraly`, `to:john`, `in:important`) that allow users to combine free-text search with structured filters in a single query string.

### The Answer: Yes, It Makes Sense for ALL Search Modes

| Search Mode | Relevance | Reason |
|-------------|-----------|--------|
| **Keyword** | :white_check_mark: High | Operators pre-filter the corpus before keyword matching |
| **Semantic** | :white_check_mark: High | Operators filter which chunks get vector similarity comparison |
| **Hybrid** | :white_check_mark: High | Both keyword and semantic benefit from corpus filtering |
| **Rerank** | :white_check_mark: High | Reduces candidate set before expensive reranking |
| **HyDE** | :white_check_mark: High | Focuses hypothetical document matching on relevant subset |
| **Auto** | :white_check_mark: High | Query router analyzes text-only portion; filters apply after routing |

**Key Insight:** Operators are **pre-filters** applied before any search mode executes. They reduce the search space, improving both relevance and performance for ALL modes.

---

## 2. Proposed Operators

### Core Operators (Priority 1)

| Operator | Syntax | Description | Maps To |
|----------|--------|-------------|---------|
| `from:` | `from:kiraly` | Sender contains value | `sender` filter |
| `to:` | `to:john` | Recipient contains value | NEW: `recipient` filter |
| `subject:` | `subject:meeting` | Subject contains value | `subject_contains` filter |
| `in:` | `in:important` | Email folder/label | NEW: `folder` filter |
| `has:` | `has:attachment` | Has attachments | `has_attachments=True` |

### Extended Operators (Priority 2)

| Operator | Syntax | Description | Maps To |
|----------|--------|-------------|---------|
| `cc:` | `cc:team` | CC field contains value | NEW: `cc` filter |
| `after:` | `after:2024-01-01` | Date after | `date_after` filter |
| `before:` | `before:2024-12-31` | Date before | `date_before` filter |
| `thread:` | `thread:abc123` | Thread ID | `thread_id` filter |
| `is:` | `is:unread` | Email state | Future: requires state tracking |

### Operator Syntax Rules

```
operator    := name ":" value
name        := "from" | "to" | "cc" | "subject" | "in" | "has" | "after" | "before" | "thread"
value       := quoted_string | unquoted_token
quoted_string := '"' [^"]* '"'
unquoted_token := [^\s]+
```

**Examples:**
- `from:kiraly` - Simple unquoted value
- `from:"Kiraly Attila"` - Quoted value with spaces
- `subject:"Q4 Report"` - Subject with spaces
- `in:inbox from:boss urgent` - Multiple operators + free text

---

## 3. Architecture

### 3.1 Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│  User Query: "from:kiraly project deadline has:attachment"  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  QueryOperatorParser (NEW)                                  │
│  ├─ Extract operators: {from: "kiraly", has: "attachment"} │
│  └─ Extract free text: "project deadline"                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  search_emails() in tools.py                                │
│  ├─ Convert operators to existing filter params             │
│  ├─ Pass free text as query                                │
│  └─ Execute search with filters                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  HybridSearchEngine.search()                                │
│  ├─ Apply filters (reduced search space)                   │
│  └─ Execute search mode (keyword/semantic/hybrid/etc.)     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 New Module: `backend/core/query_parser.py`

```python
"""
Query Operator Parser for Outlook/Gmail-style search operators.

Extracts structured operators from query strings while preserving
free-text search terms for full-text search.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class ParsedQuery:
    """Result of parsing a query with operators"""

    # Free text for full-text search (operators removed)
    free_text: str

    # Extracted operators
    operators: Dict[str, str] = field(default_factory=dict)

    # Original query for logging/debugging
    original_query: str = ""

    # Parsing metadata
    operator_count: int = 0
    parse_warnings: List[str] = field(default_factory=list)

    def has_operators(self) -> bool:
        return self.operator_count > 0

    def get_filter_params(self) -> Dict[str, any]:
        """Convert operators to search_emails filter parameters"""
        params = {}

        # Direct mappings
        if 'from' in self.operators:
            params['sender'] = self.operators['from']
        if 'to' in self.operators:
            params['recipient'] = self.operators['to']
        if 'cc' in self.operators:
            params['cc'] = self.operators['cc']
        if 'subject' in self.operators:
            params['subject_contains'] = self.operators['subject']
        if 'in' in self.operators:
            params['folder'] = self.operators['in']
        if 'thread' in self.operators:
            params['thread_id'] = self.operators['thread']
        if 'after' in self.operators:
            params['date_after'] = self.operators['after']
        if 'before' in self.operators:
            params['date_before'] = self.operators['before']

        # Boolean operators
        if 'has' in self.operators:
            value = self.operators['has'].lower()
            if value in ('attachment', 'attachments'):
                params['has_attachments'] = True

        return params


class QueryOperatorParser:
    """
    Parser for extracting search operators from query strings.

    Supports Outlook/Gmail-style operators:
    - from:sender - Filter by sender
    - to:recipient - Filter by recipient
    - cc:recipient - Filter by CC
    - subject:text - Filter by subject
    - in:folder - Filter by folder/label
    - has:attachment - Filter by attachments
    - after:date - Filter by date (after)
    - before:date - Filter by date (before)
    - thread:id - Filter by thread ID
    """

    # Supported operators
    OPERATORS = {
        'from', 'to', 'cc', 'subject', 'in',
        'has', 'after', 'before', 'thread'
    }

    # Regex pattern for operator:value (handles quoted values)
    # Matches: operator:"quoted value" or operator:unquoted_value
    OPERATOR_PATTERN = re.compile(
        r'\b(' + '|'.join(OPERATORS) + r'):("([^"]+)"|(\S+))',
        re.IGNORECASE
    )

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse query string and extract operators.

        Args:
            query: Raw query string with potential operators

        Returns:
            ParsedQuery with operators extracted and free text remaining
        """
        if not query:
            return ParsedQuery(free_text="", original_query="")

        operators = {}
        warnings = []

        # Find all operator matches
        matches = list(self.OPERATOR_PATTERN.finditer(query))

        for match in matches:
            op_name = match.group(1).lower()
            # Group 3 is quoted value (without quotes), Group 4 is unquoted
            value = match.group(3) or match.group(4)

            if op_name in operators:
                warnings.append(f"Duplicate operator '{op_name}' - using last value")

            operators[op_name] = value

        # Remove operators from query to get free text
        free_text = self.OPERATOR_PATTERN.sub('', query).strip()
        # Normalize whitespace
        free_text = ' '.join(free_text.split())

        return ParsedQuery(
            free_text=free_text,
            operators=operators,
            original_query=query,
            operator_count=len(operators),
            parse_warnings=warnings
        )

    def extract_and_validate(
        self,
        query: str,
        allowed_operators: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, str], List[str]]:
        """
        Extract operators with validation.

        Args:
            query: Raw query string
            allowed_operators: Optional whitelist of allowed operators

        Returns:
            Tuple of (free_text, valid_operators, warnings)
        """
        parsed = self.parse(query)

        if allowed_operators:
            allowed_set = set(op.lower() for op in allowed_operators)
            invalid = set(parsed.operators.keys()) - allowed_set
            for op in invalid:
                parsed.parse_warnings.append(
                    f"Unsupported operator '{op}' ignored"
                )
                del parsed.operators[op]

        return (
            parsed.free_text,
            parsed.operators,
            parsed.parse_warnings
        )
```

### 3.3 Integration with `search_emails()`

**Modified `backend/mcp_server/tools.py`:**

```python
from backend.core.query_parser import QueryOperatorParser

# Module-level parser instance
query_parser = QueryOperatorParser()

@mcp.tool()
def search_emails(
    query: str,
    # ... existing parameters ...
    parse_operators: bool = True,  # NEW: Enable/disable operator parsing
) -> dict:
    """
    Search emails with email-specific filters.

    Supports inline search operators (Outlook/Gmail style):
    - from:sender - Emails from sender (partial match)
    - to:recipient - Emails to recipient (partial match)
    - cc:recipient - Emails with CC recipient (partial match)
    - subject:text - Subject contains text
    - in:folder - Emails in folder/label
    - has:attachment - Emails with attachments
    - after:YYYY-MM-DD - Emails after date
    - before:YYYY-MM-DD - Emails before date
    - thread:id - Emails in thread

    Examples:
        search_emails("from:kiraly project deadline")
        search_emails("in:important has:attachment budget")
        search_emails('subject:"Q4 Report" from:finance')

    Args:
        query: Search text with optional operators
        parse_operators: Whether to parse inline operators (default: True)
        ... existing args ...
    """
    try:
        # Parse operators from query
        parsed_operators = {}
        search_query = query
        parse_warnings = []

        if parse_operators:
            parsed = query_parser.parse(query)
            search_query = parsed.free_text or query  # Fallback to full query if no text
            parsed_operators = parsed.get_filter_params()
            parse_warnings = parsed.parse_warnings

        # Merge parsed operators with explicit parameters
        # Explicit parameters take precedence
        effective_sender = sender or parsed_operators.get('sender')
        effective_recipient = parsed_operators.get('recipient')  # NEW
        effective_cc = parsed_operators.get('cc')  # NEW
        effective_subject = subject_contains or parsed_operators.get('subject_contains')
        effective_folder = parsed_operators.get('folder')  # NEW
        effective_thread = thread_id or parsed_operators.get('thread_id')
        effective_date_after = date_after or parsed_operators.get('date_after')
        effective_date_before = date_before or parsed_operators.get('date_before')

        # has:attachment handling
        effective_has_attachments = has_attachments
        if effective_has_attachments is None and 'has_attachments' in parsed_operators:
            effective_has_attachments = parsed_operators['has_attachments']

        # Execute search with merged filters
        results = search_engine.search(
            query=search_query,
            file_types=['.eml'],
            max_results=min(max_results, 50) * candidate_multiplier,
            mode=mode,
            include_parent_context=include_parent_context,
            enhance_results=False,
            max_per_document=max_per_document
        )

        # Apply email-specific filters (extended)
        filtered_results = _apply_email_filters(
            results,
            sender=effective_sender,
            recipient=effective_recipient,  # NEW
            cc=effective_cc,  # NEW
            thread_id=effective_thread,
            subject_contains=effective_subject,
            folder=effective_folder,  # NEW
            has_attachments=effective_has_attachments,
            date_after=effective_date_after,
            date_before=effective_date_before
        )

        # ... rest of function ...

        return {
            "results": filtered_results[:max_results],
            "total": len(filtered_results),
            "query": search_query,  # Show parsed query
            "original_query": query,  # Keep original for debugging
            "parsed_operators": parsed_operators if parse_operators else {},
            "parse_warnings": parse_warnings,
            "search_mode": actual_mode,
            "filters": { ... }
        }
```

---

## 4. New Metadata: Folder Support

### 4.1 Problem

Currently, EML files don't capture folder information. The `in:` operator needs folder metadata.

### 4.2 Solutions

#### Option A: Extract from File Path (Recommended)

If PST exports maintain folder hierarchy in the file system:

```
docs/
  emails/
    inbox/
      message1.eml
    important/
      message2.eml
    sent/
      message3.eml
```

**Implementation in `EMLParser.parse()`:**

```python
def parse(self, file_path: str) -> ParseResult:
    # ... existing code ...

    # Extract folder from path
    # e.g., docs/emails/important/msg.eml -> "important"
    path_parts = Path(file_path).parts
    folder = self._infer_folder(path_parts)

    metadata = {
        # ... existing fields ...
        'folder': folder,  # NEW
    }
```

**Folder inference logic:**

```python
def _infer_folder(self, path_parts: Tuple[str, ...]) -> str:
    """
    Infer email folder from file path.

    Examples:
        ('docs', 'emails', 'inbox', 'msg.eml') -> 'inbox'
        ('docs', 'emails', 'sent', '2024', 'msg.eml') -> 'sent'
        ('docs', 'emails', 'msg.eml') -> 'root'
    """
    # Look for common email folder names
    KNOWN_FOLDERS = {
        'inbox', 'sent', 'drafts', 'deleted', 'trash',
        'archive', 'important', 'spam', 'junk',
        'outbox', 'flagged', 'starred'
    }

    for part in reversed(path_parts[:-1]):  # Exclude filename
        if part.lower() in KNOWN_FOLDERS:
            return part.lower()

    # Fallback: use immediate parent directory
    if len(path_parts) > 1:
        parent = path_parts[-2].lower()
        if parent not in ('emails', 'eml', 'mail'):
            return parent

    return 'root'
```

#### Option B: X-Folder Header

Some email clients add custom headers like `X-Folder` during export:

```python
def _extract_headers(self, msg) -> dict:
    # ... existing headers ...
    headers['folder'] = self._safe_header_string(msg.get('X-Folder', ''))
```

#### Option C: User-Specified in Frontmatter

For manually organized emails, allow folder specification in a sidecar file or filename pattern:

```
# message.meta.yaml
folder: important
priority: high
```

### 4.3 Recommended Approach

**Phase 1:** Option A (path-based) - Works immediately with existing data
**Phase 2:** Option B (X-Folder) - Check during parsing, use if available
**Phase 3:** Option C (sidecar) - For power users who need custom organization

---

## 5. Extended Filter Support

### 5.1 New Filters for `_apply_email_filters()`

```python
def _apply_email_filters(
    results: List[dict],
    sender: Optional[str] = None,
    recipient: Optional[str] = None,  # NEW: to field
    cc: Optional[str] = None,  # NEW: cc field
    thread_id: Optional[str] = None,
    subject_contains: Optional[str] = None,
    folder: Optional[str] = None,  # NEW: folder/label
    has_attachments: Optional[bool] = None,
    date_after: Optional[str] = None,
    date_before: Optional[str] = None
) -> List[dict]:
    """Apply email-specific filters to search results"""
    filtered = []

    for result in results:
        metadata = result.get('metadata', {})

        # ... existing filters ...

        # NEW: Recipient filter (to field)
        if recipient:
            result_to = metadata.get('to', '')
            if recipient.lower() not in result_to.lower():
                continue

        # NEW: CC filter
        if cc:
            result_cc = metadata.get('cc', '')
            if cc.lower() not in result_cc.lower():
                continue

        # NEW: Folder filter
        if folder:
            result_folder = metadata.get('folder', '')
            if folder.lower() != result_folder.lower():
                continue

        filtered.append(result)

    return filtered
```

---

## 6. Search Mode Integration Details

### 6.1 How Operators Work with Each Mode

```
┌────────────────────────────────────────────────────────────────┐
│  Query: "from:kiraly budget report has:attachment"            │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  Step 1: Parse Operators                                       │
│  ├─ Operators: {from: "kiraly", has: "attachment"}            │
│  └─ Free text: "budget report"                                │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  Step 2: Execute Search Mode on "budget report"               │
│                                                                │
│  KEYWORD MODE:                                                 │
│  └─ BM25 search for "budget report" in all emails            │
│                                                                │
│  SEMANTIC MODE:                                                │
│  └─ Vector similarity for "budget report" embedding          │
│                                                                │
│  HYBRID MODE:                                                  │
│  └─ RRF fusion of keyword + semantic for "budget report"     │
│                                                                │
│  HYDE MODE:                                                    │
│  └─ Generate hypothetical answer for "budget report"         │
│  └─ Vector similarity for hypothetical + query embedding     │
│                                                                │
│  AUTO MODE:                                                    │
│  └─ Analyze "budget report" → route to optimal mode          │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  Step 3: Post-Filter Results                                   │
│  ├─ Apply from:kiraly filter                                  │
│  ├─ Apply has:attachment filter                               │
│  └─ Return filtered, ranked results                           │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 Performance Optimization: Pre-filtering vs Post-filtering

**Current Design (Post-filtering):**
- Operators applied AFTER search completes
- Simple, works with existing architecture
- May retrieve unnecessary results

**Future Optimization (Pre-filtering):**
- Operators filter ChromaDB `where` clause
- Fewer vectors compared
- Significant speedup for selective filters

```python
# Future: Pre-filter in ChromaDB
results = self.vector_db.query(
    query_embedding=embedding,
    n_results=50,
    where={
        "$and": [
            {"metadata.from": {"$contains": "kiraly"}},
            {"metadata.has_attachments": {"$eq": True}}
        ]
    }
)
```

**Recommendation:** Start with post-filtering (simpler), add pre-filtering in Phase 2 for performance.

---

## 7. Edge Cases & Validation

### 7.1 Edge Cases

| Scenario | Handling |
|----------|----------|
| Query with only operators | Use operators as filters, empty free text |
| Unknown operator | Warn user, treat as regular text |
| Duplicate operators | Use last value, warn user |
| Malformed date | Warn user, skip filter |
| Empty operator value | Warn user, skip filter |
| Quoted value with quotes inside | Escape with backslash |
| Case sensitivity | Operators lowercase, values case-insensitive match |

### 7.2 Validation Examples

```python
# Only operators, no free text
parse("from:kiraly to:john")
# → ParsedQuery(free_text="", operators={from: "kiraly", to: "john"})
# → Searches ALL emails, filters by sender and recipient

# Unknown operator
parse("custom:value project")
# → ParsedQuery(free_text="custom:value project", operators={})
# → Warning: "Unknown operator 'custom' treated as search text"

# Malformed date
parse("after:not-a-date meeting")
# → ParsedQuery(free_text="meeting", operators={after: "not-a-date"})
# → Warning: "Invalid date format for 'after' operator"

# Empty operator value
parse("from: project")  # Note space after colon
# → ParsedQuery(free_text="from: project", operators={})
# → Treated as regular text (no value after colon)
```

---

## 8. User Experience

### 8.1 Help Text / Documentation

**MCP Tool Description:**

```
Supports inline search operators (Outlook/Gmail style):
- from:sender      Emails from sender (partial match)
- to:recipient     Emails to recipient (partial match)
- subject:text     Subject contains text
- in:folder        Emails in folder (inbox, sent, important, etc.)
- has:attachment   Emails with attachments
- after:YYYY-MM-DD Emails after date
- before:YYYY-MM-DD Emails before date

Examples:
  "from:kiraly project deadline"
  "in:important has:attachment budget"
  "subject:\"Q4 Report\" from:finance after:2024-01-01"
```

### 8.2 Web UI Integration

Add operator autocomplete in `frontend/librarian-ui/app.js`:

```javascript
const EMAIL_OPERATORS = [
    { op: 'from:', hint: 'sender name/email' },
    { op: 'to:', hint: 'recipient name/email' },
    { op: 'subject:', hint: 'subject text' },
    { op: 'in:', hint: 'folder (inbox, sent, important)' },
    { op: 'has:', hint: 'attachment' },
    { op: 'after:', hint: 'YYYY-MM-DD' },
    { op: 'before:', hint: 'YYYY-MM-DD' },
];

// Show autocomplete when user types an operator prefix
function showOperatorHints(inputValue) {
    const lastWord = inputValue.split(' ').pop();
    const matches = EMAIL_OPERATORS.filter(
        op => op.op.startsWith(lastWord.toLowerCase())
    );
    // Display matches as autocomplete suggestions
}
```

---

## 9. Implementation Plan

### Phase 1: Core Operators (Priority 1)
1. Create `backend/core/query_parser.py` with `QueryOperatorParser`
2. Integrate parser into `search_emails()` in `tools.py`
3. Add `recipient` (to:) and `cc:` filters to `_apply_email_filters()`
4. Add folder extraction to `EMLParser` (path-based)
5. Add `folder` filter to `_apply_email_filters()`
6. Update MCP tool documentation
7. Add unit tests for parser

### Phase 2: Extended Operators & UX (Priority 2)
8. Add operator autocomplete to Web UI
9. Add X-Folder header support in EMLParser
10. Add pre-filtering in ChromaDB for performance
11. Add `is:` operator for future state tracking

### Phase 3: Advanced Features
12. Negation operators (`-from:spam`)
13. OR logic (`from:alice OR from:bob`)
14. Saved searches / search shortcuts

---

## 10. Testing Strategy

### Unit Tests

```python
# test_query_parser.py

def test_parse_simple_operator():
    parser = QueryOperatorParser()
    result = parser.parse("from:kiraly project")
    assert result.free_text == "project"
    assert result.operators == {"from": "kiraly"}

def test_parse_quoted_value():
    parser = QueryOperatorParser()
    result = parser.parse('subject:"Q4 Report" urgent')
    assert result.free_text == "urgent"
    assert result.operators == {"subject": "Q4 Report"}

def test_parse_multiple_operators():
    parser = QueryOperatorParser()
    result = parser.parse("from:alice to:bob meeting")
    assert result.free_text == "meeting"
    assert result.operators == {"from": "alice", "to": "bob"}

def test_parse_has_attachment():
    parser = QueryOperatorParser()
    result = parser.parse("has:attachment budget")
    params = result.get_filter_params()
    assert params["has_attachments"] == True

def test_parse_only_operators():
    parser = QueryOperatorParser()
    result = parser.parse("from:kiraly to:john")
    assert result.free_text == ""
    assert len(result.operators) == 2

def test_parse_unknown_operator():
    parser = QueryOperatorParser()
    result = parser.parse("custom:value project")
    assert result.free_text == "custom:value project"
    assert result.operators == {}
```

### Integration Tests

```python
# test_email_search_operators.py

def test_search_with_from_operator():
    result = search_emails("from:kiraly project deadline")
    assert all("kiraly" in r["metadata"]["from"].lower()
               for r in result["results"])

def test_search_with_folder_operator():
    result = search_emails("in:important budget")
    assert all(r["metadata"]["folder"] == "important"
               for r in result["results"])

def test_search_operators_with_semantic_mode():
    result = search_emails(
        "from:kiraly machine learning concepts",
        mode="semantic"
    )
    # Verify semantic search worked on free text
    # while from filter was applied
```

---

## 11. Summary

### Benefits
- **Familiar UX**: Users know Outlook/Gmail-style operators
- **Power User Feature**: Combines free-text with structured filters naturally
- **Works with ALL search modes**: Operators are pre-filters, compatible with keyword/semantic/hybrid/hyde/auto
- **Incremental Value**: Can start with core operators, extend later

### Technical Approach
- **Parser Module**: New `query_parser.py` for operator extraction
- **Post-filtering**: Start simple, optimize to pre-filtering later
- **Folder Metadata**: Extract from file path initially
- **Backward Compatible**: `parse_operators=True` by default, can disable

### Effort Estimate
- Phase 1 (Core): ~4-6 hours implementation + testing
- Phase 2 (Extended): ~4 hours
- Phase 3 (Advanced): ~8 hours

---

## Appendix: Comparison with Existing Systems

| Feature | Gmail | Outlook | Librarian (Proposed) |
|---------|-------|---------|---------------------|
| from: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| to: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| subject: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| in:/folder: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| has:attachment | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| after:/before: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| is:unread | :white_check_mark: | :white_check_mark: | Phase 2 |
| label: | :white_check_mark: | :x: | Consider |
| Negation (-) | :white_check_mark: | :white_check_mark: | Phase 3 |
| OR logic | :white_check_mark: | :white_check_mark: | Phase 3 |
