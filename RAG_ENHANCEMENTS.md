# RAG Enhancements - Implementation Plan

**Status:** Planning
**Target Phase:** Phase 3 - RAG Quality Improvements
**Estimated Effort:** 2-3 days total

## Overview

This document outlines three key improvements to enhance the RAG (Retrieval-Augmented Generation) capabilities of Librarian MCP:

1. **Document Chunking** - Split long documents into semantic chunks for better embedding quality
2. **Reranking** - Refine semantic search results with keyword-based scoring
3. **Enhanced Metadata Filtering** - Richer metadata for more precise document filtering

## Priority & Complexity

| Feature | Impact | Complexity | Estimated Time | Priority |
|---------|--------|------------|----------------|----------|
| Reranking | Medium | Low ⭐ | 2-4 hours | 1 (Quick win) |
| Document Chunking | High | Medium-High ⭐⭐⭐ | 1-2 days | 2 (Biggest impact) |
| Metadata Filtering | Low-Medium | Low-Medium ⭐⭐ | 4-6 hours | 3 (Nice to have) |

**Recommended Order:** Reranking → Chunking → Metadata Filtering

---

## Feature 1: Reranking 🔄

### Problem Statement

Current hybrid search combines keyword and semantic results linearly. However, semantic search can return conceptually similar but contextually irrelevant documents with high scores. We need a two-stage approach:
1. Cast a wide net with semantic search
2. Refine with precise keyword matching

### Benefits

- **Improved Relevance:** Filters out false positives from semantic search
- **Better Precision:** Keyword scoring ensures query terms are actually present
- **Minimal Overhead:** Reuses existing search engines, no new dependencies

### Technical Approach

**Current Flow:**
```
Query → [Keyword Search] ─┐
                          ├→ Merge → Weighted Average → Results
Query → [Semantic Search] ┘
```

**New Flow (Reranking):**
```
Query → [Semantic Search] → Top N candidates → [Keyword Reranking] → Filtered Results
```

### Implementation Details

#### Files to Modify

1. **`backend/core/hybrid_search.py`** (primary changes)
   - Add new `rerank` mode alongside `keyword`, `semantic`, `hybrid`
   - Implement `_rerank_search()` method

2. **`backend/config/settings.py`**
   - Add `rerank_candidates` config (default: 50)
   - Add `rerank_mode` to search config

3. **`config.json`**
   - Add reranking configuration section

#### Step-by-Step Implementation

**Step 1: Update Config**

Add to `config.json`:
```json
{
  "search": {
    "mode": "rerank",
    "rerank_candidates": 50,
    "rerank_keyword_threshold": 0.1
  }
}
```

**Step 2: Implement Reranking Logic**

In `backend/core/hybrid_search.py`, add new method:

```python
def _rerank_search(
    self,
    query: str,
    max_results: int,
    product: Optional[str] = None,
    component: Optional[str] = None
) -> List[SearchResult]:
    """
    Two-stage reranking search:
    1. Semantic search for broad recall (50-100 candidates)
    2. Keyword scoring for precision (top N)
    """
    # Stage 1: Semantic search (broad recall)
    candidates_limit = self.settings.search.get('rerank_candidates', 50)
    semantic_results = self.semantic_engine.search(
        query=query,
        max_results=candidates_limit,
        product=product,
        component=component
    )

    # Stage 2: Keyword reranking
    reranked = []
    for result in semantic_results:
        # Get keyword score for this document
        keyword_score = self._calculate_keyword_score(
            query=query,
            doc_id=result.document_id,
            content=result.content
        )

        # Filter out results with no keyword matches
        threshold = self.settings.search.get('rerank_keyword_threshold', 0.1)
        if keyword_score < threshold:
            continue

        # Combine scores (70% semantic, 30% keyword)
        combined_score = 0.7 * result.score + 0.3 * keyword_score

        reranked.append(SearchResult(
            document_id=result.document_id,
            product=result.product,
            component=result.component,
            file_path=result.file_path,
            content=result.content,
            score=combined_score,
            metadata=result.metadata
        ))

    # Sort by combined score and return top N
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked[:max_results]
```

**Step 3: Add Helper Method**

```python
def _calculate_keyword_score(self, query: str, doc_id: str, content: str) -> float:
    """Calculate keyword relevance score for a document"""
    keywords = query.lower().split()
    content_lower = content.lower()

    score = 0
    for keyword in keywords:
        if len(keyword) < 2:
            continue

        # Count occurrences (capped at 5)
        count = min(content_lower.count(keyword), 5)
        score += count

    # Normalize by max possible score
    max_score = len(keywords) * 5
    return score / max_score if max_score > 0 else 0.0
```

**Step 4: Update Main Search Method**

Modify `search()` method to support rerank mode:

```python
def search(self, query: str, max_results: int = 10, **kwargs) -> List[SearchResult]:
    mode = self.settings.search.get('mode', 'hybrid')

    if mode == 'rerank':
        return self._rerank_search(query, max_results, **kwargs)
    elif mode == 'keyword':
        # ... existing code
    elif mode == 'semantic':
        # ... existing code
    elif mode == 'hybrid':
        # ... existing code
```

### Testing

**Unit Tests** (`backend/tests/test_hybrid_search.py`):

```python
def test_rerank_search(hybrid_engine):
    """Test reranking mode"""
    results = hybrid_engine.search(
        query="authentication API",
        max_results=10,
        mode='rerank'
    )

    # All results should contain at least one query keyword
    for result in results:
        content_lower = result.content.lower()
        assert 'authentication' in content_lower or 'api' in content_lower

def test_rerank_filters_irrelevant(hybrid_engine):
    """Test that reranking filters out irrelevant semantic matches"""
    # This should get semantic matches but filter them with keywords
    results = hybrid_engine.search(
        query="specific_function_name",
        max_results=10,
        mode='rerank'
    )

    # Should have fewer results than pure semantic search
    semantic_results = hybrid_engine.search(
        query="specific_function_name",
        max_results=10,
        mode='semantic'
    )

    assert len(results) <= len(semantic_results)
```

**Integration Tests:**

```bash
# Test reranking with real docs
pytest backend/tests/test_rag_integration.py::test_rerank_mode -v
```

### Success Criteria

- ✅ Rerank mode returns only documents containing query keywords
- ✅ Results are ordered by combined semantic + keyword score
- ✅ Performance: <200ms for typical queries (50 candidates)
- ✅ Backward compatible: existing modes still work
- ✅ Configurable via `config.json` and environment variables

---

## Feature 2: Document Chunking 📄→🧩

### Problem Statement

Long documents (10+ pages) produce diluted embeddings that represent many topics poorly rather than one topic well. When a user searches for a specific concept, the entire document's embedding may not match well even if one section is highly relevant.

**Example:**
- Document: 50-page architecture guide covering database, API, frontend, deployment
- Query: "database connection pooling"
- Current: Single embedding for entire doc → weak match
- With chunking: 200 chunks, one specifically about connection pooling → strong match

### Benefits

- **Better Semantic Search Quality:** Each chunk focuses on a narrow topic
- **More Precise Results:** Can return specific sections, not just entire documents
- **Improved Context:** Overlap between chunks preserves context at boundaries
- **Scalability:** Handle very large documents (100+ pages)

### Technical Approach

**Chunking Strategy:**
1. **Semantic Boundaries:** Split on headings (## H2, ### H3) when possible
2. **Fallback to Fixed Size:** If no headings, split at 512 tokens with 50-token overlap
3. **Preserve Metadata:** Each chunk inherits product, component, file_path from parent document
4. **Chunk Metadata:** Add `chunk_id`, `chunk_index`, `parent_doc`, `heading` metadata

**Token Calculation:**
```
Average: 1 token ≈ 4 characters for English
Chunk size: 512 tokens ≈ 2048 characters
Overlap: 50 tokens ≈ 200 characters
```

### Implementation Details

#### Files to Modify

1. **`backend/core/chunker.py`** (NEW FILE)
   - `DocumentChunker` class
   - Semantic and fixed-size chunking strategies

2. **`backend/core/indexer.py`**
   - Modify `FileIndexer.index_document()` to chunk before embedding
   - Store chunk relationships

3. **`backend/core/vector_db.py`**
   - Support chunk metadata
   - Add `get_parent_document()` method

4. **`backend/core/semantic_search.py`**
   - Aggregate results from multiple chunks of same document
   - Return best matching chunk + surrounding chunks

5. **`backend/core/hybrid_search.py`**
   - Update to work with chunked results

6. **`config.json`**
   - Add chunking configuration

#### Step-by-Step Implementation

**Step 1: Create Chunker Module**

Create `backend/core/chunker.py`:

```python
"""
Document chunking for improved embedding quality
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    chunk_id: str  # e.g., "path/to/doc.md#chunk_0"
    parent_doc: str  # Original document path
    content: str  # Chunk text content
    chunk_index: int  # Position in document (0, 1, 2, ...)
    heading: Optional[str]  # Associated heading (if any)
    metadata: Dict[str, Any]  # Inherits from parent + chunk-specific


class DocumentChunker:
    """Split documents into smaller chunks for better embeddings"""

    def __init__(
        self,
        chunk_size: int = 512,  # tokens
        chunk_overlap: int = 50,  # tokens
        strategy: str = "semantic"  # "semantic" or "fixed"
    ):
        """
        Initialize chunker

        Args:
            chunk_size: Target size in tokens (approximate)
            chunk_overlap: Overlap between chunks in tokens
            strategy: "semantic" (split on headings) or "fixed" (fixed size)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        # Approximate token calculation: 1 token ≈ 4 chars
        self.chunk_chars = chunk_size * 4
        self.overlap_chars = chunk_overlap * 4

    def chunk_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Split document into chunks

        Args:
            doc_id: Document identifier (file path)
            content: Full document content
            metadata: Document metadata (product, component, etc.)

        Returns:
            List of document chunks
        """
        if not content or len(content) < self.chunk_chars:
            # Document is small enough, return as single chunk
            return [DocumentChunk(
                chunk_id=f"{doc_id}#chunk_0",
                parent_doc=doc_id,
                content=content,
                chunk_index=0,
                heading=None,
                metadata={**metadata, 'is_chunked': False}
            )]

        if self.strategy == "semantic":
            chunks = self._semantic_chunking(doc_id, content, metadata)
        else:
            chunks = self._fixed_size_chunking(doc_id, content, metadata)

        logger.debug(f"Split {doc_id} into {len(chunks)} chunks")
        return chunks

    def _semantic_chunking(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Split on semantic boundaries (headings)
        Falls back to fixed-size if no headings found
        """
        # Extract sections by heading
        sections = self._extract_sections(content)

        if len(sections) <= 1:
            # No meaningful headings, use fixed size
            return self._fixed_size_chunking(doc_id, content, metadata)

        chunks = []
        for idx, (heading, section_content) in enumerate(sections):
            # If section is too large, split it further
            if len(section_content) > self.chunk_chars * 2:
                sub_chunks = self._split_large_section(
                    section_content,
                    start_idx=idx * 10  # Offset chunk indices
                )

                for sub_idx, sub_content in enumerate(sub_chunks):
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc_id}#chunk_{idx}_{sub_idx}",
                        parent_doc=doc_id,
                        content=sub_content,
                        chunk_index=idx * 10 + sub_idx,
                        heading=heading,
                        metadata={
                            **metadata,
                            'is_chunked': True,
                            'chunk_method': 'semantic',
                            'heading': heading
                        }
                    ))
            else:
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}#chunk_{idx}",
                    parent_doc=doc_id,
                    content=section_content,
                    chunk_index=idx,
                    heading=heading,
                    metadata={
                        **metadata,
                        'is_chunked': True,
                        'chunk_method': 'semantic',
                        'heading': heading
                    }
                ))

        return chunks

    def _extract_sections(self, content: str) -> List[tuple[Optional[str], str]]:
        """
        Extract sections by markdown headings

        Returns:
            List of (heading, content) tuples
        """
        # Match markdown headings (## or ###)
        heading_pattern = r'^(#{2,3})\s+(.+)$'

        sections = []
        current_heading = None
        current_content = []

        for line in content.split('\n'):
            match = re.match(heading_pattern, line)

            if match:
                # Save previous section
                if current_content:
                    sections.append((
                        current_heading,
                        '\n'.join(current_content).strip()
                    ))

                # Start new section
                current_heading = match.group(2).strip()
                current_content = [line]  # Include heading in content
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            sections.append((
                current_heading,
                '\n'.join(current_content).strip()
            ))

        return sections

    def _fixed_size_chunking(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Split into fixed-size chunks with overlap
        """
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(content):
            # Extract chunk
            end = start + self.chunk_chars
            chunk_content = content[start:end]

            # Try to end at sentence boundary if possible
            if end < len(content):
                # Look for sentence ending (. ! ?) in last 200 chars
                last_period = chunk_content.rfind('. ', -200)
                if last_period > 0:
                    end = start + last_period + 1
                    chunk_content = content[start:end]

            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}#chunk_{chunk_idx}",
                parent_doc=doc_id,
                content=chunk_content.strip(),
                chunk_index=chunk_idx,
                heading=None,
                metadata={
                    **metadata,
                    'is_chunked': True,
                    'chunk_method': 'fixed'
                }
            ))

            # Move to next chunk with overlap
            start = end - self.overlap_chars
            chunk_idx += 1

        return chunks

    def _split_large_section(self, content: str, start_idx: int = 0) -> List[str]:
        """Split a large section into smaller fixed-size chunks"""
        sub_chunks = []
        start = 0

        while start < len(content):
            end = start + self.chunk_chars
            sub_chunk = content[start:end]

            # Try to end at sentence boundary
            if end < len(content):
                last_period = sub_chunk.rfind('. ', -200)
                if last_period > 0:
                    end = start + last_period + 1
                    sub_chunk = content[start:end]

            sub_chunks.append(sub_chunk.strip())
            start = end - self.overlap_chars

        return sub_chunks
```

**Step 2: Update Indexer**

Modify `backend/core/indexer.py`:

```python
from .chunker import DocumentChunker

class FileIndexer:
    def __init__(self, config: Dict[str, Any], embedding_generator=None, vector_db=None):
        # ... existing code ...

        # Add chunker
        chunk_config = config.get('chunking', {})
        if chunk_config.get('enabled', False):
            self.chunker = DocumentChunker(
                chunk_size=chunk_config.get('chunk_size', 512),
                chunk_overlap=chunk_config.get('chunk_overlap', 50),
                strategy=chunk_config.get('strategy', 'semantic')
            )
        else:
            self.chunker = None

    def index_document(self, file_path: Path) -> bool:
        """Index a single document (with chunking if enabled)"""
        try:
            # ... existing parsing code ...

            # Store in keyword index (full document)
            self.index.add_document(doc_id, doc_data)

            # Generate embeddings (with chunking)
            if self.embedding_generator and self.vector_db:
                if self.chunker:
                    # Chunk the document
                    chunks = self.chunker.chunk_document(
                        doc_id=doc_id,
                        content=content,
                        metadata={
                            'product': product,
                            'component': component,
                            'file_type': file_path.suffix
                        }
                    )

                    # Generate embeddings for each chunk
                    for chunk in chunks:
                        embedding = self.embedding_generator.encode_document(
                            chunk.content
                        )

                        self.vector_db.add_document(
                            doc_id=chunk.chunk_id,
                            embedding=embedding,
                            metadata=chunk.metadata
                        )
                else:
                    # No chunking, embed full document
                    # ... existing code ...

        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")
            return False
```

**Step 3: Update Semantic Search**

Modify `backend/core/semantic_search.py` to aggregate chunk results:

```python
def search(self, query: str, max_results: int = 10, **kwargs) -> List[SearchResult]:
    """Search with chunk aggregation"""
    # ... existing query embedding code ...

    # Search vector DB (may return multiple chunks from same doc)
    raw_results = self.vector_db.search(
        query_embedding=query_embedding,
        n_results=max_results * 3,  # Get more to account for chunking
        where=where_filter
    )

    # Aggregate chunks by parent document
    doc_scores = {}  # parent_doc -> best chunk score
    doc_chunks = {}  # parent_doc -> list of matching chunks

    for result in raw_results:
        metadata = result['metadata']
        parent_doc = metadata.get('parent_doc', result['id'])
        score = result['similarity']

        if parent_doc not in doc_scores:
            doc_scores[parent_doc] = score
            doc_chunks[parent_doc] = [result]
        else:
            # Keep best score, accumulate chunks
            doc_scores[parent_doc] = max(doc_scores[parent_doc], score)
            doc_chunks[parent_doc].append(result)

    # Create results from best chunks
    results = []
    for parent_doc, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:max_results]:
        best_chunk = max(doc_chunks[parent_doc], key=lambda x: x['similarity'])

        # Get surrounding chunks for context (optional)
        chunk_content = self._get_chunk_with_context(
            best_chunk,
            doc_chunks[parent_doc]
        )

        results.append(SearchResult(
            document_id=parent_doc,
            score=score,
            content=chunk_content,
            # ... other fields ...
        ))

    return results

def _get_chunk_with_context(self, best_chunk, all_chunks):
    """Get best chunk plus surrounding chunks for context"""
    # Sort chunks by index
    sorted_chunks = sorted(all_chunks, key=lambda x: x['metadata'].get('chunk_index', 0))

    # Find best chunk index
    best_idx = sorted_chunks.index(best_chunk)

    # Get best chunk ± 1 surrounding chunk
    start_idx = max(0, best_idx - 1)
    end_idx = min(len(sorted_chunks), best_idx + 2)

    context_chunks = sorted_chunks[start_idx:end_idx]

    # Combine content
    return '\n\n'.join([c['metadata'].get('content', '') for c in context_chunks])
```

**Step 4: Update Config**

Add to `config.json`:

```json
{
  "chunking": {
    "enabled": true,
    "chunk_size": 512,
    "chunk_overlap": 50,
    "strategy": "semantic"
  }
}
```

### Testing

**Unit Tests** (`backend/tests/test_chunker.py`):

```python
def test_semantic_chunking():
    """Test chunking on headings"""
    chunker = DocumentChunker(strategy="semantic")

    content = """
# Title

Some intro text.

## Section 1

Content for section 1.

## Section 2

Content for section 2.
"""

    chunks = chunker.chunk_document("test.md", content, {})

    assert len(chunks) >= 2  # At least 2 sections
    assert any("Section 1" in c.content for c in chunks)
    assert any("Section 2" in c.content for c in chunks)

def test_fixed_size_chunking():
    """Test fixed-size chunking with overlap"""
    chunker = DocumentChunker(chunk_size=128, chunk_overlap=20, strategy="fixed")

    content = "word " * 500  # 2500 chars, should split into ~2 chunks

    chunks = chunker.chunk_document("test.txt", content, {})

    assert len(chunks) >= 2
    # Check overlap exists
    assert chunks[0].content[-50:] in chunks[1].content[:100]

def test_small_document_no_chunking():
    """Test that small docs aren't chunked"""
    chunker = DocumentChunker()

    content = "Small document."
    chunks = chunker.chunk_document("test.txt", content, {})

    assert len(chunks) == 1
    assert chunks[0].metadata['is_chunked'] is False
```

**Integration Tests:**

```bash
pytest backend/tests/test_indexer.py::test_chunked_indexing -v
```

### Migration Plan

**For existing indexed documents:**

1. Add `rebuild_index` flag to config
2. On startup, if chunking enabled and index version mismatch:
   - Clear vector DB
   - Re-index all documents with chunking
   - Update index version

**Backward Compatibility:**

- Chunking is opt-in via config
- If disabled, behaves exactly as before
- Keyword search unaffected (still uses full documents)

### Success Criteria

- ✅ Long documents (>2048 chars) split into multiple chunks
- ✅ Semantic chunking creates chunks at heading boundaries
- ✅ Fixed-size chunking maintains 50-token overlap
- ✅ Search results aggregate chunks from same document
- ✅ Best matching chunk + context returned
- ✅ Backward compatible: chunking can be disabled
- ✅ Performance: Indexing time increases <2x

---

## Feature 3: Enhanced Metadata Filtering 🔍

### Problem Statement

Current metadata is limited to `product`, `component`, and `file_type`. Users cannot filter by:
- Document modification date (find recent docs)
- Heading level (find high-level architecture vs. detailed API docs)
- Document size (exclude large files)
- Custom tags

### Benefits

- **Precise Filtering:** Find "API docs modified in last 30 days"
- **Better UX:** MCP tools expose more filter options to LLMs
- **Temporal Queries:** "What changed recently?" type questions
- **Hierarchical Filtering:** Filter by heading level to get overviews vs. details

### Technical Approach

**New Metadata Fields:**
- `last_modified`: ISO timestamp of file modification
- `file_size`: Size in bytes
- `heading_level`: For chunked docs, the heading level (h2, h3, etc.)
- `tags`: User-defined tags from frontmatter or filename patterns
- `indexed_at`: When the document was indexed
- `doc_type`: Inferred type (api, guide, architecture, reference, etc.)

### Implementation Details

#### Files to Modify

1. **`backend/core/indexer.py`**
   - Extract richer metadata during indexing
   - Parse frontmatter for tags
   - Infer document type from filename/content

2. **`backend/core/vector_db.py`**
   - Store additional metadata fields

3. **`backend/mcp_server/tools.py`**
   - Update tool schemas to accept new filter parameters
   - Add date range filtering

4. **`backend/core/search.py` and `semantic_search.py`**
   - Support new filter parameters

5. **`config.json`**
   - Add metadata extraction configuration

#### Step-by-Step Implementation

**Step 1: Extract Enhanced Metadata**

Modify `backend/core/indexer.py`:

```python
import os
from datetime import datetime
from pathlib import Path

def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
    """Extract enhanced metadata from file"""

    # Basic metadata
    stats = os.stat(file_path)
    metadata = {
        'product': self._extract_product(file_path),
        'component': self._extract_component(file_path),
        'file_type': file_path.suffix,
        'file_size': stats.st_size,
        'last_modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
        'indexed_at': datetime.now().isoformat(),
    }

    # Extract tags from frontmatter (YAML)
    tags = self._extract_frontmatter_tags(content)
    if tags:
        metadata['tags'] = tags

    # Infer document type
    metadata['doc_type'] = self._infer_doc_type(file_path, content)

    return metadata

def _extract_frontmatter_tags(self, content: str) -> List[str]:
    """Extract tags from YAML frontmatter"""
    # Match YAML frontmatter: ---\n...\n---
    import re
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)

    if not match:
        return []

    try:
        import yaml
        frontmatter = yaml.safe_load(match.group(1))
        return frontmatter.get('tags', [])
    except:
        return []

def _infer_doc_type(self, file_path: Path, content: str) -> str:
    """Infer document type from filename and content"""
    filename_lower = file_path.name.lower()
    content_lower = content.lower()

    # Check filename patterns
    if 'api' in filename_lower or 'endpoint' in filename_lower:
        return 'api'
    elif 'architecture' in filename_lower or 'design' in filename_lower:
        return 'architecture'
    elif 'guide' in filename_lower or 'tutorial' in filename_lower:
        return 'guide'
    elif 'reference' in filename_lower:
        return 'reference'

    # Check content patterns
    if 'class ' in content_lower or 'function ' in content_lower:
        return 'api'
    elif 'architecture' in content_lower[:500]:  # In first 500 chars
        return 'architecture'

    return 'documentation'  # default
```

**Step 2: Update MCP Tools**

Modify `backend/mcp_server/tools.py`:

```python
@mcp.tool()
def search_documentation(
    query: str,
    product: str = None,
    component: str = None,
    file_types: list[str] = None,
    doc_type: str = None,  # NEW
    modified_after: str = None,  # NEW: ISO date string
    modified_before: str = None,  # NEW: ISO date string
    tags: list[str] = None,  # NEW
    max_results: int = 10
) -> dict:
    """
    Search documentation with enhanced filtering

    Args:
        query: Search query
        product: Filter by product name
        component: Filter by component name
        file_types: Filter by file extensions (.md, .txt, .docx)
        doc_type: Filter by document type (api, guide, architecture, reference)
        modified_after: Only docs modified after this date (ISO format: 2024-01-01)
        modified_before: Only docs modified before this date
        tags: Filter by tags (documents must have at least one matching tag)
        max_results: Maximum number of results

    Returns:
        Search results with metadata
    """

    # Build filter dict
    filters = {}
    if product:
        filters['product'] = product
    if component:
        filters['component'] = component
    if doc_type:
        filters['doc_type'] = doc_type
    if tags:
        filters['tags'] = tags  # ChromaDB supports array contains

    # Date filtering (done post-search for simplicity)
    results = search_engine.search(
        query=query,
        max_results=max_results * 2,  # Get more to filter by date
        **filters
    )

    # Apply date filters
    if modified_after or modified_before:
        results = filter_by_date(results, modified_after, modified_before)

    return {
        'results': results[:max_results],
        'total': len(results),
        'filters_applied': filters
    }

def filter_by_date(results, after=None, before=None):
    """Filter results by last_modified date"""
    from datetime import datetime

    filtered = []
    for result in results:
        modified_str = result.metadata.get('last_modified')
        if not modified_str:
            continue

        modified = datetime.fromisoformat(modified_str)

        if after:
            after_dt = datetime.fromisoformat(after)
            if modified < after_dt:
                continue

        if before:
            before_dt = datetime.fromisoformat(before)
            if modified > before_dt:
                continue

        filtered.append(result)

    return filtered
```

**Step 3: Update Search Engines**

Modify `backend/core/semantic_search.py` to support new filters:

```python
def search(
    self,
    query: str,
    max_results: int = 10,
    product: Optional[str] = None,
    component: Optional[str] = None,
    doc_type: Optional[str] = None,  # NEW
    tags: Optional[List[str]] = None  # NEW
) -> List[SearchResult]:
    """Search with enhanced metadata filtering"""

    # Build ChromaDB where clause
    where_filter = {}

    if product:
        where_filter['product'] = product
    if component:
        where_filter['component'] = component
    if doc_type:
        where_filter['doc_type'] = doc_type
    if tags:
        # ChromaDB array contains syntax
        where_filter['tags'] = {'$contains': tags[0]}  # At least one tag matches

    # ... rest of search logic
```

**Step 4: Update Config**

Add to `config.json`:

```json
{
  "metadata": {
    "extract_tags": true,
    "infer_doc_type": true,
    "track_modifications": true
  }
}
```

### Testing

**Unit Tests** (`backend/tests/test_metadata.py`):

```python
def test_extract_frontmatter_tags():
    """Test tag extraction from YAML frontmatter"""
    content = """---
tags: [api, authentication, security]
title: Auth API
---

# Authentication API
"""

    indexer = FileIndexer(config)
    tags = indexer._extract_frontmatter_tags(content)

    assert tags == ['api', 'authentication', 'security']

def test_infer_doc_type():
    """Test document type inference"""
    indexer = FileIndexer(config)

    # Test filename patterns
    assert indexer._infer_doc_type(Path("api-reference.md"), "") == 'api'
    assert indexer._infer_doc_type(Path("architecture-guide.md"), "") == 'architecture'

    # Test content patterns
    content_with_class = "class AuthService { ... }"
    assert indexer._infer_doc_type(Path("auth.md"), content_with_class) == 'api'

def test_date_filtering():
    """Test filtering by modification date"""
    from datetime import datetime, timedelta

    today = datetime.now()
    week_ago = today - timedelta(days=7)

    # Create test results with different dates
    results = [
        SearchResult(metadata={'last_modified': today.isoformat()}),
        SearchResult(metadata={'last_modified': week_ago.isoformat()}),
    ]

    # Filter: only docs from last 3 days
    three_days_ago = (today - timedelta(days=3)).isoformat()
    filtered = filter_by_date(results, after=three_days_ago)

    assert len(filtered) == 1  # Only today's result
```

**Integration Tests:**

```python
def test_search_with_metadata_filters(search_engine):
    """Test search with enhanced metadata filters"""

    # Search: API docs modified in last 30 days
    from datetime import datetime, timedelta
    thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()

    results = search_engine.search(
        query="authentication",
        doc_type="api",
        modified_after=thirty_days_ago,
        max_results=10
    )

    for result in results:
        assert result.metadata['doc_type'] == 'api'
        modified = datetime.fromisoformat(result.metadata['last_modified'])
        assert modified > datetime.fromisoformat(thirty_days_ago)
```

### Success Criteria

- ✅ All documents have `last_modified`, `file_size`, `indexed_at`, `doc_type` metadata
- ✅ YAML frontmatter tags are extracted correctly
- ✅ Date range filtering works (modified_after, modified_before)
- ✅ Document type inference achieves >80% accuracy
- ✅ MCP tools expose new filter parameters
- ✅ ChromaDB metadata filtering works correctly
- ✅ Backward compatible: old indexed docs still work

---

## Configuration Changes

### Updated `config.json`

```json
{
  "docs": {
    "root_path": "./docs",
    "file_extensions": [".md", ".txt", ".docx"],
    "max_file_size_mb": 10,
    "watch_for_changes": true,
    "index_on_startup": true
  },
  "search": {
    "max_results": 50,
    "snippet_length": 200,
    "context_lines": 3,
    "min_keyword_length": 2,
    "mode": "rerank",
    "rerank_candidates": 50,
    "rerank_keyword_threshold": 0.1
  },
  "embeddings": {
    "enabled": true,
    "model": "all-MiniLM-L6-v2",
    "persist_directory": null,
    "semantic_weight": 0.5
  },
  "chunking": {
    "enabled": true,
    "chunk_size": 512,
    "chunk_overlap": 50,
    "strategy": "semantic"
  },
  "metadata": {
    "extract_tags": true,
    "infer_doc_type": true,
    "track_modifications": true
  },
  "mcp": {
    "transport": "http-sse",
    "host": "127.0.0.1",
    "port": 3001,
    "endpoint": "/mcp"
  }
}
```

### Environment Variables

```bash
# Search mode
SEARCH_MODE=rerank  # keyword | semantic | hybrid | rerank

# Chunking
ENABLE_CHUNKING=true
CHUNK_SIZE=512
CHUNK_OVERLAP=50
CHUNK_STRATEGY=semantic  # semantic | fixed

# Metadata
EXTRACT_TAGS=true
INFER_DOC_TYPE=true
```

---

## Testing Strategy

### Unit Tests

- `backend/tests/test_reranking.py` - Reranking logic
- `backend/tests/test_chunker.py` - Document chunking
- `backend/tests/test_metadata.py` - Enhanced metadata extraction

### Integration Tests

- `backend/tests/test_rag_integration.py` - End-to-end RAG pipeline with all features

### Performance Tests

- Measure indexing time with chunking (target: <2x slowdown)
- Measure search latency with reranking (target: <200ms)
- Memory usage with chunks (target: <2x increase)

### Regression Tests

- Ensure backward compatibility when features disabled
- Verify existing tests still pass

---

## Migration & Rollout

### Phase 1: Reranking (Week 1)

1. Implement reranking mode
2. Add config option `search.mode = "rerank"`
3. Test with existing index
4. Deploy if performance acceptable

### Phase 2: Chunking (Week 2-3)

1. Implement chunker module
2. Update indexer to chunk documents
3. Add rebuild index mechanism
4. Test with sample docs
5. **Migration step:** Rebuild index with chunking enabled
6. Deploy

### Phase 3: Enhanced Metadata (Week 3)

1. Add metadata extraction
2. Update MCP tools
3. Test filtering
4. **Migration step:** Rebuild index with enhanced metadata
5. Deploy

### Rollback Plan

Each feature can be independently disabled via config:
- `search.mode = "hybrid"` (disable reranking)
- `chunking.enabled = false` (disable chunking)
- `metadata.extract_tags = false` (disable enhanced metadata)

---

## Success Metrics

### Search Quality (Measured with test queries)

- **Precision@5:** Percentage of top 5 results that are relevant
  - Target: >85% (up from current ~70%)
- **NDCG@10:** Normalized Discounted Cumulative Gain
  - Target: >0.80

### Performance

- **Indexing time:** <2x increase with chunking
- **Search latency:** <200ms for p95
- **Memory usage:** <2x increase

### User Experience (MCP usage)

- **Query success rate:** Percentage of queries returning relevant results
  - Target: >90%
- **Filter usage:** Percentage of queries using new filters
  - Target: >20% adoption within 1 month

---

## Open Questions

1. **Chunking strategy:** Should we use semantic (heading-based) or fixed-size by default?
   - **Recommendation:** Semantic, fallback to fixed-size

2. **Reranking weights:** What's the optimal keyword vs. semantic weight?
   - **Recommendation:** Start with 70% semantic, 30% keyword, make configurable

3. **Chunk overlap:** 50 tokens or more?
   - **Recommendation:** 50 tokens (200 chars) is good starting point

4. **Document type taxonomy:** What categories beyond api/guide/architecture?
   - **Recommendation:** Start simple, expand based on user needs

---

## Implementation Checklist

### Reranking
- [ ] Add rerank mode to `hybrid_search.py`
- [ ] Implement `_rerank_search()` method
- [ ] Add config options
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update documentation

### Chunking
- [ ] Create `chunker.py` module
- [ ] Implement `DocumentChunker` class
- [ ] Implement semantic chunking
- [ ] Implement fixed-size chunking with overlap
- [ ] Update `indexer.py` to use chunker
- [ ] Update `semantic_search.py` to aggregate chunks
- [ ] Add config options
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Implement index rebuild mechanism
- [ ] Update documentation

### Enhanced Metadata
- [ ] Add metadata extraction to `indexer.py`
- [ ] Implement frontmatter tag parsing
- [ ] Implement document type inference
- [ ] Update `tools.py` MCP tool schemas
- [ ] Implement date filtering
- [ ] Update search engines to support new filters
- [ ] Add config options
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update documentation

### Testing & Deployment
- [ ] Run full test suite
- [ ] Performance benchmarking
- [ ] Create migration guide
- [ ] Update CLAUDE.md with new features
- [ ] Deploy to production
- [ ] Monitor metrics

---

## References

- **Current Implementation:** `PHASE2_IMPLEMENTATION.md`
- **Architecture:** `INDEXING_ARCHITECTURE.md`
- **Testing:** `TEST_REPORT.md`
- **Project Overview:** `CLAUDE.md`

---

**Document Version:** 1.0
**Last Updated:** 2025-12-04
**Author:** Claude Code Planning
**Status:** Ready for Implementation
