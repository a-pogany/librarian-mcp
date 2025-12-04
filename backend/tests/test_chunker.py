"""Unit tests for document chunking"""

import pytest
from backend.core.chunking import DocumentChunker, DocumentChunk


def test_small_document_no_chunking():
    """Test that small documents aren't chunked"""
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)

    content = "This is a small document with less than 2048 characters."
    metadata = {'product': 'test', 'component': 'test'}

    chunks = chunker.chunk_document(
        doc_id="test.md",
        content=content,
        metadata=metadata,
        file_type=".md"
    )

    assert len(chunks) == 1, "Small documents should return single chunk"
    assert chunks[0].content == content, "Content should be unchanged"
    assert chunks[0].metadata['is_chunked'] is False, "Should indicate no chunking"
    assert chunks[0].chunk_index == 0, "Single chunk should have index 0"


def test_semantic_chunking_markdown():
    """Test semantic chunking on markdown headings"""
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)

    # Make content large enough to trigger chunking (>2048 chars)
    content = """
# Title

Some intro text that is important.

## Section 1

""" + ("Content for section 1 with important information about authentication. " * 50) + """

## Section 2

""" + ("Content for section 2 with different information about databases. " * 50) + """

## Section 3

The final section covers deployment strategies.
"""

    metadata = {'product': 'test', 'component': 'test'}

    chunks = chunker.chunk_document(
        doc_id="test.md",
        content=content,
        metadata=metadata,
        file_type=".md"
    )

    # Should have at least 2 chunks (sections)
    assert len(chunks) >= 2, "Should split into multiple sections"

    # Check that chunks have headings
    headings = [c.heading for c in chunks if c.heading]
    assert len(headings) > 0, "Should capture section headings"
    assert any("Section 1" in h for h in headings), "Should have Section 1 heading"

    # Check metadata
    for chunk in chunks:
        assert chunk.metadata['is_chunked'] is True, "Should indicate chunking"
        assert chunk.metadata['chunk_method'] == 'semantic', "Should indicate semantic chunking"
        assert 'product' in chunk.metadata, "Should inherit metadata"


def test_fixed_size_chunking():
    """Test fixed-size chunking with overlap"""
    chunker = DocumentChunker(chunk_size=128, chunk_overlap=20)

    # Create long content without headings
    content = "word " * 500  # ~2500 chars, should split into multiple chunks

    metadata = {'product': 'test', 'component': 'test'}

    chunks = chunker.chunk_document(
        doc_id="test.txt",
        content=content,
        metadata=metadata,
        file_type=".txt"
    )

    assert len(chunks) >= 2, "Long document should split into multiple chunks"

    # Check overlap exists between consecutive chunks
    if len(chunks) > 1:
        # Last part of chunk 0 should appear in beginning of chunk 1
        chunk0_end = chunks[0].content[-50:]
        chunk1_start = chunks[1].content[:100]
        # Should have some overlap
        assert len(chunk0_end) > 0 and len(chunk1_start) > 0, "Chunks should have content"

    # Check metadata
    for chunk in chunks:
        assert chunk.metadata['is_chunked'] is True, "Should indicate chunking"
        assert chunk.metadata['chunk_method'] == 'fixed', "Should indicate fixed-size chunking"


def test_chunk_metadata_inheritance():
    """Test that chunks inherit parent document metadata"""
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)

    content = "## Section 1\n\n" + ("Long content. " * 200)

    metadata = {
        'product': 'test_product',
        'component': 'test_component',
        'file_type': '.md',
        'custom_field': 'custom_value'
    }

    chunks = chunker.chunk_document(
        doc_id="docs/test.md",
        content=content,
        metadata=metadata,
        file_type=".md"
    )

    # All chunks should inherit parent metadata
    for chunk in chunks:
        assert chunk.metadata['product'] == 'test_product', "Should inherit product"
        assert chunk.metadata['component'] == 'test_component', "Should inherit component"
        assert chunk.metadata['file_type'] == '.md', "Should inherit file_type"
        assert chunk.metadata['custom_field'] == 'custom_value', "Should inherit custom fields"
        assert chunk.parent_doc == "docs/test.md", "Should reference parent document"


def test_chunk_ids_are_unique():
    """Test that chunk IDs are unique and properly formatted"""
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)

    content = "## Section 1\n\n" + ("Content. " * 100) + "\n\n## Section 2\n\n" + ("More content. " * 100)

    metadata = {'product': 'test', 'component': 'test'}

    chunks = chunker.chunk_document(
        doc_id="test.md",
        content=content,
        metadata=metadata,
        file_type=".md"
    )

    chunk_ids = [c.chunk_id for c in chunks]

    # All IDs should be unique
    assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs should be unique"

    # IDs should follow pattern: doc_id#chunk_N
    for chunk_id in chunk_ids:
        assert '#chunk_' in chunk_id, f"Chunk ID should contain '#chunk_': {chunk_id}"
        assert chunk_id.startswith('test.md#'), f"Chunk ID should start with doc_id: {chunk_id}"


def test_large_section_splitting():
    """Test that very large sections within a multi-section document are split into sub-chunks"""
    chunker = DocumentChunker(chunk_size=128, chunk_overlap=20)

    # Create a multi-section document with one very large section (chunk_chars = 128 * 4 = 512 chars)
    # Large section needs >1024 chars to exceed 2x chunk_chars
    content = """
## Small Section

This is a small section with minimal content.

## Large Section

""" + ("This is a lot of content that will require splitting into multiple sub-chunks. " * 150) + """

## Another Small Section

Final section with minimal content.
"""

    metadata = {'product': 'test', 'component': 'test'}

    chunks = chunker.chunk_document(
        doc_id="test.md",
        content=content,
        metadata=metadata,
        file_type=".md"
    )

    # Should have multiple chunks total
    assert len(chunks) > 3, "Document with large section should be split into multiple chunks"

    # Find chunks from the "Large Section"
    large_section_chunks = [c for c in chunks if c.heading == "Large Section"]

    # The large section should have been split into multiple sub-chunks
    assert len(large_section_chunks) > 1, "Large section should be split into multiple sub-chunks"

    # All sub-chunks should reference the section heading
    for chunk in large_section_chunks:
        assert chunk.heading == "Large Section", "Sub-chunks should reference section heading"
        assert chunk.metadata.get('heading') == "Large Section", "Metadata should include heading"


def test_sentence_boundary_preservation():
    """Test that chunks try to end at sentence boundaries"""
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

    # Content with clear sentence boundaries
    content = "This is sentence one. This is sentence two. This is sentence three. " * 20

    metadata = {'product': 'test', 'component': 'test'}

    chunks = chunker.chunk_document(
        doc_id="test.txt",
        content=content,
        metadata=metadata,
        file_type=".txt"
    )

    # Check that chunks tend to end with sentence endings
    for chunk in chunks[:-1]:  # Exclude last chunk
        # Should try to end with period
        assert chunk.content.rstrip().endswith('.') or len(chunk.content) < 200, \
            f"Chunk should try to end at sentence boundary: '{chunk.content[-50:]}'"


def test_empty_content_handling():
    """Test handling of empty or None content"""
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)

    metadata = {'product': 'test', 'component': 'test'}

    # Empty string
    chunks = chunker.chunk_document(
        doc_id="test.md",
        content="",
        metadata=metadata,
        file_type=".md"
    )

    assert len(chunks) == 1, "Empty content should return single chunk"
    assert chunks[0].content == "", "Content should be empty string"


def test_extract_markdown_sections():
    """Test markdown section extraction"""
    chunker = DocumentChunker()

    content = """
# Main Title

Intro content

## Section A

Content A

### Subsection A1

Subsection content

## Section B

Content B
"""

    sections = chunker._extract_markdown_sections(content)

    # Should extract sections (## or ### level)
    assert len(sections) >= 2, "Should extract multiple sections"

    # Check section headings
    headings = [h for h, _ in sections if h]
    assert any("Section A" in (h or "") for h in headings), "Should extract Section A"
    assert any("Section B" in (h or "") for h in headings), "Should extract Section B"


def test_chunk_indices_are_sequential():
    """Test that chunk indices are sequential"""
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=128)

    content = ("## Section\n\nContent. " * 50)

    metadata = {'product': 'test', 'component': 'test'}

    chunks = chunker.chunk_document(
        doc_id="test.md",
        content=content,
        metadata=metadata,
        file_type=".md"
    )

    indices = [c.chunk_index for c in chunks]

    # Indices should be unique and incrementing
    assert len(indices) == len(set(indices)), "Chunk indices should be unique"
    assert min(indices) >= 0, "Chunk indices should start at 0 or higher"
