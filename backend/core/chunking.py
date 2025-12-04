"""
Hierarchical document chunking with structure awareness
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
import re
import logging

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
    """
    Smart document chunking that preserves structure
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        respect_boundaries: bool = True
    ):
        """
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks (for context continuity)
            respect_boundaries: Don't split across sections/tables
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_boundaries = respect_boundaries

        # Approximate token calculation: 1 token ≈ 4 chars
        self.chunk_chars = chunk_size * 4
        self.overlap_chars = chunk_overlap * 4

    def chunk_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        file_type: str = None
    ) -> List[DocumentChunk]:
        """
        Split document into chunks (works with all file types)

        Args:
            doc_id: Document identifier (file path)
            content: Full document content
            metadata: Document metadata (product, component, etc.)
            file_type: File extension (.md, .txt, .docx)

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

        # Use semantic chunking for markdown files
        if file_type == '.md':
            return self._semantic_chunking_markdown(doc_id, content, metadata)
        else:
            # Fixed-size chunking for other file types
            return self._fixed_size_chunking(doc_id, content, metadata)

    def _semantic_chunking_markdown(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Split markdown on semantic boundaries (headings)
        Falls back to fixed-size if no headings found
        """
        # Extract sections by heading
        sections = self._extract_markdown_sections(content)

        if len(sections) <= 1:
            # No meaningful headings, use fixed size
            return self._fixed_size_chunking(doc_id, content, metadata)

        chunks = []
        for idx, (heading, section_content) in enumerate(sections):
            # If section is too large, split it further
            if len(section_content) > self.chunk_chars * 2:
                sub_chunks = self._split_large_section(section_content)

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

    def _extract_markdown_sections(self, content: str) -> List[tuple]:
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

    def _split_large_section(self, content: str) -> List[str]:
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

    def chunk_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract hierarchical chunks from DOCX with metadata

        Returns:
            List of chunks with metadata:
            - content: chunk text
            - metadata: {page, section, heading, chunk_type, position}
        """
        doc = Document(file_path)
        chunks = []
        current_section = "Introduction"
        current_page = 1
        chunk_index = 0

        # Process document elements in order
        for element in doc.element.body:
            if isinstance(element, CT_P):  # Paragraph
                para = element._element

                # Check if it's a heading
                if self._is_heading(para):
                    heading_text = self._get_paragraph_text(para)
                    current_section = heading_text

                    # Headings become their own chunks (for structure search)
                    chunks.append({
                        'content': heading_text,
                        'metadata': {
                            'chunk_type': 'heading',
                            'section': current_section,
                            'heading_level': self._get_heading_level(para),
                            'page': current_page,
                            'chunk_index': chunk_index
                        }
                    })
                    chunk_index += 1

                else:
                    # Regular paragraph - add to current chunk buffer
                    para_text = self._get_paragraph_text(para)
                    if para_text.strip():
                        # Create chunks with overlap
                        para_chunks = self._create_overlapping_chunks(
                            para_text,
                            current_section,
                            current_page,
                            chunk_index
                        )
                        chunks.extend(para_chunks)
                        chunk_index += len(para_chunks)

            elif isinstance(element, CT_Tbl):  # Table
                # Extract table as structured text
                table_text = self._extract_table_text(element)

                # Tables become their own chunks (don't split)
                chunks.append({
                    'content': table_text,
                    'metadata': {
                        'chunk_type': 'table',
                        'section': current_section,
                        'page': current_page,
                        'chunk_index': chunk_index
                    }
                })
                chunk_index += 1

        logger.debug(f"Created {len(chunks)} chunks from {file_path}")
        return chunks

    def _create_overlapping_chunks(
        self,
        text: str,
        section: str,
        page: int,
        start_index: int
    ) -> List[Dict[str, Any]]:
        """
        Split long text into overlapping chunks

        Overlap ensures context continuity:
        Chunk 1: [tokens 0-512]
        Chunk 2: [tokens 384-896]  (128 token overlap)
        Chunk 3: [tokens 768-1280]
        """
        # Approximate tokenization (1 token ≈ 4 characters)
        words = text.split()
        chunks = []

        # Calculate chunks needed
        words_per_chunk = self.chunk_size * 0.75  # ~0.75 words per token
        overlap_words = self.chunk_overlap * 0.75

        start = 0
        chunk_num = 0

        while start < len(words):
            end = int(start + words_per_chunk)
            chunk_words = words[start:end]

            if chunk_words:
                chunks.append({
                    'content': ' '.join(chunk_words),
                    'metadata': {
                        'chunk_type': 'text',
                        'section': section,
                        'page': page,
                        'chunk_index': start_index + chunk_num,
                        'overlap_start': start > 0,  # Has overlap with previous
                        'overlap_end': end < len(words)  # Has overlap with next
                    }
                })

            # Move start forward with overlap
            start = int(end - overlap_words)
            chunk_num += 1

        return chunks

    def _is_heading(self, paragraph) -> bool:
        """Check if paragraph is a heading"""
        style = paragraph.style
        if style:
            style_name = style.name_val if hasattr(style, 'name_val') else str(style)
            return 'heading' in style_name.lower()
        return False

    def _get_heading_level(self, paragraph) -> int:
        """Extract heading level (1-9)"""
        style = paragraph.style
        if style:
            style_name = style.name_val if hasattr(style, 'name_val') else str(style)
            style_lower = style_name.lower()
            match = re.search(r'heading (\d)', style_lower)
            return int(match.group(1)) if match else 0
        return 0

    def _get_paragraph_text(self, paragraph) -> str:
        """Extract text from paragraph"""
        # Get text content
        text_nodes = []
        for node in paragraph.iter():
            if hasattr(node, 'text') and node.text:
                text_nodes.append(node.text)
        return ''.join(text_nodes)

    def _extract_table_text(self, table) -> str:
        """
        Extract table as formatted text

        Example output:
        | Header 1 | Header 2 | Header 3 |
        | Value 1  | Value 2  | Value 3  |
        """
        table_obj = Table(table, None)

        rows = []
        for row in table_obj.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(' | '.join(cells))

        return '\n'.join(rows)
