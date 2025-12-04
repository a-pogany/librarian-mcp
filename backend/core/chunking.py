"""
Hierarchical document chunking with structure awareness
"""

from typing import List, Dict, Any, Optional
from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
import re
import logging

logger = logging.getLogger(__name__)


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
