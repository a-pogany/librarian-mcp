"""File parsers for different document formats"""

from abc import ABC, abstractmethod
from typing import Dict, List
from pathlib import Path
import re
import chardet
from docx import Document as DOCXDocument


class ParseResult:
    """Result of parsing a document"""

    def __init__(self, content: str, headings: List[str], metadata: Dict):
        self.content = content
        self.headings = headings
        self.metadata = metadata

    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'headings': self.headings,
            'metadata': self.metadata
        }


class Parser(ABC):
    """Abstract parser interface"""

    @abstractmethod
    def parse(self, file_path: str) -> ParseResult:
        """
        Parse file and extract content

        Args:
            file_path: Path to file to parse

        Returns:
            ParseResult with content, headings, metadata
        """
        pass


class MarkdownParser(Parser):
    """Parse Markdown files"""

    def parse(self, file_path: str) -> ParseResult:
        """Parse Markdown file"""
        # Detect encoding
        with open(file_path, 'rb') as f:
            raw = f.read()
            encoding = chardet.detect(raw)['encoding'] or 'utf-8'

        # Read file with detected encoding
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()

        # Extract headings
        headings = self._extract_headings(content)

        return ParseResult(
            content=content,
            headings=headings,
            metadata={'encoding': encoding}
        )

    def _extract_headings(self, content: str) -> List[str]:
        """Extract markdown headings"""
        headings = []
        for line in content.split('\n'):
            if line.strip().startswith('#'):
                # Remove # symbols and clean
                heading = re.sub(r'^#+\s*', '', line).strip()
                if heading:
                    headings.append(heading)
        return headings


class TextParser(Parser):
    """Parse plain text files"""

    def parse(self, file_path: str) -> ParseResult:
        """Parse text file"""
        # Detect encoding
        with open(file_path, 'rb') as f:
            raw = f.read()
            encoding = chardet.detect(raw)['encoding'] or 'utf-8'

        # Read file with detected encoding
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()

        return ParseResult(
            content=content,
            headings=[],
            metadata={'encoding': encoding}
        )


class DOCXParser(Parser):
    """Parse DOCX files"""

    def parse(self, file_path: str) -> ParseResult:
        """Parse DOCX file"""
        doc = DOCXDocument(file_path)

        content_parts = []
        headings = []

        # Extract paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # Check if it's a heading
            if para.style.name.startswith('Heading'):
                headings.append(text)

            content_parts.append(text)

        # Extract tables
        for table in doc.tables:
            content_parts.append(self._format_table(table))

        # Extract metadata
        metadata = self._extract_metadata(doc)

        return ParseResult(
            content='\n'.join(content_parts),
            headings=headings,
            metadata=metadata
        )

    def _format_table(self, table) -> str:
        """Format table as text"""
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(' | '.join(cells))
        return '\n'.join(rows)

    def _extract_metadata(self, doc: DOCXDocument) -> Dict:
        """Extract document metadata"""
        try:
            core_props = doc.core_properties
            return {
                'title': core_props.title or '',
                'author': core_props.author or '',
                'created': core_props.created.isoformat() if core_props.created else None,
                'modified': core_props.modified.isoformat() if core_props.modified else None
            }
        except Exception:
            return {}
