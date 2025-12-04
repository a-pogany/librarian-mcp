"""
Folder metadata extraction and management for hierarchical search
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from collections import Counter
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class FolderMetadata:
    """Represents metadata for a single folder"""

    def __init__(
        self,
        path: str,
        description: str = "",
        topics: Optional[List[str]] = None,
        doc_count: int = 0,
        file_types: Optional[Set[str]] = None,
        last_updated: Optional[str] = None,
        parent_folder: Optional[str] = None
    ):
        self.path = path
        self.description = description
        self.topics = topics or []
        self.doc_count = doc_count
        self.file_types = file_types or set()
        self.last_updated = last_updated
        self.parent_folder = parent_folder

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'path': self.path,
            'description': self.description,
            'topics': self.topics,
            'doc_count': self.doc_count,
            'file_types': list(self.file_types),
            'last_updated': self.last_updated,
            'parent_folder': self.parent_folder
        }

    def get_search_text(self) -> str:
        """
        Get combined text for semantic search embedding

        This text will be embedded and used for folder-level semantic search
        """
        parts = [
            f"Folder: {self.path}",
            f"Description: {self.description}",
            f"Topics: {', '.join(self.topics)}"
        ]
        return " | ".join(parts)


class FolderMetadataExtractor:
    """Extract and generate folder metadata from document contents"""

    def __init__(self, indexer):
        """
        Initialize folder metadata extractor

        Args:
            indexer: FileIndexer instance with indexed documents
        """
        self.indexer = indexer
        self.folder_metadata: Dict[str, FolderMetadata] = {}

    def build_folder_metadata(self) -> Dict[str, FolderMetadata]:
        """
        Build metadata for all folders by analyzing contained documents

        Returns:
            Dictionary mapping folder paths to FolderMetadata objects
        """
        logger.info("Building folder metadata from indexed documents")

        # Group documents by folder
        folder_docs: Dict[str, List[Dict]] = {}

        for doc_path, doc in self.indexer.index.documents.items():
            # Extract folder path (product/component level)
            folder_path = self._extract_folder_path(doc_path)

            if folder_path not in folder_docs:
                folder_docs[folder_path] = []
            folder_docs[folder_path].append(doc)

        # Generate metadata for each folder
        for folder_path, docs in folder_docs.items():
            metadata = self._generate_folder_metadata(folder_path, docs)
            self.folder_metadata[folder_path] = metadata

        logger.info(f"Generated metadata for {len(self.folder_metadata)} folders")
        return self.folder_metadata

    def _extract_folder_path(self, doc_path: str) -> str:
        """
        Extract folder path from document path

        Args:
            doc_path: Document path like "product/component/file.md"

        Returns:
            Folder path like "product/component"
        """
        parts = Path(doc_path).parts

        # Return product/component (first two levels)
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        elif len(parts) == 1:
            return parts[0]
        else:
            return "root"

    def _generate_folder_metadata(
        self,
        folder_path: str,
        docs: List[Dict]
    ) -> FolderMetadata:
        """
        Generate metadata for a folder from its documents

        Args:
            folder_path: Path to folder
            docs: List of documents in folder

        Returns:
            FolderMetadata object
        """
        # Extract basic stats
        doc_count = len(docs)
        file_types = set(doc['file_type'] for doc in docs)

        # Find most recent modification
        last_updated = max(
            (doc['last_modified'] for doc in docs),
            default=datetime.now().isoformat()
        )

        # Generate description from document contents
        description = self._generate_description(folder_path, docs)

        # Extract key topics
        topics = self._extract_topics(docs)

        # Determine parent folder
        parent = str(Path(folder_path).parent) if '/' in folder_path else None

        return FolderMetadata(
            path=folder_path,
            description=description,
            topics=topics,
            doc_count=doc_count,
            file_types=file_types,
            last_updated=last_updated,
            parent_folder=parent
        )

    def _generate_description(
        self,
        folder_path: str,
        docs: List[Dict],
        max_length: int = 300
    ) -> str:
        """
        Generate a descriptive summary of the folder's purpose

        Uses heuristics:
        1. Check for README or index files
        2. Extract common heading patterns
        3. Analyze document titles and first paragraphs

        Args:
            folder_path: Folder path
            docs: Documents in folder
            max_length: Maximum description length

        Returns:
            Generated description string
        """
        # Strategy 1: Look for README or index files
        readme_doc = self._find_readme(docs)
        if readme_doc:
            description = self._extract_first_paragraph(readme_doc['content'])
            if description:
                return self._truncate(description, max_length)

        # Strategy 2: Extract common themes from headings
        all_headings = []
        for doc in docs:
            if doc.get('headings'):
                all_headings.extend(doc['headings'])

        if all_headings:
            # Use first few unique headings
            unique_headings = []
            seen = set()
            for heading in all_headings[:20]:
                normalized = heading.lower().strip()
                if normalized not in seen:
                    unique_headings.append(heading)
                    seen.add(normalized)

            if unique_headings:
                description = "Documentation covering: " + ", ".join(unique_headings[:5])
                return self._truncate(description, max_length)

        # Strategy 3: Use folder name and file names
        folder_name = Path(folder_path).name
        description = f"Documentation for {folder_name} including "

        # Add representative file names
        file_names = [doc['file_name'].replace('.md', '').replace('.txt', '').replace('-', ' ')
                      for doc in docs[:5]]
        description += ", ".join(file_names)

        return self._truncate(description, max_length)

    def _find_readme(self, docs: List[Dict]) -> Optional[Dict]:
        """Find README or index document in folder"""
        readme_patterns = ['readme', 'index', 'overview']

        for doc in docs:
            name_lower = doc['file_name'].lower()
            for pattern in readme_patterns:
                if pattern in name_lower:
                    return doc
        return None

    def _extract_first_paragraph(self, content: str) -> str:
        """Extract first substantial paragraph from content"""
        lines = content.split('\n')

        paragraph = []
        for line in lines:
            line = line.strip()

            # Skip headings and empty lines
            if not line or line.startswith('#'):
                if paragraph:  # If we've started collecting, stop at next heading
                    break
                continue

            paragraph.append(line)

            # Stop if we have enough text
            if len(' '.join(paragraph)) > 200:
                break

        return ' '.join(paragraph)

    def _extract_topics(
        self,
        docs: List[Dict],
        max_topics: int = 10
    ) -> List[str]:
        """
        Extract key topics from document contents using term frequency

        Args:
            docs: Documents in folder
            max_topics: Maximum topics to return

        Returns:
            List of topic keywords
        """
        # Collect all words from headings and content
        all_words = []

        for doc in docs:
            # Heavily weight headings
            if doc.get('headings'):
                for heading in doc['headings']:
                    words = self._extract_keywords(heading)
                    all_words.extend(words * 3)  # Triple weight for headings

            # Also use file names
            name = doc['file_name'].replace('.md', '').replace('.txt', '').replace('.docx', '')
            words = self._extract_keywords(name)
            all_words.extend(words * 2)  # Double weight for filenames

            # Sample content (to avoid overwhelming with content words)
            content_sample = doc.get('content', '')[:2000]
            words = self._extract_keywords(content_sample)
            all_words.extend(words)

        # Count word frequencies
        word_counts = Counter(all_words)

        # Remove very common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        filtered_counts = {
            word: count for word, count in word_counts.items()
            if word not in stop_words and len(word) > 2
        }

        # Get top keywords
        top_topics = [word for word, _ in
                     sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:max_topics]]

        return top_topics

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Convert to lowercase and split on non-alphanumeric
        words = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return words

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."

    def get_folder_metadata(self, folder_path: str) -> Optional[FolderMetadata]:
        """Get metadata for specific folder"""
        return self.folder_metadata.get(folder_path)

    def list_all_folders(self) -> List[str]:
        """Get list of all folder paths"""
        return list(self.folder_metadata.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about folder metadata"""
        if not self.folder_metadata:
            return {'total_folders': 0}

        return {
            'total_folders': len(self.folder_metadata),
            'total_documents': sum(m.doc_count for m in self.folder_metadata.values()),
            'avg_docs_per_folder': sum(m.doc_count for m in self.folder_metadata.values()) / len(self.folder_metadata),
            'folders': [m.to_dict() for m in self.folder_metadata.values()]
        }
