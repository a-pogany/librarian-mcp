"""
Hierarchical search engine with folder-level filtering

Performs two-stage search:
1. Search folders by metadata/description
2. Search documents within top matching folders
"""

import logging
from typing import List, Dict, Optional, Any
from .embeddings import EmbeddingGenerator
from .folder_metadata import FolderMetadataExtractor
from .folder_vector_db import FolderVectorDatabase
from .semantic_search import SemanticSearchEngine

logger = logging.getLogger(__name__)


class HierarchicalSearchEngine:
    """
    Two-stage hierarchical search engine

    Stage 1: Search folders based on query
    Stage 2: Search documents within top folders
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        folder_metadata_extractor: FolderMetadataExtractor,
        folder_vector_db: FolderVectorDatabase,
        semantic_search_engine: SemanticSearchEngine,
        enable_folder_filtering: bool = True
    ):
        """
        Initialize hierarchical search engine

        Args:
            embedding_generator: Embedding generator instance
            folder_metadata_extractor: Folder metadata extractor
            folder_vector_db: Folder vector database
            semantic_search_engine: Document-level semantic search
            enable_folder_filtering: Enable folder-level filtering stage
        """
        self.embedding_generator = embedding_generator
        self.folder_metadata = folder_metadata_extractor
        self.folder_vector_db = folder_vector_db
        self.semantic_search = semantic_search_engine
        self.enable_folder_filtering = enable_folder_filtering

        logger.info(f"Hierarchical search initialized (folder filtering: {enable_folder_filtering})")

    def search(
        self,
        query: str,
        product: Optional[str] = None,
        component: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        max_results: int = 10,
        max_folders: int = 3,
        folder_similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform hierarchical search

        Args:
            query: Search query
            product: Filter by product
            component: Filter by component
            file_types: Filter by file types
            max_results: Maximum documents to return
            max_folders: Maximum folders to search (stage 1)
            folder_similarity_threshold: Minimum folder similarity score

        Returns:
            List of matching documents with scores
        """
        if not query.strip():
            return []

        # Stage 1: Find relevant folders (if enabled)
        relevant_folders = None
        if self.enable_folder_filtering:
            try:
                relevant_folders = self._search_folders(
                    query=query,
                    product=product,
                    max_results=max_folders,
                    min_similarity=folder_similarity_threshold
                )

                if relevant_folders:
                    folder_names = [f['folder_path'] for f in relevant_folders]
                    scores = [f"{f['similarity']:.2f}" for f in relevant_folders]
                    logger.info(f"Stage 1: Found {len(relevant_folders)} relevant folders: "
                              f"{', '.join(f'{n}({s})' for n, s in zip(folder_names, scores))}")
                else:
                    logger.info("Stage 1: No folders matched, falling back to full search")

            except Exception as e:
                logger.warning(f"Folder search failed, falling back to full search: {e}")
                relevant_folders = None

        # Stage 2: Search documents (scoped to folders if available)
        documents = self._search_documents(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            relevant_folders=relevant_folders,
            max_results=max_results
        )

        # Add folder context to results
        if relevant_folders:
            folder_scores = {f['folder_path']: f['similarity'] for f in relevant_folders}

            for doc in documents:
                doc_folder = self._extract_folder_from_doc(doc['file_path'])

                # Add folder relevance score
                if doc_folder in folder_scores:
                    doc['folder_similarity'] = folder_scores[doc_folder]
                    doc['folder_matched'] = True

                    # Boost document score based on folder relevance
                    original_score = doc.get('relevance_score', 0.5)
                    folder_boost = folder_scores[doc_folder] * 0.2  # 20% weight to folder match
                    doc['relevance_score'] = round(original_score + folder_boost, 2)
                else:
                    doc['folder_matched'] = False

        return documents

    def _search_folders(
        self,
        query: str,
        product: Optional[str] = None,
        max_results: int = 3,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Search for relevant folders

        Args:
            query: Search query
            product: Filter by product
            max_results: Maximum folders to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of matching folders with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.encode_query(query)

        # Build folder filter
        where_filter = {}
        if product:
            where_filter['product'] = product

        # Search folder vectors
        folder_results = self.folder_vector_db.search(
            query_embedding=query_embedding,
            n_results=max_results,
            where=where_filter if where_filter else None
        )

        # Filter by minimum similarity
        filtered_folders = [
            f for f in folder_results
            if f['similarity'] >= min_similarity
        ]

        return filtered_folders

    def _search_documents(
        self,
        query: str,
        product: Optional[str],
        component: Optional[str],
        file_types: Optional[List[str]],
        relevant_folders: Optional[List[Dict[str, Any]]],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Search documents (optionally scoped to folders)

        Args:
            query: Search query
            product: Product filter
            component: Component filter
            file_types: File type filter
            relevant_folders: List of folders to scope search to
            max_results: Maximum results

        Returns:
            List of matching documents
        """
        if relevant_folders:
            # Search within specific folders
            all_results = []

            for folder_info in relevant_folders:
                folder_path = folder_info['folder_path']

                # Parse product/component from folder path
                parts = folder_path.split('/')
                folder_product = parts[0] if len(parts) > 0 else None
                folder_component = parts[1] if len(parts) > 1 else None

                # Search this folder
                try:
                    folder_results = self.semantic_search.search(
                        query=query,
                        product=folder_product,
                        component=folder_component,
                        file_types=file_types,
                        max_results=max_results * 2  # Get more per folder, dedupe later
                    )

                    # Tag with folder info
                    for result in folder_results:
                        result['source_folder'] = folder_path
                        result['folder_similarity'] = folder_info['similarity']

                    all_results.extend(folder_results)

                except Exception as e:
                    logger.error(f"Error searching folder {folder_path}: {e}")

            # Deduplicate and sort by relevance
            seen_paths = set()
            unique_results = []
            for result in sorted(all_results, key=lambda x: x.get('relevance_score', 0), reverse=True):
                if result['file_path'] not in seen_paths:
                    seen_paths.add(result['file_path'])
                    unique_results.append(result)

            return unique_results[:max_results]

        else:
            # No folder filtering, search all documents
            return self.semantic_search.search(
                query=query,
                product=product,
                component=component,
                file_types=file_types,
                max_results=max_results
            )

    def _extract_folder_from_doc(self, doc_path: str) -> str:
        """Extract folder path from document path"""
        from pathlib import Path
        parts = Path(doc_path).parts

        # Return product/component level
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        elif len(parts) == 1:
            return parts[0]
        else:
            return "root"

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            'type': 'hierarchical',
            'folder_filtering_enabled': self.enable_folder_filtering,
            'folder_stats': self.folder_vector_db.get_stats(),
            'document_stats': self.semantic_search.get_stats()
        }
