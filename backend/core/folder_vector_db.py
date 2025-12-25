"""
Vector database for folder-level embeddings
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class FolderVectorDatabase:
    """
    Manages vector embeddings for folders using ChromaDB

    Separate from document-level embeddings to allow hierarchical search
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "folders_v1",
        enable_compression: bool = True
    ):
        """
        Initialize folder vector database

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of ChromaDB collection
            enable_compression: Enable embedding compression
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb")

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB client
        if persist_directory:
            logger.info(f"Initializing persistent folder vector DB at: {persist_directory}")
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        else:
            logger.info("Initializing in-memory folder vector DB")
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )

        # Create or get collection with optimized settings
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,  # Higher for better quality
                    "hnsw:M": 16,  # Moderate connections
                    "hnsw:search_ef": 100  # Search-time accuracy
                }
            )
        except Exception as e:
            logger.warning(f"Failed to apply HNSW metadata, retrying with defaults: {e}")
            self.collection = self.client.get_or_create_collection(
                name=collection_name
            )

        logger.info(f"Folder vector database initialized: {collection_name}")

    def add_folders_batch(
        self,
        folder_paths: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 100
    ):
        """
        Add folder embeddings in batches

        Args:
            folder_paths: List of folder paths (used as IDs)
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            batch_size: Number of folders per batch
        """
        if not folder_paths:
            logger.warning("No folders to add")
            return

        if len(folder_paths) != len(embeddings) or len(folder_paths) != len(metadatas):
            raise ValueError("Lengths of folder_paths, embeddings, and metadatas must match")

        logger.info(f"Adding {len(folder_paths)} folder embeddings to vector DB")

        # Convert embeddings to list format for ChromaDB
        embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb
                          for emb in embeddings]

        # Add in batches
        for i in range(0, len(folder_paths), batch_size):
            batch_end = min(i + batch_size, len(folder_paths))

            try:
                self.collection.add(
                    ids=folder_paths[i:batch_end],
                    embeddings=embeddings_list[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )
                logger.debug(f"Added folder batch {i}-{batch_end}")

            except Exception as e:
                logger.error(f"Error adding folder batch {i}-{batch_end}: {e}")
                raise

        logger.info(f"Successfully added {len(folder_paths)} folders")

    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar folders

        Args:
            query_embedding: Query vector embedding
            n_results: Maximum results to return
            where: Optional metadata filter

        Returns:
            List of matching folders with scores
        """
        try:
            # Convert to list if numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=['metadatas', 'distances']
            )

            # Parse results
            folders = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    folder_path = results['ids'][0][i]
                    distance = results['distances'][0][i]
                    metadata = results['metadatas'][0][i]

                    # Convert distance to similarity score (0-1)
                    similarity = 1 - (distance / 2)  # Cosine distance to similarity

                    folders.append({
                        'folder_path': folder_path,
                        'similarity': max(0.0, min(1.0, similarity)),
                        'distance': distance,
                        'metadata': metadata
                    })

            logger.debug(f"Found {len(folders)} matching folders")
            return folders

        except Exception as e:
            logger.error(f"Error searching folders: {e}")
            raise

    def update_folder(
        self,
        folder_path: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ):
        """
        Update or insert a single folder

        Args:
            folder_path: Folder path (used as ID)
            embedding: Embedding vector
            metadata: Folder metadata
        """
        try:
            # Convert to list if numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            self.collection.upsert(
                ids=[folder_path],
                embeddings=[embedding],
                metadatas=[metadata]
            )

            logger.debug(f"Updated folder: {folder_path}")

        except Exception as e:
            logger.error(f"Error updating folder {folder_path}: {e}")
            raise

    def delete_folder(self, folder_path: str):
        """
        Delete a folder from the database

        Args:
            folder_path: Folder path to delete
        """
        try:
            self.collection.delete(ids=[folder_path])
            logger.debug(f"Deleted folder: {folder_path}")

        except Exception as e:
            logger.error(f"Error deleting folder {folder_path}: {e}")
            raise

    def clear(self):
        """Clear all folders from the database"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                    "hnsw:search_ef": 100
                }
            )
            logger.info("Cleared folder vector database")

        except Exception as e:
            logger.error(f"Error clearing folder database: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            count = self.collection.count()

            return {
                'collection_name': self.collection_name,
                'total_folders': count,
                'persist_directory': self.persist_directory
            }

        except Exception as e:
            logger.error(f"Error getting folder DB stats: {e}")
            return {'error': str(e)}

    def get_count(self) -> int:
        """Get number of folders in the database"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting folder count: {e}")
            return 0
