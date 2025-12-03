"""
Vector database wrapper for ChromaDB
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorDatabase:
    """ChromaDB wrapper for vector similarity search"""

    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "documents"):
        """
        Initialize vector database

        Args:
            persist_directory: Directory to persist database (None for in-memory)
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings

            logger.info("Initializing ChromaDB")

            if self.persist_directory:
                # Persistent storage
                persist_path = Path(self.persist_directory)
                persist_path.mkdir(parents=True, exist_ok=True)

                self.client = chromadb.PersistentClient(
                    path=str(persist_path),
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info(f"Using persistent storage: {persist_path}")
            else:
                # In-memory storage
                self.client = chromadb.Client(
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info("Using in-memory storage")

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

            logger.info(f"Collection '{self.collection_name}' ready")

        except ImportError:
            logger.error("chromadb not installed. Install with: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise

    def add_documents(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]]
    ):
        """
        Add multiple documents to the database

        Args:
            ids: List of document IDs (file paths)
            embeddings: numpy array of embeddings (shape: [n_docs, embedding_dim])
            metadatas: List of metadata dictionaries
        """
        if len(ids) != len(embeddings) or len(ids) != len(metadatas):
            raise ValueError("ids, embeddings, and metadatas must have the same length")

        try:
            # Convert numpy array to list for ChromaDB
            embeddings_list = embeddings.tolist()

            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas
            )

            logger.debug(f"Added {len(ids)} documents to vector database")

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def add_document(
        self,
        doc_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ):
        """
        Add a single document to the database

        Args:
            doc_id: Document ID (file path)
            embedding: Document embedding vector
            metadata: Document metadata
        """
        self.add_documents(
            ids=[doc_id],
            embeddings=np.array([embedding]),
            metadatas=[metadata]
        )

    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filter (e.g., {"product": "symphony"})

        Returns:
            List of results with document IDs, distances, and metadata
        """
        try:
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where,
                include=["metadatas", "distances"]
            )

            # Parse results
            parsed_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    distance = results['distances'][0][i]
                    metadata = results['metadatas'][0][i]

                    # Convert distance to similarity score (0-1, higher is better)
                    # ChromaDB returns cosine distance (0-2), convert to similarity
                    similarity = 1.0 - (distance / 2.0)

                    parsed_results.append({
                        'id': doc_id,
                        'similarity': max(0.0, min(1.0, similarity)),  # Clamp to [0, 1]
                        'distance': distance,
                        'metadata': metadata
                    })

            return parsed_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    def update_document(
        self,
        doc_id: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update a document's embedding or metadata

        Args:
            doc_id: Document ID
            embedding: New embedding (optional)
            metadata: New metadata (optional)
        """
        try:
            update_kwargs = {"ids": [doc_id]}

            if embedding is not None:
                update_kwargs["embeddings"] = [embedding.tolist()]

            if metadata is not None:
                update_kwargs["metadatas"] = [metadata]

            self.collection.update(**update_kwargs)
            logger.debug(f"Updated document: {doc_id}")

        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise

    def delete_document(self, doc_id: str):
        """
        Delete a document from the database

        Args:
            doc_id: Document ID to delete
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document: {doc_id}")

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise

    def delete_documents(self, doc_ids: List[str]):
        """
        Delete multiple documents

        Args:
            doc_ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=doc_ids)
            logger.debug(f"Deleted {len(doc_ids)} documents")

        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def clear(self):
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared collection '{self.collection_name}'")

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    def get_count(self) -> int:
        """Get number of documents in the database"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting count: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'collection_name': self.collection_name,
            'document_count': self.get_count(),
            'persist_directory': self.persist_directory,
            'storage_type': 'persistent' if self.persist_directory else 'in-memory'
        }
