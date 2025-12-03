"""
Embedding generation using sentence-transformers for semantic search
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate vector embeddings for documents and queries"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator

        Args:
            model_name: Name of sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # MiniLM-L6-v2 dimension
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Get actual dimension from model
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")

        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of text strings to encode
            batch_size: Number of texts to process at once
            show_progress: Show progress bar during encoding

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])

        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            return embeddings

        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Text string to encode

        Returns:
            numpy array of shape (dimension,)
        """
        if not text:
            return np.zeros(self.dimension)

        embeddings = self.encode([text])
        return embeddings[0]

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a search query

        Args:
            query: Query string

        Returns:
            Query embedding vector
        """
        return self.encode_single(query)

    def encode_document(self, content: str, max_length: int = 512) -> np.ndarray:
        """
        Encode a document with optional truncation

        Args:
            content: Document content
            max_length: Maximum number of tokens (approximate)

        Returns:
            Document embedding vector
        """
        # Simple truncation by characters (rough approximation)
        # 1 token ≈ 4 characters for English text
        max_chars = max_length * 4

        if len(content) > max_chars:
            content = content[:max_chars]
            logger.debug(f"Truncated document to {max_chars} characters")

        return self.encode_single(content)

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

    def get_model_name(self) -> str:
        """Get model name"""
        return self.model_name
