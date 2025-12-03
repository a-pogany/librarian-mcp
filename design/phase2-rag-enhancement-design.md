# Phase 2: RAG Enhancement - Design Specification

## Overview

Phase 2 enhances Phase 1's keyword search with semantic search using vector embeddings (RAG - Retrieval Augmented Generation). This enables better handling of vague queries and conceptual searches.

**Goals:**
- Add vector embeddings for all documents using sentence-transformers
- Implement semantic search using ChromaDB vector database
- Create hybrid search combining keyword + semantic ranking
- Improve results for conceptual/vague queries
- Maintain backward compatibility with Phase 1

**Time Estimate:** 8-12 hours

---

## System Architecture

### Enhanced Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                  Documentation Files                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   File Indexer (Phase 1)                     │
│                         +                                     │
│              Embedding Generator (NEW)                       │
│  ┌────────────────────────────────────────────────┐         │
│  │  sentence-transformers (all-MiniLM-L6-v2)      │         │
│  │  • Converts text → 384-dim vectors             │         │
│  │  • Batch processing for efficiency             │         │
│  └────────────────────────────────────────────────┘         │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ↓                       ↓
┌──────────────────────┐  ┌───────────────────────┐
│  In-Memory Index     │  │  Vector Database      │
│  (Keyword Search)    │  │  (ChromaDB)           │
│                      │  │                       │
│  • documents: {}     │  │  • embeddings: []     │
│  • products: {}      │  │  • metadata: []       │
│  • components: {}    │  │  • HNSW index         │
└──────────┬───────────┘  └───────┬───────────────┘
           │                      │
           └──────────┬───────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│            Hybrid Search Engine (NEW)                        │
│  ┌─────────────────────────────────────────────────┐        │
│  │  Keyword Search  →  keyword_score (0-1)         │        │
│  │  Semantic Search →  semantic_score (0-1)        │        │
│  │  Combine:                                        │        │
│  │    hybrid = (1-w)*keyword + w*semantic          │        │
│  └─────────────────────────────────────────────────┘        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (Phase 1)                      │
│              + Enhanced search_documentation                 │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow: Indexing with Embeddings
```
[Document File]
    │
    ├─→ [Phase 1: FileIndexer]
    │       ↓
    │   [Parser] → {content, headings, metadata}
    │       ↓
    │   [DocumentIndex.add_document()]
    │
    └─→ [Phase 2: Embedding Pipeline]
            ↓
        [EmbeddingGenerator.encode(content)]
            ↓
        384-dimensional vector
            ↓
        [VectorDatabase.add_document()]
            ├─ embedding: [0.12, -0.34, ...]
            ├─ metadata: {product, component, file_type}
            └─ doc_id: "product/component/file.md"
```

### Data Flow: Hybrid Search
```
[User Query: "how do we authenticate users?"]
    │
    ├─→ [Keyword Search]
    │       ↓
    │   Extract keywords: ["authenticate", "users"]
    │       ↓
    │   Match against content, headings, filenames
    │       ↓
    │   keyword_results: [{doc, keyword_score}, ...]
    │
    └─→ [Semantic Search]
            ↓
        [EmbeddingGenerator.encode(query)]
            ↓
        Query vector: [0.08, -0.22, ...]
            ↓
        [VectorDatabase.search(query_vector)]
            ↓
        Cosine similarity ranking
            ↓
        semantic_results: [{doc, similarity}, ...]
    │
    ↓
[Hybrid Scoring & Re-ranking]
    │
    ├─ Merge results from both searches
    ├─ Calculate hybrid_score = (1-w)*keyword + w*semantic
    ├─ Sort by hybrid_score descending
    └─ Return top N results
```

---

## Component Design

### 1. Embedding Generator (`backend/core/embeddings.py`)

**Purpose:** Generate vector embeddings for documents and queries using pre-trained sentence transformers

#### Class: EmbeddingGenerator
```python
class EmbeddingGenerator:
    """Generate vector embeddings using sentence-transformers"""

    # Dependencies
    model: SentenceTransformer              # Loaded model
    dimension: int                          # Embedding dimension (384 for MiniLM)
    model_name: str                         # Model identifier

    # Methods
    encode(texts: List[str]) -> np.ndarray  # Batch encoding
    encode_single(text: str) -> np.ndarray  # Single text encoding
```

**Supported Models:**
```python
MODELS = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "max_length": 256,
        "speed": "fast",
        "quality": "good",
        "size_mb": 80,
        "use_case": "default, production"
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "max_length": 384,
        "speed": "medium",
        "quality": "excellent",
        "size_mb": 420,
        "use_case": "high-quality requirements"
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "dimension": 384,
        "max_length": 512,
        "speed": "fast",
        "quality": "good",
        "size_mb": 80,
        "use_case": "question-answering focused"
    }
}
```

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model

        Args:
            model_name: HuggingFace model name
                First run downloads ~80MB model to cache
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for batch of texts

        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once

        Returns:
            Array of embeddings (shape: [len(texts), dimension])

        Performance:
            - ~100 texts/second on CPU
            - ~1000 texts/second on GPU
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text

        Args:
            text: Text string

        Returns:
            Embedding vector (shape: [dimension])
        """
        return self.encode([text])[0]
```

**Text Preprocessing:**
```python
def preprocess_for_embedding(text: str, max_length: int = 512) -> str:
    """
    Prepare text for embedding generation

    Processing:
    1. Truncate to max_length (model limit: 256-512 tokens)
    2. Remove excessive whitespace
    3. Keep punctuation (helps with semantic meaning)
    """
    # Truncate to approximate token limit (1 token ≈ 4 chars)
    char_limit = max_length * 4
    if len(text) > char_limit:
        text = text[:char_limit]

    # Normalize whitespace
    text = ' '.join(text.split())

    return text
```

**Batch Processing Strategy:**
```python
def batch_encode_documents(documents: List[Document], batch_size: int = 32):
    """
    Efficiently encode large document collections

    Strategy:
    1. Extract text content from all documents
    2. Preprocess each text
    3. Encode in batches (reduces overhead)
    4. Return embeddings mapped to document IDs
    """
    texts = [preprocess_for_embedding(doc.content) for doc in documents]

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = self.encode(batch, batch_size=batch_size)
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)
```

---

### 2. Vector Database (`backend/core/vector_db.py`)

**Purpose:** Store and search document embeddings using ChromaDB with HNSW indexing

#### Class: VectorDatabase
```python
class VectorDatabase:
    """ChromaDB wrapper for document embeddings"""

    # Dependencies
    client: chromadb.Client                 # ChromaDB client
    collection: chromadb.Collection         # Document collection
    persist_directory: str                  # Persistence path

    # Methods
    add_document(doc_id, embedding, text, metadata) -> None
    add_batch(doc_ids, embeddings, texts, metadatas) -> None
    search(query_embedding, n_results, filter_dict) -> Dict
    delete_document(doc_id: str) -> None
    clear() -> None
    get_stats() -> Dict
```

**Implementation:**
```python
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid

class VectorDatabase:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents"
    ):
        """
        Initialize ChromaDB

        Args:
            persist_directory: Path for persistence
            collection_name: Collection identifier
        """
        self.persist_directory = persist_directory
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False,
            allow_reset=True
        ))

        # Create or get collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine distance metric
        )

    def add_document(
        self,
        doc_id: str,
        embedding: List[float],
        text: str,
        metadata: Dict
    ) -> None:
        """
        Add single document to vector database

        Args:
            doc_id: Unique document identifier (path)
            embedding: Vector embedding (384 or 768 dims)
            text: Original text (stored for retrieval)
            metadata: Product, component, file_type, etc.
        """
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )

    def add_batch(
        self,
        doc_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict]
    ) -> None:
        """
        Add multiple documents in batch (more efficient)

        Args:
            doc_ids: List of document IDs
            embeddings: List of embedding vectors
            texts: List of text contents
            metadatas: List of metadata dicts

        Performance:
            - Batch inserts ~10x faster than individual adds
            - Recommended batch size: 100-500 documents
        """
        self.collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar documents using cosine similarity

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            filter_dict: Metadata filters
                Example: {"product": "symphony"}
                         {"$and": [{"product": "symphony"}, {"component": "PAM"}]}

        Returns:
            {
                "ids": [[doc_id1, doc_id2, ...]],
                "distances": [[0.12, 0.25, ...]],  # Lower = more similar
                "documents": [[text1, text2, ...]],
                "metadatas": [[meta1, meta2, ...]]
            }
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict,
            include=["distances", "documents", "metadatas"]
        )

        return results

    def delete_document(self, doc_id: str) -> None:
        """Remove document from database"""
        self.collection.delete(ids=[doc_id])

    def clear(self) -> None:
        """Clear all documents and recreate collection"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def get_stats(self) -> Dict:
        """Get database statistics"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection.name,
            "persist_directory": self.persist_directory
        }
```

**HNSW Index Configuration:**
```python
# Hierarchical Navigable Small World (HNSW) parameters
# ChromaDB uses these defaults:

HNSW_CONFIG = {
    "hnsw:space": "cosine",        # Distance metric (cosine, l2, ip)
    "hnsw:construction_ef": 200,   # Build-time accuracy (higher = slower build, better quality)
    "hnsw:search_ef": 100,         # Search-time accuracy (higher = slower search, better recall)
    "hnsw:M": 16                   # Max connections per node (higher = more memory, faster search)
}

# For production tuning:
# - High accuracy: construction_ef=400, search_ef=200, M=32
# - Balanced: construction_ef=200, search_ef=100, M=16 (default)
# - Fast: construction_ef=100, search_ef=50, M=8
```

**Metadata Filtering Examples:**
```python
# Single condition
filter_dict = {"product": "symphony"}

# Multiple conditions (AND)
filter_dict = {
    "$and": [
        {"product": "symphony"},
        {"component": "PAM"}
    ]
}

# OR condition
filter_dict = {
    "$or": [
        {"product": "symphony"},
        {"product": "project-x"}
    ]
}

# Complex nested
filter_dict = {
    "$and": [
        {"product": "symphony"},
        {
            "$or": [
                {"component": "PAM"},
                {"component": "Auth"}
            ]
        }
    ]
}
```

---

### 3. Hybrid Search Engine (`backend/core/search_v2.py`)

**Purpose:** Combine keyword and semantic search with configurable weighting

#### Class: HybridSearchEngine
```python
class HybridSearchEngine:
    """Hybrid search combining keyword and semantic approaches"""

    # Dependencies
    indexer: FileIndexer                    # Phase 1 indexer
    keyword_search: SearchEngine            # Phase 1 keyword search
    embeddings: EmbeddingGenerator          # Embedding generator
    vector_db: VectorDatabase               # Vector database
    config: Dict                            # Configuration

    # Methods
    index_document(doc: Document) -> None
    index_all_documents() -> None
    search(
        query: str,
        product: Optional[str],
        component: Optional[str],
        file_types: Optional[List[str]],
        max_results: int,
        hybrid_weight: float
    ) -> List[SearchResult]

    # Internal
    _combine_results(keyword_results, vector_results, weight) -> List[SearchResult]
```

**Implementation:**
```python
from .search import SearchEngine as KeywordSearch
from .embeddings import EmbeddingGenerator
from .vector_db import VectorDatabase
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    def __init__(self, indexer, config: Dict = None):
        """
        Initialize hybrid search engine

        Args:
            indexer: FileIndexer instance (Phase 1)
            config: Configuration dict
        """
        self.indexer = indexer
        self.config = config or {}

        # Phase 1: Keyword search
        self.keyword_search = KeywordSearch(indexer)

        # Phase 2: Semantic search
        model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.embeddings = EmbeddingGenerator(model_name=model_name)

        vector_db_path = config.get('vector_db_path', './chroma_db')
        self.vector_db = VectorDatabase(persist_directory=vector_db_path)

    def index_document(self, doc: Document) -> None:
        """
        Index single document with embeddings

        Args:
            doc: Document object from FileIndexer
        """
        # Generate embedding
        embedding = self.embeddings.encode_single(doc.content)

        # Store in vector DB
        self.vector_db.add_document(
            doc_id=doc.path,
            embedding=embedding.tolist(),
            text=doc.content,
            metadata={
                'product': doc.product,
                'component': doc.component,
                'file_type': doc.file_type,
                'file_name': doc.file_name
            }
        )

        logger.debug(f"Indexed with embedding: {doc.path}")

    def index_all_documents(self, batch_size: int = 32) -> Dict:
        """
        Generate embeddings for all indexed documents

        Args:
            batch_size: Number of documents to process at once

        Returns:
            Statistics about indexing process
        """
        start_time = time.time()
        documents = list(self.indexer.index.documents.values())

        if not documents:
            logger.warning("No documents to index")
            return {"status": "empty", "count": 0}

        # Batch processing for efficiency
        doc_ids = []
        texts = []
        metadatas = []

        for doc in documents:
            doc_ids.append(doc.path)
            texts.append(preprocess_for_embedding(doc.content))
            metadatas.append({
                'product': doc.product,
                'component': doc.component,
                'file_type': doc.file_type,
                'file_name': doc.file_name
            })

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.encode(batch_texts)
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings).tolist()

        # Store in vector database
        self.vector_db.add_batch(
            doc_ids=doc_ids,
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas
        )

        duration = time.time() - start_time

        logger.info(f"Indexed {len(documents)} documents in {duration:.2f}s")

        return {
            "status": "complete",
            "documents_indexed": len(documents),
            "duration_seconds": duration,
            "embeddings_dimension": self.embeddings.dimension
        }

    def search(
        self,
        query: str,
        product: Optional[str] = None,
        component: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        max_results: int = 10,
        hybrid_weight: float = 0.5
    ) -> List[Dict]:
        """
        Hybrid search combining keyword and semantic approaches

        Args:
            query: Search query
            product: Product filter
            component: Component filter
            file_types: File type filter
            max_results: Maximum results to return
            hybrid_weight: Weight for semantic search (0-1)
                0.0 = pure keyword search
                0.5 = balanced hybrid (default)
                1.0 = pure semantic search

        Returns:
            Combined and re-ranked search results
        """
        # 1. Keyword search (Phase 1)
        keyword_results = self.keyword_search.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=max_results * 2  # Get more for re-ranking
        )

        # 2. Semantic search (Phase 2)
        query_embedding = self.embeddings.encode_single(query)

        # Build metadata filter
        filter_dict = self._build_filter(product, component, file_types)

        vector_results = self.vector_db.search(
            query_embedding=query_embedding.tolist(),
            n_results=max_results * 2,
            filter_dict=filter_dict
        )

        # 3. Combine and re-rank
        combined = self._combine_results(
            keyword_results,
            vector_results,
            hybrid_weight
        )

        return combined[:max_results]

    def _build_filter(
        self,
        product: Optional[str],
        component: Optional[str],
        file_types: Optional[List[str]]
    ) -> Optional[Dict]:
        """Build ChromaDB metadata filter"""
        conditions = []

        if product:
            conditions.append({"product": product})

        if component:
            conditions.append({"component": component})

        if file_types:
            # ChromaDB doesn't support $in, use $or
            type_conditions = [{"file_type": ft} for ft in file_types]
            if len(type_conditions) == 1:
                conditions.append(type_conditions[0])
            else:
                conditions.append({"$or": type_conditions})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def _combine_results(
        self,
        keyword_results: List[Dict],
        vector_results: Dict,
        weight: float
    ) -> List[Dict]:
        """
        Combine keyword and semantic results with hybrid scoring

        Algorithm:
        1. Create score map for all documents
        2. Add keyword scores (already normalized 0-1)
        3. Add semantic scores (convert distance to similarity)
        4. Calculate hybrid score: (1-weight)*keyword + weight*semantic
        5. Sort by hybrid score descending

        Args:
            keyword_results: Results from keyword search
            vector_results: Results from vector search
            weight: Semantic weight (0-1)

        Returns:
            Combined and re-ranked results
        """
        # Score map: {doc_id: {keyword, semantic, result}}
        scores = {}

        # Add keyword scores
        for result in keyword_results:
            doc_id = result['file_path']
            keyword_score = result['relevance_score']
            scores[doc_id] = {
                'keyword': keyword_score,
                'semantic': 0.0,
                'result': result
            }

        # Add semantic scores
        if vector_results['ids'] and vector_results['ids'][0]:
            for i, doc_id in enumerate(vector_results['ids'][0]):
                # Convert cosine distance to similarity
                # ChromaDB returns distance (lower = more similar)
                # Similarity = 1 - distance
                distance = vector_results['distances'][0][i]
                similarity = 1.0 - distance

                if doc_id in scores:
                    # Document in both results
                    scores[doc_id]['semantic'] = similarity
                else:
                    # Document only in semantic results
                    doc = self.indexer.index.documents.get(doc_id)
                    if doc:
                        scores[doc_id] = {
                            'keyword': 0.0,
                            'semantic': similarity,
                            'result': {
                                'id': doc_id,
                                'file_path': doc_id,
                                'product': doc['product'],
                                'component': doc['component'],
                                'file_name': doc['file_name'],
                                'file_type': doc['file_type'],
                                'snippet': doc['content'][:200],
                                'last_modified': doc['last_modified']
                            }
                        }

        # Calculate hybrid scores
        results = []
        for doc_id, score_data in scores.items():
            hybrid_score = (
                (1 - weight) * score_data['keyword'] +
                weight * score_data['semantic']
            )

            result = score_data['result'].copy()
            result['relevance_score'] = round(hybrid_score, 2)
            result['keyword_score'] = round(score_data['keyword'], 2)
            result['semantic_score'] = round(score_data['semantic'], 2)
            results.append(result)

        # Sort by hybrid score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        return results
```

**Hybrid Scoring Examples:**
```
Query: "how to authenticate users?"

Document 1: "api-authentication.md"
  Keyword score: 0.85 (strong keyword match)
  Semantic score: 0.92 (high conceptual similarity)

  With weight=0.5:
    hybrid = (1-0.5)*0.85 + 0.5*0.92 = 0.425 + 0.46 = 0.885

Document 2: "oauth-spec.md"
  Keyword score: 0.45 (weak keyword match - "OAuth" not in query)
  Semantic score: 0.88 (high conceptual similarity - OAuth is authentication)

  With weight=0.5:
    hybrid = (1-0.5)*0.45 + 0.5*0.88 = 0.225 + 0.44 = 0.665

Document 3: "database-schema.md"
  Keyword score: 0.10 (minimal keyword match)
  Semantic score: 0.15 (low conceptual similarity)

  With weight=0.5:
    hybrid = (1-0.5)*0.10 + 0.5*0.15 = 0.05 + 0.075 = 0.125

Ranking: Document 1 (0.885) > Document 2 (0.665) > Document 3 (0.125)
```

**Weight Tuning Guidance:**
```python
WEIGHT_PROFILES = {
    "keyword_focused": {
        "weight": 0.2,
        "use_case": "Technical terms, exact phrases, code snippets",
        "example": "JWT token validation"
    },
    "balanced": {
        "weight": 0.5,
        "use_case": "General queries, default setting",
        "example": "authentication methods"
    },
    "semantic_focused": {
        "weight": 0.8,
        "use_case": "Vague queries, conceptual searches",
        "example": "how do we secure API access?"
    },
    "pure_semantic": {
        "weight": 1.0,
        "use_case": "Natural language questions",
        "example": "what are our options for user login?"
    }
}
```

---

## Enhanced MCP Tools

### Updated search_documentation Tool
```python
@mcp.tool()
def search_documentation(
    query: str,
    product: Optional[str] = None,
    component: Optional[str] = None,
    file_types: Optional[List[str]] = None,
    max_results: int = 10,
    search_mode: str = "hybrid",        # NEW
    hybrid_weight: float = 0.5          # NEW
) -> dict:
    """
    Search documentation with keyword, semantic, or hybrid search.

    Args:
        query: Search query
        product: Filter by product name
        component: Filter by component name
        file_types: Filter by file extensions
        max_results: Maximum results (1-50)
        search_mode: "keyword", "semantic", or "hybrid" (default)
        hybrid_weight: Semantic weight for hybrid mode (0-1, default 0.5)
            0.0 = pure keyword
            0.5 = balanced
            1.0 = pure semantic

    Returns:
        Search results with relevance scores

    Examples:
        # Hybrid search (default)
        search_documentation(query="authentication methods")

        # Keyword-only search
        search_documentation(
            query="OAuth2 implementation",
            search_mode="keyword"
        )

        # Semantic-focused search for vague queries
        search_documentation(
            query="how do we handle user sessions?",
            search_mode="hybrid",
            hybrid_weight=0.8
        )
    """
    if search_mode == "hybrid":
        results = hybrid_search_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=min(max_results, 50),
            hybrid_weight=hybrid_weight
        )
    elif search_mode == "keyword":
        results = keyword_search_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=min(max_results, 50)
        )
    elif search_mode == "semantic":
        # Pure semantic search (hybrid_weight=1.0)
        results = hybrid_search_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=min(max_results, 50),
            hybrid_weight=1.0
        )
    else:
        return {
            "error": "Invalid search_mode",
            "valid_modes": ["keyword", "semantic", "hybrid"]
        }

    return {
        "results": results,
        "total": len(results),
        "query": query,
        "mode": search_mode,
        "hybrid_weight": hybrid_weight if search_mode == "hybrid" else None
    }
```

---

## Configuration Updates

### Enhanced config.json
```json
{
  "system": {
    "name": "Documentation Search MCP",
    "version": "2.0.0"
  },
  "docs": {
    "root_path": "/path/to/docs",
    "file_extensions": [".md", ".txt", ".docx"],
    "max_file_size_mb": 10,
    "watch_for_changes": true,
    "index_on_startup": true
  },
  "search": {
    "mode": "hybrid",
    "max_results": 50,
    "snippet_length": 200,
    "hybrid_weight": 0.5
  },
  "embeddings": {
    "model": "all-MiniLM-L6-v2",
    "dimension": 384,
    "batch_size": 32,
    "max_text_length": 512
  },
  "vector_db": {
    "path": "./chroma_db",
    "collection_name": "documents",
    "distance_metric": "cosine",
    "hnsw_config": {
      "construction_ef": 200,
      "search_ef": 100,
      "M": 16
    }
  },
  "mcp": {
    "transport": "http-sse",
    "host": "127.0.0.1",
    "port": 3001,
    "endpoint": "/mcp"
  },
  "logging": {
    "level": "info",
    "file": "mcp_server.log"
  }
}
```

---

## Updated requirements.txt

```txt
# Core (Phase 1)
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0

# MCP
mcp==0.9.0

# File processing
python-docx==1.1.0
markdown==3.5.1
chardet==5.2.0
watchdog==3.0.0

# Phase 2: Embeddings
sentence-transformers==2.2.2
torch==2.1.0
transformers==4.35.0
huggingface-hub==0.19.0

# Phase 2: Vector database
chromadb==0.4.18
hnswlib==0.7.0
numpy==1.24.3

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.2
```

---

## Migration from Phase 1 to Phase 2

### Migration Steps

1. **Backup existing data**
```bash
# Backup Phase 1 index (if persistence added)
cp -r ./data ./data.backup
```

2. **Install new dependencies**
```bash
pip install sentence-transformers==2.2.2
pip install chromadb==0.4.18
pip install torch==2.1.0
```

3. **Update configuration**
```bash
# Add embeddings and vector_db sections to config.json
# See "Enhanced config.json" above
```

4. **Generate embeddings for existing documents**
```python
# Run once after Phase 2 deployment
from backend.core.indexer import FileIndexer
from backend.core.search_v2 import HybridSearchEngine

indexer = FileIndexer(config['docs']['root_path'])
indexer.build_index()

hybrid_search = HybridSearchEngine(indexer, config)
result = hybrid_search.index_all_documents(batch_size=32)

print(f"Indexed {result['documents_indexed']} documents")
print(f"Duration: {result['duration_seconds']:.2f}s")
```

5. **Update MCP server**
```python
# backend/main.py
from core.search_v2 import HybridSearchEngine

# Initialize hybrid search instead of keyword-only
hybrid_search_engine = HybridSearchEngine(indexer, config)
hybrid_search_engine.index_all_documents()
```

6. **Test hybrid search**
```bash
# Test with Claude Desktop
# Try vague queries like "how do we authenticate?"
# Compare with keyword-only results
```

### Backward Compatibility

Phase 2 maintains full backward compatibility:
- Keyword-only search still available via `search_mode="keyword"`
- Existing MCP tool parameters unchanged (new parameters are optional)
- Phase 1 FileIndexer and SearchEngine unchanged
- No breaking changes to API contracts

---

## Performance Optimization

### Indexing Performance

**Batch Processing:**
```python
# Bad: Individual document indexing
for doc in documents:
    embedding = embeddings.encode_single(doc.content)  # 100ms each
    vector_db.add_document(...)
# Total: 100ms × 1000 = 100 seconds

# Good: Batch indexing
texts = [doc.content for doc in documents]
embeddings_batch = embeddings.encode(texts, batch_size=32)  # 10 seconds
vector_db.add_batch(...)
# Total: ~12 seconds (8x faster)
```

**GPU Acceleration:**
```python
# Automatic GPU detection
import torch

if torch.cuda.is_available():
    model = SentenceTransformer(model_name, device='cuda')
    # 10-50x faster than CPU
else:
    model = SentenceTransformer(model_name, device='cpu')
```

**Memory Management:**
```python
# For large document sets (>10,000), process in chunks
chunk_size = 1000

for i in range(0, len(documents), chunk_size):
    chunk = documents[i:i + chunk_size]
    index_batch(chunk)
    # Optional: Force garbage collection
    gc.collect()
```

### Search Performance

**Query Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_embedding(query: str) -> np.ndarray:
    """Cache query embeddings for repeated searches"""
    return embeddings.encode_single(query)
```

**HNSW Tuning:**
```python
# Fast searches (lower recall)
collection = client.create_collection(
    name="documents",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:search_ef": 50  # Default: 100
    }
)

# Accurate searches (higher recall)
collection = client.create_collection(
    name="documents",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:search_ef": 200  # Higher = slower but more accurate
    }
)
```

---

## Testing Strategy

### Unit Tests

**Test: Embedding Generation**
```python
def test_embedding_generation():
    embeddings = EmbeddingGenerator()

    text = "This is a test document about authentication."
    embedding = embeddings.encode_single(text)

    assert embedding.shape == (384,)  # MiniLM dimension
    assert embedding.dtype == np.float32
    assert not np.isnan(embedding).any()

def test_batch_encoding():
    embeddings = EmbeddingGenerator()

    texts = ["Text 1", "Text 2", "Text 3"]
    batch_embeddings = embeddings.encode(texts)

    assert batch_embeddings.shape == (3, 384)
    assert batch_embeddings.dtype == np.float32
```

**Test: Vector Database**
```python
def test_vector_db_add_search():
    vdb = VectorDatabase(persist_directory="./test_db")

    # Add document
    embedding = np.random.rand(384).tolist()
    vdb.add_document(
        doc_id="test_doc",
        embedding=embedding,
        text="Test content",
        metadata={"product": "test"}
    )

    # Search
    results = vdb.search(
        query_embedding=embedding,
        n_results=1
    )

    assert len(results['ids'][0]) == 1
    assert results['ids'][0][0] == "test_doc"
    assert results['distances'][0][0] < 0.01  # Should be nearly identical

def test_vector_db_filtering():
    vdb = VectorDatabase()

    # Add multiple documents
    for i in range(5):
        vdb.add_document(
            doc_id=f"doc{i}",
            embedding=np.random.rand(384).tolist(),
            text=f"Content {i}",
            metadata={"product": "test" if i < 3 else "other"}
        )

    # Search with filter
    results = vdb.search(
        query_embedding=np.random.rand(384).tolist(),
        n_results=10,
        filter_dict={"product": "test"}
    )

    assert len(results['ids'][0]) == 3
```

**Test: Hybrid Search**
```python
def test_hybrid_search_combination(hybrid_search_engine):
    # Test query that should rank highly in both keyword and semantic
    results = hybrid_search_engine.search(
        query="authentication API",
        max_results=5,
        hybrid_weight=0.5
    )

    assert len(results) > 0
    assert all('relevance_score' in r for r in results)
    assert all('keyword_score' in r for r in results)
    assert all('semantic_score' in r for r in results)

    # Check score calculation
    for result in results:
        expected_hybrid = (
            0.5 * result['keyword_score'] +
            0.5 * result['semantic_score']
        )
        assert abs(result['relevance_score'] - expected_hybrid) < 0.01

def test_hybrid_weight_extremes(hybrid_search_engine):
    query = "authentication"

    # Pure keyword (weight=0)
    keyword_results = hybrid_search_engine.search(
        query=query,
        hybrid_weight=0.0
    )

    # Pure semantic (weight=1)
    semantic_results = hybrid_search_engine.search(
        query=query,
        hybrid_weight=1.0
    )

    # Rankings may differ significantly
    assert keyword_results != semantic_results
```

### Integration Tests

**Test: End-to-End Indexing and Search**
```python
@pytest.mark.asyncio
async def test_e2e_hybrid_search(temp_docs):
    # Setup
    indexer = FileIndexer(str(temp_docs))
    indexer.build_index()

    hybrid_search = HybridSearchEngine(indexer, config)
    hybrid_search.index_all_documents()

    # Search
    results = hybrid_search.search(
        query="API authentication methods",
        max_results=5,
        hybrid_weight=0.5
    )

    assert len(results) > 0
    assert all('relevance_score' in r for r in results)

    # Verify hybrid scoring
    assert all(
        0 <= r['relevance_score'] <= 1
        for r in results
    )
```

### Performance Benchmarks

**Benchmark: Embedding Speed**
```python
def benchmark_embedding_speed():
    embeddings = EmbeddingGenerator()

    texts = ["Test text " * 50] * 100  # 100 documents

    start = time.time()
    embeddings.encode(texts, batch_size=32)
    duration = time.time() - start

    docs_per_second = 100 / duration
    print(f"Embedding speed: {docs_per_second:.1f} docs/sec")

    assert docs_per_second > 50  # Minimum acceptable speed
```

**Benchmark: Search Latency**
```python
def benchmark_search_latency(hybrid_search_engine):
    # Index 1000 documents
    # ...

    query = "authentication methods"

    # Warm-up
    hybrid_search_engine.search(query, max_results=10)

    # Benchmark
    latencies = []
    for _ in range(10):
        start = time.time()
        hybrid_search_engine.search(query, max_results=10)
        latencies.append(time.time() - start)

    avg_latency = np.mean(latencies)
    print(f"Average search latency: {avg_latency*1000:.1f}ms")

    assert avg_latency < 0.5  # <500ms target
```

---

## Acceptance Criteria

### AC1: Embeddings Work ✓
- [x] Documents embedded on indexing with sentence-transformers
- [x] Embeddings stored in ChromaDB with metadata
- [x] Embedding generation <100ms per document (CPU)
- [x] Batch processing significantly faster than individual
- [x] GPU acceleration works if available

### AC2: Semantic Search Works ✓
- [x] Similar documents found by conceptual meaning
- [x] Works well for vague/natural language queries
- [x] Better than keyword-only for concept queries
- [x] Metadata filtering works correctly
- [x] Cosine similarity ranking is sensible

### AC3: Hybrid Search Works ✓
- [x] Combines keyword + semantic scores correctly
- [x] Configurable weighting (0-1 range)
- [x] Hybrid results better than either method alone
- [x] Weight=0 equivalent to pure keyword
- [x] Weight=1 equivalent to pure semantic

### AC4: Performance ✓
- [x] Search responds in <500ms (hybrid mode)
- [x] Index rebuild <2 minutes for 500 documents
- [x] Memory usage <500MB for 1000 documents
- [x] Batch indexing 5-10x faster than individual
- [x] HNSW search scales sub-linearly

### AC5: Backward Compatibility ✓
- [x] Phase 1 keyword search still available
- [x] Existing MCP tool signatures unchanged
- [x] New parameters are optional
- [x] No breaking changes to API
- [x] Phase 1 tests still pass

### AC6: Claude Integration ✓
- [x] Claude can choose search mode (keyword/semantic/hybrid)
- [x] Better context for vague questions
- [x] No regression on keyword query quality
- [x] Hybrid mode is helpful default
- [x] Tool documentation clear for Claude

---

## Next Steps (Phase 3 Preview)

Phase 2 provides enhanced search capabilities. Phase 3 will add:
- **REST API** for HTTP access (not just MCP)
- **React Web UI** for human users
- **Advanced features**: search history, favorites, annotations
- **Dual server**: MCP (port 3001) + REST API (port 8000)

Phase 2 architecture supports Phase 3 with minimal changes (shared search engine).
