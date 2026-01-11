"""File indexing system for documentation"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from .parsers import MarkdownParser, DOCXParser, TextParser, ParseResult
from .email_parser import EMLParser

logger = logging.getLogger(__name__)


class DocumentIndex:
    """In-memory document index"""

    def __init__(self):
        self.documents: Dict[str, Dict] = {}
        self.products: Dict[str, Dict] = {}
        self.components: Dict[str, List[str]] = {}
        self.last_indexed: Optional[datetime] = None

    def add_document(self, doc: Dict):
        """Add document to index"""
        path = doc['path']
        self.documents[path] = doc

        # Update product index
        product = doc['product']
        if product not in self.products:
            self.products[product] = {
                'name': product,
                'doc_count': 0,
                'components': set()
            }
        self.products[product]['doc_count'] += 1
        self.products[product]['components'].add(doc['component'])

        # Update component index
        component_key = f"{product}/{doc['component']}"
        if component_key not in self.components:
            self.components[component_key] = []
        self.components[component_key].append(path)

    def remove_document(self, path: str):
        """Remove document from index"""
        if path in self.documents:
            doc = self.documents[path]
            product = doc['product']

            # Update counts
            if product in self.products:
                self.products[product]['doc_count'] -= 1
                if self.products[product]['doc_count'] == 0:
                    del self.products[product]

            # Remove from component index
            component_key = f"{product}/{doc['component']}"
            if component_key in self.components:
                self.components[component_key].remove(path)
                if not self.components[component_key]:
                    del self.components[component_key]

            del self.documents[path]

    def get_document(self, path: str) -> Optional[Dict]:
        """Get document by path"""
        return self.documents.get(path)

    def clear(self):
        """Clear all documents"""
        self.documents.clear()
        self.products.clear()
        self.components.clear()
        self.last_indexed = None


class FileWatcher(FileSystemEventHandler):
    """Watch for file changes and update index"""

    def __init__(self, indexer):
        self.indexer = indexer

    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"File created: {event.src_path}")
            try:
                self.indexer.index_file(event.src_path)
            except Exception as e:
                logger.error(f"Error indexing created file {event.src_path}: {e}")

    def on_modified(self, event):
        if not event.is_directory:
            logger.info(f"File modified: {event.src_path}")
            try:
                self.indexer.index_file(event.src_path)
            except Exception as e:
                logger.error(f"Error indexing modified file {event.src_path}: {e}")

    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"File deleted: {event.src_path}")
            try:
                rel_path = self.indexer.get_relative_path(event.src_path)
                self.indexer.index.remove_document(rel_path)

                # Remove from vector database if RAG is enabled
                if self.indexer.enable_embeddings and self.indexer.vector_db:
                    try:
                        self.indexer.vector_db.delete_document(rel_path)
                    except Exception as e:
                        logger.error(f"Error removing embeddings for {rel_path}: {e}")

            except Exception as e:
                logger.error(f"Error removing deleted file {event.src_path}: {e}")


class FileIndexer:
    """Index documentation files"""

    def __init__(self, docs_root: str, config: Dict = None, enable_embeddings: bool = False):
        self.docs_root = Path(docs_root)
        self.config = config or {}
        self.index = DocumentIndex()
        self.parsers = {
            '.md': MarkdownParser(),
            '.txt': TextParser(),
            '.docx': DOCXParser(),
            '.eml': EMLParser()
        }
        self.observer: Optional[Observer] = None

        # RAG/Semantic search components (Phase 2)
        self.enable_embeddings = enable_embeddings
        self.embedding_generator = None
        self.vector_db = None
        self.chunker = None  # Hierarchical chunking

        # Folder metadata components (Phase 2.5 - Hierarchical Search)
        self.enable_folder_metadata = enable_embeddings and self.config.get('folder_metadata', {}).get('enabled', True)
        self.folder_metadata_extractor = None
        self.folder_vector_db = None

        if enable_embeddings:
            self._initialize_rag_components()
            if self.enable_folder_metadata:
                self._initialize_folder_metadata_components()

    def _initialize_rag_components(self):
        """Initialize embedding generator, vector database, and chunker for RAG"""
        try:
            from .embeddings import EmbeddingGenerator
            from .vector_db import VectorDatabase
            from .chunking import DocumentChunker

            logger.info("Initializing RAG components")

            # Get configuration
            embeddings_config = self.config.get('embeddings', {})
            chunking_config = self.config.get('chunking', {})

            # Get model name from config
            model_name = embeddings_config.get('model', 'all-MiniLM-L6-v2')

            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator(model_name=model_name)

            # Initialize vector database with optimizations
            persist_dir = embeddings_config.get('persist_directory')
            enable_compression = embeddings_config.get('enable_compression', True)
            self.vector_db = VectorDatabase(
                persist_directory=persist_dir,
                collection_name="documents_v2",
                enable_compression=enable_compression
            )

            # Initialize hierarchical chunker
            chunk_size = embeddings_config.get('chunk_size', 512)
            chunk_overlap = embeddings_config.get('chunk_overlap', 128)
            self.chunker = DocumentChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                respect_boundaries=chunking_config.get('respect_boundaries', True)
            )

            logger.info("RAG components initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import RAG dependencies: {e}")
            logger.error("Install with: pip install sentence-transformers chromadb")
            self.enable_embeddings = False
            raise
        except Exception as e:
            logger.error(f"Error initializing RAG components: {e}")
            self.enable_embeddings = False
            raise

    def _initialize_folder_metadata_components(self):
        """Initialize folder metadata extractor and vector database"""
        try:
            from .folder_metadata import FolderMetadataExtractor
            from .folder_vector_db import FolderVectorDatabase

            logger.info("Initializing folder metadata components")

            # Get configuration
            folder_config = self.config.get('folder_metadata', {})
            embeddings_config = self.config.get('embeddings', {})

            # Initialize folder metadata extractor
            self.folder_metadata_extractor = FolderMetadataExtractor(indexer=self)

            # Initialize folder vector database
            persist_dir = embeddings_config.get('persist_directory')
            self.folder_vector_db = FolderVectorDatabase(
                persist_directory=persist_dir,
                collection_name="folders_v1",
                enable_compression=True
            )

            logger.info("Folder metadata components initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import folder metadata dependencies: {e}")
            self.enable_folder_metadata = False
            raise
        except Exception as e:
            logger.error(f"Error initializing folder metadata components: {e}")
            self.enable_folder_metadata = False
            raise

    def _index_embeddings_with_chunks(self, file_path: Path, parsed_content: str, product: str, component: str, file_type: str, enhanced_metadata: Dict[str, Any] = None):
        """Generate and store embeddings for document chunks"""
        chunks = []

        # Use chunker if enabled
        if self.chunker:
            try:
                # Get relative path for doc_id
                relative_path = self.get_relative_path(str(file_path))

                # Basic metadata
                base_metadata = {
                    'product': product,
                    'component': component,
                    'file_type': file_type
                }

                # Merge with enhanced metadata
                if enhanced_metadata:
                    base_metadata.update(enhanced_metadata)

                # Use hierarchical chunking for DOCX files
                if file_type == '.docx':
                    chunks = self.chunker.chunk_docx(str(file_path))
                    logger.debug(f"Created {len(chunks)} chunks for {file_path.name}")
                else:
                    # Use new chunk_document method for all file types
                    chunk_objects = self.chunker.chunk_document(
                        doc_id=relative_path,
                        content=parsed_content,
                        metadata=base_metadata,
                        file_type=file_type
                    )

                    # Convert DocumentChunk objects to dict format
                    chunks = []
                    for chunk_obj in chunk_objects:
                        chunks.append({
                            'content': chunk_obj.content,
                            'metadata': chunk_obj.metadata
                        })

                    logger.debug(f"Created {len(chunks)} chunks for {file_path.name}")

            except Exception as e:
                logger.error(f"Chunking failed for {file_path}, falling back to full document: {e}")
                # Fall back to single chunk
                chunks = [{
                    'content': parsed_content,
                    'metadata': {'chunk_type': 'full_document', 'chunk_index': 0, 'is_chunked': False}
                }]
        else:
            # Chunking disabled, use full document
            chunks = [{
                'content': parsed_content,
                'metadata': {'chunk_type': 'full_document', 'chunk_index': 0, 'is_chunked': False}
            }]

        if not chunks:
            return

        # Generate embeddings for all chunks in batch
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_generator.encode(chunk_texts, batch_size=32)

        # Prepare data for batch insertion
        ids = []
        metadatas = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create unique chunk ID
            relative_path = self.get_relative_path(str(file_path))
            chunk_id = f"{relative_path}#chunk{i}"

            # Merge metadata
            metadata = {
                'file_path': relative_path,
                'product': product,
                'component': component,
                'file_type': file_type,
                'chunk_index': i,
                'total_chunks': len(chunks),
                **chunk['metadata']  # Add chunk-specific metadata
            }

            ids.append(chunk_id)
            metadatas.append(metadata)

        # Batch insert into vector database
        self.vector_db.add_documents_batch(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            batch_size=1000
        )

        logger.debug(f"Indexed {len(chunks)} chunks for: {file_path.name}")

    def _build_folder_metadata(self) -> int:
        """
        Build folder metadata and embeddings after documents are indexed

        Returns:
            Number of folders indexed
        """
        if not self.folder_metadata_extractor or not self.folder_vector_db:
            logger.warning("Folder metadata components not initialized")
            return 0

        logger.info("Building folder metadata and embeddings")
        start_time = datetime.now()

        # Extract metadata from indexed documents
        folder_metadata_dict = self.folder_metadata_extractor.build_folder_metadata()

        if not folder_metadata_dict:
            logger.warning("No folder metadata generated")
            return 0

        # Generate embeddings for all folders
        folder_paths = []
        folder_embeddings = []
        folder_metadatas = []

        for folder_path, metadata in folder_metadata_dict.items():
            # Get search text for embedding
            search_text = metadata.get_search_text()

            # Generate embedding
            embedding = self.embedding_generator.encode_query(search_text)

            # Prepare metadata for vector DB
            folder_meta = {
                'product': folder_path.split('/')[0] if '/' in folder_path else folder_path,
                'description': metadata.description,
                'doc_count': metadata.doc_count,
                'topics': ', '.join(metadata.topics[:5])  # Store as string for ChromaDB
            }

            folder_paths.append(folder_path)
            folder_embeddings.append(embedding)
            folder_metadatas.append(folder_meta)

        # Store in vector database
        self.folder_vector_db.add_folders_batch(
            folder_paths=folder_paths,
            embeddings=folder_embeddings,
            metadatas=folder_metadatas,
            batch_size=100
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Folder metadata built: {len(folder_paths)} folders, {duration:.2f}s")

        return len(folder_paths)

    def build_index(self, force_reindex: bool = False) -> Dict:
        """Build complete index of all documents
        
        Args:
            force_reindex: If True, rebuild vector embeddings even if they exist.
                          If False (default), reuse existing embeddings from persistent storage.
        """
        start_time = datetime.now()
        logger.info(f"Building index from: {self.docs_root}")

        # Check if vector DB has existing data (persistence check)
        existing_embeddings_count = 0
        skip_embeddings = False
        
        if self.enable_embeddings and self.vector_db:
            existing_embeddings_count = self.vector_db.get_count()
            if existing_embeddings_count > 0 and not force_reindex:
                logger.info(f"Found {existing_embeddings_count} existing embeddings in persistent storage")
                logger.info("Skipping embedding regeneration (use force_reindex=True to rebuild)")
                skip_embeddings = True

        # First, collect all files to get total count
        all_files = self._scan_files()
        total_files = len(all_files)
        logger.info(f"Found {total_files} files to index")

        file_count = 0
        error_count = 0
        last_progress_log = 0
        progress_interval = max(1, total_files // 20)  # Log every 5% or at least every file

        # Process files with progress reporting
        for file_path in all_files:
            try:
                self._index_file_internal(str(file_path), skip_embeddings=skip_embeddings)
                file_count += 1
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
                error_count += 1
                file_count += 1  # Still count as processed

            # Progress reporting
            if file_count - last_progress_log >= progress_interval or file_count == total_files:
                elapsed = (datetime.now() - start_time).total_seconds()
                percent = (file_count / total_files * 100) if total_files > 0 else 100
                rate = file_count / elapsed if elapsed > 0 else 0
                remaining_files = total_files - file_count
                eta_seconds = remaining_files / rate if rate > 0 else 0

                if eta_seconds > 60:
                    eta_str = f"{eta_seconds / 60:.1f} min"
                else:
                    eta_str = f"{eta_seconds:.0f} sec"

                logger.info(f"Progress: {file_count}/{total_files} ({percent:.1f}%) - "
                           f"{rate:.1f} files/sec - ETA: {eta_str}")
                last_progress_log = file_count

        self.index.last_indexed = datetime.now()
        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"Index built: {file_count} files, {error_count} errors, {duration:.2f}s")
        if skip_embeddings:
            logger.info(f"Reused {existing_embeddings_count} existing embeddings from persistent storage")

        # Build folder metadata and embeddings if enabled
        folder_count = 0
        if self.enable_folder_metadata and not skip_embeddings:
            try:
                folder_count = self._build_folder_metadata()
            except Exception as e:
                logger.error(f"Error building folder metadata: {e}")
        elif self.enable_folder_metadata and skip_embeddings:
            # Check for existing folder metadata
            if self.folder_vector_db:
                folder_count = self.folder_vector_db.get_count()
                if folder_count > 0:
                    logger.info(f"Reused {folder_count} existing folder embeddings")

        # Start file watcher if configured
        if self.config.get('docs', {}).get('watch_for_changes', True):
            self.start_watching()

        return {
            'status': 'complete',
            'files_indexed': file_count,
            'folders_indexed': folder_count,
            'errors': error_count,
            'duration_seconds': round(duration, 2),
            'embeddings_reused': skip_embeddings,
            'existing_embeddings': existing_embeddings_count
        }

    def _extract_frontmatter_tags(self, content: str) -> List[str]:
        """Extract tags from YAML frontmatter"""
        import re

        # Match YAML frontmatter: ---\n...\n---
        match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)

        if not match:
            return []

        try:
            import yaml
            frontmatter = yaml.safe_load(match.group(1))
            tags = frontmatter.get('tags', [])
            # Ensure tags is a list
            if isinstance(tags, str):
                # Support comma-separated tags
                return [t.strip() for t in tags.split(',')]
            elif isinstance(tags, list):
                return tags
            else:
                return []
        except Exception as e:
            logger.debug(f"Error parsing frontmatter: {e}")
            return []

    def _infer_doc_type(self, file_path: Path, content: str) -> str:
        """Infer document type from filename and content"""
        filename_lower = file_path.name.lower()
        content_lower = content.lower()

        # Check filename patterns
        if 'api' in filename_lower or 'endpoint' in filename_lower:
            return 'api'
        elif 'architecture' in filename_lower or 'design' in filename_lower:
            return 'architecture'
        elif 'guide' in filename_lower or 'tutorial' in filename_lower:
            return 'guide'
        elif 'reference' in filename_lower or 'spec' in filename_lower:
            return 'reference'
        elif 'readme' in filename_lower:
            return 'readme'

        # Check content patterns (first 500 chars)
        content_sample = content_lower[:500]
        if 'class ' in content_sample or 'function ' in content_sample:
            return 'api'
        elif 'architecture' in content_sample or 'system design' in content_sample:
            return 'architecture'
        elif 'tutorial' in content_sample or 'guide' in content_sample:
            return 'guide'

        return 'documentation'  # default

    def _index_file_internal(self, file_path: str, skip_embeddings: bool = False):
        """Index a single file with optional embedding generation control
        
        Args:
            file_path: Path to the file to index
            skip_embeddings: If True, skip generating embeddings (for persistence reuse)
        """
        path = Path(file_path)

        # Check if file should be indexed
        if not self._should_index(path):
            return

        # Get relative path
        rel_path = self.get_relative_path(file_path)

        # Extract product and component from path
        parts = Path(rel_path).parts
        
        if len(parts) == 0:
            logger.warning(f"Empty path structure: {rel_path}")
            return
        elif len(parts) == 1:
            product = self.docs_root.name
            component = 'root'
        elif len(parts) == 2:
            product = parts[0]
            component = 'root'
        else:
            product = parts[0]
            component = parts[1]

        # Parse file content
        parser = self.parsers.get(path.suffix)
        if not parser:
            logger.warning(f"No parser for {path.suffix}: {path}")
            return

        try:
            parsed = parser.parse(str(path))
        except Exception as e:
            logger.error(f"Parse error for {path}: {e}")
            return

        # Extract enhanced metadata
        tags = self._extract_frontmatter_tags(parsed.content)
        doc_type = self._infer_doc_type(path, parsed.content)

        # Create document entry
        doc = {
            'path': rel_path,
            'product': product,
            'component': component,
            'file_name': path.name,
            'file_type': path.suffix,
            'content': parsed.content,
            'headings': parsed.headings,
            'metadata': parsed.metadata,
            'size_bytes': path.stat().st_size,
            'last_modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            'indexed_at': datetime.now().isoformat(),
            'doc_type': doc_type,
            'tags': tags
        }

        # Add to in-memory keyword index (always needed for keyword search)
        self.index.add_document(doc)
        logger.debug(f"Indexed: {rel_path}")

        # Generate and store embeddings with chunking if enabled (Phase 2)
        # Skip if embeddings already exist in persistent storage
        if not skip_embeddings and self.enable_embeddings and self.embedding_generator and self.vector_db:
            try:
                enhanced_metadata = {
                    'doc_type': doc_type,
                    'tags': tags,
                    'last_modified': doc['last_modified'],
                    'indexed_at': doc['indexed_at'],
                    'file_size': doc['size_bytes']
                }

                # Add email-specific fields for .eml files (enables pre-filtering in ChromaDB)
                if path.suffix == '.eml' and parsed.metadata:
                    email_meta = parsed.metadata
                    # ChromaDB only supports str, int, float, bool - convert lists to strings
                    enhanced_metadata['email_from'] = email_meta.get('from', '') or ''
                    enhanced_metadata['email_to'] = ','.join(email_meta.get('to', []) or [])
                    enhanced_metadata['email_cc'] = ','.join(email_meta.get('cc', []) or [])
                    enhanced_metadata['email_folder'] = (email_meta.get('folder', '') or '').lower()  # lowercase for case-insensitive search
                    enhanced_metadata['email_subject'] = email_meta.get('subject', '') or ''
                    enhanced_metadata['email_date'] = email_meta.get('date', '') or ''
                    enhanced_metadata['email_thread_id'] = email_meta.get('thread_id', '') or ''
                    enhanced_metadata['email_has_attachments'] = bool(email_meta.get('has_attachments', False))
                self._index_embeddings_with_chunks(
                    path,
                    parsed.content,
                    product,
                    component,
                    path.suffix,
                    enhanced_metadata
                )
            except Exception as e:
                logger.error(f"Error generating embeddings for {rel_path}: {e}")

    def index_file(self, file_path: str):
        """Index a single file (public method for file watcher and external calls)
        
        Note: This always generates embeddings for new/updated files.
        For bulk indexing with persistence awareness, use build_index().
        """
        self._index_file_internal(file_path, skip_embeddings=False)

    def _scan_files(self) -> List[Path]:
        """Recursively scan for indexable files"""
        extensions = self.config.get('docs', {}).get('file_extensions', ['.md', '.txt', '.docx', '.eml'])
        max_size = self.config.get('docs', {}).get('max_file_size_mb', 10) * 1024 * 1024

        files = []
        for ext in extensions:
            for file_path in self.docs_root.rglob(f'*{ext}'):
                if file_path.is_file() and file_path.stat().st_size <= max_size:
                    files.append(file_path)

        return files

    def _should_index(self, path: Path) -> bool:
        """Check if file should be indexed"""
        extensions = self.config.get('docs', {}).get('file_extensions', ['.md', '.txt', '.docx', '.eml'])
        max_size = self.config.get('docs', {}).get('max_file_size_mb', 10) * 1024 * 1024

        if path.suffix not in extensions:
            return False

        if not path.is_file():
            return False

        try:
            if path.stat().st_size > max_size:
                logger.warning(f"File too large: {path} ({path.stat().st_size} bytes)")
                return False
        except Exception:
            return False

        return True

    def get_relative_path(self, file_path: str) -> str:
        """Get path relative to docs root"""
        return str(Path(file_path).relative_to(self.docs_root))

    def start_watching(self):
        """Start watching for file changes"""
        if self.observer:
            return

        self.observer = Observer()
        event_handler = FileWatcher(self)
        self.observer.schedule(event_handler, str(self.docs_root), recursive=True)
        self.observer.start()
        logger.info(f"File watcher started for: {self.docs_root}")

    def stop_watching(self):
        """Stop watching for file changes"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("File watcher stopped")

    def get_products(self) -> List[Dict]:
        """Get list of all products"""
        return [
            {
                'name': name,
                'doc_count': info['doc_count'],
                'components': sorted(list(info['components']))
            }
            for name, info in self.index.products.items()
        ]

    def get_components(self, product: str) -> Optional[List[Dict]]:
        """Get components for a product"""
        if product not in self.index.products:
            return None

        components = []
        for comp in self.index.products[product]['components']:
            key = f"{product}/{comp}"
            doc_count = len(self.index.components.get(key, []))
            components.append({
                'name': comp,
                'doc_count': doc_count
            })

        return components

    def get_status(self) -> Dict:
        """Get index status"""
        return {
            'status': 'ready',
            'total_documents': len(self.index.documents),
            'products': len(self.index.products),
            'last_indexed': self.index.last_indexed.isoformat() if self.index.last_indexed else None,
            'watching': self.observer is not None
        }
