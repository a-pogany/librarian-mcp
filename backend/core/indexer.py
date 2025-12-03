"""File indexing system for documentation"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from .parsers import MarkdownParser, DOCXParser, TextParser, ParseResult

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
            except Exception as e:
                logger.error(f"Error removing deleted file {event.src_path}: {e}")


class FileIndexer:
    """Index documentation files"""

    def __init__(self, docs_root: str, config: Dict = None):
        self.docs_root = Path(docs_root)
        self.config = config or {}
        self.index = DocumentIndex()
        self.parsers = {
            '.md': MarkdownParser(),
            '.txt': TextParser(),
            '.docx': DOCXParser()
        }
        self.observer: Optional[Observer] = None

    def build_index(self) -> Dict:
        """Build complete index of all documents"""
        start_time = datetime.now()
        logger.info(f"Building index from: {self.docs_root}")

        file_count = 0
        error_count = 0

        # Scan all files
        for file_path in self._scan_files():
            try:
                self.index_file(str(file_path))
                file_count += 1
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
                error_count += 1

        self.index.last_indexed = datetime.now()
        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"Index built: {file_count} files, {error_count} errors, {duration:.2f}s")

        # Start file watcher if configured
        if self.config.get('watch_for_changes', True):
            self.start_watching()

        return {
            'status': 'complete',
            'files_indexed': file_count,
            'errors': error_count,
            'duration_seconds': round(duration, 2)
        }

    def index_file(self, file_path: str):
        """Index a single file"""
        path = Path(file_path)

        # Check if file should be indexed
        if not self._should_index(path):
            return

        # Get relative path
        rel_path = self.get_relative_path(file_path)

        # Extract product and component from path
        parts = Path(rel_path).parts
        if len(parts) < 2:
            logger.warning(f"Invalid path structure: {rel_path}")
            return

        product = parts[0]
        component = parts[1] if len(parts) > 1 else 'root'

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
            'last_modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        }

        # Add to index
        self.index.add_document(doc)
        logger.debug(f"Indexed: {rel_path}")

    def _scan_files(self) -> List[Path]:
        """Recursively scan for indexable files"""
        extensions = self.config.get('file_extensions', ['.md', '.txt', '.docx'])
        max_size = self.config.get('max_file_size_mb', 10) * 1024 * 1024

        files = []
        for ext in extensions:
            for file_path in self.docs_root.rglob(f'*{ext}'):
                if file_path.is_file() and file_path.stat().st_size <= max_size:
                    files.append(file_path)

        return files

    def _should_index(self, path: Path) -> bool:
        """Check if file should be indexed"""
        extensions = self.config.get('file_extensions', ['.md', '.txt', '.docx'])
        max_size = self.config.get('max_file_size_mb', 10) * 1024 * 1024

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
