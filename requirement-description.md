# Generic Documentation Search System - Librarian - Complete 3-Phase Requirements

## System Overview

**Product name: Librarian**

A documentation search system that makes technical documentation accessible to LLMs and humans through:
- **Phase 1:** HTTP/SSE MCP server with keyword search
- **Phase 2:** RAG-enhanced semantic search
- **Phase 3:** Web UI for human access

**Core Value:** Enable LLMs (Claude, GPT, local models) to autonomously retrieve documentation during architecture/design discussions.

---

# PHASE 1: MCP Server + Basic Search

**Goal:** Working HTTP/SSE MCP server that Claude Desktop/Cline can query for documentation

**Time Estimate:** 8-10 hours (Weekend 1)

## Architecture
```
Documentation Files
    ↓
File Indexer → In-Memory Index
    ↓
Search Engine (Keyword-based)
    ↓
MCP Server (HTTP/SSE)
    ↓
Claude Desktop / Cline (HTTP client)
```

## Project Structure
```
doc-search-mcp/
├── backend/
│   ├── main.py                 # HTTP/SSE MCP server entry
│   ├── core/
│   │   ├── __init__.py
│   │   ├── indexer.py          # File indexing
│   │   ├── search.py           # Keyword search
│   │   └── parsers.py          # MD/DOCX/TXT parsing
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py           # MCP HTTP/SSE server
│   │   └── tools.py            # MCP tool definitions
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Configuration
│   ├── tests/
│   │   ├── test_indexer.py
│   │   ├── test_search.py
│   │   ├── test_parsers.py
│   │   └── test_mcp_tools.py
│   ├── requirements.txt
│   └── README.md
├── docs/                       # Documentation repository
│   ├── product-a/
│   └── product-b/
├── scripts/                    # System management scripts
│   ├── start.sh                # Start the system (frontend, backend, MCP)
│   ├── stop.sh                 # Stop the system
│   ├── restart.sh              # Restart the system
│   └── status.sh               # Check system health status (frontend, backend, MCP)
├── config.json
└── README.md
```

## Documentation Folder Structure
```
/docs/
├── {product-name}/           # e.g., symphony, project-x
│   ├── {component}/          # e.g., PAM, auth, database
│   │   ├── *.md
│   │   ├── *.docx
│   │   └── *.txt
│   └── architecture/
├── meetings/
│   └── {product-name}/
└── shared/                   # Cross-product docs
```

## Configuration

### config.json
```json
{
  "system": {
    "name": "Documentation Search MCP",
    "version": "1.0.0"
  },
  "docs": {
    "root_path": "/path/to/docs",
    "file_extensions": [".md", ".txt", ".docx"],
    "max_file_size_mb": 10,
    "watch_for_changes": true,
    "index_on_startup": true
  },
  "search": {
    "max_results": 50,
    "snippet_length": 200,
    "context_lines": 3,
    "min_keyword_length": 2
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

### .env
```bash
DOCS_ROOT_PATH=/path/to/docs
MCP_HOST=127.0.0.1
MCP_PORT=3001
LOG_LEVEL=info
```

## Technical Stack

**Core:**
- Python 3.10+
- FastAPI (HTTP server)
- MCP Python SDK (HTTP/SSE transport)

**File Processing:**
- python-docx (DOCX parsing)
- python-markdown (Markdown parsing)
- chardet (encoding detection)
- watchdog (file watching)

**Testing:**
- pytest
- pytest-asyncio
- httpx (API testing)
- pytest-cov (coverage)

**Utilities:**
- pydantic (validation)
- python-dotenv (env vars)
- uvicorn (ASGI server)

### requirements.txt
```txt
# Core
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

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.2
```

## MCP Server Implementation

### HTTP/SSE Transport
```python
# backend/mcp/server.py

from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI
import uvicorn

# Create FastAPI app
app = FastAPI(title="Documentation Search MCP")

# Create MCP server with HTTP/SSE transport
mcp = FastMCP("doc-search-mcp")

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "doc-search-mcp"}

# Mount MCP endpoints
app.mount("/mcp", mcp.get_asgi_app())

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=3001,
        log_level="info"
    )
```

### MCP Tools
```python
# backend/mcp/tools.py

from mcp.server.fastmcp import FastMCP
from core.search import SearchEngine
from core.indexer import FileIndexer
from typing import Optional, List
import os

# Initialize components
config = load_config()
indexer = FileIndexer(config['docs']['root_path'])
search_engine = SearchEngine(indexer)

# Initialize index on startup
indexer.build_index()

mcp = FastMCP("doc-search-mcp")

@mcp.tool()
def search_documentation(
    query: str,
    product: Optional[str] = None,
    component: Optional[str] = None,
    file_types: Optional[List[str]] = None,
    max_results: int = 10
) -> dict:
    """
    Search across all documentation using keyword matching.
    
    Args:
        query: Search keywords (space-separated)
        product: Filter by product name (e.g., "symphony")
        component: Filter by component (e.g., "PAM")
        file_types: Filter by file extensions (e.g., [".md", ".docx"])
        max_results: Maximum number of results (default: 10, max: 50)
    
    Returns:
        Dictionary with search results including file paths, snippets, and relevance scores
    
    Example:
        search_documentation(
            query="authentication OAuth",
            product="symphony",
            component="PAM",
            max_results=5
        )
    """
    results = search_engine.search(
        query=query,
        product=product,
        component=component,
        file_types=file_types,
        max_results=min(max_results, 50)
    )
    
    return {
        "results": results,
        "total": len(results),
        "query": query,
        "filters": {
            "product": product,
            "component": component,
            "file_types": file_types
        }
    }

@mcp.tool()
def get_document(
    path: str,
    section: Optional[str] = None
) -> dict:
    """
    Retrieve full content of a specific document.
    
    Args:
        path: Relative path from docs root (e.g., "symphony/PAM/api-spec.md")
        section: Optional section heading to extract (e.g., "Authentication")
    
    Returns:
        Document content, metadata, and structure
    
    Example:
        get_document(
            path="symphony/PAM/api-spec.md",
            section="Endpoints"
        )
    """
    doc = search_engine.get_document(path, section)
    
    if not doc:
        return {
            "error": "Document not found",
            "path": path
        }
    
    return doc

@mcp.tool()
def list_products() -> dict:
    """
    List all available products in the documentation.
    
    Returns:
        List of products with component counts and metadata
    
    Example:
        list_products()
    """
    products = indexer.get_products()
    
    return {
        "products": products,
        "total": len(products)
    }

@mcp.tool()
def list_components(product: str) -> dict:
    """
    List all components for a specific product.
    
    Args:
        product: Product name (e.g., "symphony")
    
    Returns:
        List of components with document counts
    
    Example:
        list_components(product="symphony")
    """
    components = indexer.get_components(product)
    
    if not components:
        return {
            "error": f"Product '{product}' not found",
            "product": product
        }
    
    return {
        "product": product,
        "components": components,
        "total": len(components)
    }

@mcp.tool()
def get_index_status() -> dict:
    """
    Get current indexing status and statistics.
    
    Returns:
        Index status, file counts, and last update time
    
    Example:
        get_index_status()
    """
    status = indexer.get_status()
    
    return status
```

## File Indexer
```python
# backend/core/indexer.py

import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from .parsers import MarkdownParser, DOCXParser, TextParser

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
            
            del self.documents[path]

class FileWatcher(FileSystemEventHandler):
    """Watch for file changes and update index"""
    
    def __init__(self, indexer):
        self.indexer = indexer
    
    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"File created: {event.src_path}")
            self.indexer.index_file(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory:
            logger.info(f"File modified: {event.src_path}")
            self.indexer.index_file(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"File deleted: {event.src_path}")
            rel_path = self.indexer.get_relative_path(event.src_path)
            self.indexer.index.remove_document(rel_path)

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
        self.observer = None
    
    def build_index(self) -> Dict:
        """Build complete index of all documents"""
        start_time = datetime.now()
        logger.info(f"Building index from: {self.docs_root}")
        
        file_count = 0
        error_count = 0
        
        # Scan all files
        for file_path in self._scan_files():
            try:
                self.index_file(file_path)
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
            'duration_seconds': duration
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
            'content': parsed['content'],
            'headings': parsed.get('headings', []),
            'metadata': parsed.get('metadata', {}),
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
        
        if path.stat().st_size > max_size:
            logger.warning(f"File too large: {path} ({path.stat().st_size} bytes)")
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
```

## Search Engine
```python
# backend/core/search.py

import re
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SearchEngine:
    """Keyword-based document search"""
    
    def __init__(self, indexer):
        self.indexer = indexer
    
    def search(
        self,
        query: str,
        product: Optional[str] = None,
        component: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search documents using keyword matching
        
        Args:
            query: Search keywords
            product: Filter by product
            component: Filter by component
            file_types: Filter by file extensions
            max_results: Maximum results to return
        
        Returns:
            List of matching documents with snippets
        """
        # Parse query into keywords
        keywords = self._parse_query(query)
        
        if not keywords:
            return []
        
        # Search across all documents
        results = []
        
        for path, doc in self.indexer.index.documents.items():
            # Apply filters
            if product and doc['product'] != product:
                continue
            
            if component and doc['component'] != component:
                continue
            
            if file_types and doc['file_type'] not in file_types:
                continue
            
            # Calculate relevance score
            score = self._calculate_score(doc, keywords, query)
            
            if score > 0:
                # Extract snippet
                snippet = self._extract_snippet(doc['content'], keywords)
                
                results.append({
                    'id': path,
                    'file_path': path,
                    'product': doc['product'],
                    'component': doc['component'],
                    'file_name': doc['file_name'],
                    'file_type': doc['file_type'],
                    'snippet': snippet,
                    'relevance_score': score,
                    'last_modified': doc['last_modified']
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Limit results
        return results[:max_results]
    
    def get_document(
        self,
        path: str,
        section: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get full document content
        
        Args:
            path: Relative path to document
            section: Optional section heading to extract
        
        Returns:
            Document content and metadata
        """
        doc = self.indexer.index.documents.get(path)
        
        if not doc:
            return None
        
        content = doc['content']
        
        # Extract specific section if requested
        if section and doc.get('headings'):
            content = self._extract_section(doc['content'], section, doc['headings'])
        
        return {
            'file_path': path,
            'product': doc['product'],
            'component': doc['component'],
            'file_name': doc['file_name'],
            'file_type': doc['file_type'],
            'content': content,
            'headings': doc.get('headings', []),
            'metadata': doc.get('metadata', {}),
            'size_bytes': doc['size_bytes'],
            'last_modified': doc['last_modified']
        }
    
    def _parse_query(self, query: str) -> List[str]:
        """Parse query into keywords"""
        # Remove special characters
        cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
        
        # Split into words
        words = cleaned.split()
        
        # Filter short words
        keywords = [w for w in words if len(w) >= 2]
        
        return keywords
    
    def _calculate_score(self, doc: Dict, keywords: List[str], original_query: str) -> float:
        """
        Calculate relevance score for document
        
        Scoring:
        - File name match: 3 points per keyword
        - Heading match: 2 points per keyword
        - Content match: 1 point per keyword
        - Phrase match bonus: 5 points
        """
        score = 0.0
        content_lower = doc['content'].lower()
        file_name_lower = doc['file_name'].lower()
        
        # Check for phrase match
        if original_query.lower() in content_lower:
            score += 5.0
        
        # Score keywords
        for keyword in keywords:
            # File name matches
            if keyword in file_name_lower:
                score += 3.0
            
            # Heading matches
            for heading in doc.get('headings', []):
                if keyword in heading.lower():
                    score += 2.0
            
            # Content matches (count occurrences, but cap contribution)
            count = content_lower.count(keyword)
            score += min(count, 5) * 1.0
        
        # Normalize score (0-1 range)
        max_possible = len(keywords) * 10
        normalized = min(score / max_possible, 1.0) if max_possible > 0 else 0.0
        
        return round(normalized, 2)
    
    def _extract_snippet(self, content: str, keywords: List[str], max_length: int = 200) -> str:
        """Extract relevant snippet containing keywords"""
        lines = content.split('\n')
        
        # Find line with most keyword matches
        best_line = ""
        max_matches = 0
        
        for line in lines:
            line_lower = line.lower()
            matches = sum(1 for kw in keywords if kw in line_lower)
            
            if matches > max_matches:
                max_matches = matches
                best_line = line
        
        # If no matches, return first non-empty line
        if not best_line:
            for line in lines:
                if line.strip():
                    best_line = line
                    break
        
        # Truncate if too long
        snippet = best_line.strip()
        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "..."
        
        return snippet
    
    def _extract_section(self, content: str, section: str, headings: List[str]) -> str:
        """Extract specific section from document"""
        lines = content.split('\n')
        section_lower = section.lower()
        
        # Find section heading
        in_section = False
        section_lines = []
        
        for line in lines:
            # Check if this is the target section
            if line.strip().lower().startswith('#') and section_lower in line.lower():
                in_section = True
                section_lines.append(line)
                continue
            
            # Check if we've reached next section
            if in_section and line.strip().startswith('#'):
                break
            
            if in_section:
                section_lines.append(line)
        
        return '\n'.join(section_lines) if section_lines else content
```

## File Parsers
```python
# backend/core/parsers.py

from abc import ABC, abstractmethod
from typing import Dict, List
from pathlib import Path
import re
import chardet
from docx import Document

class Parser(ABC):
    """Abstract parser interface"""
    
    @abstractmethod
    def parse(self, file_path: str) -> Dict:
        """Parse file and return structured content"""
        pass

class MarkdownParser(Parser):
    """Parse Markdown files"""
    
    def parse(self, file_path: str) -> Dict:
        """Parse Markdown file"""
        with open(file_path, 'rb') as f:
            raw = f.read()
            encoding = chardet.detect(raw)['encoding'] or 'utf-8'
        
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        # Extract headings
        headings = self._extract_headings(content)
        
        return {
            'content': content,
            'headings': headings,
            'metadata': {
                'encoding': encoding
            }
        }
    
    def _extract_headings(self, content: str) -> List[str]:
        """Extract markdown headings"""
        headings = []
        for line in content.split('\n'):
            if line.strip().startswith('#'):
                # Remove # symbols and clean
                heading = re.sub(r'^#+\s*', '', line).strip()
                headings.append(heading)
        return headings

class TextParser(Parser):
    """Parse plain text files"""
    
    def parse(self, file_path: str) -> Dict:
        """Parse text file"""
        with open(file_path, 'rb') as f:
            raw = f.read()
            encoding = chardet.detect(raw)['encoding'] or 'utf-8'
        
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        return {
            'content': content,
            'headings': [],
            'metadata': {
                'encoding': encoding
            }
        }

class DOCXParser(Parser):
    """Parse DOCX files"""
    
    def parse(self, file_path: str) -> Dict:
        """Parse DOCX file"""
        doc = Document(file_path)
        
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
        
        return {
            'content': '\n'.join(content_parts),
            'headings': headings,
            'metadata': metadata
        }
    
    def _format_table(self, table) -> str:
        """Format table as text"""
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(' | '.join(cells))
        return '\n'.join(rows)
    
    def _extract_metadata(self, doc: Document) -> Dict:
        """Extract document metadata"""
        try:
            core_props = doc.core_properties
            return {
                'title': core_props.title or '',
                'author': core_props.author or '',
                'created': core_props.created.isoformat() if core_props.created else None,
                'modified': core_props.modified.isoformat() if core_props.modified else None
            }
        except:
            return {}
```

## Testing

### Test: Indexer
```python
# backend/tests/test_indexer.py

import pytest
from pathlib import Path
from core.indexer import FileIndexer, DocumentIndex
import tempfile
import shutil

@pytest.fixture
def temp_docs(tmp_path):
    """Create temporary documentation structure"""
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    
    # Create product/component structure
    (docs_root / "product-a" / "component-1").mkdir(parents=True)
    (docs_root / "product-a" / "component-2").mkdir(parents=True)
    (docs_root / "product-b" / "api").mkdir(parents=True)
    
    # Create sample files
    (docs_root / "product-a" / "component-1" / "readme.md").write_text("# Component 1\nThis is a test.")
    (docs_root / "product-a" / "component-2" / "spec.md").write_text("# Spec\nAPI specification.")
    (docs_root / "product-b" / "api" / "endpoints.md").write_text("# Endpoints\nGET /api/v1")
    
    return docs_root

def test_indexer_initialization(temp_docs):
    """Test indexer initializes correctly"""
    indexer = FileIndexer(str(temp_docs))
    assert indexer.docs_root == temp_docs
    assert isinstance(indexer.index, DocumentIndex)

def test_build_index(temp_docs):
    """Test building complete index"""
    indexer = FileIndexer(str(temp_docs), {'watch_for_changes': False})
    result = indexer.build_index()
    
    assert result['status'] == 'complete'
    assert result['files_indexed'] == 3
    assert len(indexer.index.documents) == 3

def test_index_products(temp_docs):
    """Test product indexing"""
    indexer = FileIndexer(str(temp_docs), {'watch_for_changes': False})
    indexer.build_index()
    
    products = indexer.get_products()
    assert len(products) == 2
    
    product_names = [p['name'] for p in products]
    assert 'product-a' in product_names
    assert 'product-b' in product_names

def test_index_components(temp_docs):
    """Test component indexing"""
    indexer = FileIndexer(str(temp_docs), {'watch_for_changes': False})
    indexer.build_index()
    
    components = indexer.get_components('product-a')
    assert len(components) == 2
    
    component_names = [c['name'] for c in components]
    assert 'component-1' in component_names
    assert 'component-2' in component_names

def test_get_status(temp_docs):
    """Test index status"""
    indexer = FileIndexer(str(temp_docs), {'watch_for_changes': False})
    indexer.build_index()
    
    status = indexer.get_status()
    assert status['status'] == 'ready'
    assert status['total_documents'] == 3
    assert status['products'] == 2
```

### Test: Search
```python
# backend/tests/test_search.py

import pytest
from core.search import SearchEngine
from core.indexer import FileIndexer

@pytest.fixture
def search_engine(temp_docs):
    """Create search engine with indexed docs"""
    indexer = FileIndexer(str(temp_docs), {'watch_for_changes': False})
    indexer.build_index()
    return SearchEngine(indexer)

def test_basic_search(search_engine):
    """Test basic keyword search"""
    results = search_engine.search("API")
    
    assert len(results) > 0
    assert any('api' in r['file_path'].lower() for r in results)

def test_search_with_product_filter(search_engine):
    """Test search with product filter"""
    results = search_engine.search("test", product="product-a")
    
    assert all(r['product'] == 'product-a' for r in results)

def test_search_relevance_scoring(search_engine):
    """Test relevance scoring"""
    results = search_engine.search("API specification")
    
    # Results should be sorted by relevance
    if len(results) > 1:
        scores = [r['relevance_score'] for r in results]
        assert scores == sorted(scores, reverse=True)

def test_get_document(search_engine):
    """Test document retrieval"""
    # Build path (depends on temp_docs structure)
    doc = search_engine.get_document("product-a/component-1/readme.md")
    
    assert doc is not None
    assert 'content' in doc
    assert 'Component 1' in doc['content']

def test_get_nonexistent_document(search_engine):
    """Test retrieving non-existent document"""
    doc = search_engine.get_document("nonexistent/path.md")
    assert doc is None

def test_empty_query(search_engine):
    """Test empty query returns no results"""
    results = search_engine.search("")
    assert len(results) == 0

def test_max_results_limit(search_engine):
    """Test max results limit"""
    results = search_engine.search("test", max_results=1)
    assert len(results) <= 1
```

### Test: Parsers
```python
# backend/tests/test_parsers.py

import pytest
from pathlib import Path
from core.parsers import MarkdownParser, TextParser, DOCXParser
from docx import Document

def test_markdown_parser():
    """Test Markdown parsing"""
    # Create temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Heading 1\n\nContent here.\n\n## Heading 2\n")
        temp_path = f.name
    
    try:
        parser = MarkdownParser()
        result = parser.parse(temp_path)
        
        assert 'content' in result
        assert 'headings' in result
        assert len(result['headings']) == 2
        assert 'Heading 1' in result['headings']
        assert 'Heading 2' in result['headings']
    finally:
        Path(temp_path).unlink()

def test_text_parser():
    """Test text file parsing"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Plain text content\nMultiple lines\n")
        temp_path = f.name
    
    try:
        parser = TextParser()
        result = parser.parse(temp_path)
        
        assert 'content' in result
        assert 'Plain text content' in result['content']
    finally:
        Path(temp_path).unlink()

def test_docx_parser():
    """Test DOCX parsing"""
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.docx', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Create DOCX
        doc = Document()
        doc.add_heading('Test Heading', 1)
        doc.add_paragraph('Test paragraph')
        doc.save(temp_path)
        
        # Parse
        parser = DOCXParser()
        result = parser.parse(temp_path)
        
        assert 'content' in result
        assert 'headings' in result
        assert 'Test Heading' in result['headings']
        assert 'Test paragraph' in result['content']
    finally:
        Path(temp_path).unlink()
```

## Claude Desktop Configuration

### MCP Server Config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or equivalent:
```json
{
  "mcpServers": {
    "doc-search": {
      "url": "http://127.0.0.1:3001/mcp"
    }
  }
}
```

### Cline Configuration

Add to Cline MCP settings:
```json
{
  "mcpServers": [
    {
      "name": "doc-search",
      "url": "http://127.0.0.1:3001/mcp"
    }
  ]
}
```

## Usage Examples

### Example 1: Architecture Design

**User to Claude:**
```
Design a new authentication endpoint for the payment component
```

**Claude (automatically):**
```
[Calls MCP: search_documentation("authentication payment")]
[Calls MCP: get_document("product-x/auth/api-spec.md")]

Based on your existing authentication patterns, here's the design:

POST /api/v1/payment/auth

Following OAuth 2.0 as specified in your auth-spec.md...
```

### Example 2: Database Schema

**User to Claude:**
```
Add a transactions table to the database
```

**Claude (automatically):**
```
[Calls MCP: search_documentation("database schema tables")]
[Calls MCP: get_document("product-x/database/schema.md")]

Based on your existing schema conventions:

CREATE TABLE transactions (
    id UUID PRIMARY KEY,
    ...
)

Following the naming pattern from your existing tables...
```

## Phase 1 Acceptance Criteria

### AC1: MCP Server Runs
- ✅ Server starts on configured port
- ✅ Health endpoint responds
- ✅ MCP endpoints accessible via HTTP

### AC2: File Indexing Works
- ✅ All .md, .txt, .docx files indexed
- ✅ Products and components identified
- ✅ Index builds in under 10 seconds (for <500 files)
- ✅ File watcher detects changes

### AC3: Search Functions
- ✅ Keyword search returns relevant results
- ✅ Results ranked by relevance
- ✅ Snippets include matched keywords
- ✅ Filters (product, component) work

### AC4: MCP Tools Work
- ✅ search_documentation callable from Claude
- ✅ get_document returns full content
- ✅ list_products returns all products
- ✅ All tools handle errors gracefully

### AC5: Claude Desktop Integration
- ✅ Claude Desktop connects to MCP server
- ✅ Tools appear in Claude's available tools
- ✅ Claude can search docs automatically
- ✅ Results display correctly

### AC6: Testing
- ✅ All tests pass
- ✅ Code coverage >80%
- ✅ Integration tests successful

### AC7: Documentation
- ✅ README with setup instructions
- ✅ Configuration examples
- ✅ Usage examples
- ✅ API documentation

## Phase 1 Implementation Tasks

### Day 1: Setup & Core (3 hours)
- [ ] Create project structure
- [ ] Setup virtual environment
- [ ] Install dependencies
- [ ] Create config.json
- [ ] Implement basic FileIndexer
- [ ] Write indexer tests

### Day 2: Parsing (2 hours)
- [ ] Implement MarkdownParser
- [ ] Implement DOCXParser
- [ ] Implement TextParser
- [ ] Write parser tests
- [ ] Handle encoding issues

### Day 3: Search (2 hours)
- [ ] Implement SearchEngine
- [ ] Keyword search algorithm
- [ ] Result ranking
- [ ] Snippet extraction
- [ ] Write search tests

### Day 4: MCP Server (3 hours)
- [ ] Setup FastAPI + MCP
- [ ] Implement HTTP/SSE transport
- [ ] Define MCP tools
- [ ] Test tools locally
- [ ] Write MCP tests

### Day 5: Integration & Testing (2 hours)
- [ ] Test with Claude Desktop
- [ ] Test with Cline
- [ ] Fix bugs
- [ ] Write documentation
- [ ] Final testing

---

# PHASE 2: RAG Enhancement

**Goal:** Add semantic search using vector embeddings for better context retrieval

**Time Estimate:** 8-12 hours (Weekend 2)

## What Changes
```
Phase 1 Architecture
    +
Vector Embeddings (sentence-transformers)
    +
Vector Database (ChromaDB)
    +
Hybrid Search (keyword + semantic)
    =
Better Results for Vague Queries
```

## New Components

### Vector Embeddings
```python
# backend/core/embeddings.py

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingGenerator:
    """Generate vector embeddings for documents"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: Hugging Face model name
                - all-MiniLM-L6-v2: Fast, good quality (default)
                - all-mpnet-base-v2: Higher quality, slower
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of text strings
        
        Returns:
            Array of embeddings (shape: [len(texts), dimension])
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        return self.model.encode([text], show_progress_bar=False)[0]
```

### Vector Database
```python
# backend/core/vector_db.py

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid

class VectorDatabase:
    """ChromaDB wrapper for document embeddings"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB"""
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(
        self,
        doc_id: str,
        embedding: List[float],
        text: str,
        metadata: Dict
    ):
        """Add document to vector database"""
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector
            n_results: Number of results
            filter_dict: Metadata filters (e.g., {"product": "symphony"})
        
        Returns:
            Query results with documents and distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        return results
    
    def delete_document(self, doc_id: str):
        """Remove document from database"""
        self.collection.delete(ids=[doc_id])
    
    def clear(self):
        """Clear all documents"""
        self.client.delete_collection("documents")
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
```

### Enhanced Search Engine
```python
# backend/core/search_v2.py

from .search import SearchEngine as KeywordSearch
from .embeddings import EmbeddingGenerator
from .vector_db import VectorDatabase
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """Hybrid search combining keyword and semantic search"""
    
    def __init__(self, indexer, config: Dict = None):
        self.indexer = indexer
        self.config = config or {}
        
        # Keyword search
        self.keyword_search = KeywordSearch(indexer)
        
        # Semantic search
        self.embeddings = EmbeddingGenerator(
            model_name=config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        self.vector_db = VectorDatabase(
            persist_directory=config.get('vector_db_path', './chroma_db')
        )
    
    def index_document(self, doc: Dict):
        """Index document with embeddings"""
        # Generate embedding
        embedding = self.embeddings.encode_single(doc['content'])
        
        # Store in vector DB
        self.vector_db.add_document(
            doc_id=doc['path'],
            embedding=embedding.tolist(),
            text=doc['content'],
            metadata={
                'product': doc['product'],
                'component': doc['component'],
                'file_type': doc['file_type']
            }
        )
    
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
        Hybrid search combining keyword and semantic search
        
        Args:
            query: Search query
            product: Product filter
            component: Component filter
            file_types: File type filter
            max_results: Maximum results
            hybrid_weight: Weight for semantic search (0-1)
                0 = pure keyword, 1 = pure semantic, 0.5 = balanced
        
        Returns:
            Combined and re-ranked results
        """
        # Keyword search
        keyword_results = self.keyword_search.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=max_results * 2  # Get more for re-ranking
        )
        
        # Semantic search
        query_embedding = self.embeddings.encode_single(query)
        
        # Build filter
        filter_dict = {}
        if product:
            filter_dict['product'] = product
        if component:
            filter_dict['component'] = component
        
        vector_results = self.vector_db.search(
            query_embedding=query_embedding.tolist(),
            n_results=max_results * 2,
            filter_dict=filter_dict if filter_dict else None
        )
        
        # Combine and re-rank
        combined = self._combine_results(
            keyword_results,
            vector_results,
            hybrid_weight
        )
        
        return combined[:max_results]
    
    def _combine_results(
        self,
        keyword_results: List[Dict],
        vector_results: Dict,
        weight: float
    ) -> List[Dict]:
        """
        Combine keyword and semantic results with hybrid scoring
        
        Args:
            keyword_results: Results from keyword search
            vector_results: Results from vector search
            weight: Weight for semantic scores (0-1)
        
        Returns:
            Combined and re-ranked results
        """
        # Create score map
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
        if vector_results['ids']:
            for i, doc_id in enumerate(vector_results['ids'][0]):
                # Convert distance to similarity (1 - distance for cosine)
                similarity = 1.0 - vector_results['distances'][0][i]
                
                if doc_id in scores:
                    scores[doc_id]['semantic'] = similarity
                else:
                    # Need to fetch document info
                    doc = self.indexer.index.documents.get(doc_id)
                    if doc:
                        scores[doc_id] = {
                            'keyword': 0.0,
                            'semantic': similarity,
                            'result': {
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

## Updated Configuration
```json
{
  "search": {
    "mode": "hybrid",
    "keyword_weight": 0.5,
    "semantic_weight": 0.5,
    "max_results": 50
  },
  "embeddings": {
    "model": "all-MiniLM-L6-v2",
    "batch_size": 32
  },
  "vector_db": {
    "path": "./chroma_db",
    "collection": "documents"
  }
}
```

## New Dependencies
```txt
# Add to requirements.txt

# Embeddings
sentence-transformers==2.2.2
torch==2.1.0

# Vector database
chromadb==0.4.18
```

## Updated MCP Tools
```python
# backend/mcp/tools.py - Updated search tool

@mcp.tool()
def search_documentation(
    query: str,
    product: Optional[str] = None,
    component: Optional[str] = None,
    file_types: Optional[List[str]] = None,
    max_results: int = 10,
    search_mode: str = "hybrid"  # NEW: "keyword", "semantic", "hybrid"
) -> dict:
    """
    Search documentation with hybrid keyword + semantic search
    
    Args:
        query: Search query
        product: Filter by product
        component: Filter by component
        file_types: Filter by file extensions
        max_results: Maximum results (1-50)
        search_mode: Search mode - "hybrid" (default), "keyword", or "semantic"
    
    Returns:
        Search results with relevance scores
    """
    if search_mode == "hybrid":
        results = hybrid_search_engine.search(
            query=query,
            product=product,
            component=component,
            file_types=file_types,
            max_results=min(max_results, 50)
        )
    elif search_mode == "keyword":
        results = keyword_search_engine.search(...)
    else:  # semantic
        results = semantic_only_search(...)
    
    return {
        "results": results,
        "total": len(results),
        "mode": search_mode
    }
```

## Phase 2 Tasks

### Day 1: Embeddings (3 hours)
- [ ] Install sentence-transformers
- [ ] Implement EmbeddingGenerator
- [ ] Test embedding generation
- [ ] Benchmark embedding speed

### Day 2: Vector Database (2 hours)
- [ ] Install ChromaDB
- [ ] Implement VectorDatabase wrapper
- [ ] Test CRUD operations
- [ ] Test similarity search

### Day 3: Hybrid Search (3 hours)
- [ ] Implement HybridSearchEngine
- [ ] Score combination logic
- [ ] Re-ranking algorithm
- [ ] Write tests

### Day 4: Integration (2-3 hours)
- [ ] Update indexer to generate embeddings
- [ ] Update MCP tools
- [ ] Test with Claude Desktop
- [ ] Performance optimization

### Day 5: Testing & Documentation (2 hours)
- [ ] Compare keyword vs semantic vs hybrid
- [ ] Benchmark queries
- [ ] Update documentation
- [ ] Write migration guide

## Phase 2 Acceptance Criteria

### AC1: Embeddings Work
- ✅ Documents embedded on indexing
- ✅ Embeddings stored in vector DB
- ✅ Embedding generation <100ms per document

### AC2: Semantic Search Works
- ✅ Similar documents found by meaning
- ✅ Works for vague queries
- ✅ Better than keyword-only for concept queries

### AC3: Hybrid Search Works
- ✅ Combines keyword + semantic
- ✅ Configurable weighting
- ✅ Better results than either alone

### AC4: Performance
- ✅ Search responds in <1 second
- ✅ Index rebuild <2 minutes (500 docs)
- ✅ Memory usage <500MB

### AC5: Claude Integration
- ✅ Claude can choose search mode
- ✅ Better context for vague questions
- ✅ No regression on keyword queries

---

# PHASE 3: Web UI

**Goal:** Human-friendly web interface for documentation search

**Time Estimate:** 12-15 hours (Weekend 3-4)

## Architecture
```
React Frontend (Port 3000)
    ↓ HTTP
REST API Backend (Port 8000)
    ↓
Shared Search Engine (from Phase 1/2)
    ↓
Documentation Files
```

## Project Structure Updates
```
doc-search-system/
├── backend/
│   ├── main.py              # FastAPI app (REST API + MCP)
│   ├── api/                 # NEW: REST API
│   │   ├── routes.py
│   │   ├── models.py
│   │   └── deps.py
│   ├── mcp/                 # Existing MCP server
│   ├── core/                # Shared search engine
│   └── ...
├── frontend/                # NEW: React app
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── ...
│   └── ...
├── scripts/                 # System management scripts
│   ├── start.sh             # Start the system (frontend, backend, MCP)
│   ├── stop.sh              # Stop the system
│   ├── restart.sh           # Restart the system
│   └── status.sh            # Check system health status
└── ...
```

## Backend: REST API

### FastAPI Routes
```python
# backend/api/routes.py

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from .models import (
    SearchRequest,
    SearchResponse,
    DocumentResponse,
    ProductsResponse
)
from core.search_v2 import HybridSearchEngine

router = APIRouter(prefix="/api/v1")

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search documentation"""
    try:
        results = hybrid_search_engine.search(
            query=request.query,
            product=request.product,
            component=request.component,
            file_types=request.file_types,
            max_results=request.max_results,
            hybrid_weight=request.semantic_weight
        )
        
        return SearchResponse(
            results=results,
            total=len(results),
            query_time_ms=0  # TODO: measure
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{product}/{component}/{filename}")
async def get_document(
    product: str,
    component: str,
    filename: str
):
    """Get full document content"""
    path = f"{product}/{component}/{filename}"
    doc = hybrid_search_engine.keyword_search.get_document(path)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return doc

@router.get("/products", response_model=ProductsResponse)
async def list_products():
    """List all products"""
    products = indexer.get_products()
    return ProductsResponse(products=products)

@router.get("/products/{product}/components")
async def list_components(product: str):
    """List components for a product"""
    components = indexer.get_components(product)
    
    if not components:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return {"product": product, "components": components}

@router.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}
```

### Pydantic Models
```python
# backend/api/models.py

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    product: Optional[str] = None
    component: Optional[str] = None
    file_types: Optional[List[str]] = None
    max_results: int = Field(10, ge=1, le=50)
    semantic_weight: float = Field(0.5, ge=0.0, le=1.0)

class SearchResult(BaseModel):
    id: str
    file_path: str
    product: str
    component: str
    file_name: str
    file_type: str
    snippet: str
    relevance_score: float
    keyword_score: Optional[float] = None
    semantic_score: Optional[float] = None
    last_modified: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query_time_ms: float

class DocumentResponse(BaseModel):
    file_path: str
    product: str
    component: str
    content: str
    headings: List[str]
    metadata: dict
    last_modified: str

class Product(BaseModel):
    name: str
    doc_count: int
    components: List[str]

class ProductsResponse(BaseModel):
    products: List[Product]
```

### CORS Configuration
```python
# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import routes

app = FastAPI(title="Documentation Search System")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(routes.router)

# MCP server runs separately
```

## Frontend: React UI

### Technology Stack
- React 18 + TypeScript
- Vite (build tool)
- TanStack Query (data fetching)
- Tailwind CSS (styling)
- Zustand (state management)
- React Router (navigation)
- React Markdown (rendering)

### Key Components

#### SearchPage
```typescript
// frontend/src/pages/SearchPage.tsx

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { SearchBar } from '../components/SearchBar';
import { FilterPanel } from '../components/FilterPanel';
import { ResultsList } from '../components/ResultsList';
import { DocumentViewer } from '../components/DocumentViewer';
import { searchDocuments } from '../services/api';

export function SearchPage() {
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState({
    product: null,
    component: null,
    fileTypes: null
  });
  const [selectedDoc, setSelectedDoc] = useState(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ['search', query, filters],
    queryFn: () => searchDocuments({ query, ...filters }),
    enabled: query.length > 0
  });

  return (
    <div className="flex h-screen">
      {/* Left: Search & Results */}
      <div className="w-1/3 border-r flex flex-col">
        <div className="p-4 border-b">
          <SearchBar onSearch={setQuery} />
        </div>
        
        <FilterPanel filters={filters} onChange={setFilters} />
        
        <div className="flex-1 overflow-auto">
          {isLoading && <div>Loading...</div>}
          {error && <div>Error: {error.message}</div>}
          {data && (
            <ResultsList
              results={data.results}
              onSelect={setSelectedDoc}
              selectedId={selectedDoc?.id}
            />
          )}
        </div>
      </div>

      {/* Right: Document Viewer */}
      <div className="flex-1">
        {selectedDoc ? (
          <DocumentViewer document={selectedDoc} />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-400">
            Search to get started
          </div>
        )}
      </div>
    </div>
  );
}
```

#### SearchBar
```typescript
// frontend/src/components/SearchBar.tsx

import { useState } from 'react';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';

interface SearchBarProps {
  onSearch: (query: string) => void;
}

export function SearchBar({ onSearch }: SearchBarProps) {
  const [value, setValue] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim()) {
      onSearch(value);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Search documentation..."
          className="w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500"
        />
      </div>
    </form>
  );
}
```

#### ResultsList
```typescript
// frontend/src/components/ResultsList.tsx

import { DocumentIcon } from '@heroicons/react/24/outline';

interface Result {
  id: string;
  file_name: string;
  product: string;
  component: string;
  snippet: string;
  relevance_score: number;
}

interface ResultsListProps {
  results: Result[];
  onSelect: (result: Result) => void;
  selectedId?: string;
}

export function ResultsList({ results, onSelect, selectedId }: ResultsListProps) {
  return (
    <div className="divide-y">
      {results.map((result) => (
        <button
          key={result.id}
          onClick={() => onSelect(result)}
          className={`w-full p-4 text-left hover:bg-gray-50 ${
            selectedId === result.id ? 'bg-indigo-50' : ''
          }`}
        >
          <div className="flex items-start gap-3">
            <DocumentIcon className="h-5 w-5 text-gray-400 flex-shrink-0 mt-1" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <p className="font-medium truncate">{result.file_name}</p>
                <span className="text-xs text-gray-500">
                  {Math.round(result.relevance_score * 100)}%
                </span>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {result.product} / {result.component}
              </p>
              <p className="text-sm text-gray-600 mt-2 line-clamp-2">
                {result.snippet}
              </p>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}
```

#### DocumentViewer
```typescript
// frontend/src/components/DocumentViewer.tsx

import { useQuery } from '@tanstack/react-query';
import ReactMarkdown from 'react-markdown';
import { getDocument } from '../services/api';

interface DocumentViewerProps {
  document: {
    file_path: string;
  };
}

export function DocumentViewer({ document }: DocumentViewerProps) {
  const { data, isLoading } = useQuery({
    queryKey: ['document', document.file_path],
    queryFn: () => getDocument(document.file_path)
  });

  if (isLoading) return <div className="p-8">Loading...</div>;
  if (!data) return null;

  return (
    <div className="h-full overflow-auto">
      <div className="p-8 max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-6 pb-6 border-b">
          <h1 className="text-2xl font-bold">{data.file_name}</h1>
          <p className="text-sm text-gray-500 mt-2">
            {data.product} / {data.component}
          </p>
        </div>

        {/* Content */}
        <div className="prose prose-slate max-w-none">
          {data.file_type === '.md' ? (
            <ReactMarkdown>{data.content}</ReactMarkdown>
          ) : (
            <pre className="whitespace-pre-wrap">{data.content}</pre>
          )}
        </div>
      </div>
    </div>
  );
}
```

### API Service
```typescript
// frontend/src/services/api.ts

const API_BASE = 'http://localhost:8000/api/v1';

export async function searchDocuments(params: {
  query: string;
  product?: string;
  component?: string;
  max_results?: number;
}) {
  const response = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  
  if (!response.ok) throw new Error('Search failed');
  return response.json();
}

export async function getDocument(path: string) {
  const [product, component, filename] = path.split('/');
  const response = await fetch(
    `${API_BASE}/documents/${product}/${component}/${filename}`
  );
  
  if (!response.ok) throw new Error('Document not found');
  return response.json();
}

export async function listProducts() {
  const response = await fetch(`${API_BASE}/products`);
  if (!response.ok) throw new Error('Failed to fetch products');
  return response.json();
}
```

## Phase 3 Tasks

### Week 1: Backend API (4 hours)
- [ ] Create FastAPI routes
- [ ] Define Pydantic models
- [ ] Setup CORS
- [ ] Write API tests
- [ ] API documentation

### Week 2: Frontend Setup (2 hours)
- [ ] Create Vite + React project
- [ ] Install dependencies
- [ ] Setup Tailwind CSS
- [ ] Create component structure

### Week 3: Core UI (4 hours)
- [ ] SearchBar component
- [ ] ResultsList component
- [ ] DocumentViewer component
- [ ] FilterPanel component
- [ ] Layout and routing

### Week 4: Integration (3 hours)
- [ ] Setup TanStack Query
- [ ] API service layer
- [ ] Connect components to API
- [ ] Error handling
- [ ] Loading states

### Week 5: Polish (2-3 hours)
- [ ] Keyboard shortcuts
- [ ] Responsive design
- [ ] UI refinements
- [ ] Testing
- [ ] Documentation

## Phase 3 Acceptance Criteria

### AC1: REST API Works
- ✅ All endpoints functional
- ✅ CORS configured correctly
- ✅ Error handling
- ✅ API tests pass

### AC2: Search UI Works
- ✅ Search returns results
- ✅ Filters work correctly
- ✅ Results update in real-time

### AC3: Document Viewing Works
- ✅ Full document loads
- ✅ Markdown renders correctly
- ✅ Code syntax highlighted

### AC4: UX
- ✅ Fast search (<500ms)
- ✅ Keyboard shortcuts work
- ✅ Responsive layout
- ✅ Clear error messages

### AC5: Production Ready
- ✅ Build process works
- ✅ Environment configuration
- ✅ Deployment documentation

---

## Summary: 3-Phase Roadmap

**Phase 1 (Weekend 1): MCP + Basic Search**
- HTTP/SSE MCP server
- Keyword search
- DOCX parsing
- Claude Desktop integration
- **Value:** Claude auto-fetches docs

**Phase 2 (Weekend 2): RAG Enhancement**
- Vector embeddings
- Semantic search
- Hybrid ranking
- **Value:** Better results for vague queries

**Phase 3 (Weekend 3-4): Web UI**
- React frontend
- REST API
- Human-friendly interface
- **Value:** Non-MCP users can search

**Total Time: 28-37 hours across 3-4 weekends**
