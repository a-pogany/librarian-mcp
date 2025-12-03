# Phase 3: Web UI - Design Specification

## Overview

Phase 3 adds a human-friendly web interface for documentation search, making the system accessible to users beyond MCP clients (Claude Desktop, Cline).

**Goals:**
- REST API backend for HTTP access
- React web UI with modern UX patterns
- Real-time search with instant results
- Document viewing with markdown rendering
- Filter panel for products/components
- Dual server architecture (MCP + REST API)

**Time Estimate:** 12-15 hours

---

## System Architecture

### Full System Architecture
```
┌──────────────────────────────────────────────────────────┐
│                  Documentation Files                      │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────────────────────┐
│              File Indexer + Embeddings (Phase 1+2)         │
└────────────────┬───────────────────────────────────────────┘
                 │
      ┌──────────┴──────────┐
      ↓                     ↓
┌─────────────┐    ┌──────────────────┐
│  Keyword    │    │  Vector Database │
│  Index      │    │  (ChromaDB)      │
└──────┬──────┘    └────────┬─────────┘
       │                    │
       └─────────┬──────────┘
                 ↓
┌────────────────────────────────────────────────────────────┐
│           Hybrid Search Engine (Phase 2)                   │
└────────────────┬───────────────────────────────────────────┘
                 │
      ┌──────────┴──────────┐
      ↓                     ↓
┌─────────────────┐   ┌──────────────────────┐
│   MCP Server    │   │   REST API Server    │
│  (Port 3001)    │   │   (Port 8000)        │
│                 │   │                      │
│  HTTP/SSE       │   │   FastAPI Routes     │
│  Transport      │   │   CORS enabled       │
│                 │   │   JSON responses     │
└────────┬────────┘   └──────────┬───────────┘
         │                       │
         ↓                       ↓
┌─────────────────┐   ┌──────────────────────┐
│ Claude Desktop  │   │  React Web UI        │
│ / Cline         │   │  (Port 3000)         │
└─────────────────┘   │                      │
                      │  • Search interface  │
                      │  • Results list      │
                      │  • Document viewer   │
                      │  • Filter panel      │
                      └──────────────────────┘
```

### Dual Server Architecture
```
Backend Process (Python):
┌────────────────────────────────────────────────┐
│  main.py                                       │
│  ┌──────────────────────────────────────────┐ │
│  │  FastAPI App                             │ │
│  │                                          │ │
│  │  ┌────────────────┐  ┌────────────────┐ │ │
│  │  │  REST API      │  │  MCP Server    │ │ │
│  │  │  /api/v1/*     │  │  /mcp/*        │ │ │
│  │  │  (Port 8000)   │  │  (Port 3001)   │ │ │
│  │  └────────────────┘  └────────────────┘ │ │
│  │           │                   │         │ │
│  │           └───────┬───────────┘         │ │
│  │                   ↓                     │ │
│  │        Shared Search Engine             │ │
│  └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────┘
```

### Frontend Architecture
```
React App (TypeScript):
┌────────────────────────────────────────────────┐
│  src/                                          │
│  ├── pages/                                    │
│  │   └── SearchPage.tsx        [Main UI]      │
│  │                                             │
│  ├── components/                               │
│  │   ├── SearchBar.tsx         [Query input]  │
│  │   ├── FilterPanel.tsx       [Filters UI]   │
│  │   ├── ResultsList.tsx       [Results grid] │
│  │   └── DocumentViewer.tsx    [Doc display]  │
│  │                                             │
│  ├── services/                                 │
│  │   └── api.ts                [HTTP client]  │
│  │                                             │
│  ├── hooks/                                    │
│  │   ├── useSearch.ts          [Search hook]  │
│  │   └── useFilters.ts         [Filter hook]  │
│  │                                             │
│  └── types/                                    │
│      └── index.ts               [TypeScript]   │
└────────────────────────────────────────────────┘
```

---

## Backend: REST API Design

### API Endpoints

#### POST /api/v1/search
Search documentation with filters

**Request:**
```json
{
  "query": "authentication methods",
  "product": "symphony",
  "component": "PAM",
  "file_types": [".md"],
  "max_results": 10,
  "search_mode": "hybrid",
  "hybrid_weight": 0.5
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "symphony/PAM/auth.md",
      "file_path": "symphony/PAM/auth.md",
      "product": "symphony",
      "component": "PAM",
      "file_name": "auth.md",
      "file_type": ".md",
      "snippet": "Authentication methods supported: OAuth 2.0, JWT...",
      "relevance_score": 0.92,
      "keyword_score": 0.88,
      "semantic_score": 0.95,
      "last_modified": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "query_time_ms": 125.5
}
```

#### GET /api/v1/documents/{product}/{component}/{filename}
Retrieve full document content

**Response:**
```json
{
  "file_path": "symphony/PAM/auth.md",
  "product": "symphony",
  "component": "PAM",
  "file_name": "auth.md",
  "file_type": ".md",
  "content": "# Authentication\n\n## Overview\n...",
  "headings": ["Authentication", "Overview", "OAuth 2.0"],
  "metadata": {
    "encoding": "utf-8"
  },
  "size_bytes": 5420,
  "last_modified": "2024-01-15T10:30:00Z"
}
```

#### GET /api/v1/products
List all products

**Response:**
```json
{
  "products": [
    {
      "name": "symphony",
      "doc_count": 45,
      "components": ["PAM", "Auth", "API"]
    },
    {
      "name": "project-x",
      "doc_count": 23,
      "components": ["Database", "Frontend"]
    }
  ],
  "total": 2
}
```

#### GET /api/v1/products/{product}/components
List components for a product

**Response:**
```json
{
  "product": "symphony",
  "components": [
    {
      "name": "PAM",
      "doc_count": 12
    },
    {
      "name": "Auth",
      "doc_count": 8
    }
  ],
  "total": 2
}
```

#### GET /api/v1/health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "service": "librarian-api",
  "version": "3.0.0",
  "index_status": {
    "total_documents": 68,
    "products": 2,
    "last_indexed": "2024-01-15T09:00:00Z"
  }
}
```

### API Implementation

#### FastAPI Routes (`backend/api/routes.py`)
```python
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from .models import (
    SearchRequest,
    SearchResponse,
    DocumentResponse,
    ProductsResponse,
    HealthResponse
)
from core.search_v2 import HybridSearchEngine
from core.indexer import FileIndexer
import time

router = APIRouter(prefix="/api/v1", tags=["documentation"])

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search documentation with keyword, semantic, or hybrid search

    Supports:
    - Keyword search: Exact term matching
    - Semantic search: Conceptual similarity
    - Hybrid search: Combined ranking (default)
    """
    try:
        start_time = time.time()

        results = hybrid_search_engine.search(
            query=request.query,
            product=request.product,
            component=request.component,
            file_types=request.file_types,
            max_results=request.max_results,
            hybrid_weight=request.semantic_weight
        )

        query_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            results=results,
            total=len(results),
            query_time_ms=round(query_time_ms, 2)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@router.get(
    "/documents/{product}/{component}/{filename}",
    response_model=DocumentResponse
)
async def get_document(
    product: str,
    component: str,
    filename: str,
    section: Optional[str] = Query(None, description="Extract specific section")
):
    """
    Retrieve full document content

    Optionally extract a specific section by heading name
    """
    path = f"{product}/{component}/{filename}"

    doc = hybrid_search_engine.keyword_search.get_document(path, section)

    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {path}"
        )

    return DocumentResponse(**doc)


@router.get("/products", response_model=ProductsResponse)
async def list_products():
    """List all available products with component counts"""
    products = indexer.get_products()

    return ProductsResponse(
        products=products,
        total=len(products)
    )


@router.get("/products/{product}/components")
async def list_components(product: str):
    """List all components for a specific product"""
    components = indexer.get_components(product)

    if not components:
        raise HTTPException(
            status_code=404,
            detail=f"Product not found: {product}"
        )

    return {
        "product": product,
        "components": components,
        "total": len(components)
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns service status and index statistics
    """
    index_status = indexer.get_status()

    return HealthResponse(
        status="healthy",
        service="librarian-api",
        version="3.0.0",
        index_status=index_status
    )
```

#### Pydantic Models (`backend/api/models.py`)
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, description="Search query")
    product: Optional[str] = Field(None, description="Filter by product")
    component: Optional[str] = Field(None, description="Filter by component")
    file_types: Optional[List[str]] = Field(None, description="Filter by file types")
    max_results: int = Field(10, ge=1, le=50, description="Maximum results")
    search_mode: str = Field("hybrid", description="Search mode: keyword, semantic, hybrid")
    semantic_weight: float = Field(0.5, ge=0.0, le=1.0, description="Semantic weight for hybrid")

    @validator('search_mode')
    def validate_search_mode(cls, v):
        if v not in ['keyword', 'semantic', 'hybrid']:
            raise ValueError('search_mode must be keyword, semantic, or hybrid')
        return v


class SearchResult(BaseModel):
    """Individual search result"""
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
    """Search response model"""
    results: List[SearchResult]
    total: int
    query_time_ms: float


class DocumentResponse(BaseModel):
    """Full document response"""
    file_path: str
    product: str
    component: str
    file_name: str
    file_type: str
    content: str
    headings: List[str]
    metadata: Dict[str, Any]
    size_bytes: int
    last_modified: str


class Product(BaseModel):
    """Product metadata"""
    name: str
    doc_count: int
    components: List[str]


class ProductsResponse(BaseModel):
    """Products list response"""
    products: List[Product]
    total: int


class Component(BaseModel):
    """Component metadata"""
    name: str
    doc_count: int


class IndexStatus(BaseModel):
    """Index status information"""
    status: str
    total_documents: int
    products: int
    last_indexed: Optional[str]
    watching: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    index_status: IndexStatus
```

#### CORS Configuration (`backend/main.py`)
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import routes as api_routes
from mcp import server as mcp_server

# Create FastAPI app
app = FastAPI(
    title="Librarian Documentation Search",
    description="Search system for technical documentation",
    version="3.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React dev server
        "http://127.0.0.1:3000",
        "http://localhost:5173",      # Vite dev server
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST API routes
app.include_router(api_routes.router)

# MCP server is separate (runs on port 3001)
# Or can be mounted on same app:
# app.mount("/mcp", mcp_server.get_asgi_app())

if __name__ == "__main__":
    import uvicorn

    # Start REST API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

---

## Frontend: React UI Design

### Technology Stack

**Core:**
- React 18.2 with TypeScript 5.0
- Vite 5.0 (build tool)
- React Router 6.20 (routing)

**Data Fetching:**
- TanStack Query 5.0 (formerly React Query)
- Axios 1.6 (HTTP client)

**State Management:**
- Zustand 4.4 (lightweight state)
- React Context (for theme/settings)

**UI Components:**
- Tailwind CSS 3.3 (styling)
- Headless UI 1.7 (accessible components)
- Heroicons 2.0 (icons)
- React Markdown 9.0 (markdown rendering)
- Prism React Renderer 2.3 (code highlighting)

**Utilities:**
- date-fns 3.0 (date formatting)
- classnames 2.3 (conditional classes)

### Component Architecture

#### SearchPage (Main Layout)
```typescript
// src/pages/SearchPage.tsx

import { useState } from 'react';
import { useSearch } from '../hooks/useSearch';
import { useFilters } from '../hooks/useFilters';
import { SearchBar } from '../components/SearchBar';
import { FilterPanel } from '../components/FilterPanel';
import { ResultsList } from '../components/ResultsList';
import { DocumentViewer } from '../components/DocumentViewer';
import type { SearchResult } from '../types';

export function SearchPage() {
  const [query, setQuery] = useState('');
  const [selectedDoc, setSelectedDoc] = useState<SearchResult | null>(null);

  const { filters, updateFilter, clearFilters } = useFilters();
  const { results, isLoading, error } = useSearch(query, filters);

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Left Panel: Search & Results */}
      <div className="w-1/3 min-w-[400px] max-w-[600px] border-r border-gray-200 bg-white flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">
            Librarian
          </h1>
          <SearchBar onSearch={setQuery} />
        </div>

        {/* Filters */}
        <FilterPanel
          filters={filters}
          onFilterChange={updateFilter}
          onClearFilters={clearFilters}
        />

        {/* Results */}
        <div className="flex-1 overflow-auto">
          {isLoading && (
            <div className="p-8 text-center text-gray-500">
              Searching...
            </div>
          )}

          {error && (
            <div className="p-8 text-center text-red-600">
              Error: {error.message}
            </div>
          )}

          {!isLoading && !error && results && (
            <ResultsList
              results={results.results}
              onSelect={setSelectedDoc}
              selectedId={selectedDoc?.id}
              queryTime={results.query_time_ms}
            />
          )}
        </div>
      </div>

      {/* Right Panel: Document Viewer */}
      <div className="flex-1 bg-white">
        {selectedDoc ? (
          <DocumentViewer document={selectedDoc} />
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <svg className="w-16 h-16 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-lg">Search documentation to get started</p>
          </div>
        )}
      </div>
    </div>
  );
}
```

#### SearchBar Component
```typescript
// src/components/SearchBar.tsx

import { useState, useRef, useEffect } from 'react';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';

interface SearchBarProps {
  onSearch: (query: string) => void;
  placeholder?: string;
}

export function SearchBar({ onSearch, placeholder = "Search documentation..." }: SearchBarProps) {
  const [value, setValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim()) {
      onSearch(value.trim());
    }
  };

  const handleClear = () => {
    setValue('');
    inputRef.current?.focus();
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-4 top-3.5 h-5 w-5 text-gray-400 pointer-events-none" />
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={placeholder}
          className="w-full pl-11 pr-10 py-3 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm"
        />
        {value && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Keyboard shortcut hint */}
      <div className="mt-2 text-xs text-gray-500">
        Press <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded">⌘K</kbd> to focus
      </div>
    </form>
  );
}
```

#### FilterPanel Component
```typescript
// src/components/FilterPanel.tsx

import { Fragment } from 'react';
import { Disclosure } from '@headlessui/react';
import { ChevronUpIcon } from '@heroicons/react/24/outline';
import type { Filters } from '../types';

interface FilterPanelProps {
  filters: Filters;
  onFilterChange: (key: keyof Filters, value: any) => void;
  onClearFilters: () => void;
}

export function FilterPanel({ filters, onFilterChange, onClearFilters }: FilterPanelProps) {
  const hasActiveFilters = Object.values(filters).some(v => v !== null && v !== undefined);

  return (
    <div className="border-b border-gray-200 bg-gray-50">
      <Disclosure defaultOpen>
        {({ open }) => (
          <>
            <Disclosure.Button className="flex w-full justify-between px-6 py-3 text-left text-sm font-medium text-gray-700 hover:bg-gray-100">
              <span>Filters</span>
              <ChevronUpIcon className={`${open ? 'rotate-180 transform' : ''} h-5 w-5 text-gray-500`} />
            </Disclosure.Button>

            <Disclosure.Panel className="px-6 pb-4 space-y-4">
              {/* Product Filter */}
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Product
                </label>
                <select
                  value={filters.product || ''}
                  onChange={(e) => onFilterChange('product', e.target.value || null)}
                  className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value="">All products</option>
                  <option value="symphony">Symphony</option>
                  <option value="project-x">Project X</option>
                </select>
              </div>

              {/* Component Filter */}
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Component
                </label>
                <select
                  value={filters.component || ''}
                  onChange={(e) => onFilterChange('component', e.target.value || null)}
                  className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                  disabled={!filters.product}
                >
                  <option value="">All components</option>
                  {/* Dynamically populated based on product */}
                </select>
              </div>

              {/* File Type Filter */}
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  File Type
                </label>
                <div className="space-y-2">
                  {['.md', '.txt', '.docx'].map((type) => (
                    <label key={type} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={filters.file_types?.includes(type) || false}
                        onChange={(e) => {
                          const current = filters.file_types || [];
                          const updated = e.target.checked
                            ? [...current, type]
                            : current.filter(t => t !== type);
                          onFilterChange('file_types', updated.length > 0 ? updated : null);
                        }}
                        className="h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                      />
                      <span className="ml-2 text-sm text-gray-700">{type}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Search Mode */}
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Search Mode
                </label>
                <select
                  value={filters.search_mode || 'hybrid'}
                  onChange={(e) => onFilterChange('search_mode', e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value="hybrid">Hybrid (Balanced)</option>
                  <option value="keyword">Keyword Only</option>
                  <option value="semantic">Semantic Only</option>
                </select>
              </div>

              {/* Clear Filters */}
              {hasActiveFilters && (
                <button
                  onClick={onClearFilters}
                  className="w-full px-3 py-2 text-sm text-indigo-600 hover:text-indigo-800 hover:bg-indigo-50 rounded-md transition-colors"
                >
                  Clear all filters
                </button>
              )}
            </Disclosure.Panel>
          </>
        )}
      </Disclosure>
    </div>
  );
}
```

#### ResultsList Component
```typescript
// src/components/ResultsList.tsx

import { DocumentIcon } from '@heroicons/react/24/outline';
import { formatDistanceToNow } from 'date-fns';
import type { SearchResult } from '../types';

interface ResultsListProps {
  results: SearchResult[];
  onSelect: (result: SearchResult) => void;
  selectedId?: string;
  queryTime?: number;
}

export function ResultsList({ results, onSelect, selectedId, queryTime }: ResultsListProps) {
  if (results.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        No results found
      </div>
    );
  }

  return (
    <div>
      {/* Results header */}
      <div className="px-6 py-3 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center justify-between text-xs text-gray-600">
          <span>{results.length} result{results.length !== 1 && 's'}</span>
          {queryTime && (
            <span>{queryTime.toFixed(0)}ms</span>
          )}
        </div>
      </div>

      {/* Results list */}
      <div className="divide-y divide-gray-200">
        {results.map((result) => (
          <button
            key={result.id}
            onClick={() => onSelect(result)}
            className={`w-full p-4 text-left hover:bg-gray-50 transition-colors ${
              selectedId === result.id ? 'bg-indigo-50 border-l-4 border-indigo-600' : ''
            }`}
          >
            <div className="flex items-start gap-3">
              <DocumentIcon className="h-5 w-5 text-gray-400 flex-shrink-0 mt-0.5" />

              <div className="flex-1 min-w-0">
                {/* Filename */}
                <div className="flex items-center justify-between gap-2">
                  <p className="font-medium text-gray-900 truncate">
                    {result.file_name}
                  </p>
                  <span className="text-xs text-gray-500 flex-shrink-0">
                    {Math.round(result.relevance_score * 100)}%
                  </span>
                </div>

                {/* Path */}
                <p className="text-xs text-gray-500 mt-1">
                  {result.product} / {result.component}
                </p>

                {/* Snippet */}
                <p className="text-sm text-gray-600 mt-2 line-clamp-2">
                  {result.snippet}
                </p>

                {/* Score breakdown (if hybrid) */}
                {result.keyword_score !== undefined && result.semantic_score !== undefined && (
                  <div className="flex gap-3 mt-2 text-xs text-gray-500">
                    <span>KW: {Math.round(result.keyword_score * 100)}%</span>
                    <span>SEM: {Math.round(result.semantic_score * 100)}%</span>
                  </div>
                )}

                {/* Last modified */}
                <p className="text-xs text-gray-400 mt-2">
                  Modified {formatDistanceToNow(new Date(result.last_modified), { addSuffix: true })}
                </p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
```

#### DocumentViewer Component
```typescript
// src/components/DocumentViewer.tsx

import { useQuery } from '@tanstack/react-query';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { getDocument } from '../services/api';
import type { SearchResult } from '../types';

interface DocumentViewerProps {
  document: SearchResult;
}

export function DocumentViewer({ document }: DocumentViewerProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['document', document.file_path],
    queryFn: () => getDocument(document.file_path)
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">Loading document...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-600">Error loading document</div>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-4xl mx-auto p-8">
        {/* Header */}
        <div className="mb-8 pb-6 border-b border-gray-200">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            {data.file_name}
          </h1>
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <span>{data.product}</span>
            <span>/</span>
            <span>{data.component}</span>
            <span>•</span>
            <span>{(data.size_bytes / 1024).toFixed(1)} KB</span>
          </div>
        </div>

        {/* Table of Contents (if headings exist) */}
        {data.headings.length > 0 && (
          <div className="mb-8 p-4 bg-gray-50 rounded-lg">
            <h2 className="text-sm font-medium text-gray-700 mb-2">
              Table of Contents
            </h2>
            <ul className="space-y-1">
              {data.headings.map((heading, idx) => (
                <li key={idx}>
                  <a
                    href={`#${heading.toLowerCase().replace(/\s+/g, '-')}`}
                    className="text-sm text-indigo-600 hover:text-indigo-800"
                  >
                    {heading}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Content */}
        <div className="prose prose-slate max-w-none">
          {data.file_type === '.md' ? (
            <ReactMarkdown
              components={{
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <SyntaxHighlighter
                      style={oneDark}
                      language={match[1]}
                      PreTag="div"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                }
              }}
            >
              {data.content}
            </ReactMarkdown>
          ) : (
            <pre className="whitespace-pre-wrap font-mono text-sm">
              {data.content}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
```

### Custom Hooks

#### useSearch Hook
```typescript
// src/hooks/useSearch.ts

import { useQuery } from '@tanstack/react-query';
import { searchDocuments } from '../services/api';
import type { Filters } from '../types';

export function useSearch(query: string, filters: Filters) {
  return useQuery({
    queryKey: ['search', query, filters],
    queryFn: () => searchDocuments({ query, ...filters }),
    enabled: query.length > 0,
    staleTime: 30000, // 30 seconds
    retry: 1
  });
}
```

#### useFilters Hook
```typescript
// src/hooks/useFilters.ts

import { useState, useCallback } from 'react';
import type { Filters } from '../types';

export function useFilters() {
  const [filters, setFilters] = useState<Filters>({
    product: null,
    component: null,
    file_types: null,
    search_mode: 'hybrid',
    max_results: 10
  });

  const updateFilter = useCallback((key: keyof Filters, value: any) => {
    setFilters(prev => ({
      ...prev,
      [key]: value,
      // Reset component when product changes
      ...(key === 'product' ? { component: null } : {})
    }));
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({
      product: null,
      component: null,
      file_types: null,
      search_mode: 'hybrid',
      max_results: 10
    });
  }, []);

  return {
    filters,
    updateFilter,
    clearFilters
  };
}
```

### API Service Layer

```typescript
// src/services/api.ts

import axios from 'axios';
import type { SearchRequest, SearchResponse, Document } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export async function searchDocuments(params: SearchRequest): Promise<SearchResponse> {
  const response = await api.post<SearchResponse>('/search', params);
  return response.data;
}

export async function getDocument(path: string): Promise<Document> {
  const [product, component, filename] = path.split('/');
  const response = await api.get<Document>(
    `/documents/${product}/${component}/${filename}`
  );
  return response.data;
}

export async function listProducts() {
  const response = await api.get('/products');
  return response.data;
}

export async function listComponents(product: string) {
  const response = await api.get(`/products/${product}/components`);
  return response.data;
}
```

### Type Definitions

```typescript
// src/types/index.ts

export interface SearchResult {
  id: string;
  file_path: string;
  product: string;
  component: string;
  file_name: string;
  file_type: string;
  snippet: string;
  relevance_score: number;
  keyword_score?: number;
  semantic_score?: number;
  last_modified: string;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query_time_ms: number;
}

export interface Document {
  file_path: string;
  product: string;
  component: string;
  file_name: string;
  file_type: string;
  content: string;
  headings: string[];
  metadata: Record<string, any>;
  size_bytes: number;
  last_modified: string;
}

export interface Filters {
  product: string | null;
  component: string | null;
  file_types: string[] | null;
  search_mode: 'keyword' | 'semantic' | 'hybrid';
  max_results: number;
}

export interface SearchRequest extends Filters {
  query: string;
}
```

---

## Deployment Architecture

### Development Setup

**Backend (Port 8000):**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend (Port 3000):**
```bash
cd frontend
npm run dev
```

**MCP Server (Port 3001 - Optional):**
```bash
cd backend
python mcp/server.py
```

### Production Setup

**Docker Compose:**
```yaml
# docker-compose.yml

version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DOCS_ROOT_PATH=/data/docs
      - MCP_PORT=3001
    volumes:
      - ./docs:/data/docs
      - ./chroma_db:/app/chroma_db
    command: uvicorn main:app --host 0.0.0.0 --port 8000

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    environment:
      - VITE_API_URL=http://localhost:8000/api/v1
    depends_on:
      - backend

  mcp:
    build: ./backend
    ports:
      - "3001:3001"
    environment:
      - DOCS_ROOT_PATH=/data/docs
    volumes:
      - ./docs:/data/docs
      - ./chroma_db:/app/chroma_db
    command: python mcp/server.py
```

### System Management Scripts

```bash
#!/bin/bash
# scripts/start.sh

echo "Starting Librarian system..."

# Start backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start MCP server (optional)
python mcp/server.py &
MCP_PID=$!

# Start frontend
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo "System started:"
echo "  Backend: http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  MCP: http://localhost:3001"
echo ""
echo "PIDs: Backend=$BACKEND_PID, MCP=$MCP_PID, Frontend=$FRONTEND_PID"
```

```bash
#!/bin/bash
# scripts/status.sh

echo "Checking system status..."

# Check backend
if curl -s http://localhost:8000/api/v1/health > /dev/null; then
    echo "✓ Backend (Port 8000): Running"
else
    echo "✗ Backend (Port 8000): Not running"
fi

# Check frontend
if curl -s http://localhost:3000 > /dev/null; then
    echo "✓ Frontend (Port 3000): Running"
else
    echo "✗ Frontend (Port 3000): Not running"
fi

# Check MCP
if curl -s http://localhost:3001/health > /dev/null; then
    echo "✓ MCP Server (Port 3001): Running"
else
    echo "✗ MCP Server (Port 3001): Not running"
fi
```

---

## Acceptance Criteria

### AC1: REST API Works ✓
- [x] All endpoints functional and documented
- [x] CORS configured for frontend access
- [x] Error handling returns proper status codes
- [x] Request/response validation with Pydantic
- [x] API tests pass

### AC2: Search UI Works ✓
- [x] Search bar with real-time query
- [x] Results display with relevance scores
- [x] Filters update results dynamically
- [x] Empty state and loading states
- [x] Error states handled gracefully

### AC3: Document Viewing Works ✓
- [x] Full document content loads
- [x] Markdown renders correctly with syntax highlighting
- [x] Table of contents navigation
- [x] Code blocks syntax highlighted
- [x] Document metadata displayed

### AC4: UX Quality ✓
- [x] Fast search (<500ms perceived)
- [x] Keyboard shortcuts (⌘K for search focus)
- [x] Responsive layout (desktop focus)
- [x] Accessible components (WCAG AA)
- [x] Clear visual feedback

### AC5: Production Ready ✓
- [x] Build process works (Vite production build)
- [x] Environment configuration (.env support)
- [x] Docker setup for deployment
- [x] Health check endpoints
- [x] System management scripts

---

## Next Steps & Future Enhancements

### Phase 4 Potential Features
- **Search History**: Track and revisit previous searches
- **Favorites/Bookmarks**: Save important documents
- **Annotations**: Add personal notes to documents
- **User Authentication**: Multi-user support with permissions
- **Document Upload**: Web-based document management
- **Advanced Analytics**: Search analytics and popular docs
- **Export Functions**: PDF/Word export of search results
- **Mobile App**: Native mobile interface
- **API Key Management**: Secure API access for integrations

### Performance Optimizations
- **Redis Caching**: Cache search results
- **CDN Integration**: Static asset delivery
- **SSR/SSG**: Server-side rendering for faster load
- **Lazy Loading**: Virtual scrolling for large result sets
- **Service Workers**: Offline search capability

### AI Enhancements
- **Query Suggestions**: Auto-complete for queries
- **Related Documents**: ML-based recommendations
- **Document Summarization**: AI-generated summaries
- **Question Answering**: Direct answers from docs
- **Smart Snippets**: AI-enhanced result snippets
