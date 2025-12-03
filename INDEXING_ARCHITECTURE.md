# Indexing Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIBRARIAN MCP SERVER                          │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              INDEXING SYSTEM                            │    │
│  │                                                          │    │
│  │  ┌──────────────┐    ┌──────────────┐   ┌───────────┐ │    │
│  │  │   Startup    │    │     File     │   │  Manual   │ │    │
│  │  │   Indexer    │    │   Watcher    │   │  Rebuild  │ │    │
│  │  │              │    │              │   │           │ │    │
│  │  │ Scans docs/  │    │ Monitors     │   │ On server │ │    │
│  │  │ on start     │    │ changes      │   │ restart   │ │    │
│  │  └──────┬───────┘    └──────┬───────┘   └─────┬─────┘ │    │
│  │         │                   │                  │        │    │
│  │         └───────────────────┴──────────────────┘        │    │
│  │                            │                             │    │
│  │                            ▼                             │    │
│  │                  ┌──────────────────┐                   │    │
│  │                  │  File Processors │                   │    │
│  │                  │                  │                   │    │
│  │                  │  ┌─────────────┐│                   │    │
│  │                  │  │ Markdown    ││ Extract content   │    │
│  │                  │  │ Parser      ││ & metadata        │    │
│  │                  │  └─────────────┘│                   │    │
│  │                  │  ┌─────────────┐│                   │    │
│  │                  │  │ DOCX        ││ Extract content   │    │
│  │                  │  │ Parser      ││ & metadata        │    │
│  │                  │  └─────────────┘│                   │    │
│  │                  │  ┌─────────────┐│                   │    │
│  │                  │  │ Text        ││ Extract content   │    │
│  │                  │  │ Parser      ││ & metadata        │    │
│  │                  │  └─────────────┘│                   │    │
│  │                  └────────┬─────────┘                   │    │
│  │                           │                              │    │
│  │                           ▼                              │    │
│  │                  ┌──────────────────┐                   │    │
│  │                  │ Document Index   │                   │    │
│  │                  │ (In-Memory)      │                   │    │
│  │                  │                  │                   │    │
│  │                  │ ┌──────────────┐│                   │    │
│  │                  │ │ Products     ││ Hierarchy         │    │
│  │                  │ │ Components   ││ Organization      │    │
│  │                  │ │ Documents    ││                   │    │
│  │                  │ └──────────────┘│                   │    │
│  │                  └────────┬─────────┘                   │    │
│  └───────────────────────────┼──────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│                    ┌──────────────────┐                          │
│                    │  Search Engine   │                          │
│                    │                  │                          │
│                    │ - Keyword search │                          │
│                    │ - Relevance      │                          │
│                    │   scoring        │                          │
│                    │ - Filtering      │                          │
│                    └────────┬─────────┘                          │
│                             │                                    │
│                             ▼                                    │
│                    ┌──────────────────┐                          │
│                    │   MCP Tools      │                          │
│                    │                  │                          │
│                    │ - search_docs    │                          │
│                    │ - get_document   │                          │
│                    │ - list_products  │                          │
│                    │ - list_components│                          │
│                    │ - get_status     │                          │
│                    └──────────────────┘                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Claude Desktop  │
                    │                  │
                    │  User queries → │
                    │  ← Answers       │
                    └──────────────────┘
```

## Indexing Flow

### 1. Server Startup Flow

```
Server Start
    │
    ├─→ Load config.json
    │   └─→ index_on_startup: true
    │
    ├─→ Initialize FileIndexer(docs_root, config)
    │   │
    │   ├─→ Create DocumentIndex (in-memory)
    │   │   ├─→ documents: {}
    │   │   ├─→ products: {}
    │   │   └─→ components: {}
    │   │
    │   └─→ Create parsers
    │       ├─→ MarkdownParser
    │       ├─→ DOCXParser
    │       └─→ TextParser
    │
    ├─→ Build index (if index_on_startup: true)
    │   │
    │   ├─→ Scan docs/ directory recursively
    │   │   │
    │   │   ├─→ For each file:
    │   │   │   ├─→ Check extension (.md, .txt, .docx)
    │   │   │   ├─→ Check size (< max_file_size_mb)
    │   │   │   ├─→ Extract product from path (dir1)
    │   │   │   ├─→ Extract component from path (dir2)
    │   │   │   └─→ Parse file content
    │   │   │
    │   │   └─→ Add to DocumentIndex
    │   │
    │   └─→ Log: "Index built: N files in X.XXs"
    │
    ├─→ Start file watcher (if watch_for_changes: true)
    │   │
    │   └─→ Monitor docs/ for changes
    │       ├─→ File created → Index new file
    │       ├─→ File modified → Re-index file
    │       └─→ File deleted → Remove from index
    │
    └─→ Initialize SearchEngine(indexer)
        └─→ Ready for queries
```

### 2. File Watching Flow

```
File System Change Detected
    │
    ├─→ File Created
    │   ├─→ Check if supported extension
    │   ├─→ Parse file
    │   ├─→ Add to index
    │   └─→ Log: "Indexed new file: path"
    │
    ├─→ File Modified
    │   ├─→ Remove old version from index
    │   ├─→ Parse updated file
    │   ├─→ Add to index
    │   └─→ Log: "Re-indexed: path"
    │
    └─→ File Deleted
        ├─→ Remove from index
        ├─→ Update product/component counts
        └─→ Log: "Removed from index: path"
```

### 3. Search Query Flow

```
User Query via Claude Desktop
    │
    ├─→ MCP Tool: search_documentation(query, filters)
    │
    └─→ SearchEngine.search()
        │
        ├─→ Parse keywords from query
        │   └─→ Split on spaces, min_length: 2
        │
        ├─→ Apply filters
        │   ├─→ Product filter (if specified)
        │   ├─→ Component filter (if specified)
        │   └─→ File type filter (if specified)
        │
        ├─→ Calculate relevance scores
        │   │
        │   ├─→ For each document:
        │   │   │
        │   │   ├─→ Phrase match in content: +5 points
        │   │   ├─→ Keyword in filename: +3 points each
        │   │   ├─→ Keyword in headings: +2 points each
        │   │   └─→ Keyword in content: +1 point each (max 5)
        │   │
        │   └─→ Normalize scores to 0.0-1.0
        │
        ├─→ Sort by relevance (highest first)
        │
        ├─→ Extract snippets (context around keywords)
        │
        ├─→ Limit results (max_results: 50 default)
        │
        └─→ Return results to Claude
```

## Data Structures

### DocumentIndex Structure

```python
{
    # All indexed documents
    "documents": {
        "dge/file1.md": {
            "path": "dge/file1.md",
            "product": "dge",
            "component": "(root)",
            "file_type": ".md",
            "content": "Full document content...",
            "headings": ["Introduction", "Setup", "Usage"],
            "size": 2048,
            "indexed_at": "2025-11-30T14:30:00"
        },
        # ... more documents
    },

    # Product hierarchy
    "products": {
        "dge": {
            "name": "dge",
            "doc_count": 7,
            "components": {"(root)", "migration", "api"}
        },
        # ... more products
    },

    # Component mapping
    "components": {
        "dge/(root)": ["dge/file1.md", "dge/file2.txt"],
        "dge/migration": ["dge/migration/guide.md"],
        # ... more components
    },

    "last_indexed": "2025-11-30T14:30:00"
}
```

### Search Result Structure

```python
{
    "results": [
        {
            "file_path": "dge/DS_DGE_migration.md",
            "snippet": "...DGE migration process involves...",
            "relevance_score": 0.92,
            "product": "dge",
            "component": "(root)",
            "file_type": ".md",
            "matched_keywords": ["migration", "DGE"]
        },
        # ... more results
    ],
    "total": 5,
    "query": "DGE migration",
    "filters": {
        "product": null,
        "component": null,
        "file_types": null
    }
}
```

## Performance Characteristics

### Indexing Speed

| Collection Size | Initial Index | Real-time Update |
|----------------|---------------|------------------|
| 1-100 files    | < 1 second    | < 100ms          |
| 100-1K files   | 1-5 seconds   | < 200ms          |
| 1K-10K files   | 5-30 seconds  | < 500ms          |

### Search Performance

| Operation        | Average Time | Notes                    |
|-----------------|--------------|--------------------------|
| Keyword search  | < 200ms      | In-memory search         |
| Filtering       | < 50ms       | Hash table lookups       |
| Snippet extract | < 100ms      | Context-aware extraction |

### Memory Usage

| Collection Size | Memory (approx) |
|----------------|-----------------|
| 100 files      | ~10 MB          |
| 1K files       | ~100 MB         |
| 10K files      | ~1 GB           |

## Configuration Impact

### index_on_startup: true (Default)
- ✅ Immediate availability after server start
- ✅ All documents indexed before first query
- ⚠️  Startup time increases with collection size
- 💡 Recommended for production

### index_on_startup: false
- ⚠️  Empty index on startup
- ❌ Requires manual indexing
- 💡 Only for testing/development

### watch_for_changes: true (Default)
- ✅ Real-time updates (1-2 second latency)
- ✅ No manual re-indexing needed
- ⚠️  Minor CPU overhead for file monitoring
- 💡 Recommended for production

### watch_for_changes: false
- ❌ Changes not detected automatically
- ⚠️  Requires server restart to re-index
- 💡 Only if you never modify docs during runtime

## Current System Status

Based on your setup:

```
📊 Your Configuration
├─ Total Documents: 7 files
├─ Products: 1 (dge)
├─ Components: 1 ((root))
├─ Formats: .md (1), .txt (1), .docx (5)
├─ Auto-indexing: ENABLED ✅
└─ File watching: ENABLED ✅

📈 Expected Performance
├─ Index build time: < 1 second
├─ Search latency: < 200ms
├─ Real-time updates: < 2 seconds
└─ Memory usage: ~5 MB
```

## Adding Documents - Impact

### Scenario 1: Add 1 file
```
Action: cp new-doc.md docs/dge/
Effect: Indexed in 1-2 seconds (file watcher)
Impact: Immediate search availability
```

### Scenario 2: Add 100 files
```
Action: cp -r project-docs/* docs/new-product/
Effect: Indexed in 2-3 seconds (file watcher batch)
Impact: Search available within 3-5 seconds
```

### Scenario 3: Organize existing files
```
Action: mv docs/dge/*.docx docs/dge/migration/
Effect: Deleted + Re-indexed in 3-4 seconds
Impact: Component structure updated
```

## Best Practices

1. **Structure First**: Create product/component folders before adding files
2. **Batch Adds**: Copy multiple files at once (file watcher batches them)
3. **Monitor Logs**: Check `mcp_server.log` for indexing confirmation
4. **Verify Status**: Use `./scripts/check_index.sh` periodically
5. **Trust Auto-Index**: Don't manually rebuild unless troubleshooting
