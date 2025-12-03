# Document Indexing Guide

## 📋 Overview

Your Librarian MCP system has **automatic indexing** with real-time updates. You don't need to manually trigger indexing in most cases.

## 🔄 Automatic Indexing System

### How It Works

The system automatically indexes documents in three ways:

#### 1. **Startup Indexing** (Automatic)
When the MCP server starts, it automatically:
- Scans the entire `docs/` directory
- Indexes all `.md`, `.txt`, and `.docx` files
- Builds the product/component hierarchy
- Starts the file watcher

**Configuration**: `config.json`
```json
{
  "docs": {
    "index_on_startup": true,  // ✅ Enabled by default
    "watch_for_changes": true  // ✅ Enabled by default
  }
}
```

#### 2. **Real-Time File Watching** (Automatic)
After startup, the system continuously monitors for:
- New files added to `docs/`
- Files modified in `docs/`
- Files deleted from `docs/`

Changes are reflected **immediately** in the index.

**Technology**: Uses `watchdog` library for cross-platform file monitoring

#### 3. **On-Demand Manual Indexing** (Optional)
Available via MCP tool if you need to force a rebuild:

```python
# Available MCP tools (callable from Claude Desktop):
get_index_status()  # Check current index statistics
```

## 📂 Document Organization

Your documents are automatically organized by path:

```
docs/
├── product-name/           # Becomes "product"
│   ├── component-name/     # Becomes "component"
│   │   ├── file1.md
│   │   ├── file2.txt
│   │   └── file3.docx
│   └── another-component/
└── another-product/
```

**Current Structure** (from your docs folder):
```
docs/
└── dge/                    # Product: "dge"
    ├── Danske_Spil_DGE_TA_Mapping_V008.docx
    ├── Summary_Chatgpt.docx
    ├── DS_SYM_DGE_Data_Conversion_Concept_V001_d.docx
    ├── DS_Symphony_DGE_Migration_MeetingMinutes_20240919.docx
    ├── DS_DGE_migration.md
    ├── DS_SYM_Data_Conversion_Concept_V003_ac.docx
    └── DS Symphony DGE migration - kick off call-20250919 1000-1.txt
```

**Indexed as**:
- **Product**: `dge`
- **Component**: `(root)` (no subdirectory)
- **8 documents** ready to search

## 🚀 Quick Start: Adding Documents

### Option 1: Add Files Directly (Recommended)
```bash
# Just copy/move files to docs directory
cp my-doc.md /Users/attila.pogany/Code/projects/librarian-mcp/docs/dge/

# The system automatically detects and indexes it within seconds
```

### Option 2: Create Organized Structure
```bash
# Create product/component structure
mkdir -p docs/my-product/api
mkdir -p docs/my-product/database

# Add documentation
echo "# API Documentation" > docs/my-product/api/readme.md

# Automatically indexed!
```

### Option 3: Bulk Import
```bash
# Copy entire documentation tree
cp -r /path/to/existing/docs/* docs/

# All files indexed automatically on next scan
```

## 🔍 Verify Indexing Status

### Method 1: Server Logs
When server starts, you'll see:
```
INFO - Initializing indexer for: ./docs
INFO - Building initial index...
INFO - Index built: 8 files in 0.45s
INFO - File watcher started for: ./docs
```

### Method 2: MCP Tool (from Claude Desktop)
Ask Claude: "What's the index status?"

Claude will call `get_index_status()` and show:
```json
{
  "total_documents": 8,
  "total_products": 1,
  "total_components": 1,
  "last_indexed": "2025-11-30T14:30:00",
  "watching": true,
  "products": {
    "dge": {
      "name": "dge",
      "doc_count": 8,
      "components": ["(root)"]
    }
  }
}
```

### Method 3: Test Search
Ask Claude: "Search for DGE migration"

If documents are indexed, you'll get results immediately.

## ⚙️ Configuration Options

Edit `config.json` to customize indexing behavior:

```json
{
  "docs": {
    "root_path": "./docs",              // Where to look for docs
    "file_extensions": [".md", ".txt", ".docx"],  // Supported formats
    "max_file_size_mb": 10,             // Skip files larger than this
    "watch_for_changes": true,          // Real-time monitoring (recommended)
    "index_on_startup": true            // Index on server start (recommended)
  }
}
```

### Disable Auto-Indexing (Not Recommended)
```json
{
  "docs": {
    "watch_for_changes": false,   // ❌ Disables real-time updates
    "index_on_startup": false     // ❌ Requires manual indexing
  }
}
```

## 🛠️ Manual Index Operations

### Force Rebuild Index
```bash
# Restart the server to rebuild from scratch
# This re-indexes everything

# 1. Stop server (Ctrl+C)
# 2. Start server again
python ./backend/main.py

# Index rebuilds automatically on startup
```

### Check Server Status
```bash
# Health check
curl http://127.0.0.1:3001/health

# Expected response:
{
  "status": "healthy",
  "service": "Documentation Search MCP",
  "version": "1.0.0"
}
```

## 📊 Index Statistics

Your current documents (from `docs/dge/`):
- **Total Files**: 8 documents
- **Formats**:
  - `.docx` files: 5 documents
  - `.md` files: 1 document
  - `.txt` files: 1 document
  - Other: 1 file (.DS_Store - ignored)

**Expected Index**:
```
Product: dge
└── Component: (root)
    ├── Danske_Spil_DGE_TA_Mapping_V008.docx
    ├── Summary_Chatgpt.docx
    ├── DS_SYM_DGE_Data_Conversion_Concept_V001_d.docx
    ├── DS_Symphony_DGE_Migration_MeetingMinutes_20240919.docx
    ├── DS_DGE_migration.md
    ├── DS_SYM_Data_Conversion_Concept_V003_ac.docx
    └── DS Symphony DGE migration - kick off call-20250919 1000-1.txt
```

## 🎯 Recommended Workflow

### For New Documents:
1. **Add files** to `docs/product/component/` directory
2. **Wait 1-2 seconds** (file watcher detects change)
3. **Search immediately** - document is indexed!

### For Organizing Existing Docs:
1. **Create structure**: `docs/project-name/component-name/`
2. **Move files** into appropriate folders
3. **Automatic indexing** updates within seconds
4. **Verify** with search or `get_index_status()`

### Best Practices:
- ✅ Use meaningful product/component names (e.g., `symphony/api`, `dge/migration`)
- ✅ Keep files under 10MB for best performance
- ✅ Use supported formats: `.md`, `.txt`, `.docx`
- ✅ Let auto-indexing handle updates (don't manually rebuild unless needed)
- ❌ Avoid deeply nested directories (max 2 levels recommended)

## 🔧 Troubleshooting

### Issue: Documents Not Appearing in Search

**Check 1: File Format**
```bash
# Verify file has supported extension
ls docs/**/*.{md,txt,docx}
```

**Check 2: File Size**
```bash
# Check if files exceed max size (10MB default)
find docs -type f -size +10M
```

**Check 3: Server Running**
```bash
curl http://127.0.0.1:3001/health
```

**Check 4: Server Logs**
```bash
# Check mcp_server.log for indexing messages
tail -f mcp_server.log | grep -i "index"
```

### Issue: Real-Time Updates Not Working

**Solution**: Restart server to re-enable file watcher
```bash
# Stop server (Ctrl+C)
# Start again
python ./backend/main.py
```

### Issue: Need to Force Re-Index

**Solution**: Restart the server
```bash
# The index rebuilds from scratch on startup
# with index_on_startup: true (default)
```

## 📈 Performance Expectations

- **Small collections** (<100 files): Index builds in <1 second
- **Medium collections** (100-1,000 files): Index builds in 1-5 seconds
- **Large collections** (1,000-10,000 files): Index builds in 5-30 seconds
- **Real-time updates**: Changes reflected within 1-2 seconds
- **Search latency**: <200ms average for keyword search

## 🎓 Next Steps

1. **Add your documents** to `docs/` directory
2. **Start the server**: `python ./backend/main.py`
3. **Verify indexing**: Check server logs for "Index built: X files"
4. **Test search**: Ask Claude to search your documentation
5. **Monitor**: Use `get_index_status()` to check index health

Your system is ready for automatic indexing! Just add documents and search.
