#!/bin/bash

# Check Index Status Script

set -e

echo "🔍 Librarian MCP - Index Status Check"
echo "======================================"
echo ""

# Check if server is running
echo "1. Checking server status..."
if curl -s http://127.0.0.1:3001/health > /dev/null 2>&1; then
    echo "   ✅ Server is running"
    SERVER_INFO=$(curl -s http://127.0.0.1:3001/health)
    echo "   📊 $SERVER_INFO"
else
    echo "   ❌ Server is NOT running"
    echo "   💡 Start with: python ./backend/main.py"
    exit 1
fi

echo ""
echo "2. Checking documentation directory..."
DOCS_PATH="./docs"

if [ -d "$DOCS_PATH" ]; then
    echo "   ✅ Docs directory exists: $DOCS_PATH"

    # Count files by type
    MD_COUNT=$(find "$DOCS_PATH" -type f -name "*.md" 2>/dev/null | wc -l | xargs)
    TXT_COUNT=$(find "$DOCS_PATH" -type f -name "*.txt" 2>/dev/null | wc -l | xargs)
    DOCX_COUNT=$(find "$DOCS_PATH" -type f -name "*.docx" 2>/dev/null | wc -l | xargs)
    TOTAL=$((MD_COUNT + TXT_COUNT + DOCX_COUNT))

    echo "   📄 Markdown files: $MD_COUNT"
    echo "   📄 Text files: $TXT_COUNT"
    echo "   📄 DOCX files: $DOCX_COUNT"
    echo "   📊 Total indexable files: $TOTAL"
else
    echo "   ❌ Docs directory not found: $DOCS_PATH"
    exit 1
fi

echo ""
echo "3. Checking directory structure..."

# Show product/component hierarchy
echo "   📁 Directory structure:"
find "$DOCS_PATH" -type d -mindepth 1 -maxdepth 2 | while read -r dir; do
    REL_PATH="${dir#$DOCS_PATH/}"
    LEVEL=$(echo "$REL_PATH" | tr '/' '\n' | wc -l | xargs)

    if [ "$LEVEL" -eq 1 ]; then
        DOC_COUNT=$(find "$dir" -type f \( -name "*.md" -o -name "*.txt" -o -name "*.docx" \) 2>/dev/null | wc -l | xargs)
        echo "   └── 📦 Product: $REL_PATH ($DOC_COUNT docs)"
    elif [ "$LEVEL" -eq 2 ]; then
        DOC_COUNT=$(find "$dir" -type f \( -name "*.md" -o -name "*.txt" -o -name "*.docx" \) 2>/dev/null | wc -l | xargs)
        COMPONENT=$(basename "$dir")
        echo "       └── 🔧 Component: $COMPONENT ($DOC_COUNT docs)"
    fi
done

echo ""
echo "4. Checking configuration..."
if [ -f "config.json" ]; then
    echo "   ✅ Configuration file exists"

    # Extract key settings
    INDEX_ON_STARTUP=$(grep -o '"index_on_startup"[[:space:]]*:[[:space:]]*[^,}]*' config.json | cut -d: -f2 | tr -d ' ')
    WATCH_CHANGES=$(grep -o '"watch_for_changes"[[:space:]]*:[[:space:]]*[^,}]*' config.json | cut -d: -f2 | tr -d ' ')

    if [ "$INDEX_ON_STARTUP" = "true" ]; then
        echo "   ✅ Auto-index on startup: ENABLED"
    else
        echo "   ⚠️  Auto-index on startup: DISABLED"
    fi

    if [ "$WATCH_CHANGES" = "true" ]; then
        echo "   ✅ File watching: ENABLED"
    else
        echo "   ⚠️  File watching: DISABLED"
    fi
else
    echo "   ⚠️  Configuration file not found"
fi

echo ""
echo "5. Recent server logs (last 10 index-related lines)..."
if [ -f "mcp_server.log" ]; then
    echo "   📋 Log excerpts:"
    grep -i "index" mcp_server.log | tail -10 | while read -r line; do
        echo "      $line"
    done
else
    echo "   ⚠️  Log file not found (server may not have been started yet)"
fi

echo ""
echo "======================================"
echo "✅ Index Status Check Complete"
echo ""
echo "💡 Next steps:"
echo "   - To add documents: cp your-doc.md $DOCS_PATH/product/component/"
echo "   - To search: Ask Claude in Claude Desktop"
echo "   - To check status: Use 'get_index_status()' MCP tool"
echo ""
