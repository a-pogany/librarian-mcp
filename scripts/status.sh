#!/bin/bash

# Check status of Librarian MCP server

echo "🔍 Checking Librarian MCP Server status..."
echo ""

# Check if server is running
if curl -s http://localhost:3001/health > /dev/null 2>&1; then
    echo "✅ MCP Server (Port 3001): Running"

    # Get health info
    health=$(curl -s http://localhost:3001/health)
    echo "   Status: $(echo $health | grep -o '"status":"[^"]*"' | cut -d'"' -f4)"
    echo "   Service: $(echo $health | grep -o '"service":"[^"]*"' | cut -d'"' -f4)"
    echo "   Version: $(echo $health | grep -o '"version":"[^"]*"' | cut -d'"' -f4)"
else
    echo "❌ MCP Server (Port 3001): Not running"
    echo "   Run: ./scripts/start.sh"
fi

echo ""

# Check if docs directory exists and has files
if [ -d "docs" ]; then
    doc_count=$(find docs -type f \( -name "*.md" -o -name "*.txt" -o -name "*.docx" \) 2>/dev/null | wc -l)
    echo "📚 Documentation: $doc_count files"
else
    echo "⚠️  Documentation directory not found"
fi

echo ""
