#!/bin/bash

# Start the Librarian MCP server

set -e

echo "🚀 Starting Librarian MCP Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if config exists
if [ ! -f "config.json" ]; then
    echo "⚠️  config.json not found, using defaults"
fi

# Create docs directory if it doesn't exist
mkdir -p docs

# Start server
cd backend
echo "📚 Starting MCP server on http://127.0.0.1:3001"
python main.py
