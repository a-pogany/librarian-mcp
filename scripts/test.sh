#!/bin/bash

# Run tests for Librarian MCP

set -e

echo "🧪 Running Librarian MCP tests..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Run tests
cd backend

echo ""
echo "Running unit tests..."
pytest tests/ -v

echo ""
echo "Running tests with coverage..."
pytest tests/ --cov=core --cov=mcp --cov-report=term-missing

echo ""
echo "✅ All tests passed!"
