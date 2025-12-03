#!/bin/bash

# Setup script for Librarian MCP

set -e

echo "🔧 Setting up Librarian MCP..."

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.10 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r backend/requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env to configure your documentation path"
else
    echo "✅ .env already exists"
fi

# Create docs directory
mkdir -p docs
echo "✅ Documentation directory created: ./docs"

# Create sample documentation
if [ ! -f "docs/.gitkeep" ]; then
    touch docs/.gitkeep
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env to set your DOCS_ROOT_PATH"
echo "  2. Add documentation to the docs/ folder"
echo "  3. Run: ./scripts/start.sh"
echo ""
