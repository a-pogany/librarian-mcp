#!/bin/bash
# Re-index script for Librarian MCP
#
# Rebuilds the vector database with updated email metadata fields.
# Use after upgrading to versions with new metadata schema.
#
# Usage:
#   ./scripts/reindex.sh [--clear-only] [--force]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Run the Python reindex script
python "$SCRIPT_DIR/reindex.py" "$@"
