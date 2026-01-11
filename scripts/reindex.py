#!/usr/bin/env python3
"""
Re-index script for Librarian MCP

Use this script to rebuild the index with the new email metadata fields.
This is necessary after upgrading to versions that add new metadata to ChromaDB.

Usage:
    python scripts/reindex.py [--clear-only] [--force]

Options:
    --clear-only    Only clear the vector database, don't rebuild
    --force         Skip confirmation prompt
"""

import sys
import os
import argparse

# Add backend to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
backend_path = os.path.join(project_root, 'backend')
sys.path.insert(0, backend_path)

from core.indexer import FileIndexer
from config.settings import load_config


def main():
    parser = argparse.ArgumentParser(description='Re-index Librarian MCP documents')
    parser.add_argument('--clear-only', action='store_true',
                       help='Only clear the vector database, do not rebuild')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Skip confirmation prompt')
    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = load_config()

    docs_root = config.get('docs', {}).get('root_path', './docs')
    embeddings_enabled = config.get('embeddings', {}).get('enabled', True)

    print(f"  Docs root: {docs_root}")
    print(f"  Embeddings enabled: {embeddings_enabled}")

    if not embeddings_enabled:
        print("\nWarning: Embeddings are disabled in config.")
        print("This script is primarily for rebuilding the vector database.")
        if not args.force:
            response = input("Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return 1

    # Confirm action
    if not args.force:
        if args.clear_only:
            print("\nThis will CLEAR the vector database (ChromaDB).")
            print("You will need to run this script again without --clear-only to rebuild.")
        else:
            print("\nThis will:")
            print("  1. Clear the existing vector database (ChromaDB)")
            print("  2. Re-scan all documents")
            print("  3. Rebuild embeddings with updated metadata")

        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1

    # Create indexer (this initializes the vector DB)
    print("\nInitializing indexer...")
    indexer = FileIndexer(docs_root, config)

    # Clear vector database if it exists
    if indexer.vector_db:
        print("\nClearing vector database...")
        try:
            count_before = indexer.vector_db.get_count()
            print(f"  Current document count: {count_before}")

            indexer.vector_db.clear()
            print("  Vector database cleared.")
        except Exception as e:
            print(f"  Error clearing vector database: {e}")
            return 1
    else:
        print("\nNo vector database found (embeddings may be disabled).")

    if args.clear_only:
        print("\nVector database cleared. Run without --clear-only to rebuild.")
        return 0

    # Rebuild index
    print("\nRebuilding index with force_reindex=True...")
    print("This will regenerate all embeddings with updated metadata.\n")

    try:
        result = indexer.build_index(force_reindex=True)

        print("\n" + "=" * 50)
        print("Re-indexing complete!")
        print("=" * 50)
        print(f"  Files indexed: {result.get('files_indexed', 0)}")
        print(f"  Folders indexed: {result.get('folders_indexed', 0)}")
        print(f"  Errors: {result.get('errors', 0)}")
        print(f"  Duration: {result.get('duration_seconds', 0):.2f} seconds")

        if indexer.vector_db:
            count_after = indexer.vector_db.get_count()
            print(f"  Chunks in vector DB: {count_after}")

        if result.get('errors', 0) > 0:
            print("\nWarning: Some files had errors. Check the logs for details.")
            return 1

        return 0

    except Exception as e:
        print(f"\nError during re-indexing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
