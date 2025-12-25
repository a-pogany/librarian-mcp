"""Configuration management"""

import json
import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.json") -> Dict:
    """
    Load configuration from config.json and override with environment variables

    Args:
        config_path: Path to config.json file

    Returns:
        Configuration dictionary
    """
    # Load config.json
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = _get_default_config()
    else:
        with open(config_file) as f:
            config = json.load(f)

    # Override with environment variables
    # Note: docs.root_path is configured in config.json only (no env override)
    # This avoids confusion from dual configuration sources

    if os.getenv('MCP_HOST'):
        config['mcp']['host'] = os.getenv('MCP_HOST')

    if os.getenv('MCP_PORT'):
        config['mcp']['port'] = int(os.getenv('MCP_PORT'))

    if os.getenv('LOG_LEVEL'):
        config['logging']['level'] = os.getenv('LOG_LEVEL')

    # RAG/Semantic search configuration
    if os.getenv('SEARCH_MODE'):
        if 'search' not in config:
            config['search'] = {}
        config['search']['mode'] = os.getenv('SEARCH_MODE')

    if os.getenv('ENABLE_EMBEDDINGS'):
        if 'embeddings' not in config:
            config['embeddings'] = {}
        config['embeddings']['enabled'] = os.getenv('ENABLE_EMBEDDINGS').lower() == 'true'

    return config


def _get_default_config() -> Dict:
    """Get default configuration"""
    return {
        "system": {
            "name": "Documentation Search MCP",
            "version": "1.0.0"
        },
        "docs": {
            "root_path": "./docs",
            "file_extensions": [".md", ".txt", ".docx"],
            "max_file_size_mb": 10,
            "watch_for_changes": True,
            "index_on_startup": True
        },
        "search": {
            "max_results": 50,
            "snippet_length": 200,
            "context_lines": 3,
            "min_keyword_length": 2,
            "mode": "hybrid"
        },
        "embeddings": {
            "enabled": True,
            "model": "all-MiniLM-L6-v2",
            "persist_directory": None,
            "semantic_weight": 0.5
        },
        "mcp": {
            "transport": "http-sse",
            "host": "127.0.0.1",
            "port": 3001,
            "endpoint": "/mcp"
        },
        "logging": {
            "level": "info",
            "file": "mcp_server.log"
        }
    }


def setup_logging(config: Dict):
    """Setup logging configuration"""
    log_level = config['logging']['level'].upper()
    log_file = config['logging'].get('file', 'mcp_server.log')

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Silence noisy access/request logs from polling
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("mcp.server").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
