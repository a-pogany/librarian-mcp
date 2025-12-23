#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_DIR="$ROOT_DIR/agent_layer"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is required. Install Node 18+ and rerun." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required. Install npm and rerun." >&2
  exit 1
fi

cd "$AGENT_DIR"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created agent_layer/.env from .env.example"
fi

npm install

echo "Agent layer dependencies installed."
