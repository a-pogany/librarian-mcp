#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="$ROOT_DIR/scripts/.pids"

stop_process() {
  local name="$1"
  local pid_file="$2"

  if [ ! -f "$pid_file" ]; then
    echo "$name not running"
    return
  fi

  local pid
  pid=$(cat "$pid_file")
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid"
    echo "Stopped $name (pid $pid)"
  else
    echo "$name not running (stale pid $pid)"
  fi

  rm -f "$pid_file"
}

stop_process "agent-layer" "$PID_DIR/agent.pid"
stop_process "frontend" "$PID_DIR/frontend.pid"
