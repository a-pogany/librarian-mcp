#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_DIR="$ROOT_DIR/agent_layer"
FRONTEND_DIR="$ROOT_DIR/frontend/librarian-ui"
PID_DIR="$ROOT_DIR/scripts/.pids"

AGENT_PORT="${AGENT_PORT:-4010}"
FRONTEND_PORT="${FRONTEND_PORT:-4170}"

mkdir -p "$PID_DIR"

start_process() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  local cmd="$4"

  if [ -f "$pid_file" ]; then
    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "$name already running (pid $pid)"
      return
    fi
  fi

  nohup bash -c "$cmd" > "$log_file" 2>&1 &
  echo $! > "$pid_file"
  echo "Started $name (pid $(cat "$pid_file"))"
}

start_process "agent-layer" "$PID_DIR/agent.pid" "$PID_DIR/agent.log" "cd '$AGENT_DIR' && AGENT_PORT=$AGENT_PORT npm start"
start_process "frontend" "$PID_DIR/frontend.pid" "$PID_DIR/frontend.log" "python3 -m http.server $FRONTEND_PORT --directory '$FRONTEND_DIR'"

echo "Frontend: http://127.0.0.1:$FRONTEND_PORT"
echo "Agent API: http://127.0.0.1:$AGENT_PORT"
