#!/usr/bin/env bash
# Starts JARVIS: the Python brain and the web interface. Stop with Ctrl+C.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "First run: creating the Python environment (a few minutes)..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r backend/requirements.txt
fi
if [ ! -d frontend/jarvis-ui/node_modules ]; then
  echo "First run: installing the interface..."
  (cd frontend/jarvis-ui && npm install)
fi

.venv/bin/python backend/server.py &
BRAIN=$!
trap 'kill $BRAIN 2>/dev/null' EXIT INT TERM
cd frontend/jarvis-ui && npm run dev -- --open
