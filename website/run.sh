#!/bin/bash
# Starts the website on this computer at http://localhost:7860
#   ./run.sh           local only (works without internet after the first run)
#   ./run.sh --public  also opens a temporary public link through cloudflared
set -e
cd "$(dirname "$0")"
VENV="../.venv"
PORT=${PORT:-7860}
SITE="https://tanish-237-ai-image-forensics.static.hf.space"

if [ ! -x "$VENV/bin/python" ]; then
  echo "First run: creating the Python environment (a few minutes)..."
  PY=$(command -v python3.12 || command -v python3)
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
fi
if ! "$VENV/bin/python" -c "import torch, fastapi, uvicorn, multipart, pillow_heif, transformers, c2pa" 2> /dev/null; then
  echo "Installing the required packages (a few minutes)..."
  "$VENV/bin/pip" install --quiet torch fastapi uvicorn python-multipart pillow pillow-heif transformers huggingface_hub numpy c2pa-python
fi

"$VENV/bin/uvicorn" app:app --host 127.0.0.1 --port $PORT &
SERVER=$!
trap 'kill $SERVER $WATCHER 2>/dev/null; pkill -f "cloudflared tunnel --url http://localhost:$PORT" 2>/dev/null' EXIT

echo "Loading models..."
until curl -s "http://localhost:$PORT/api/info" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "The server stopped; see the messages above."; exit 1; }
  sleep 2
done
echo
echo "Website running at http://localhost:$PORT"
[ -z "$NO_OPEN" ] && open "http://localhost:$PORT" 2>/dev/null || true

if [ "$1" == "--public" ]; then
  command -v cloudflared > /dev/null || { echo "Install cloudflared first: brew install cloudflared"; exit 1; }
  open_tunnel() {
    LOG=$(mktemp)
    cloudflared tunnel --url "http://localhost:$PORT" > "$LOG" 2>&1 &
    TUNNEL=$!
    until URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1) && [ -n "$URL" ]; do sleep 1; done
    echo "Opening the public link (this can take a minute)..."
    until curl -s -m 10 "$URL/api/info" | grep -q max_tiles; do sleep 3; done
    echo
    echo "Public link (while this window stays open): $URL"
    echo "Hosted page with live analysis:             $SITE/?api=$URL"
    echo
  }
  open_tunnel
  # Quick tunnels end when the computer sleeps or changes network; reopen and print the new link.
  (
    while kill -0 $SERVER 2>/dev/null; do
      sleep 30
      if ! curl -s -m 10 "$URL/api/info" | grep -q max_tiles && ! curl -s -m 10 "$URL/api/info" | grep -q max_tiles; then
        echo "The public link stopped working; opening a new one..."
        kill $TUNNEL 2>/dev/null || true
        open_tunnel
      fi
    done
  ) &
  WATCHER=$!
fi

echo "Press Ctrl+C to stop."
wait $SERVER
