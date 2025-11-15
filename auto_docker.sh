#!/bin/bash

cd /app

# Start background processes
node src/websocket-server.js &
WS_PID=$!

npm run dev &
DEV_PID=$!

python rl/train.py &
PY_PID=$!

# Kill background processes when auto.sh exits (Ctrl+C, error, normal exit)
cleanup() {
    echo "Stopping background processes..."
    kill $WS_PID $DEV_PID $PY_PID 2>/dev/null
}
trap cleanup EXIT

# Keep script alive until interrupted
wait
