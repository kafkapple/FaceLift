#!/bin/bash
# Auto-detect free local port and open SSH tunnel to mask_annotator on gpu03.
#
# Usage:
#   bash open_viewer_tunnel.sh          # default remote port 8770
#   bash open_viewer_tunnel.sh 8771     # custom remote port

REMOTE_PORT="${1:-8770}"
SERVER="gpu03"
START_PORT=8770
MAX_TRIES=20

# Find a free local port starting from START_PORT
for (( port=START_PORT; port<START_PORT+MAX_TRIES; port++ )); do
    if ! lsof -i :"$port" &>/dev/null; then
        LOCAL_PORT=$port
        break
    fi
done

if [ -z "$LOCAL_PORT" ]; then
    echo "❌ No free port found in range ${START_PORT}-$((START_PORT+MAX_TRIES-1))"
    exit 1
fi

# Kill any existing tunnel to the same remote port
existing=$(ps aux | grep "ssh.*-L.*:localhost:${REMOTE_PORT}.*${SERVER}" | grep -v grep | awk '{print $2}')
if [ -n "$existing" ]; then
    kill $existing 2>/dev/null
    echo "Killed old tunnel (PID $existing)"
fi

# Open tunnel
ssh -f -N -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "$SERVER"

if [ $? -eq 0 ]; then
    echo "✅ Tunnel: localhost:${LOCAL_PORT} → ${SERVER}:${REMOTE_PORT}"
    echo "🌐 Open: http://localhost:${LOCAL_PORT}"

    # Try to open in browser (macOS)
    if command -v open &>/dev/null; then
        open "http://localhost:${LOCAL_PORT}"
    fi
else
    echo "❌ SSH tunnel failed"
    exit 1
fi
