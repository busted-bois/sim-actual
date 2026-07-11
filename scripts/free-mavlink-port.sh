#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-14550}"

if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -tiUDP:"$PORT" 2>/dev/null || true)"
elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser -n udp "$PORT" 2>/dev/null || true)"
else
    echo "Need lsof or fuser to find the UDP $PORT holder." >&2
    exit 1
fi

if [ -z "${pids// /}" ]; then
    echo "UDP $PORT is free."
    exit 0
fi

for pid in $pids; do
    if kill -9 "$pid" 2>/dev/null; then
        echo "Killed PID $pid holding UDP $PORT."
    else
        echo "Could not kill PID $pid."
    fi
done
