#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Installing/checking dependencies..."
pip3 install -q -r requirements.txt

echo ""
echo "Starting Subtitle Generator at http://localhost:8766"
echo "Press Ctrl+C to stop."
echo ""

open http://localhost:8766 2>/dev/null || true
python3 main.py
