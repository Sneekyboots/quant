#!/bin/bash

# QX Quantum Hospital — Startup Script
# Cleans up old processes and launches both dashboards in parallel.

# Exit on error
set -e

echo "🧹 [1/3] Cleaning up existing dashboard processes on 8050/8051..."
fuser -k 8050/tcp 8051/tcp 2>/dev/null || true

# Identify the local virtual environment
VENV_PYTHON="./.venv/bin/python3"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: Virtual environment not found at ./.venv"
    echo "Please ensure you have run the setup/installation steps."
    exit 1
fi

# Function to handle graceful shutdown
cleanup() {
    echo ""
    echo "🛑 [3/3] Shutting down QX dashboards..."
    fuser -k 8050/tcp 8051/tcp 2>/dev/null || true
    echo "👋 System offline."
    exit
}

# Trap Ctrl+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

echo "🚀 [2/3] Launching QX Quantum Hospital Dashboards..."

# Launch both dashboards in background and disown them
nohup $VENV_PYTHON ui/hospital_dashboard.py > /dev/null 2>&1 &
nohup $VENV_PYTHON ui/quantum_dashboard.py > /dev/null 2>&1 &

echo "-----------------------------------------------------------"
echo "🏨 Hospital Command Center  : http://localhost:8050"
echo "⚛️  Quantum Engine Visualizer: http://localhost:8051"
echo "-----------------------------------------------------------"
echo "📡 Dashboards started in background."
echo "-----------------------------------------------------------"
