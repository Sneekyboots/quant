#!/bin/bash
# 🛑 stop.sh - Terminate all running Dash dashboard processes

echo "-----------------------------------------------------------"
echo "🛑 Stopping QX Quantum Hospital Dashboards..."

# Find and kill processes running our dashboard scripts
pkill -f "python.*dashboard.py"

# Verify ports are clear
echo "🔍 Checking ports 8050 and 8051..."
fuser -k 8050/tcp 2>/dev/null
fuser -k 8051/tcp 2>/dev/null

echo "✅ All dashboard processes terminated."
echo "-----------------------------------------------------------"
