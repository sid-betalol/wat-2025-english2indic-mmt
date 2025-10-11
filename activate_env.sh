#!/bin/bash
# Script to activate the virtual environment
# This can be sourced or executed directly

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

echo "Virtual environment activated!"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
