#!/bin/bash

echo "Setting up Deep Research Python Environment..."

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found. Please install Python 3.10+"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
echo "To start, run:"
echo "  source venv/bin/activate"
echo "  cp .env.example .env  # (And edit your API keys)"
echo "  python main.py \"Your query\""
