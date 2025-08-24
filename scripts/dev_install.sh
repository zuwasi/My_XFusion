#!/bin/bash
# XFusion Ultra-Lite Development Environment Setup

set -e

echo "Setting up XFusion Ultra-Lite development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
if [ "$python_version" != "3.10" ]; then
    echo "Warning: Expected Python 3.10, found $python_version"
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing XFusion in development mode..."
pip install -e .

echo "Setup complete! Run 'source venv/bin/activate' to activate the environment."
echo "Test with: python xfusion_lite.py validate"
