#!/bin/bash
# Automated setup script for Stereo Vision System
# This script installs uv and all dependencies

set -e  # Exit on error

echo "=================================================="
echo "Stereo Vision System - Automated Setup"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Found Python $PYTHON_VERSION"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo ""
    echo "📦 Installing uv (fast package manager)..."
    
    # Detect OS and install accordingly
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Add to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        
        echo "✓ uv installed successfully"
    else
        echo "⚠ Automatic uv installation not supported on this OS"
        echo "Please install uv manually: https://github.com/astral-sh/uv"
        exit 1
    fi
else
    echo "✓ uv is already installed"
fi

echo ""
echo "📚 Installing Python dependencies..."

# Install dependencies using uv
if command -v uv &> /dev/null; then
    echo "Using uv for installation..."
    uv venv
    uv pip install --python .venv/bin/python opencv-python numpy open3d matplotlib
    echo "✓ Dependencies installed successfully with uv"
    echo "✓ Virtual environment created at .venv/"
    echo ""
    echo "To activate the virtual environment:"
    echo "  source .venv/bin/activate"
else
    echo "⚠ uv not found, falling back to pip"
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
    echo "✓ Dependencies installed successfully with pip"
    echo "✓ Virtual environment created at .venv/"
fi

echo ""
echo "🧪 Verifying installation..."

# Activate venv for testing
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Verify key packages
python -c "import cv2" 2>/dev/null && echo "  ✓ OpenCV" || echo "  ❌ OpenCV failed"
python -c "import numpy" 2>/dev/null && echo "  ✓ NumPy" || echo "  ❌ NumPy failed"
python -c "import open3d" 2>/dev/null && echo "  ✓ Open3D" || echo "  ❌ Open3D failed"
python -c "import matplotlib" 2>/dev/null && echo "  ✓ Matplotlib" || echo "  ❌ Matplotlib failed"

# PiCamera2 is optional (only needed on Raspberry Pi)
if python -c "import picamera2" 2>/dev/null; then
    echo "  ✓ PiCamera2 (for Raspberry Pi cameras)"
else
    echo "  ⚠ PiCamera2 not available (only needed on Raspberry Pi)"
fi

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Connect your Raspberry Pi cameras"
echo "  2. Run the pipeline: uv run main.py --full-pipeline"
echo "  3. View your point cloud: stereo_output/point_cloud.ply"
echo ""
echo "Documentation:"
echo "  - Full Guide: README.md"
echo ""

