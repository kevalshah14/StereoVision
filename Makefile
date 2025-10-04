# Makefile for Stereo Vision System
# Convenient commands for common tasks

.PHONY: help install install-uv install-dev test test-cameras calibrate clean format lint

help:
	@echo "Stereo Vision System - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies (macOS/Linux)"
	@echo "  make install-pi       Install with PiCamera2 (Raspberry Pi)"
	@echo "  make install-uv       Install uv package manager"
	@echo "  make install-dev      Install with dev dependencies"
	@echo "  make setup            Complete setup (install uv + dependencies)"
	@echo ""
	@echo "Testing:"
	@echo "  make test-cameras     Test camera connection and alignment"
	@echo "  make pattern-info     Show calibration pattern information"
	@echo ""
	@echo "Pipeline:"
	@echo "  make run              Run main.py with uv (recommended)"
	@echo "  make calibrate        Run camera calibration"
	@echo "  make full-pipeline    Run complete stereo vision pipeline"
	@echo "  make process-live     Process live camera capture"
	@echo ""
	@echo "Development:"
	@echo "  make format           Format code with black"
	@echo "  make lint             Lint code with ruff"
	@echo "  make clean            Clean generated files"
	@echo "  make clean-all        Clean everything including venv"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs             Show documentation info"
	@echo "  make version          Show version information"
	@echo ""

# Installation
install-uv:
	@echo "Installing uv..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "✓ uv installed"

install:
	@echo "Installing dependencies..."
	@uv venv
	@uv pip install --python .venv/bin/python opencv-python numpy open3d matplotlib
	@echo "✓ Dependencies installed"
	@echo "✓ Virtual environment created at .venv/"
	@echo ""
	@echo "For Raspberry Pi with cameras, also run: make install-pi"

install-pi:
	@echo "Installing PiCamera2 for Raspberry Pi..."
	@uv pip install --python .venv/bin/python picamera2
	@echo "✓ PiCamera2 installed"

install-dev:
	@echo "Installing with dev dependencies..."
	@uv sync --extra dev
	@echo "✓ Dev environment ready"

# Testing
test-cameras:
	@. .venv/bin/activate && python test_cameras.py --test all

pattern-info:
	@. .venv/bin/activate && python test_cameras.py --pattern-info

# Pipeline commands
calibrate:
	@. .venv/bin/activate && python stereo_vision_pipeline.py --capture-calibration --num-images 20
	@. .venv/bin/activate && python stereo_vision_pipeline.py --calibrate

run:
	@uv run main.py --full-pipeline

full-pipeline:
	@. .venv/bin/activate && python stereo_vision_pipeline.py --full-pipeline

process-live:
	@. .venv/bin/activate && python stereo_vision_pipeline.py --process-live

# Development
format:
	@echo "Formatting code..."
	@black *.py
	@echo "✓ Code formatted"

lint:
	@echo "Linting code..."
	@ruff check *.py
	@echo "✓ Linting complete"

# Cleanup
clean:
	@echo "Cleaning generated files..."
	@rm -rf __pycache__ .pytest_cache *.pyc
	@rm -rf calibration_images/ stereo_output/ test_images/
	@rm -f stereo_calibration.pkl
	@rm -f test_*.jpg continuous_*.jpg side_by_side.jpg
	@rm -f *.npy
	@echo "✓ Cleaned"

clean-all: clean
	@echo "Removing virtual environment and lock file..."
	@rm -rf .venv uv.lock
	@echo "✓ Deep cleaned"

# Documentation
docs:
	@echo "Project Documentation:"
	@echo "  README.md      - Complete documentation"
	@cat README.md | head -20

# Version info
version:
	@echo "Stereo Vision System v1.0.0"
	@python --version
	@uv --version 2>/dev/null || echo "uv: not installed"

# Setup everything
setup: install-uv install
	@echo ""
	@echo "✅ Setup complete!"
	@echo "Run 'make test-cameras' to verify your cameras"

