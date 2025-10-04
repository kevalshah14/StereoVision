# Stereo Vision System

A complete stereo vision pipeline for generating 3D point clouds using two Raspberry Pi Camera Module 3 cameras.

## Setup

```bash
# Install dependencies (works on macOS/Linux)
uv venv
uv pip install --python .venv/bin/python opencv-python numpy open3d matplotlib

# On Raspberry Pi, also install:
# uv pip install --python .venv/bin/python picamera2

# Run the pipeline
uv run main.py --full-pipeline
```

That's it! The pipeline will:
1. Guide you through capturing calibration images (20 pairs)
2. Calibrate the stereo camera system
3. Capture and process a test image to generate a point cloud

## Requirements

**Hardware:**
- 2x Raspberry Pi Camera Module 3
- Raspberry Pi 4 or 5
- Chessboard calibration pattern (9x6 internal corners, 25mm squares)
  - Download: https://markhedleyjones.com/projects/calibration-checkerboard-collection

**Software:**
- Python 3.8+
- uv package manager ([install here](https://github.com/astral-sh/uv))

## Usage

```bash
# Full automated pipeline (recommended for first run)
uv run main.py --full-pipeline

# Capture calibration images only
uv run main.py --capture-calibration --num-images 25

# Calibrate from existing images
uv run main.py --calibrate

# Generate point cloud from live capture
uv run main.py --process-live

# Generate point cloud from existing images
uv run main.py --process-files left.jpg right.jpg
```

## Output

After running, you'll find:
- `calibration_images/` - Calibration image pairs
- `stereo_calibration.pkl` - Camera calibration parameters
- `stereo_output/point_cloud.ply` - 3D point cloud (open with MeshLab, CloudCompare, or Blender)

## Troubleshooting

**Camera not found:**
```bash
libcamera-hello --list-cameras
```

**Poor calibration results:**
- Capture 25-30 image pairs
- Ensure chessboard is fully visible in both cameras
- Vary positions and angles
- Use good lighting

**Low point cloud quality:**
- Recalibrate with more images
- Add texture to the scene
- Improve lighting
- Ensure cameras are stable

## Project Structure

```
StereoVision/
├── main.py                      # Entry point (uv run main.py)
├── stereo_vision_pipeline.py    # Main pipeline logic
├── camera_capture.py             # Dual camera interface
├── stereo_calibration.py         # Camera calibration
├── stereo_matcher.py             # Stereo matching
└── point_cloud_generator.py     # Point cloud generation
```

## Advanced

Individual module usage:

```bash
# Test cameras
python camera_capture.py

# Run calibration
python stereo_calibration.py calibration_images/

# Compute disparity
python stereo_matcher.py left.jpg right.jpg

# Generate point cloud
python point_cloud_generator.py stereo_disparity.npy left.jpg
```

---

**Happy 3D Vision! 📷📷➡️🌐**
