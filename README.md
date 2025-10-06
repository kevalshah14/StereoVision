# Stereo Vision System

A complete stereo vision pipeline for generating 3D point clouds from dual camera systems. Supports Raspberry Pi Camera Module 3 and standard USB/webcams.

## Quick Start

### 1. Install Dependencies

```bash
# Clone or navigate to the project
cd StereoVision

# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv pip install --python .venv/bin/python -r requirements.txt

# On Raspberry Pi, also install picamera2:
# uv pip install --python .venv/bin/python picamera2
```

### 2. Configure Your Setup

Edit `config.py` to match your hardware:

```python
# Set camera indices (usually 0 and 1)
CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 1

# Set your camera resolution
IMAGE_RESOLUTION = (1920, 1080)

# Set your chessboard pattern dimensions
CHESSBOARD_COLUMNS = 9  # Internal corners (horizontal)
CHESSBOARD_ROWS = 6     # Internal corners (vertical)
SQUARE_SIZE_MM = 25.0   # Size of each square in mm
```

### 3. Run the Complete Pipeline

```bash
# Full automated pipeline - recommended for first run
python main.py --full-pipeline
```

This will:
1. Guide you through capturing 20 calibration image pairs
2. Automatically calibrate your stereo camera system
3. Capture a live stereo pair and generate a 3D point cloud
4. Display the point cloud in an interactive 3D viewer

## Step-by-Step Usage

### Option 1: Individual Steps

```bash
# Step 1: Capture calibration images (move chessboard to different positions)
python main.py --capture-calibration --num-images 25

# Step 2: Calibrate the stereo camera system
python main.py --calibrate

# Step 3: Capture and process live images
python main.py --process-live

# Step 4: View the results
python viewer.py --dir stereo_output
```

### Option 2: Process Existing Images

```bash
# If you already have calibration data, process image pairs directly
python main.py --process-files left.jpg right.jpg

# View the results
python viewer.py --dir stereo_output --mode all
```

### Option 3: Process MiddEval3 Dataset

The project includes a script to process the MiddEval3 stereo benchmark dataset:

```bash
# List available scenes
python process_middeval3.py --list

# Process a single scene
python process_middeval3.py --scene Motorcycle --set training

# Process all test scenes
python process_middeval3.py --all --set test --no-visualize

# View results
python viewer.py --dir middeval3_output/Motorcycle
```

## Hardware Requirements

**Option 1: Raspberry Pi Setup**
- 2x Raspberry Pi Camera Module 3
- Raspberry Pi 4 or 5 (8GB RAM recommended)
- Camera mounting bracket for stable baseline
- Chessboard calibration pattern

**Option 2: USB Webcam Setup**
- 2x USB webcams
- macOS, Linux, or Windows PC
- Chessboard calibration pattern

**Calibration Pattern:**
- 9x6 internal corners (10x7 squares)
- 25mm square size (adjustable in config.py)
- Print on rigid, flat surface
- Download: https://markhedleyjones.com/projects/calibration-checkerboard-collection

## Software Requirements

- Python 3.8 or higher
- uv package manager ([install guide](https://github.com/astral-sh/uv))
- Dependencies: opencv-python, numpy, open3d, matplotlib
- picamera2 (Raspberry Pi only)

## Output Files

After processing, you'll find:

```
stereo_output/
├── stereo_left_rectified.jpg      # Rectified left image
├── stereo_right_rectified.jpg     # Rectified right image
├── stereo_disparity.jpg           # Colored disparity visualization
├── stereo_disparity.npy           # Raw disparity data
├── stereo_depth.npy               # Depth map in millimeters
└── point_cloud.ply                # 3D point cloud (open with MeshLab, CloudCompare, Blender)
```

## Viewer Controls

The `viewer.py` script provides multiple viewing modes:

```bash
# View everything (2D maps + 3D point cloud)
python viewer.py --dir stereo_output --mode all

# View only 2D depth/disparity maps
python viewer.py --dir stereo_output --mode 2d

# View only 3D point cloud
python viewer.py --dir stereo_output --mode 3d

# Adjust point cloud sampling and depth range
python viewer.py --dir stereo_output --sample 5 --max-depth 5000
```

**Interactive 3D Controls:**
- Left-click + drag: Rotate view
- Right-click + drag: Zoom
- Scroll wheel: Zoom in/out

## Configuration

The `config.py` file contains all adjustable parameters:

**Camera Settings:**
- Camera indices and resolution
- Stereo baseline distance

**Calibration Settings:**
- Chessboard pattern dimensions
- Number of calibration images
- Square size in mm

**Stereo Matching:**
- Algorithm selection (BM or SGBM)
- Disparity parameters
- Window sizes

**Point Cloud:**
- Filtering and downsampling
- Outlier removal
- Normal estimation

View current configuration:
```bash
python config.py
```

## Troubleshooting

### Cameras Not Detected
```bash
# On Raspberry Pi, check available cameras
libcamera-hello --list-cameras

# On other systems, check video devices
ls /dev/video*

# Test camera capture
python camera_capture.py
```

### Poor Calibration Results

**Symptoms:** High RMS error (> 1.0), distorted rectification

**Solutions:**
- Capture 25-30 image pairs (not just 20)
- Ensure entire chessboard visible in BOTH cameras
- Vary positions: center, edges, corners, near, far
- Vary angles: tilted, rotated, at different depths
- Use bright, uniform lighting
- Keep cameras and chessboard perfectly still during capture
- Print chessboard on rigid material (foam board, not paper)

### Low Point Cloud Quality

**Symptoms:** Noisy points, holes, poor detail

**Solutions:**
- Re-run calibration with more images
- Add texture to the scene (avoid plain white walls)
- Improve lighting (avoid shadows and glare)
- Ensure cameras are rigidly mounted (no movement)
- Adjust stereo matching parameters in `config.py`
- Increase `SGBM_NUM_DISPARITIES` for closer objects
- Decrease `SGBM_WINDOW_SIZE` for more detail

### Memory Issues

If you encounter out-of-memory errors:
- Reduce `IMAGE_RESOLUTION` in `config.py`
- Increase `VOXEL_SIZE` for more downsampling
- Use `--no-visualize` flag when processing multiple scenes

## Project Structure

```
StereoVision/
├── main.py                      # Main entry point
├── stereo_vision_pipeline.py    # Complete pipeline orchestration
├── camera_capture.py            # Dual camera capture (Pi Camera / USB)
├── stereo_calibration.py        # Camera calibration using chessboard
├── stereo_matcher.py            # Disparity computation (BM/SGBM)
├── point_cloud_generator.py    # 3D point cloud generation
├── viewer.py                    # Interactive result viewer
├── process_middeval3.py         # MiddEval3 dataset processor
├── config.py                    # Configuration parameters
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── MiddEval3/                   # Optional: benchmark dataset
```

## Advanced Usage

### Individual Module Testing

```bash
# Test camera capture
python camera_capture.py

# Run calibration on specific directory
python stereo_calibration.py calibration_images/

# Compute disparity from image pair
python stereo_matcher.py left.jpg right.jpg

# Generate point cloud from disparity
python point_cloud_generator.py stereo_disparity.npy left.jpg
```

### Custom Parameters

```bash
# Calibrate with custom chessboard
python main.py --calibrate \
  --chessboard-size 7 5 \
  --square-size 30.0

# Process with custom output directory
python main.py --process-live \
  --output-dir custom_output \
  --no-visualize
```

### Batch Processing

```bash
# Process multiple MiddEval3 scenes
python process_middeval3.py --all --set training --algorithm sgbm
```

## Tips for Best Results

1. **Calibration is Critical:** Spend time getting a good calibration. More images from varied positions = better results.

2. **Lighting Matters:** Use diffuse, uniform lighting. Avoid direct sunlight, shadows, and reflections.

3. **Texture Helps:** Stereo matching works better on textured surfaces. Plain walls and featureless objects are difficult.

4. **Stable Mounting:** Any camera movement between left/right capture ruins stereo matching. Use a rigid mount.

5. **Baseline Distance:** Larger baseline = better depth accuracy but smaller overlapping field of view. 60-120mm is typical for desktop scenes.

6. **Start with SGBM:** Semi-Global Block Matching (SGBM) is slower but produces better results than Block Matching (BM).

## Resources

- [OpenCV Stereo Vision Tutorial](https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html)
- [MiddEval3 Dataset](https://vision.middlebury.edu/stereo/data/)
- [Calibration Checkerboards](https://markhedleyjones.com/projects/calibration-checkerboard-collection)
- [Open3D Documentation](http://www.open3d.org/docs/)

---

