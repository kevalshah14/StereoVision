"""
Configuration file for stereo vision system
Adjust these parameters according to your setup
"""

# ==============================================================================
# CAMERA CONFIGURATION
# ==============================================================================

# Camera indices (usually 0 and 1 for dual camera setup)
CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 1

# Image resolution (width, height)
# Pi Camera Module 3 supports: (1920, 1080), (2304, 1296), (4608, 2592)
# Higher resolution = better quality but slower processing
IMAGE_RESOLUTION = (1920, 1080)


# ==============================================================================
# CALIBRATION CONFIGURATION
# ==============================================================================

# Chessboard pattern configuration
# Count the INTERNAL corners (not squares)
# For a standard 9x6 pattern, there are 10x7 squares with 9x6 internal corners
CHESSBOARD_COLUMNS = 9  # Internal corners in horizontal direction
CHESSBOARD_ROWS = 6     # Internal corners in vertical direction

# Square size in millimeters
# Measure the size of one square on your printed/physical chessboard
SQUARE_SIZE_MM = 25.0

# Number of calibration image pairs to capture
# More images = better calibration (recommended: 20-30)
NUM_CALIBRATION_IMAGES = 20

# Directories
CALIBRATION_IMAGES_DIR = "calibration_images"
CALIBRATION_FILE = "stereo_calibration.pkl"


# ==============================================================================
# STEREO MATCHING CONFIGURATION
# ==============================================================================

# Stereo matching algorithm
# Options: "bm" (Block Matching - faster) or "sgbm" (Semi-Global - better quality)
STEREO_ALGORITHM = "sgbm"

# Stereo matching parameters for SGBM
SGBM_WINDOW_SIZE = 5
SGBM_MIN_DISPARITY = 0
SGBM_NUM_DISPARITIES = 160  # Must be divisible by 16; higher = better for close objects
SGBM_UNIQUENESS_RATIO = 10
SGBM_SPECKLE_WINDOW_SIZE = 100
SGBM_SPECKLE_RANGE = 32
SGBM_DISP12_MAX_DIFF = 1
SGBM_PRE_FILTER_CAP = 63

# Stereo matching parameters for BM
BM_NUM_DISPARITIES = 160  # Must be divisible by 16
BM_BLOCK_SIZE = 15
BM_PRE_FILTER_CAP = 31
BM_TEXTURE_THRESHOLD = 10
BM_UNIQUENESS_RATIO = 15
BM_SPECKLE_RANGE = 32
BM_SPECKLE_WINDOW_SIZE = 100


# ==============================================================================
# POINT CLOUD CONFIGURATION
# ==============================================================================

# Disparity filtering
MIN_DISPARITY_THRESHOLD = 1.0  # Filter out invalid points
MAX_DEPTH_MM = 10000          # Maximum depth in mm (10 meters)

# Point cloud filtering
ENABLE_FILTERING = True
VOXEL_SIZE = 5.0              # Voxel size for downsampling (mm); larger = fewer points
OUTLIER_NB_NEIGHBORS = 20     # Number of neighbors for outlier removal
OUTLIER_STD_RATIO = 2.0       # Standard deviation ratio for outlier removal

# Normal estimation
ESTIMATE_NORMALS = True
NORMAL_RADIUS = 10.0          # Search radius for normal estimation (mm)
NORMAL_MAX_NN = 30            # Maximum nearest neighbors for normal estimation

# Visualization
ENABLE_VISUALIZATION = True


# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

# Output directories
OUTPUT_DIR = "stereo_output"
TEST_IMAGES_DIR = "test_images"

# Output file formats
POINT_CLOUD_FORMAT = "ply"  # Options: "ply", "pcd", "xyz"

# Save intermediate results
SAVE_RECTIFIED_IMAGES = True
SAVE_DISPARITY_MAP = True
SAVE_DEPTH_MAP = True
SAVE_DISPARITY_VISUALIZATION = True


# ==============================================================================
# ADVANCED SETTINGS
# ==============================================================================

# OpenCV calibration flags
CALIB_FLAGS = 0  # Additional calibration flags if needed

# Stereo rectification alpha
# 0 = no black pixels (cropped view)
# 1 = all pixels retained (black borders)
RECTIFY_ALPHA = 0

# Color space
# OpenCV uses BGR by default, but some applications may need RGB
USE_RGB = False  # Set to True if your application needs RGB instead of BGR


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_camera_config():
    """Get camera configuration dictionary"""
    return {
        'left_index': CAMERA_LEFT_INDEX,
        'right_index': CAMERA_RIGHT_INDEX,
        'resolution': IMAGE_RESOLUTION
    }


def get_calibration_config():
    """Get calibration configuration dictionary"""
    return {
        'chessboard_size': (CHESSBOARD_COLUMNS, CHESSBOARD_ROWS),
        'square_size': SQUARE_SIZE_MM,
        'num_images': NUM_CALIBRATION_IMAGES,
        'calibration_dir': CALIBRATION_IMAGES_DIR,
        'calibration_file': CALIBRATION_FILE
    }


def get_stereo_matcher_config():
    """Get stereo matcher configuration dictionary"""
    if STEREO_ALGORITHM == "sgbm":
        return {
            'algorithm': STEREO_ALGORITHM,
            'window_size': SGBM_WINDOW_SIZE,
            'min_disparity': SGBM_MIN_DISPARITY,
            'num_disparities': SGBM_NUM_DISPARITIES,
            'uniqueness_ratio': SGBM_UNIQUENESS_RATIO,
            'speckle_window_size': SGBM_SPECKLE_WINDOW_SIZE,
            'speckle_range': SGBM_SPECKLE_RANGE,
            'disp12_max_diff': SGBM_DISP12_MAX_DIFF,
            'pre_filter_cap': SGBM_PRE_FILTER_CAP
        }
    else:
        return {
            'algorithm': STEREO_ALGORITHM,
            'num_disparities': BM_NUM_DISPARITIES,
            'block_size': BM_BLOCK_SIZE,
            'pre_filter_cap': BM_PRE_FILTER_CAP,
            'texture_threshold': BM_TEXTURE_THRESHOLD,
            'uniqueness_ratio': BM_UNIQUENESS_RATIO,
            'speckle_range': BM_SPECKLE_RANGE,
            'speckle_window_size': BM_SPECKLE_WINDOW_SIZE
        }


def get_point_cloud_config():
    """Get point cloud configuration dictionary"""
    return {
        'min_disparity': MIN_DISPARITY_THRESHOLD,
        'max_depth': MAX_DEPTH_MM,
        'filter_cloud': ENABLE_FILTERING,
        'voxel_size': VOXEL_SIZE if ENABLE_FILTERING else None,
        'nb_neighbors': OUTLIER_NB_NEIGHBORS,
        'std_ratio': OUTLIER_STD_RATIO,
        'estimate_normals': ESTIMATE_NORMALS,
        'normal_radius': NORMAL_RADIUS,
        'normal_max_nn': NORMAL_MAX_NN,
        'visualize': ENABLE_VISUALIZATION
    }


def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("STEREO VISION SYSTEM CONFIGURATION")
    print("=" * 70)
    print("\n📷 CAMERA:")
    print(f"  Left camera index: {CAMERA_LEFT_INDEX}")
    print(f"  Right camera index: {CAMERA_RIGHT_INDEX}")
    print(f"  Resolution: {IMAGE_RESOLUTION[0]} x {IMAGE_RESOLUTION[1]}")
    
    print("\n📐 CALIBRATION:")
    print(f"  Chessboard: {CHESSBOARD_COLUMNS} x {CHESSBOARD_ROWS} internal corners")
    print(f"  Square size: {SQUARE_SIZE_MM} mm")
    print(f"  Calibration images: {NUM_CALIBRATION_IMAGES}")
    
    print("\n🔍 STEREO MATCHING:")
    print(f"  Algorithm: {STEREO_ALGORITHM.upper()}")
    if STEREO_ALGORITHM == "sgbm":
        print(f"  Disparities: {SGBM_NUM_DISPARITIES}")
        print(f"  Window size: {SGBM_WINDOW_SIZE}")
    else:
        print(f"  Disparities: {BM_NUM_DISPARITIES}")
        print(f"  Block size: {BM_BLOCK_SIZE}")
    
    print("\n☁️ POINT CLOUD:")
    print(f"  Filtering: {'Enabled' if ENABLE_FILTERING else 'Disabled'}")
    if ENABLE_FILTERING:
        print(f"  Voxel size: {VOXEL_SIZE} mm")
    print(f"  Max depth: {MAX_DEPTH_MM} mm ({MAX_DEPTH_MM/1000:.1f} m)")
    print(f"  Normal estimation: {'Enabled' if ESTIMATE_NORMALS else 'Disabled'}")
    
    print("\n💾 OUTPUT:")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Point cloud format: {POINT_CLOUD_FORMAT.upper()}")
    print(f"  Visualization: {'Enabled' if ENABLE_VISUALIZATION else 'Disabled'}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()

