"""
Configuration file for stereo vision system
"""

# ==============================================================================
# CAMERA CONFIGURATION
# ==============================================================================

CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 1

IMAGE_RESOLUTION = (1920, 1080)

STEREO_BASELINE_MM = 100.0


# ==============================================================================
# CALIBRATION CONFIGURATION
# ==============================================================================

CHESSBOARD_COLUMNS = 9
CHESSBOARD_ROWS = 6

SQUARE_SIZE_MM = 25.0

NUM_CALIBRATION_IMAGES = 20

CALIBRATION_IMAGES_DIR = "calibration_images"
CALIBRATION_FILE = "stereo_calibration.pkl"


# ==============================================================================
# STEREO MATCHING CONFIGURATION
# ==============================================================================

STEREO_ALGORITHM = "sgbm"

SGBM_WINDOW_SIZE = 5
SGBM_MIN_DISPARITY = 0
SGBM_NUM_DISPARITIES = 160
SGBM_UNIQUENESS_RATIO = 10
SGBM_SPECKLE_WINDOW_SIZE = 100
SGBM_SPECKLE_RANGE = 32
SGBM_DISP12_MAX_DIFF = 1
SGBM_PRE_FILTER_CAP = 63

BM_NUM_DISPARITIES = 160
BM_BLOCK_SIZE = 15
BM_PRE_FILTER_CAP = 31
BM_TEXTURE_THRESHOLD = 10
BM_UNIQUENESS_RATIO = 15
BM_SPECKLE_RANGE = 32
BM_SPECKLE_WINDOW_SIZE = 100


# ==============================================================================
# POINT CLOUD CONFIGURATION
# ==============================================================================

MIN_DISPARITY_THRESHOLD = 1.0
MAX_DEPTH_MM = 10000

ENABLE_FILTERING = True
VOXEL_SIZE = 5.0
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0

ESTIMATE_NORMALS = True
NORMAL_RADIUS = 10.0
NORMAL_MAX_NN = 30

ENABLE_VISUALIZATION = True


# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

OUTPUT_DIR = "stereo_output"
TEST_IMAGES_DIR = "test_images"

POINT_CLOUD_FORMAT = "ply"

SAVE_RECTIFIED_IMAGES = True
SAVE_DISPARITY_MAP = True
SAVE_DEPTH_MAP = True
SAVE_DISPARITY_VISUALIZATION = True


# ==============================================================================
# ADVANCED SETTINGS
# ==============================================================================

CALIB_FLAGS = 0

RECTIFY_ALPHA = 0

USE_RGB = False


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

