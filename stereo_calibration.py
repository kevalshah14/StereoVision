"""
Stereo Camera Calibration Module
Performs camera calibration for stereo vision using chessboard patterns
"""

import cv2
import numpy as np
import glob
from pathlib import Path
import pickle


class StereoCalibrator:
    """Class to handle stereo camera calibration"""
    
    def __init__(self, chessboard_size=(9, 6), square_size=25.0):
        """
        Initialize calibrator
        
        Args:
            chessboard_size: Tuple of (columns, rows) of internal corners
            square_size: Size of chessboard square in mm
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
    def find_chessboard_corners(self, image):
        """
        Find chessboard corners in an image
        
        Args:
            image: Input image
            
        Returns:
            tuple: (success, corners) where success is bool and corners are detected points
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        
        return ret, corners
    
    def calibrate_stereo(self, left_images_dir, right_images_dir, output_file="stereo_calibration.pkl"):
        """
        Perform stereo calibration from image pairs
        
        Args:
            left_images_dir: Directory containing left camera images
            right_images_dir: Directory containing right camera images
            output_file: Path to save calibration parameters
            
        Returns:
            dict: Calibration parameters
        """
        left_images_dir = Path(left_images_dir)
        right_images_dir = Path(right_images_dir)
        
        # Get image lists
        left_images = sorted(glob.glob(str(left_images_dir / "*.jpg")))
        right_images = sorted(glob.glob(str(right_images_dir / "*.jpg")))
        
        if len(left_images) != len(right_images):
            raise ValueError("Number of left and right images must match")
        
        if len(left_images) == 0:
            raise ValueError("No images found for calibration")
        
        print(f"Found {len(left_images)} stereo pairs for calibration")
        
        objpoints = []
        imgpoints_left = []
        imgpoints_right = []
        valid_pairs = 0
        image_size = None
        for left_path, right_path in zip(left_images, right_images):
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            
            if image_size is None:
                image_size = (left_img.shape[1], left_img.shape[0])
            
            ret_left, corners_left = self.find_chessboard_corners(left_img)
            ret_right, corners_right = self.find_chessboard_corners(right_img)
            
            if ret_left and ret_right:
                objpoints.append(self.objp)
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)
                valid_pairs += 1
                print(f"✓ Valid pair {valid_pairs}: {Path(left_path).name}")
            else:
                print(f"✗ Skipped: {Path(left_path).name} (chessboard not found)")
        
        if valid_pairs < 10:
            raise ValueError(f"Insufficient valid pairs for calibration (found {valid_pairs}, need at least 10)")
        
        print(f"\nCalibrating with {valid_pairs} valid image pairs...")
        
        print("Calibrating left camera...")
        ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            objpoints, imgpoints_left, image_size, None, None
        )
        
        print("Calibrating right camera...")
        ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            objpoints, imgpoints_right, image_size, None, None
        )
        
        print("Performing stereo calibration...")
        flags = cv2.CALIB_FIX_INTRINSIC
        
        ret_stereo, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            K_left, dist_left, K_right, dist_right,
            image_size, criteria=self.criteria, flags=flags
        )
        
        print(f"\nStereo calibration RMS error: {ret_stereo:.4f}")
        
        print("Computing rectification parameters...")
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            K_left, dist_left, K_right, dist_right,
            image_size, R, T, alpha=0
        )
        
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            K_left, dist_left, R1, P1, image_size, cv2.CV_32FC1
        )
        
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            K_right, dist_right, R2, P2, image_size, cv2.CV_32FC1
        )
        calibration_params = {
            'image_size': image_size,
            'K_left': K_left,
            'dist_left': dist_left,
            'K_right': K_right,
            'dist_right': dist_right,
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            'R1': R1,
            'R2': R2,
            'P1': P1,
            'P2': P2,
            'Q': Q,
            'roi_left': roi_left,
            'roi_right': roi_right,
            'map_left_x': map_left_x,
            'map_left_y': map_left_y,
            'map_right_x': map_right_x,
            'map_right_y': map_right_y,
            'rms_error': ret_stereo
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(calibration_params, f)
        
        print(f"\n✓ Calibration parameters saved to {output_file}")
        self._print_calibration_summary(calibration_params)
        
        return calibration_params
    
    def _print_calibration_summary(self, params):
        """Display calibration results"""
        baseline = np.linalg.norm(params['T'])
        
        print("\n" + "=" * 60)
        print("CALIBRATION SUMMARY")
        print("=" * 60)
        print(f"Image size: {params['image_size']}")
        print(f"RMS error: {params['rms_error']:.4f} pixels")
        print(f"Baseline distance: {baseline:.2f} mm ({baseline/10:.2f} cm)")
        print(f"\nLeft camera matrix:\n{params['K_left']}")
        print(f"\nRight camera matrix:\n{params['K_right']}")
        print(f"\nRotation matrix:\n{params['R']}")
        print(f"\nTranslation vector:\n{params['T'].ravel()}")
        print("=" * 60)
    
    @staticmethod
    def load_calibration(calibration_file):
        """
        Load calibration parameters from file
        
        Args:
            calibration_file: Path to calibration file
            
        Returns:
            dict: Calibration parameters
        """
        with open(calibration_file, 'rb') as f:
            params = pickle.load(f)
        print(f"Loaded calibration from {calibration_file}")
        return params


if __name__ == "__main__":
    import sys
    
    print("Stereo Camera Calibration")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        calibration_dir = sys.argv[1]
    else:
        calibration_dir = "calibration_images"
    
    left_dir = Path(calibration_dir)
    right_dir = Path(calibration_dir)
    
    if not left_dir.exists():
        print(f"Error: Directory {left_dir} not found")
        print("\nUsage: python stereo_calibration.py [calibration_images_dir]")
        sys.exit(1)
    
    # Initialize calibrator (adjust chessboard size as needed)
    calibrator = StereoCalibrator(chessboard_size=(9, 6), square_size=25.0)
    
    try:
        # Perform calibration
        params = calibrator.calibrate_stereo(left_dir, right_dir)
        print("\n✓ Calibration completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Calibration failed: {e}")
        sys.exit(1)

