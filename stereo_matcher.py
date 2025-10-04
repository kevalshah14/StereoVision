"""
Stereo Matching and Disparity Map Generation
Computes disparity maps from calibrated stereo image pairs
"""

import cv2
import numpy as np
import pickle
from pathlib import Path


class StereoMatcher:
    """Class to handle stereo matching and disparity computation"""
    
    def __init__(self, calibration_file="stereo_calibration.pkl", algorithm="sgbm"):
        """
        Initialize stereo matcher
        
        Args:
            calibration_file: Path to calibration parameters file
            algorithm: Matching algorithm ('bm' or 'sgbm')
        """
        self.calibration_file = calibration_file
        self.algorithm = algorithm.lower()
        
        # Load calibration parameters
        self.calibration = self._load_calibration()
        
        # Initialize stereo matcher
        self.matcher = self._create_matcher()
        
    def _load_calibration(self):
        """Load calibration parameters"""
        try:
            with open(self.calibration_file, 'rb') as f:
                params = pickle.load(f)
            print(f"Loaded calibration from {self.calibration_file}")
            return params
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Calibration file {self.calibration_file} not found. "
                "Please run stereo calibration first."
            )
    
    def _create_matcher(self):
        """Create stereo matcher based on selected algorithm"""
        
        if self.algorithm == "bm":
            # Block Matching - faster but less accurate
            matcher = cv2.StereoBM_create()
            matcher.setNumDisparities(16 * 10)  # Must be divisible by 16
            matcher.setBlockSize(15)
            matcher.setPreFilterCap(31)
            matcher.setMinDisparity(0)
            matcher.setTextureThreshold(10)
            matcher.setUniquenessRatio(15)
            matcher.setSpeckleRange(32)
            matcher.setSpeckleWindowSize(100)
            
        elif self.algorithm == "sgbm":
            # Semi-Global Block Matching - slower but more accurate
            window_size = 5
            min_disp = 0
            num_disp = 16 * 10  # Must be divisible by 16
            
            matcher = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=window_size,
                P1=8 * 3 * window_size ** 2,
                P2=32 * 3 * window_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}. Use 'bm' or 'sgbm'")
        
        print(f"Initialized {self.algorithm.upper()} stereo matcher")
        return matcher
    
    def rectify_images(self, left_img, right_img):
        """
        Rectify stereo image pair using calibration parameters
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            
        Returns:
            tuple: (rectified_left, rectified_right)
        """
        left_rectified = cv2.remap(
            left_img,
            self.calibration['map_left_x'],
            self.calibration['map_left_y'],
            cv2.INTER_LINEAR
        )
        
        right_rectified = cv2.remap(
            right_img,
            self.calibration['map_right_x'],
            self.calibration['map_right_y'],
            cv2.INTER_LINEAR
        )
        
        return left_rectified, right_rectified
    
    def compute_disparity(self, left_img, right_img, rectify=True):
        """
        Compute disparity map from stereo pair
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            rectify: Whether to rectify images first
            
        Returns:
            numpy.ndarray: Disparity map (normalized to 0-255)
        """
        # Rectify images if needed
        if rectify:
            left_rect, right_rect = self.rectify_images(left_img, right_img)
        else:
            left_rect, right_rect = left_img, right_img
        
        # Convert to grayscale if needed
        if len(left_rect.shape) == 3:
            left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_rect
            right_gray = right_rect
        
        # Compute disparity
        disparity = self.matcher.compute(left_gray, right_gray)
        
        # Convert to float32
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    def compute_depth_map(self, disparity):
        """
        Convert disparity map to depth map
        
        Args:
            disparity: Disparity map
            
        Returns:
            numpy.ndarray: Depth map in mm
        """
        # Get focal length and baseline from calibration
        Q = self.calibration['Q']
        
        # Avoid division by zero
        disparity_safe = np.where(disparity > 0, disparity, 0.1)
        
        # Calculate depth: depth = (focal_length * baseline) / disparity
        # Using Q matrix: Z = Q[2,3] / (disparity + Q[3,3])
        depth = Q[2, 3] / (disparity_safe + Q[3, 3])
        
        return depth
    
    def normalize_disparity(self, disparity):
        """
        Normalize disparity map for visualization
        
        Args:
            disparity: Raw disparity map
            
        Returns:
            numpy.ndarray: Normalized disparity (0-255, uint8)
        """
        # Remove negative disparities
        disparity_vis = disparity.copy()
        disparity_vis[disparity_vis < 0] = 0
        
        # Normalize to 0-255
        min_disp = np.min(disparity_vis[disparity_vis > 0]) if np.any(disparity_vis > 0) else 0
        max_disp = np.max(disparity_vis)
        
        if max_disp > min_disp:
            disparity_normalized = ((disparity_vis - min_disp) / (max_disp - min_disp) * 255).astype(np.uint8)
        else:
            disparity_normalized = np.zeros_like(disparity_vis, dtype=np.uint8)
        
        return disparity_normalized
    
    def visualize_disparity(self, disparity, colormap=cv2.COLORMAP_JET):
        """
        Create colored visualization of disparity map
        
        Args:
            disparity: Disparity map
            colormap: OpenCV colormap to apply
            
        Returns:
            numpy.ndarray: Colored disparity map
        """
        disparity_normalized = self.normalize_disparity(disparity)
        disparity_colored = cv2.applyColorMap(disparity_normalized, colormap)
        
        return disparity_colored
    
    def process_stereo_pair(self, left_img, right_img, save_dir=None, prefix="stereo"):
        """
        Complete stereo processing pipeline
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            save_dir: Optional directory to save results
            prefix: Prefix for saved files
            
        Returns:
            dict: Processing results
        """
        # Rectify images
        left_rect, right_rect = self.rectify_images(left_img, right_img)
        
        # Compute disparity
        disparity = self.compute_disparity(left_img, right_img, rectify=True)
        
        # Compute depth map
        depth_map = self.compute_depth_map(disparity)
        
        # Visualizations
        disparity_vis = self.visualize_disparity(disparity)
        
        results = {
            'left_rectified': left_rect,
            'right_rectified': right_rect,
            'disparity': disparity,
            'disparity_visualization': disparity_vis,
            'depth_map': depth_map,
            'Q_matrix': self.calibration['Q']
        }
        
        # Save results if requested
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(save_dir / f"{prefix}_left_rectified.jpg"), left_rect)
            cv2.imwrite(str(save_dir / f"{prefix}_right_rectified.jpg"), right_rect)
            cv2.imwrite(str(save_dir / f"{prefix}_disparity.jpg"), disparity_vis)
            
            # Save raw disparity as numpy array
            np.save(str(save_dir / f"{prefix}_disparity.npy"), disparity)
            np.save(str(save_dir / f"{prefix}_depth.npy"), depth_map)
            
            print(f"Results saved to {save_dir}")
        
        return results
    
    def update_matcher_params(self, **kwargs):
        """
        Update stereo matcher parameters dynamically
        
        Args:
            **kwargs: Matcher parameters to update
        """
        for param, value in kwargs.items():
            setter = f"set{param[0].upper()}{param[1:]}"
            if hasattr(self.matcher, setter):
                getattr(self.matcher, setter)(value)
                print(f"Updated {param} = {value}")
            else:
                print(f"Warning: Parameter {param} not found")


if __name__ == "__main__":
    import sys
    
    print("Stereo Matcher Test")
    print("=" * 60)
    
    # Check if calibration exists
    if not Path("stereo_calibration.pkl").exists():
        print("Error: stereo_calibration.pkl not found")
        print("Please run stereo calibration first")
        sys.exit(1)
    
    # Initialize matcher
    matcher = StereoMatcher(algorithm="sgbm")
    
    # Test with sample images if provided
    if len(sys.argv) >= 3:
        left_path = sys.argv[1]
        right_path = sys.argv[2]
        
        print(f"\nProcessing:")
        print(f"  Left: {left_path}")
        print(f"  Right: {right_path}")
        
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is None or right_img is None:
            print("Error: Could not load images")
            sys.exit(1)
        
        # Process stereo pair
        results = matcher.process_stereo_pair(left_img, right_img, save_dir="stereo_output")
        
        print("\n✓ Stereo processing completed!")
        print(f"  Disparity shape: {results['disparity'].shape}")
        print(f"  Depth range: {np.min(results['depth_map']):.1f} - {np.max(results['depth_map']):.1f} mm")
        
    else:
        print("\nUsage: python stereo_matcher.py <left_image> <right_image>")

