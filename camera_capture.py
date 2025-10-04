"""
Dual Camera Capture Module for Raspberry Pi Camera Module 3
This script handles capturing synchronized images from two Raspberry Pi cameras
Also supports standard USB/webcams via OpenCV for development on non-Pi systems
"""

import cv2
import numpy as np
import time
from pathlib import Path
import platform
import sys

# Try to import picamera2 (Raspberry Pi only)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Note: picamera2 not available - using OpenCV VideoCapture instead")

# Detect if we're on Raspberry Pi
def is_raspberry_pi():
    """Check if running on Raspberry Pi"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return 'raspberry pi' in f.read().lower()
    except:
        return False

USE_PICAMERA = PICAMERA2_AVAILABLE and is_raspberry_pi()


class DualCameraCapture:
    """Class to handle dual camera capture for stereo vision"""
    
    def __init__(self, camera_indices=[0, 1], resolution=(1920, 1080)):
        """
        Initialize dual camera setup
        
        Args:
            camera_indices: List of camera indices [left_cam, right_cam]
            resolution: Tuple of (width, height) for image resolution
        """
        self.camera_indices = camera_indices
        self.resolution = resolution
        self.cameras = []
        self.use_picamera = USE_PICAMERA
        
        print(f"Initializing cameras... (using {'picamera2' if self.use_picamera else 'OpenCV VideoCapture'})")
        self._setup_cameras()
        
    def _setup_cameras(self):
        """Setup and configure both cameras"""
        if self.use_picamera:
            self._setup_picamera()
        else:
            self._setup_opencv_cameras()
    
    def _setup_picamera(self):
        """Setup Raspberry Pi cameras using picamera2"""
        for idx in self.camera_indices:
            try:
                cam = Picamera2(idx)
                
                # Configure camera
                config = cam.create_still_configuration(
                    main={"size": self.resolution, "format": "RGB888"}
                )
                cam.configure(config)
                cam.start()
                
                # Allow camera to warm up
                time.sleep(2)
                
                self.cameras.append(cam)
                print(f"Picamera2 {idx} initialized successfully")
                
            except Exception as e:
                print(f"Error initializing picamera2 {idx}: {e}")
                raise
    
    def _setup_opencv_cameras(self):
        """Setup cameras using OpenCV VideoCapture (for USB/webcams)"""
        for idx in self.camera_indices:
            try:
                cam = cv2.VideoCapture(idx)
                
                if not cam.isOpened():
                    raise RuntimeError(f"Could not open camera {idx}")
                
                # Set resolution
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                
                # Verify actual resolution
                actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Allow camera to warm up
                time.sleep(1)
                
                self.cameras.append(cam)
                print(f"OpenCV camera {idx} initialized successfully ({actual_width}x{actual_height})")
                
            except Exception as e:
                print(f"Error initializing camera {idx}: {e}")
                raise
    
    def capture_stereo_pair(self):
        """
        Capture synchronized images from both cameras
        
        Returns:
            tuple: (left_image, right_image) as numpy arrays
        """
        if len(self.cameras) != 2:
            raise RuntimeError("Both cameras must be initialized")
        
        if self.use_picamera:
            # Capture from picamera2
            left_image = self.cameras[0].capture_array()
            right_image = self.cameras[1].capture_array()
            
            # Convert from RGB to BGR for OpenCV
            left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
            right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR)
        else:
            # Capture from OpenCV VideoCapture
            ret1, left_image = self.cameras[0].read()
            ret2, right_image = self.cameras[1].read()
            
            if not ret1 or not ret2:
                raise RuntimeError("Failed to capture from one or both cameras")
        
        return left_image, right_image
    
    def save_stereo_pair(self, save_dir, prefix="stereo"):
        """
        Capture and save stereo image pair
        
        Args:
            save_dir: Directory to save images
            prefix: Prefix for image filenames
            
        Returns:
            tuple: Paths to saved images (left_path, right_path)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        left_img, right_img = self.capture_stereo_pair()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        left_path = save_dir / f"{prefix}_left_{timestamp}.jpg"
        right_path = save_dir / f"{prefix}_right_{timestamp}.jpg"
        
        cv2.imwrite(str(left_path), left_img)
        cv2.imwrite(str(right_path), right_img)
        
        print(f"Saved stereo pair: {left_path.name}, {right_path.name}")
        
        return str(left_path), str(right_path)
    
    def capture_calibration_images(self, num_images=20, save_dir="calibration_images"):
        """
        Capture multiple stereo pairs for calibration
        
        Args:
            num_images: Number of stereo pairs to capture
            save_dir: Directory to save calibration images
        """
        print(f"\nCapturing {num_images} stereo pairs for calibration...")
        print("Move the chessboard to different positions and angles")
        print("Press Enter to capture each pair, or 'q' to quit early\n")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_images):
            input(f"Position {i+1}/{num_images} - Press Enter to capture (or Ctrl+C to stop): ")
            
            left_img, right_img = self.capture_stereo_pair()
            
            left_path = save_dir / f"left_{i:02d}.jpg"
            right_path = save_dir / f"right_{i:02d}.jpg"
            
            cv2.imwrite(str(left_path), left_img)
            cv2.imwrite(str(right_path), right_img)
            
            print(f"✓ Captured pair {i+1}/{num_images}")
        
        print(f"\nCalibration images saved to {save_dir}")
    
    def close(self):
        """Close all cameras"""
        for cam in self.cameras:
            if self.use_picamera:
                cam.stop()
                cam.close()
            else:
                cam.release()
        print("Cameras closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Example usage
    print("Dual Camera Capture Test")
    print("=" * 50)
    
    try:
        with DualCameraCapture(camera_indices=[0, 1], resolution=(1920, 1080)) as capture:
            # Test single capture
            print("\nTesting single capture...")
            left, right = capture.capture_stereo_pair()
            print(f"Left image shape: {left.shape}")
            print(f"Right image shape: {right.shape}")
            
            # Save test images
            capture.save_stereo_pair("test_images", prefix="test")
            
            print("\n" + "=" * 50)
            print("Test completed successfully!")
            
    except Exception as e:
        print(f"Error: {e}")

