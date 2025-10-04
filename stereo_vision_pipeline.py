"""
Complete Stereo Vision Pipeline
Integrated script for capturing stereo images and generating point clouds
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys

from camera_capture import DualCameraCapture
from stereo_calibration import StereoCalibrator
from stereo_matcher import StereoMatcher
from point_cloud_generator import PointCloudGenerator


class StereoVisionPipeline:
    """Complete stereo vision pipeline for point cloud generation"""
    
    def __init__(self, calibration_file="stereo_calibration.pkl"):
        """
        Initialize stereo vision pipeline
        
        Args:
            calibration_file: Path to calibration parameters file
        """
        self.calibration_file = calibration_file
        self.calibrated = Path(calibration_file).exists()
        
        if self.calibrated:
            print("✓ Found calibration file")
            self.matcher = StereoMatcher(calibration_file, algorithm="sgbm")
            self.point_cloud_gen = PointCloudGenerator(calibration_file)
        else:
            print("⚠ No calibration file found - calibration required")
            self.matcher = None
            self.point_cloud_gen = None
    
    def capture_calibration_images(self, num_images=20, output_dir="calibration_images"):
        """
        Capture images for calibration using dual cameras
        
        Args:
            num_images: Number of stereo pairs to capture
            output_dir: Directory to save calibration images
        """
        print("\n" + "=" * 70)
        print("STEP 1: CAPTURE CALIBRATION IMAGES")
        print("=" * 70)
        print("\nYou will need a chessboard calibration pattern.")
        print("Default pattern: 9x6 internal corners, 25mm squares")
        print("\nInstructions:")
        print("  1. Print a chessboard pattern or use a calibration board")
        print("  2. Position the board in front of both cameras")
        print("  3. Capture images at different angles and distances")
        print("  4. Ensure the entire board is visible in both cameras")
        print("  5. Cover different areas of the camera view")
        print("\n")
        
        try:
            with DualCameraCapture(camera_indices=[0, 1], resolution=(1920, 1080)) as capture:
                capture.capture_calibration_images(num_images, output_dir)
            print("\n✓ Calibration images captured successfully!")
            
        except Exception as e:
            print(f"\n✗ Error capturing images: {e}")
            raise
    
    def calibrate_cameras(self, calibration_dir="calibration_images", 
                         chessboard_size=(9, 6), square_size=25.0):
        """
        Perform stereo camera calibration
        
        Args:
            calibration_dir: Directory containing calibration images
            chessboard_size: Tuple of (columns, rows) of internal corners
            square_size: Size of chessboard square in mm
        """
        print("\n" + "=" * 70)
        print("STEP 2: CALIBRATE STEREO CAMERAS")
        print("=" * 70)
        print(f"\nChessboard configuration:")
        print(f"  Internal corners: {chessboard_size[0]} x {chessboard_size[1]}")
        print(f"  Square size: {square_size} mm")
        print("\n")
        
        try:
            calibrator = StereoCalibrator(chessboard_size, square_size)
            calibrator.calibrate_stereo(
                calibration_dir, 
                calibration_dir, 
                self.calibration_file
            )
            
            # Reinitialize matcher and point cloud generator
            self.matcher = StereoMatcher(self.calibration_file, algorithm="sgbm")
            self.point_cloud_gen = PointCloudGenerator(self.calibration_file)
            self.calibrated = True
            
            print("\n✓ Camera calibration completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Calibration failed: {e}")
            raise
    
    def capture_and_process_live(self, output_dir="stereo_output", visualize=True):
        """
        Capture live stereo images and generate point cloud
        
        Args:
            output_dir: Directory to save outputs
            visualize: Whether to visualize the point cloud
        """
        if not self.calibrated:
            raise RuntimeError("Cameras not calibrated. Run calibration first.")
        
        print("\n" + "=" * 70)
        print("STEP 3: CAPTURE AND PROCESS STEREO IMAGES")
        print("=" * 70)
        print("\nCapturing live stereo pair...")
        
        try:
            # Capture stereo pair
            with DualCameraCapture(camera_indices=[0, 1], resolution=(1920, 1080)) as capture:
                left_img, right_img = capture.capture_stereo_pair()
                print("✓ Stereo pair captured")
            
            # Process stereo pair
            self._process_stereo_pair(left_img, right_img, output_dir, visualize)
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            raise
    
    def process_from_files(self, left_image_path, right_image_path, 
                          output_dir="stereo_output", visualize=True):
        """
        Process stereo images from files
        
        Args:
            left_image_path: Path to left camera image
            right_image_path: Path to right camera image
            output_dir: Directory to save outputs
            visualize: Whether to visualize the point cloud
        """
        if not self.calibrated:
            raise RuntimeError("Cameras not calibrated. Run calibration first.")
        
        print("\n" + "=" * 70)
        print("PROCESS STEREO IMAGES FROM FILES")
        print("=" * 70)
        
        # Load images
        left_img = cv2.imread(left_image_path)
        right_img = cv2.imread(right_image_path)
        
        if left_img is None or right_img is None:
            raise ValueError("Could not load one or both images")
        
        print(f"\n✓ Loaded images:")
        print(f"  Left: {left_image_path}")
        print(f"  Right: {right_image_path}")
        
        # Process stereo pair
        self._process_stereo_pair(left_img, right_img, output_dir, visualize)
    
    def _process_stereo_pair(self, left_img, right_img, output_dir, visualize):
        """Internal method to process a stereo pair"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nProcessing stereo pair...")
        
        # Compute disparity
        print("  → Computing disparity map...")
        results = self.matcher.process_stereo_pair(
            left_img, right_img, 
            save_dir=output_dir, 
            prefix="stereo"
        )
        print("  ✓ Disparity map computed")
        
        # Generate point cloud
        print("  → Generating point cloud...")
        pcd = self.point_cloud_gen.process_stereo_to_point_cloud(
            results['disparity'],
            left_img,
            filter_cloud=True,
            estimate_normals=True,
            output_file=output_dir / "point_cloud.ply",
            visualize=visualize
        )
        
        print(f"\n✓ Processing complete! Results saved to {output_dir}")
        print(f"  - Rectified images")
        print(f"  - Disparity map")
        print(f"  - Point cloud (PLY format)")
        print(f"  - {len(pcd.points)} 3D points generated")


def main():
    """Main entry point for the stereo vision pipeline"""
    parser = argparse.ArgumentParser(
        description="Stereo Vision Pipeline for Point Cloud Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (capture calibration, calibrate, and process)
  python stereo_vision_pipeline.py --full-pipeline
  
  # Capture calibration images only
  python stereo_vision_pipeline.py --capture-calibration --num-images 25
  
  # Calibrate from existing images
  python stereo_vision_pipeline.py --calibrate --calibration-dir calibration_images
  
  # Process live capture
  python stereo_vision_pipeline.py --process-live
  
  # Process existing images
  python stereo_vision_pipeline.py --process-files left.jpg right.jpg
        """
    )
    
    # Operation modes
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run complete pipeline: capture, calibrate, and process')
    parser.add_argument('--capture-calibration', action='store_true',
                       help='Capture calibration images')
    parser.add_argument('--calibrate', action='store_true',
                       help='Calibrate cameras from existing images')
    parser.add_argument('--process-live', action='store_true',
                       help='Capture and process live stereo pair')
    parser.add_argument('--process-files', nargs=2, metavar=('LEFT', 'RIGHT'),
                       help='Process stereo images from files')
    
    # Configuration options
    parser.add_argument('--num-images', type=int, default=20,
                       help='Number of calibration image pairs (default: 20)')
    parser.add_argument('--calibration-dir', default='calibration_images',
                       help='Directory for calibration images')
    parser.add_argument('--calibration-file', default='stereo_calibration.pkl',
                       help='Calibration parameters file')
    parser.add_argument('--output-dir', default='stereo_output',
                       help='Output directory for results')
    parser.add_argument('--chessboard-size', nargs=2, type=int, default=[9, 6],
                       metavar=('COLS', 'ROWS'),
                       help='Chessboard internal corners (default: 9 6)')
    parser.add_argument('--square-size', type=float, default=25.0,
                       help='Chessboard square size in mm (default: 25.0)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable point cloud visualization')
    
    args = parser.parse_args()
    
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    # Initialize pipeline
    pipeline = StereoVisionPipeline(args.calibration_file)
    
    try:
        # Full pipeline
        if args.full_pipeline:
            print("\n🚀 RUNNING FULL STEREO VISION PIPELINE")
            print("=" * 70)
            
            # Step 1: Capture calibration images
            pipeline.capture_calibration_images(args.num_images, args.calibration_dir)
            
            # Step 2: Calibrate cameras
            pipeline.calibrate_cameras(
                args.calibration_dir,
                tuple(args.chessboard_size),
                args.square_size
            )
            
            # Step 3: Process live capture
            pipeline.capture_and_process_live(args.output_dir, not args.no_visualize)
            
            print("\n" + "=" * 70)
            print("✓ FULL PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 70)
        
        # Individual operations
        else:
            if args.capture_calibration:
                pipeline.capture_calibration_images(args.num_images, args.calibration_dir)
            
            if args.calibrate:
                pipeline.calibrate_cameras(
                    args.calibration_dir,
                    tuple(args.chessboard_size),
                    args.square_size
                )
            
            if args.process_live:
                pipeline.capture_and_process_live(args.output_dir, not args.no_visualize)
            
            if args.process_files:
                pipeline.process_from_files(
                    args.process_files[0],
                    args.process_files[1],
                    args.output_dir,
                    not args.no_visualize
                )
    
    except KeyboardInterrupt:
        print("\n\n⚠ Operation cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

