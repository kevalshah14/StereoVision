"""
Camera Testing and Diagnostics Script
Quick utility to test camera functionality and alignment
"""

import cv2
import numpy as np
from camera_capture import DualCameraCapture
import time
import argparse


def test_camera_connection(camera_indices=[0, 1]):
    """Test if both cameras are accessible"""
    print("\n" + "=" * 70)
    print("CAMERA CONNECTION TEST")
    print("=" * 70)
    
    try:
        print("\nAttempting to initialize cameras...")
        with DualCameraCapture(camera_indices=camera_indices, resolution=(1920, 1080)) as capture:
            print("✓ Both cameras initialized successfully")
            
            # Capture test images
            print("\nCapturing test images...")
            left, right = capture.capture_stereo_pair()
            
            print(f"✓ Left image: {left.shape}")
            print(f"✓ Right image: {right.shape}")
            
            # Save test images
            cv2.imwrite("test_left.jpg", left)
            cv2.imwrite("test_right.jpg", right)
            print("\n✓ Test images saved: test_left.jpg, test_right.jpg")
            
            return True
            
    except Exception as e:
        print(f"\n✗ Camera test failed: {e}")
        return False


def test_camera_alignment(camera_indices=[0, 1], num_samples=5):
    """Test camera alignment by comparing images"""
    print("\n" + "=" * 70)
    print("CAMERA ALIGNMENT TEST")
    print("=" * 70)
    print("\nThis test captures multiple pairs and checks for alignment issues")
    
    try:
        with DualCameraCapture(camera_indices=camera_indices, resolution=(1920, 1080)) as capture:
            print(f"\nCapturing {num_samples} stereo pairs...")
            
            brightness_diffs = []
            
            for i in range(num_samples):
                left, right = capture.capture_stereo_pair()
                
                # Calculate brightness difference
                left_brightness = np.mean(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
                right_brightness = np.mean(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
                diff = abs(left_brightness - right_brightness)
                brightness_diffs.append(diff)
                
                print(f"  Pair {i+1}: Brightness diff = {diff:.2f}")
                time.sleep(0.5)
            
            avg_diff = np.mean(brightness_diffs)
            
            print(f"\n📊 Average brightness difference: {avg_diff:.2f}")
            
            if avg_diff < 10:
                print("✓ Excellent alignment - brightness is consistent")
            elif avg_diff < 30:
                print("⚠ Good alignment - minor brightness differences")
            else:
                print("⚠ Warning - significant brightness difference detected")
                print("  Consider adjusting camera settings or lighting")
            
            return True
            
    except Exception as e:
        print(f"\n✗ Alignment test failed: {e}")
        return False


def capture_side_by_side(camera_indices=[0, 1], output_file="side_by_side.jpg"):
    """Capture and create side-by-side comparison image"""
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE CAPTURE")
    print("=" * 70)
    
    try:
        with DualCameraCapture(camera_indices=camera_indices, resolution=(1920, 1080)) as capture:
            print("\nCapturing stereo pair...")
            left, right = capture.capture_stereo_pair()
            
            # Resize for easier viewing
            scale = 0.5
            new_size = (int(left.shape[1] * scale), int(left.shape[0] * scale))
            left_small = cv2.resize(left, new_size)
            right_small = cv2.resize(right, new_size)
            
            # Create side-by-side image
            side_by_side = np.hstack([left_small, right_small])
            
            # Add labels
            cv2.putText(side_by_side, "LEFT", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(side_by_side, "RIGHT", (left_small.shape[1] + 20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Draw epipolar lines for reference (horizontal)
            h = side_by_side.shape[0]
            for y in range(0, h, h//10):
                cv2.line(side_by_side, (0, y), (side_by_side.shape[1], y), 
                        (0, 255, 255), 1)
            
            cv2.imwrite(output_file, side_by_side)
            print(f"\n✓ Side-by-side image saved: {output_file}")
            print("  Yellow lines show where corresponding points should align")
            
            return True
            
    except Exception as e:
        print(f"\n✗ Capture failed: {e}")
        return False


def continuous_capture(camera_indices=[0, 1], display_time=10):
    """Continuous capture for visual inspection"""
    print("\n" + "=" * 70)
    print("CONTINUOUS CAPTURE MODE")
    print("=" * 70)
    print(f"\nCapturing for {display_time} seconds...")
    print("Check that both cameras show the same scene")
    
    try:
        with DualCameraCapture(camera_indices=camera_indices, resolution=(1920, 1080)) as capture:
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < display_time:
                left, right = capture.capture_stereo_pair()
                frame_count += 1
                
                # Save last pair
                if time.time() - start_time >= display_time - 0.5:
                    cv2.imwrite("continuous_left.jpg", left)
                    cv2.imwrite("continuous_right.jpg", right)
                
                time.sleep(0.5)
            
            fps = frame_count / display_time
            print(f"\n✓ Captured {frame_count} frames ({fps:.1f} fps)")
            print("✓ Last pair saved: continuous_left.jpg, continuous_right.jpg")
            
            return True
            
    except Exception as e:
        print(f"\n✗ Continuous capture failed: {e}")
        return False


def generate_calibration_pattern_info():
    """Display information about calibration patterns"""
    print("\n" + "=" * 70)
    print("CALIBRATION PATTERN INFORMATION")
    print("=" * 70)
    
    print("""
📋 CHESSBOARD PATTERN

A chessboard pattern is used for camera calibration. You need to count the
INTERNAL CORNERS (where 4 squares meet), not the squares themselves.

Example patterns:

  9x6 corners (10x7 squares):  ■ □ ■ □ ■ □ ■ □ ■ □
  Most common, good for most    □ ■ □ ■ □ ■ □ ■ □ ■
  applications                  ■ □ ■ □ ■ □ ■ □ ■ □
                                □ ■ □ ■ □ ■ □ ■ □ ■
                                ■ □ ■ □ ■ □ ■ □ ■ □
                                □ ■ □ ■ □ ■ □ ■ □ ■
                                ■ □ ■ □ ■ □ ■ □ ■ □

  7x5 corners (8x6 squares):   ■ □ ■ □ ■ □ ■ □
  Smaller, easier to print     □ ■ □ ■ □ ■ □ ■
                               ■ □ ■ □ ■ □ ■ □
                               □ ■ □ ■ □ ■ □ ■
                               ■ □ ■ □ ■ □ ■ □
                               □ ■ □ ■ □ ■ □ ■

📥 WHERE TO GET PATTERNS:

1. Print your own:
   https://markhedleyjones.com/projects/calibration-checkerboard-collection
   
2. Online pattern generator:
   https://calib.io/pages/camera-calibration-pattern-generator
   
3. Buy pre-made boards:
   Search for "camera calibration chessboard" on Amazon or similar

📏 PRINTING TIPS:

- Print on rigid material (mount on cardboard/foamboard)
- Ensure pattern is FLAT (no warping or bending)
- Measure the actual square size after printing
- Use high-quality printer for sharp edges
- Keep pattern clean and undamaged

✅ CHECKLIST:

□ Pattern is flat and rigid
□ Squares are perfectly square (not rectangular)
□ Black and white contrast is clear
□ Pattern size is measured accurately
□ You know the internal corner count
□ Entire pattern fits in both camera views
    """)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test and diagnose stereo camera setup",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--test', choices=['connection', 'alignment', 'side-by-side', 
                                           'continuous', 'all'],
                       default='all', help='Test to run (default: all)')
    parser.add_argument('--cameras', nargs=2, type=int, default=[0, 1],
                       metavar=('LEFT', 'RIGHT'),
                       help='Camera indices (default: 0 1)')
    parser.add_argument('--pattern-info', action='store_true',
                       help='Show calibration pattern information')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("STEREO CAMERA TEST UTILITY")
    print("=" * 70)
    print(f"\nUsing cameras: {args.cameras[0]} (left) and {args.cameras[1]} (right)")
    
    if args.pattern_info:
        generate_calibration_pattern_info()
        return
    
    # Run tests
    results = {}
    
    if args.test in ['connection', 'all']:
        results['connection'] = test_camera_connection(args.cameras)
    
    if args.test in ['alignment', 'all']:
        results['alignment'] = test_camera_alignment(args.cameras)
    
    if args.test in ['side-by-side', 'all']:
        results['side-by-side'] = capture_side_by_side(args.cameras)
    
    if args.test in ['continuous']:
        results['continuous'] = continuous_capture(args.cameras)
    
    # Summary
    if results:
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {test_name.upper()}: {status}")
        print("=" * 70)
        
        if all(results.values()):
            print("\n🎉 All tests passed! Your stereo camera setup is ready.")
            print("\nNext steps:")
            print("  1. Review side_by_side.jpg to check camera alignment")
            print("  2. Run calibration: python stereo_vision_pipeline.py --full-pipeline")
        else:
            print("\n⚠ Some tests failed. Please check your camera connections and setup.")


if __name__ == "__main__":
    main()

