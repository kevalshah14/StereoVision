#!/usr/bin/env python3
"""
Process MiddEval3 Stereo Dataset with Stereo Vision Pipeline
Converts MiddEval3 calibration format and generates depth maps
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
import argparse
import re

from stereo_matcher import StereoMatcher
from point_cloud_generator import PointCloudGenerator


def parse_middlebury_calib(calib_file):
    """
    Parse Middlebury calibration file format
    
    Args:
        calib_file: Path to calib.txt file
        
    Returns:
        dict: Calibration parameters
    """
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    calib = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith('cam0='):
            matrix_str = line.split('=')[1].strip('[]')
            values = [float(x) for x in re.findall(r'[\d.]+', matrix_str)]
            calib['cam0'] = np.array([
                [values[0], 0, values[1]],
                [0, values[2], values[3]],
                [0, 0, 1]
            ])
        elif line.startswith('cam1='):
            matrix_str = line.split('=')[1].strip('[]')
            values = [float(x) for x in re.findall(r'[\d.]+', matrix_str)]
            calib['cam1'] = np.array([
                [values[0], 0, values[1]],
                [0, values[2], values[3]],
                [0, 0, 1]
            ])
        elif line.startswith('doffs='):
            calib['doffs'] = float(line.split('=')[1])
        elif line.startswith('baseline='):
            calib['baseline'] = float(line.split('=')[1])
        elif line.startswith('width='):
            calib['width'] = int(line.split('=')[1])
        elif line.startswith('height='):
            calib['height'] = int(line.split('=')[1])
        elif line.startswith('ndisp='):
            calib['ndisp'] = int(line.split('=')[1])
    
    return calib


def create_stereo_calibration_from_middlebury(calib_data, image_shape):
    """
    Convert Middlebury calibration to stereo calibration format
    
    Args:
        calib_data: Parsed Middlebury calibration
        image_shape: (height, width) of images
        
    Returns:
        dict: Calibration parameters
    """
    height, width = image_shape[:2]
    
    K1 = calib_data['cam0']
    K2 = calib_data['cam1']
    D1 = np.zeros(5)
    D2 = np.zeros(5)
    R = np.eye(3)
    T = np.array([[calib_data['baseline']], [0], [0]])
    E = np.zeros((3, 3))
    F = np.zeros((3, 3))
    R1 = np.eye(3)
    R2 = np.eye(3)
    P1 = np.hstack([K1, np.zeros((3, 1))])
    P2 = np.hstack([K2, np.array([[-calib_data['baseline'] * K2[0, 0]], [0], [0]])])
    
    cx1 = K1[0, 2]
    cy = K1[1, 2]
    fx = K1[0, 0]
    baseline = calib_data['baseline']
    cx2 = K2[0, 2]
    
    Q = np.array([
        [1, 0, 0, -cx1],
        [0, 1, 0, -cy],
        [0, 0, 0, fx],
        [0, 0, -1.0/baseline, (cx1 - cx2)/baseline]
    ])
    
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K1, D1, R1, K1, (width, height), cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K2, D2, R2, K2, (width, height), cv2.CV_32FC1
    )
    
    calibration = {
        'camera_matrix_left': K1,
        'dist_coeffs_left': D1,
        'camera_matrix_right': K2,
        'dist_coeffs_right': D2,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'R1': R1,
        'R2': R2,
        'P1': P1,
        'P2': P2,
        'Q': Q,
        'roi_left': (0, 0, width, height),
        'roi_right': (0, 0, width, height),
        'map_left_x': map_left_x,
        'map_left_y': map_left_y,
        'map_right_x': map_right_x,
        'map_right_y': map_right_y,
        'image_size': (width, height),
        'baseline_mm': baseline,
        'focal_length': fx
    }
    
    return calibration


def process_middlebury_scene(scene_path, output_dir, algorithm='sgbm', visualize=True, custom_baseline=None):
    """
    Process a single Middlebury scene
    
    Args:
        scene_path: Path to scene directory
        output_dir: Output directory for results
        algorithm: Stereo matching algorithm
        visualize: Display point cloud
        custom_baseline: Override baseline value
    """
    scene_path = Path(scene_path)
    scene_name = scene_path.name
    
    print("\n" + "=" * 70)
    print(f"PROCESSING: {scene_name}")
    print("=" * 70)
    
    left_img_path = scene_path / "im0.png"
    right_img_path = scene_path / "im1.png"
    calib_path = scene_path / "calib.txt"
    
    if not all([left_img_path.exists(), right_img_path.exists(), calib_path.exists()]):
        print(f"✗ Missing required files in {scene_path}")
        return False
    
    print(f"\n📷 Loading images...")
    left_img = cv2.imread(str(left_img_path))
    right_img = cv2.imread(str(right_img_path))
    
    if left_img is None or right_img is None:
        print(f"✗ Failed to load images")
        return False
    
    print(f"  ✓ Left image: {left_img.shape}")
    print(f"  ✓ Right image: {right_img.shape}")
    
    # Parse calibration
    print(f"\n📐 Parsing calibration...")
    calib_data = parse_middlebury_calib(calib_path)
    
    if custom_baseline is not None:
        print(f"  → Overriding baseline: {calib_data['baseline']:.2f} → {custom_baseline:.2f} mm")
        calib_data['baseline'] = custom_baseline
    
    print(f"  ✓ Baseline: {calib_data['baseline']:.2f} mm")
    print(f"  ✓ Focal length: {calib_data['cam0'][0, 0]:.2f} pixels")
    print(f"  ✓ Max disparity: {calib_data.get('ndisp', 'N/A')}")
    
    calibration = create_stereo_calibration_from_middlebury(calib_data, left_img.shape)
    
    temp_calib_file = f"temp_calib_{scene_name}.pkl"
    with open(temp_calib_file, 'wb') as f:
        pickle.dump(calibration, f)
    
    try:
        print(f"\n🔍 Initializing {algorithm.upper()} stereo matcher...")
        matcher = StereoMatcher(temp_calib_file, algorithm=algorithm)
        
        if 'ndisp' in calib_data:
            num_disp = ((calib_data['ndisp'] + 15) // 16) * 16
            matcher.matcher.setNumDisparities(num_disp)
            print(f"  ✓ Set num disparities to {num_disp}")
        
        if algorithm == 'sgbm':
            matcher.matcher.setBlockSize(3)
            matcher.matcher.setUniquenessRatio(5)
            matcher.matcher.setSpeckleWindowSize(50)
            matcher.matcher.setSpeckleRange(16)
            matcher.matcher.setDisp12MaxDiff(2)
            print(f"  ✓ Applied high-detail SGBM parameters")
        
        print(f"\n⚙️  Computing disparity map...")
        results = matcher.process_stereo_pair(
            left_img, 
            right_img,
            save_dir=None,
            prefix=scene_name
        )
        
        print(f"  → Recalculating depth map for MiddEval3...")
        focal_length = calibration['focal_length']
        baseline_mm = calibration['baseline_mm']
        disparity = results['disparity']
        
        disparity_safe = np.where(disparity > 0.5, disparity, 0.5)
        depth_map_corrected = (focal_length * baseline_mm) / disparity_safe
        results['depth_map'] = depth_map_corrected
        
        scene_output_dir = Path(output_dir) / scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving results to {scene_output_dir}...")
        cv2.imwrite(str(scene_output_dir / "disparity_visualization.png"), 
                   results['disparity_visualization'])
        np.save(str(scene_output_dir / "disparity.npy"), results['disparity'])
        np.save(str(scene_output_dir / "depth_map.npy"), depth_map_corrected)
        
        disparity = results['disparity']
        valid_disparity = disparity[disparity > 0]
        if len(valid_disparity) > 0:
            print(f"  ✓ Disparity range: {valid_disparity.min():.1f} - {valid_disparity.max():.1f}")
            print(f"  ✓ Valid pixels: {len(valid_disparity)} / {disparity.size} ({100*len(valid_disparity)/disparity.size:.1f}%)")
        
        if visualize:
            print(f"\n☁️  Generating point cloud...")
            point_cloud_gen = PointCloudGenerator(temp_calib_file)
            pcd = point_cloud_gen.process_stereo_to_point_cloud(
                results['disparity'],
                left_img,
                filter_cloud=True,
                estimate_normals=True,
                output_file=scene_output_dir / "point_cloud.ply",
                visualize=True
            )
            print(f"  ✓ Point cloud saved: {len(pcd.points)} points")
        
        print(f"\n✓ Successfully processed {scene_name}!")
        return True
        
    finally:
        if Path(temp_calib_file).exists():
            Path(temp_calib_file).unlink()


def list_available_scenes(middeval3_dir="MiddEval3"):
    """List all available scenes in MiddEval3 dataset"""
    middeval3_path = Path(middeval3_dir)
    
    scenes = {
        'training': [],
        'test': []
    }
    
    # Training scenes
    training_dir = middeval3_path / "trainingF"
    if training_dir.exists():
        scenes['training'] = sorted([d.name for d in training_dir.iterdir() 
                                     if d.is_dir() and (d / "im0.png").exists()])
    
    # Test scenes
    test_dir = middeval3_path / "testF"
    if test_dir.exists():
        scenes['test'] = sorted([d.name for d in test_dir.iterdir() 
                                if d.is_dir() and (d / "im0.png").exists()])
    
    return scenes


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process MiddEval3 Stereo Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available scenes
  python process_middeval3.py --list
  
  # Process a specific scene
  python process_middeval3.py --scene Motorcycle --set training
  
  # Process all test scenes
  python process_middeval3.py --all --set test
  
  # Process with different algorithm
  python process_middeval3.py --scene Piano --algorithm bm --no-visualize
        """
    )
    
    parser.add_argument('--list', action='store_true',
                       help='List all available scenes')
    parser.add_argument('--scene', type=str,
                       help='Scene name to process (e.g., Motorcycle, Piano)')
    parser.add_argument('--set', choices=['training', 'test'], default='training',
                       help='Dataset to use (training or test)')
    parser.add_argument('--all', action='store_true',
                       help='Process all scenes in the specified set')
    parser.add_argument('--algorithm', choices=['bm', 'sgbm'], default='sgbm',
                       help='Stereo matching algorithm (default: sgbm)')
    parser.add_argument('--output-dir', default='middeval3_output',
                       help='Output directory (default: middeval3_output)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable point cloud visualization')
    parser.add_argument('--middeval3-dir', default='MiddEval3',
                       help='Path to MiddEval3 dataset directory')
    parser.add_argument('--baseline', type=float,
                       help='Override baseline in mm (e.g., --baseline 35 for smartphone-like)')
    
    args = parser.parse_args()
    
    # List scenes if requested
    if args.list:
        print("\n" + "=" * 70)
        print("AVAILABLE MIDDEVAL3 SCENES")
        print("=" * 70)
        
        scenes = list_available_scenes(args.middeval3_dir)
        
        print(f"\n📚 TRAINING SET ({len(scenes['training'])} scenes):")
        for i, scene in enumerate(scenes['training'], 1):
            print(f"  {i:2d}. {scene}")
        
        print(f"\n🧪 TEST SET ({len(scenes['test'])} scenes):")
        for i, scene in enumerate(scenes['test'], 1):
            print(f"  {i:2d}. {scene}")
        
        print("\n" + "=" * 70)
        return
    
    # Get available scenes
    scenes = list_available_scenes(args.middeval3_dir)
    
    # Determine which scenes to process
    scenes_to_process = []
    
    if args.all:
        # Process all scenes in the specified set
        dataset_dir = "trainingF" if args.set == 'training' else "testF"
        scenes_to_process = [
            Path(args.middeval3_dir) / dataset_dir / scene 
            for scene in scenes[args.set]
        ]
        print(f"\n🚀 Processing all {len(scenes_to_process)} scenes from {args.set} set...")
        
    elif args.scene:
        # Process specific scene
        dataset_dir = "trainingF" if args.set == 'training' else "testF"
        scene_path = Path(args.middeval3_dir) / dataset_dir / args.scene
        
        if not scene_path.exists():
            print(f"✗ Scene '{args.scene}' not found in {args.set} set")
            print(f"\nAvailable {args.set} scenes:")
            for scene in scenes[args.set]:
                print(f"  - {scene}")
            return
        
        scenes_to_process = [scene_path]
    
    else:
        parser.print_help()
        return
    
    # Process scenes
    print("\n" + "=" * 70)
    print(f"MIDDEVAL3 STEREO PROCESSING")
    print("=" * 70)
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Output directory: {args.output_dir}")
    print(f"Visualization: {'Disabled' if args.no_visualize else 'Enabled'}")
    
    results = []
    for scene_path in scenes_to_process:
        try:
            success = process_middlebury_scene(
                scene_path,
                args.output_dir,
                algorithm=args.algorithm,
                visualize=not args.no_visualize,
                custom_baseline=args.baseline
            )
            results.append((scene_path.name, success))
        except Exception as e:
            print(f"\n✗ Error processing {scene_path.name}: {e}")
            results.append((scene_path.name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for scene_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {scene_name}: {status}")
    
    print(f"\nTotal: {successful}/{total} scenes processed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()

