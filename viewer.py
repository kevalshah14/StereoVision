#!/usr/bin/env python3
"""
Unified Stereo Vision Viewer
View depth maps, disparity maps, and 3D point clouds from your stereo camera system
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse


def load_stereo_results(output_dir):
    """Load stereo processing results"""
    output_path = Path(output_dir)
    
    depth_files = list(output_path.glob("**/stereo_depth.npy")) + list(output_path.glob("**/depth_map.npy"))
    disparity_files = list(output_path.glob("**/stereo_disparity.npy")) + list(output_path.glob("**/disparity.npy"))
    left_files = list(output_path.glob("**/stereo_left_rectified.jpg"))
    
    if not depth_files:
        raise FileNotFoundError(f"No depth map found in {output_dir}")
    
    depth_map = np.load(depth_files[0])
    disparity = np.load(disparity_files[0]) if disparity_files else None
    
    left_img = None
    if left_files:
        left_img = cv2.imread(str(left_files[0]))
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    
    return depth_map, disparity, left_img


def create_point_cloud(depth_map, color_image=None, sample_rate=5, min_depth=100, max_depth=10000):
    """
    Convert depth map to 3D point cloud with colors
    
    Args:
        depth_map: Depth values in mm
        color_image: RGB image for point colors
        sample_rate: Sample every Nth pixel
        min_depth: Minimum valid depth in mm
        max_depth: Maximum valid depth in mm
    
    Returns:
        points (Nx3), colors (Nx3)
    """
    height, width = depth_map.shape
    
    v_coords, u_coords = np.mgrid[0:height:sample_rate, 0:width:sample_rate]
    depth_sampled = depth_map[::sample_rate, ::sample_rate]
    
    valid_mask = (depth_sampled > min_depth) & (depth_sampled < max_depth)
    
    u_valid = u_coords[valid_mask]
    v_valid = v_coords[valid_mask]
    depth_valid = depth_sampled[valid_mask]
    
    x = (u_valid - width/2) * depth_valid / 2000.0
    y = (height/2 - v_valid) * depth_valid / 2000.0
    z = depth_valid
    
    points = np.stack([x, y, z], axis=1)
    
    if color_image is not None and len(color_image.shape) == 3:
        colors = color_image[::sample_rate, ::sample_rate][valid_mask].astype(np.float32) / 255.0
    else:
        colors = plt.cm.turbo((z - z.min()) / (z.max() - z.min() + 1e-6))[:, :3]
    
    return points, colors


def view_results(output_dir, view_mode='all', sample_rate=8, min_depth=100, max_depth=10000):
    """
    Display stereo vision results in 2D and 3D
    
    Args:
        output_dir: Directory containing results
        view_mode: 'all', '2d', or '3d'
        sample_rate: Point cloud sampling rate
        min_depth: Minimum depth for visualization
        max_depth: Maximum depth for visualization
    """
    print("\n" + "=" * 70)
    print("STEREO VISION VIEWER")
    print("=" * 70)
    
    print(f"\n📂 Loading results from: {output_dir}")
    try:
        depth_map, disparity, left_img = load_stereo_results(output_dir)
        print(f"  ✓ Loaded depth map: {depth_map.shape}")
        if disparity is not None:
            print(f"  ✓ Loaded disparity map: {disparity.shape}")
        if left_img is not None:
            print(f"  ✓ Loaded original image: {left_img.shape}")
    except Exception as e:
        print(f"✗ Error loading results: {e}")
        return
    
    valid_depth = depth_map[(depth_map > min_depth) & (depth_map < max_depth)]
    
    if len(valid_depth) == 0:
        print(f"✗ No valid depth values found in range {min_depth}-{max_depth}mm")
        return
    
    print(f"\n📊 Depth Statistics:")
    print(f"  Min: {valid_depth.min():.0f} mm ({valid_depth.min()/1000:.2f} m)")
    print(f"  Max: {valid_depth.max():.0f} mm ({valid_depth.max()/1000:.2f} m)")
    print(f"  Mean: {valid_depth.mean():.0f} mm ({valid_depth.mean()/1000:.2f} m)")
    print(f"  Valid pixels: {len(valid_depth):,} / {depth_map.size:,} ({100*len(valid_depth)/depth_map.size:.1f}%)")
    
    if view_mode in ['all', '2d']:
        print("\n🖼️  Opening 2D visualization...")
        plot_2d_view(depth_map, disparity, left_img, valid_depth)
    
    if view_mode in ['all', '3d']:
        print(f"\n☁️  Generating 3D point cloud (sample rate: {sample_rate})...")
        points, colors = create_point_cloud(depth_map, left_img, sample_rate, min_depth, max_depth)
        print(f"  ✓ Generated {len(points):,} 3D points")
        print("\n🎨 Opening 3D visualization...")
        plot_3d_view(points, colors)


def plot_2d_view(depth_map, disparity, left_img, valid_depth):
    """Display 2D depth and disparity maps"""
    
    n_plots = sum([left_img is not None, disparity is not None, True])
    
    fig = plt.figure(figsize=(16, 5 * ((n_plots + 1) // 2)))
    fig.suptitle('Stereo Vision Results', fontsize=16, fontweight='bold')
    
    plot_idx = 1
    
    if left_img is not None:
        ax = plt.subplot(n_plots, 2, plot_idx)
        ax.imshow(left_img)
        ax.set_title('Original Image (Left Camera)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plot_idx += 1
    
    if disparity is not None:
        ax = plt.subplot(n_plots, 2, plot_idx)
        valid_disp = disparity[disparity > 0]
        if len(valid_disp) > 0:
            vmin, vmax = np.percentile(valid_disp, [1, 99])
            im = ax.imshow(disparity, cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title('Disparity Map (Closer = Warmer Colors)', fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, label='Disparity (pixels)', fraction=0.046)
        plot_idx += 1
    
    ax = plt.subplot(n_plots, 2, plot_idx)
    vmin, vmax = np.percentile(valid_depth, [1, 99])
    im = ax.imshow(depth_map, cmap='turbo', vmin=vmin, vmax=vmax)
    ax.set_title('Depth Map (Blue = Close, Red = Far)', fontsize=12, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, label='Depth (mm)', fraction=0.046)
    
    stats_text = f"""
    Depth Range: {valid_depth.min():.0f} - {valid_depth.max():.0f} mm
    Mean Depth: {valid_depth.mean():.0f} mm ({valid_depth.mean()/1000:.2f} m)
    Valid Coverage: {100*len(valid_depth)/depth_map.size:.1f}%
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def plot_3d_view(points, colors):
    """Display interactive 3D point cloud"""
    
    max_points = 50000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
        print(f"  → Downsampled to {max_points:,} points for interactive viewing")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=1,
        alpha=0.6,
        edgecolors='none'
    )
    
    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    ax.set_zlabel('Depth (mm)', fontsize=10)
    ax.set_title(f'3D Point Cloud ({len(points):,} points)\nDrag to rotate | Right-click to zoom',
                 fontsize=12, fontweight='bold', pad=20)
    
    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.3)
    
    controls_text = """
    CONTROLS:
    • Left-click + drag: Rotate
    • Right-click + drag: Zoom
    • Scroll wheel: Zoom in/out
    """
    
    fig.text(0.02, 0.98, controls_text, fontsize=9, family='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Unified Stereo Vision Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all results (2D + 3D)
  python viewer.py --dir stereo_output
  
  # View only 2D (depth and disparity maps)
  python viewer.py --dir stereo_output --mode 2d
  
  # View only 3D point cloud
  python viewer.py --dir stereo_output --mode 3d
  
  # Adjust sampling and depth range
  python viewer.py --dir stereo_output --sample 5 --max-depth 5000
        """
    )
    
    parser.add_argument('--dir', type=str, default='stereo_output',
                       help='Directory containing stereo results (default: stereo_output)')
    parser.add_argument('--mode', choices=['all', '2d', '3d'], default='all',
                       help='Viewing mode: all, 2d (maps only), or 3d (point cloud only)')
    parser.add_argument('--sample', type=int, default=8,
                       help='Point cloud sampling rate - lower = more points (default: 8)')
    parser.add_argument('--min-depth', type=float, default=100,
                       help='Minimum depth in mm (default: 100)')
    parser.add_argument('--max-depth', type=float, default=10000,
                       help='Maximum depth in mm (default: 10000)')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not Path(args.dir).exists():
        print(f"✗ Directory not found: {args.dir}")
        print("\nPlease run the stereo vision pipeline first:")
        print("  python main.py --process-live")
        print("  or")
        print("  python main.py --process-files left.jpg right.jpg")
        return
    
    # View results
    view_results(
        args.dir,
        view_mode=args.mode,
        sample_rate=args.sample,
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

