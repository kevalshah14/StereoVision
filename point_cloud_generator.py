"""
Point Cloud Generation from Stereo Images
Converts disparity maps to 3D point clouds
"""

import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import pickle


class PointCloudGenerator:
    """Class to generate and process 3D point clouds from stereo data"""
    
    def __init__(self, calibration_file="stereo_calibration.pkl"):
        """
        Initialize point cloud generator
        
        Args:
            calibration_file: Path to calibration parameters file
        """
        self.calibration_file = calibration_file
        self.calibration = self._load_calibration()
        self.Q = self.calibration['Q']
        
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
    
    def disparity_to_point_cloud(self, disparity, left_image, min_disparity=1.0, max_depth=10000):
        """
        Convert disparity map to 3D point cloud
        
        Args:
            disparity: Disparity map
            left_image: Left camera image (for color)
            min_disparity: Minimum disparity threshold (filter out invalid points)
            max_depth: Maximum depth in mm (filter distant points)
            
        Returns:
            open3d.geometry.PointCloud: Generated point cloud
        """
        # Create mask for valid disparities
        mask = (disparity > min_disparity) & (disparity < 1000)
        
        # Reproject to 3D using Q matrix
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        
        # Extract valid points
        valid_points = points_3d[mask]
        
        # Filter by depth
        depth_mask = (valid_points[:, 2] > 0) & (valid_points[:, 2] < max_depth)
        valid_points = valid_points[depth_mask]
        
        # Get corresponding colors
        if len(left_image.shape) == 3:
            colors = left_image[mask][depth_mask]
            # Convert BGR to RGB and normalize
            colors = cv2.cvtColor(colors.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
        else:
            # Grayscale image
            gray_colors = left_image[mask][depth_mask]
            colors = np.stack([gray_colors] * 3, axis=-1) / 255.0
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        print(f"Generated point cloud with {len(valid_points)} points")
        
        return pcd
    
    def filter_point_cloud(self, pcd, voxel_size=None, nb_neighbors=20, std_ratio=2.0):
        """
        Filter and clean point cloud
        
        Args:
            pcd: Input point cloud
            voxel_size: Voxel size for downsampling (None to skip)
            nb_neighbors: Number of neighbors for statistical outlier removal
            std_ratio: Standard deviation ratio for outlier removal
            
        Returns:
            open3d.geometry.PointCloud: Filtered point cloud
        """
        pcd_filtered = pcd
        
        # Downsample if voxel size provided
        if voxel_size is not None:
            pcd_filtered = pcd_filtered.voxel_down_sample(voxel_size)
            print(f"Downsampled to {len(pcd_filtered.points)} points (voxel size: {voxel_size})")
        
        # Remove statistical outliers
        pcd_filtered, ind = pcd_filtered.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        print(f"Removed outliers: {len(pcd.points) - len(pcd_filtered.points)} points")
        
        return pcd_filtered
    
    def estimate_normals(self, pcd, search_param=None):
        """
        Estimate normals for point cloud
        
        Args:
            pcd: Input point cloud
            search_param: Search parameter for normal estimation
            
        Returns:
            open3d.geometry.PointCloud: Point cloud with normals
        """
        if search_param is None:
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
        
        pcd.estimate_normals(search_param=search_param)
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        print("Estimated normals")
        return pcd
    
    def save_point_cloud(self, pcd, filename, format='ply'):
        """
        Save point cloud to file
        
        Args:
            pcd: Point cloud to save
            filename: Output filename
            format: File format ('ply', 'pcd', 'xyz', 'xyzrgb', 'pts')
        """
        filename = Path(filename)
        
        # Ensure correct extension
        if not filename.suffix:
            filename = filename.with_suffix(f".{format}")
        
        # Save point cloud
        o3d.io.write_point_cloud(str(filename), pcd)
        print(f"Saved point cloud to {filename}")
    
    def visualize_point_cloud(self, pcd, window_name="Point Cloud Viewer"):
        """
        Visualize point cloud in interactive window
        
        Args:
            pcd: Point cloud to visualize
            window_name: Window title
        """
        print("\nVisualizing point cloud...")
        print("Controls:")
        print("  - Mouse: Rotate view")
        print("  - Scroll: Zoom")
        print("  - Ctrl + Mouse: Pan")
        print("  - Q or ESC: Close window")
        
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_name,
            width=1280,
            height=720,
            point_show_normal=False
        )
    
    def create_mesh_from_point_cloud(self, pcd, method='poisson', depth=9):
        """
        Create mesh from point cloud
        
        Args:
            pcd: Input point cloud (must have normals)
            method: Meshing method ('poisson' or 'ball_pivoting')
            depth: Depth parameter for Poisson reconstruction
            
        Returns:
            open3d.geometry.TriangleMesh: Generated mesh
        """
        if not pcd.has_normals():
            print("Computing normals for mesh generation...")
            self.estimate_normals(pcd)
        
        print(f"Creating mesh using {method} method...")
        
        if method == 'poisson':
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth
            )
            
            # Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
        elif method == 'ball_pivoting':
            # Estimate radius for ball pivoting
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist
            
            radii = [radius, radius * 2]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        
        return mesh
    
    def process_stereo_to_point_cloud(self, disparity, left_image, 
                                     filter_cloud=True, estimate_normals=False,
                                     output_file=None, visualize=False):
        """
        Complete pipeline from disparity to point cloud
        
        Args:
            disparity: Disparity map
            left_image: Left camera image
            filter_cloud: Whether to filter the point cloud
            estimate_normals: Whether to estimate normals
            output_file: Path to save point cloud (None to skip saving)
            visualize: Whether to visualize result
            
        Returns:
            open3d.geometry.PointCloud: Generated point cloud
        """
        print("\nGenerating point cloud from stereo data...")
        print("=" * 60)
        
        # Generate point cloud
        pcd = self.disparity_to_point_cloud(disparity, left_image)
        
        # Filter if requested
        if filter_cloud:
            print("\nFiltering point cloud...")
            pcd = self.filter_point_cloud(pcd, voxel_size=5, nb_neighbors=20, std_ratio=2.0)
        
        # Estimate normals if requested
        if estimate_normals:
            print("\nEstimating normals...")
            pcd = self.estimate_normals(pcd)
        
        # Save if requested
        if output_file is not None:
            print("\nSaving point cloud...")
            self.save_point_cloud(pcd, output_file)
        
        # Visualize if requested
        if visualize:
            self.visualize_point_cloud(pcd)
        
        print("\n✓ Point cloud generation completed!")
        print("=" * 60)
        
        return pcd


if __name__ == "__main__":
    import sys
    
    print("Point Cloud Generator Test")
    print("=" * 60)
    
    # Check if calibration exists
    if not Path("stereo_calibration.pkl").exists():
        print("Error: stereo_calibration.pkl not found")
        print("Please run stereo calibration first")
        sys.exit(1)
    
    # Initialize generator
    generator = PointCloudGenerator()
    
    # Test with sample data if provided
    if len(sys.argv) >= 3:
        disparity_file = sys.argv[1]
        image_file = sys.argv[2]
        
        print(f"\nLoading:")
        print(f"  Disparity: {disparity_file}")
        print(f"  Image: {image_file}")
        
        # Load data
        if disparity_file.endswith('.npy'):
            disparity = np.load(disparity_file)
        else:
            disparity = cv2.imread(disparity_file, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        left_image = cv2.imread(image_file)
        
        if disparity is None or left_image is None:
            print("Error: Could not load data")
            sys.exit(1)
        
        # Generate point cloud
        pcd = generator.process_stereo_to_point_cloud(
            disparity, left_image,
            filter_cloud=True,
            estimate_normals=True,
            output_file="output_point_cloud.ply",
            visualize=True
        )
        
    else:
        print("\nUsage: python point_cloud_generator.py <disparity.npy> <left_image.jpg>")

