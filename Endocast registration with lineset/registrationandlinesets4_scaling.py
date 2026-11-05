import open3d as o3d
import numpy as np
import os
import argparse
import time
import glob
from datetime import datetime

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Align cranial endocast meshes and associated linesets.")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference endocast PLY file")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing endocast and lineset PLY files")
    parser.add_argument("--endocast-pattern", type=str, default="*endocast*.ply", help="Filename pattern to identify endocast files")
    parser.add_argument("--lineset-left-suffix", type=str, default="_left", help="Suffix identifying left lineset files")
    parser.add_argument("--lineset-right-suffix", type=str, default="_right", help="Suffix identifying right lineset files")
    parser.add_argument("--voxel-size", type=float, default=None, help="Voxel size for downsampling. If not provided, will be automatically estimated.")
    parser.add_argument("--ransac-iter", type=int, default=100000, help="Number of RANSAC iterations")
    parser.add_argument("--no-preview", action="store_true", help="Skip alignment preview")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to save output files")
    return parser.parse_args()

def find_associated_linesets(endocast_file, input_dir, left_suffix, right_suffix):
    """Find associated left and right lineset files for a given endocast file."""
    base_name = os.path.basename(endocast_file).split('.')[0]
    # Remove "endocast" from the base name if present
    # Your lineset filenames KEEP "_endocast", so match on the full base name.
    # (Stripping "_endocast" here was why the glob found 0 linesets.)
    clean_base = base_name
    
    # Look for lineset files with similar names
    left_pattern = os.path.join(input_dir, f"*{clean_base}*{left_suffix}*.lineset")
    right_pattern = os.path.join(input_dir, f"*{clean_base}*{right_suffix}*.lineset")
    
    left_files = glob.glob(left_pattern)
    right_files = glob.glob(right_pattern)
    
    return left_files, right_files

def load_ply(filename):
    """Loads a PLY file as an Open3D mesh."""
    try:
        print(f"Loading {filename}...")
        mesh = o3d.io.read_triangle_mesh(filename)
        if len(mesh.vertices) == 0:
            raise ValueError(f"No vertices found in {filename}")
        return mesh
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None
    
def load_lineset_file(filename):
    """Load a .lineset file as an Open3D LineSet."""
    try:
        print(f"Loading lineset {filename}...")
        
        # Initialize empty lineset
        line_set = o3d.geometry.LineSet()
        all_points = []
        all_lines = []
        current_point_index = 0
        
        # Read the file
        with open(filename, 'r') as f:
            content = f.read()
        
        # Split sections based on { { ... } { ... } } pattern
        if '{ {' in content and '} {' in content and content.strip().endswith('}'):
            # Split by } {
            sections = content.split('} {')
            
            # Clean up first section
            if sections[0].startswith('# LineSet'):
                first_section_lines = sections[0].split('\n')
                # Skip header lines and find start of data
                for i, line in enumerate(first_section_lines):
                    if line.strip() == '{':
                        first_section_lines = first_section_lines[i+1:]
                        break
                sections[0] = '\n'.join(first_section_lines)
            
            # Clean up last section
            if sections[-1].endswith('} }'):
                sections[-1] = sections[-1].replace('} }', '')
            
            # Process each section (curve)
            for section in sections:
                points = []
                # Process each line in the section
                for line in section.strip().split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('{') or line.startswith('}'):
                        continue
                    
                    values = line.split()
                    if len(values) >= 3:  # At least x, y, z coordinates
                        x, y, z = float(values[0]), float(values[1]), float(values[2])
                        points.append([x, y, z])
                
                # Create lines connecting sequential points in this section
                if len(points) > 1:
                    section_lines = []
                    for i in range(len(points) - 1):
                        section_lines.append([current_point_index + i, current_point_index + i + 1])
                    
                    all_points.extend(points)
                    all_lines.extend(section_lines)
                    current_point_index += len(points)
        
        # Create Open3D LineSet
        if all_points and all_lines:
            line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
            line_set.lines = o3d.utility.Vector2iVector(np.array(all_lines))
            
            # Add color (default to blue)
            colors = [[0, 0, 1] for _ in range(len(all_lines))]
            line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
            
            print(f"Loaded lineset with {len(all_points)} points and {len(all_lines)} lines")
            return line_set
        else:
            raise ValueError(f"Invalid lineset format or no points found in {filename}")
            
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

def mesh_to_point_cloud(mesh, voxel_size=None):
    """Convert mesh to point cloud and estimate appropriate voxel size if not provided."""
    # Convert mesh to point cloud
    pcd = mesh.sample_points_uniformly(number_of_points=500000)
    
    # If voxel size not provided, estimate it based on bounding box
    if voxel_size is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_extent = bbox.get_extent()
        # Use 1/100th of the average dimension as voxel size
        voxel_size = np.mean(bbox_extent) / 100
        print(f"Automatically estimated voxel size: {voxel_size:.4f}")
    
    return pcd, voxel_size

def preprocess_point_cloud(pcd, voxel_size):
    """Downsamples and computes normals for stable alignment."""
    print(f"Preprocessing point cloud (voxel size: {voxel_size:.4f})...")
    start_time = time.time()
    
    # Downsample point cloud
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))
    
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    print(f"Downsampled point cloud has {len(pcd_down.points)} points")
    
    return pcd_down

def compute_fpfh_features(pcd, voxel_size):
    """Computes FPFH features for global registration."""
    print("Computing FPFH features...")
    start_time = time.time()
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    
    print(f"FPFH computation completed in {time.time() - start_time:.2f} seconds")
    return fpfh

def global_alignment(source, target, voxel_size, ransac_iter=100000):
    """Performs global alignment using FPFH features and RANSAC."""
    print(f"Performing global alignment (RANSAC iterations: {ransac_iter})...")
    start_time = time.time()

    # Preprocess both point clouds
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)

    # Compute FPFH features
    source_fpfh = compute_fpfh_features(source_down, voxel_size)
    target_fpfh = compute_fpfh_features(target_down, voxel_size)

    # Set up and run RANSAC registration
    distance_threshold = voxel_size * 1.5
    print(f"Running RANSAC with distance threshold {distance_threshold:.4f}...")

    # Add correspondence checkers for better matching
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(0.95)
    ]

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(True),  # True = allow isotropic scale (fossil<->extant)
        ransac_n=4,
        checkers=checkers,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iter, 0.999)
    )

    elapsed_time = time.time() - start_time
    print(f"Global alignment completed in {elapsed_time:.2f} seconds")
    print(f"RANSAC fitness: {result.fitness:.4f}, inlier RMSE: {result.inlier_rmse:.4f}")

    return result.transformation

def refine_alignment(source, target, transformation, voxel_size):
    """Refines alignment using multi-scale ICP for higher accuracy."""
    print("Refining alignment with multi-scale ICP...")
    start_time = time.time()
    
    current_transformation = transformation
    
    # Multi-scale refinement (coarse to fine)
    scales = [3.0, 1.5, 0.8, 0.4, 0.2, 0.1]  # Multiple scales for refinement
    
    for scale in scales:
        threshold = voxel_size * scale
        print(f"ICP with threshold: {threshold:.4f}")
        
        result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(True),  # scaling ICP; was PointToPlane (rigid, dropped scale)
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
        )
        
        current_transformation = result.transformation
        print(f"Scale {scale} ICP fitness: {result.fitness:.4f}, inlier RMSE: {result.inlier_rmse:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"Multi-scale ICP refinement completed in {elapsed_time:.2f} seconds")
    
    return current_transformation

def visualize_alignment(source, target, transformation):
    """Visualizes the alignment result before applying it."""
    source_temp = o3d.geometry.PointCloud()
    source_temp.points = o3d.utility.Vector3dVector(np.asarray(source.points))  # Copy the points
    source_temp.colors = o3d.utility.Vector3dVector(np.asarray(source.colors))  # Copy the colors (if any)

    source_temp.transform(transformation)  # Apply transformation to the copied point cloud

    source_temp.paint_uniform_color([1, 0, 0])  # Red = transformed model
    target.paint_uniform_color([0, 1, 0])  # Green = reference model

    print("🔍 Showing preview of alignment... Close window to continue.")
    o3d.visualization.draw_geometries([source_temp, target])

def apply_transformation(mesh, transform_matrix):
    """Applies a transformation matrix to an Open3D mesh or lineset."""
    mesh.transform(transform_matrix)
    return mesh

def save_ply(mesh, filename):
    """Saves a transformed mesh as a PLY file."""
    try:
        o3d.io.write_triangle_mesh(filename, mesh)
        print(f"✅ Saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving {filename}: {e}")
        return False

def save_lineset_with_preserved_structure(original_lineset_path, output_path, transformed_points):
    """
    Writes a new .lineset file using the transformed points but preserving the original curve structure and labels.
    """
    try:
        # Build the output content directly
        output_content = ""
        
        with open(original_lineset_path, 'r') as f:
            content = f.read()
            
        # Make sure we have proper line endings
        content = content.replace('\r\n', '\n')
        lines = content.split('\n')
        
        # Extract header and find body start after the first '{ {'
        header_lines = []
        body_start_index = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            header_lines.append(line)
            if stripped in ["{ {", "{{"]:
                body_start_index = i + 1
                break
                
        if body_start_index is None:
            raise ValueError("Invalid lineset format: Missing opening '{ {'")
        
        # Add header to output
        for line in header_lines:
            output_content += line + "\n"
        
        # Parse and transform the coordinates
        segments = []
        current_segment = []
        point_index = 0  # Track the current point index
        
        for i, line in enumerate(lines[body_start_index:]):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
                
            # Handle segment separators
            if stripped == "} {":
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            # Handle end of data
            elif stripped in ["} }", "} ", "}"]:
                if current_segment:
                    segments.append(current_segment)
                break
            # Process coordinate lines
            elif len(stripped.split()) >= 3:  # At least x, y, z coordinates
                try:
                    parts = stripped.split()
                    # Determine if there's a label
                    label = None
                    if len(parts) >= 4:
                        # Try to extract the original format of the label
                        original_label = parts[3]
                        # Check if it's an integer or has decimals
                        if '.' in original_label:
                            # Get the number of decimal places in the original
                            decimal_parts = original_label.split('.')
                            if len(decimal_parts) > 1 and len(decimal_parts[1]) > 0:
                                # If original has decimals, use integer if possible
                                try:
                                    label_value = float(original_label)
                                    if label_value.is_integer():
                                        # It's a whole number stored as float, convert to int
                                        label = str(int(label_value))
                                    else:
                                        # Keep original decimal format if not an integer
                                        label = original_label
                                except ValueError:
                                    # If conversion fails, keep original
                                    label = original_label
                            else:
                                # No meaningful decimals, treat as int
                                label = decimal_parts[0]
                        else:
                            # No decimal point, keep as is
                            label = original_label
                    
                    # Get transformed coordinate
                    if point_index < len(transformed_points):
                        new_x, new_y, new_z = transformed_points[point_index]
                        
                        # Format line with or without label
                        if label:
                            new_line = f"{new_x:.6f} {new_y:.6f} {new_z:.6f} {label}"
                        else:
                            new_line = f"{new_x:.6f} {new_y:.6f} {new_z:.6f}"
                            
                        current_segment.append(new_line)
                        point_index += 1
                    else:
                        print(f"⚠️ Warning: More points in file than in transformed data at line {i+body_start_index}")
                        # Keep original line as fallback
                        current_segment.append(stripped)
                except Exception as e:
                    print(f"⚠️ Error processing line {i+body_start_index}: {stripped}, Error: {e}")
                    # Keep original line as fallback
                    current_segment.append(stripped)
            else:
                # Keep non-coordinate lines
                current_segment.append(stripped)
        
        # Add segments with proper separators to output
        for i, segment in enumerate(segments):
            if i > 0:
                output_content += "} {\n"
                
            for line in segment:
                output_content += line + "\n"
        
        # Always use the specified ending format
        output_content += "} \n"  # Note the space after }
        output_content += "}\n"
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(output_content)
            
        print(f"✅ Transformed and saved: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"❌ Failed to transform {original_lineset_path}: {e}")
        return False


def save_transformation_matrix(transformation, filename):
    """Saves the transformation matrix to a text file for future reference."""
    try:
        np.savetxt(filename, transformation, fmt='%.6f')
        print(f"📝 Transformation matrix saved to {filename}")
    except Exception as e:
        print(f"⚠️ Could not save transformation matrix: {e}")
        
def apply_transformation_to_lineset_and_save(input_path, output_folder, transformation):
    """Apply transformation matrix to a .lineset file and save it, matching original format exactly."""
    try:
        # Build the output content directly
        output_content = ""
        
        with open(input_path, 'r') as f:
            content = f.read()
            
        # Make sure we have proper line endings
        content = content.replace('\r\n', '\n')
        lines = content.split('\n')
        
        # Extract header and find body start after the first '{ {'
        header_lines = []
        body_start_index = None
        for i, line in enumerate(lines):
            header_lines.append(line)
            if line.strip() in ["{ {", "{{"]:
                body_start_index = i + 1
                break
                
        if body_start_index is None:
            raise ValueError("Invalid lineset format: Missing opening '{ {'")
            
        # Add header to output
        for line in header_lines:
            output_content += line + "\n"
            
        # Parse and transform the coordinates
        segments = []
        current_segment = []
        
        for i, line in enumerate(lines[body_start_index:]):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
                
            # Handle segment separators
            if stripped == "} {":
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            # Handle end of data
            elif stripped in ["} ", "}"]:
                if current_segment:
                    segments.append(current_segment)
                break
            # Process coordinate lines
            elif len(stripped.split()) >= 4:
                try:
                    parts = stripped.split()
                    x, y, z = map(float, parts[:3])
                    label = parts[3].split('.')[0]  # Keep just the integer part of the label
                    
                    # Transform coordinates using homogeneous coordinates
                    coord = np.array([x, y, z, 1.0])
                    transformed = transformation @ coord
                    
                    # Format coordinates to match original precision (typically 5-6 digits)
                    new_line = f"{transformed[0]:.6f} {transformed[1]:.6f} {transformed[2]:.6f} {label}"
                    current_segment.append(new_line)
                except Exception as e:
                    print(f"⚠️ Skipped malformed line: {stripped} due to error: {e}")
        
        # Add segments with proper separators to output
        for i, segment in enumerate(segments):
            if i > 0:
                output_content += "} {\n"
                
            for line in segment:
                output_content += line + "\n"
                
        # Add explicit ending format with the space after }
        output_content += "} \n"  # Note the space after }
        output_content += "}\n"
        
        # Save to file
        base = os.path.basename(input_path)
        new_name = base.replace('.lineset', '_aligned.lineset')
        output_path = os.path.join(output_folder, new_name)
        
        with open(output_path, 'w') as f:
            f.write(output_content)
            
        print(f"✅ Transformed and saved: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"❌ Failed to transform {input_path}: {e}")
        return False
    
def manual_alignment_gui(reference_pcd, moving_pcd):
    """
    Interactive GUI for manual alignment of point clouds before automatic registration.
    
    Args:
        reference_pcd (o3d.geometry.PointCloud): The reference point cloud to align to
        moving_pcd (o3d.geometry.PointCloud): The point cloud to be manually aligned
    
    Returns:
        numpy.ndarray: 4x4 transformation matrix representing manual alignment
    """
    print("🔧 Manual Alignment Mode:")
    print("Translation:")
    print("   Arrow Keys: Move along X/Y")
    print("   PgUp/PgDown: Move along Z")
    print("Rotation:")
    print("   W/S: Rotate around X-axis")
    print("   A/D: Rotate around Y-axis")
    print("   Q/E: Rotate around Z-axis")
    print("   Spacebar: Confirm alignment")
    print("   Esc: Cancel and reset")

    # Create deep copies to avoid modifying originals
    moving_copy = o3d.geometry.PointCloud(moving_pcd)
    reference_copy = o3d.geometry.PointCloud(reference_pcd)

    # Color the point clouds for easy distinction
    reference_copy.paint_uniform_color([0.7, 0.7, 0.7])  # Grey for reference
    moving_copy.paint_uniform_color([1.0, 0.0, 0.0])    # Red for moving point cloud

    # Initial transformation matrix
    current_transform = np.eye(4)

    def update_pointcloud_transform(transform_matrix):
        """Apply the transformation to the moving point cloud."""
        nonlocal moving_copy, current_transform
        moving_copy.transform(transform_matrix)
        current_transform = transform_matrix @ current_transform
        vis.update_geometry(moving_copy)
        vis.poll_events()
        vis.update_renderer()

    def translate_pointcloud(dx=0, dy=0, dz=0):
        """Create and apply a translation transformation."""
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = [dx, dy, dz]
        update_pointcloud_transform(translation_matrix)

    def rotate_pointcloud(axis, angle_deg):
        """Create and apply a rotation transformation."""
        angle = np.deg2rad(angle_deg)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        update_pointcloud_transform(rotation_matrix)

    # Set up visualization
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Manual Point Cloud Alignment")
    vis.add_geometry(reference_copy)
    vis.add_geometry(moving_copy)

    # Translation key callbacks
    vis.register_key_callback(263, lambda vis: translate_pointcloud(dx=-5))   # Left Arrow
    vis.register_key_callback(262, lambda vis: translate_pointcloud(dx=5))    # Right Arrow
    vis.register_key_callback(264, lambda vis: translate_pointcloud(dy=-5))   # Down Arrow
    vis.register_key_callback(265, lambda vis: translate_pointcloud(dy=5))    # Up Arrow
    vis.register_key_callback(266, lambda vis: translate_pointcloud(dz=5))    # PgUp
    vis.register_key_callback(267, lambda vis: translate_pointcloud(dz=-5))   # PgDown

    # Rotation key callbacks
    vis.register_key_callback(ord('W'), lambda vis: rotate_pointcloud(np.array([1, 0, 0]), 5))   # Rotate X
    vis.register_key_callback(ord('S'), lambda vis: rotate_pointcloud(np.array([1, 0, 0]), -5))
    vis.register_key_callback(ord('A'), lambda vis: rotate_pointcloud(np.array([0, 1, 0]), 5))   # Rotate Y
    vis.register_key_callback(ord('D'), lambda vis: rotate_pointcloud(np.array([0, 1, 0]), -5))
    vis.register_key_callback(ord('Q'), lambda vis: rotate_pointcloud(np.array([0, 0, 1]), 5))   # Rotate Z
    vis.register_key_callback(ord('E'), lambda vis: rotate_pointcloud(np.array([0, 0, 1]), -5))

    # Confirm and exit
    vis.register_key_callback(32, lambda vis: vis.close())  # Spacebar

    vis.run()
    vis.destroy_window()

    print("✅ Manual alignment complete.")
    return current_transform


def transform_points_directly(points, transformation):
    """
    Apply transformation directly to an array of points using homogeneous coordinates.
    
    Args:
        points (numpy.ndarray): Array of points with shape (N, 3)
        transformation (numpy.ndarray): 4x4 transformation matrix
        
    Returns:
        numpy.ndarray: Transformed points
    """
    # Add homogeneous coordinate (1) to each point
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Apply transformation
    transformed_points_homogeneous = np.dot(homogeneous_points, transformation.T)
    
    # Return the 3D points (discard homogeneous coordinate)
    return transformed_points_homogeneous[:, :3]

def process_endocast(endocast_file, reference_mesh, left_linesets, right_linesets, args, output_dir):
    """Process a single endocast file and its associated linesets."""
    print(f"\n===== Processing {os.path.basename(endocast_file)} =====")
    
    # Load endocast mesh
    endocast_mesh = load_ply(endocast_file)
    if endocast_mesh is None:
        print(f"❌ Skipping {endocast_file} due to loading error")
        return False
    
    # Save a copy of the original mesh vertices for verification
    original_vertices = np.asarray(endocast_mesh.vertices).copy()
    
    # Proceed with automatic voxel size detection and alignment
    reference_raw_pcd, auto_voxel_size = mesh_to_point_cloud(reference_mesh)
    endocast_raw_pcd, _ = mesh_to_point_cloud(endocast_mesh)
    
    voxel_size = args.voxel_size if args.voxel_size is not None else auto_voxel_size

    # Preprocess point clouds to compute normals (needed for ICP)
    reference_pcd = preprocess_point_cloud(reference_raw_pcd, voxel_size)
    endocast_pcd = preprocess_point_cloud(endocast_raw_pcd, voxel_size)

    # Initialize combined transformation matrix
    combined_transform = np.eye(4)
    
    # Ask user if they want to manually align first
    manual_prompt = input("\n🛠️  Do you want to manually align the endocast to the reference? (yes/no): ").strip().lower()
    if manual_prompt == "yes":
        # Use preprocessed point clouds for manual alignment
        manual_transform = manual_alignment_gui(reference_pcd, endocast_pcd)
        
        # Update combined transformation
        combined_transform = manual_transform.copy()
    
        # Apply the manual transformation to both the mesh and point cloud
        endocast_mesh = apply_transformation(endocast_mesh, manual_transform)
        endocast_pcd = apply_transformation(endocast_pcd, manual_transform)
        endocast_raw_pcd = apply_transformation(endocast_raw_pcd, manual_transform)

    # Compute alignment transformation
    print("\n==== ALIGNMENT PROCESS ====")
    global_transform = global_alignment(endocast_pcd, reference_pcd, voxel_size, args.ransac_iter)
    final_transform = refine_alignment(endocast_pcd, reference_pcd, global_transform, voxel_size)
    
    # Update combined transformation
    combined_transform = final_transform @ combined_transform
    
    # Visualize alignment if not skipped
    if not args.no_preview:
        visualize_alignment(endocast_pcd, reference_pcd, final_transform)
    
    # Ask for confirmation
    confirm = input("\n✅ Does the alignment look correct? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("❌ Alignment rejected. Moving to next endocast.")
        return False
    
    # Create timestamped file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    endocast_basename = os.path.splitext(os.path.basename(endocast_file))[0]
    
    # Apply transformation to endocast
    print("\n==== APPLYING TRANSFORMATIONS ====")
    aligned_endocast = apply_transformation(endocast_mesh, final_transform)
    
    # Save transformed endocast
    aligned_endocast_path = os.path.join(output_dir, f"{endocast_basename}_aligned.ply")
    save_ply(aligned_endocast, aligned_endocast_path)
    
    # Save transformation matrix (the combined transform)
    transform_matrix_path = os.path.join(output_dir, f"{endocast_basename}_transform.txt")
    save_transformation_matrix(combined_transform, transform_matrix_path)
    
    # Sanity check - verify that applying the combined transform to the original vertices
    # gives the same result as the aligned mesh vertices
    verified_vertices = transform_points_directly(original_vertices, combined_transform)
    aligned_vertices = np.asarray(aligned_endocast.vertices)
    max_diff = np.max(np.abs(verified_vertices - aligned_vertices))
    print(f"Maximum difference between transformed vertices: {max_diff:.10f}")
    
    # Process left linesets
    for left_file in left_linesets:
        print(f"\nProcessing left lineset: {os.path.basename(left_file)}")
        
        # Load original lineset
        line_set = load_lineset_file(left_file)
        if line_set is None:
            print(f"❌ Could not load {left_file}")
            continue

        # Get the original points
        original_points = np.asarray(line_set.points)
        
        # Apply combined transformation directly to the points
        transformed_points = transform_points_directly(original_points, combined_transform)

        # Save new lineset using preserved structure
        output_left = os.path.join(output_dir, os.path.basename(left_file).replace('.lineset', '_aligned.lineset'))
        apply_transformation_to_lineset_and_save(left_file, output_dir, combined_transform)
        
    # Process right linesets
    for right_file in right_linesets:
        print(f"\nProcessing right lineset: {os.path.basename(right_file)}")
        
        # Load original lineset
        line_set = load_lineset_file(right_file)
        if line_set is None:
            print(f"❌ Could not load {right_file}")
            continue

        # Get the original points
        original_points = np.asarray(line_set.points)
        
        # Apply combined transformation directly to the points
        transformed_points = transform_points_directly(original_points, combined_transform)

        # Save new lineset using preserved structure
        output_right = os.path.join(output_dir, os.path.basename(right_file).replace('.lineset', '_aligned.lineset'))
        apply_transformation_to_lineset_and_save(right_file, output_dir, combined_transform)

    print(f"\n🎉 Alignment of {os.path.basename(endocast_file)} completed successfully!")
    return True

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Ensure output directory exists
    output_dir = args.output_dir if args.output_dir else args.input_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load reference mesh
    reference_mesh = load_ply(args.reference)
    if reference_mesh is None:
        print("❌ Cannot proceed without a valid reference mesh")
        return
    
    # Find all endocast files in the input directory
    endocast_pattern = os.path.join(args.input_dir, args.endocast_pattern)
    endocast_files = glob.glob(endocast_pattern)
    
    if not endocast_files:
        print(f"❌ No endocast files found matching pattern: {args.endocast_pattern}")
        return
    
    print(f"Found {len(endocast_files)} endocast files to process")
    
    # Process each endocast
    successful = 0
    for endocast_file in endocast_files:
        # Find associated linesets
        left_linesets, right_linesets = find_associated_linesets(
            endocast_file, args.input_dir, args.lineset_left_suffix, args.lineset_right_suffix
        )
        
        print(f"\nFor endocast {os.path.basename(endocast_file)}:")
        print(f"  Found {len(left_linesets)} left lineset(s): {[os.path.basename(f) for f in left_linesets]}")
        print(f"  Found {len(right_linesets)} right lineset(s): {[os.path.basename(f) for f in right_linesets]}")
        
        # Process this endocast and its linesets
        if process_endocast(endocast_file, reference_mesh, left_linesets, right_linesets, args, output_dir):
            successful += 1
    
    print(f"\n===== SUMMARY =====")
    print(f"Total endocasts processed: {successful}/{len(endocast_files)}")

import sys

sys.argv = [
    '',  # placeholder for script name
    '--reference', r'I:\2025\Conferences and Talks\EAAPP 2026\New folder\Reference human\human_reference.ply',
    '--input-dir', r'I:\2025\Conferences and Talks\EAAPP 2026\New folder\Homo',
    '--output-dir', r'I:\2025\Conferences and Talks\EAAPP 2026\New folder\Homo alighned human',
    '--endocast-pattern', '*.ply'  # ✅ include all .ply files in folder
]
    
main()
#python registrationandlinesets.py --reference reference_endocast.ply --input-dir ./original_files --output-dir ./aligned_files
