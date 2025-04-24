import os
import numpy as np
import open3d as o3d

# Input and output directories
input_dir = r"C:\Users\const\Projects\SAP_intro\scans"
output_dir = r"C:\Users\const\Projects\SAP_intro\processed_npy"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def convert_ply_to_npy(ply_path, output_path):
    """ Load a .ply file and save as .npy while preserving all points """
    try:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)  # Extract XYZ
        colors = np.asarray(pcd.colors)  # Extract RGB (if available)

        if colors.size > 0:  # If colors are present, concatenate
            data = np.hstack((points, colors))  # (N, 6)
        else:
            data = points  # (N, 3)

        np.save(output_path, data)
        print(f"Saved: {output_path} ({data.shape})")
    except Exception as e:
        print(f"Error processing {ply_path}: {e}")

# Traverse directories
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith("_vh_clean_2.ply"):
            scene_id = file.split("_vh_clean_2.ply")[0]  # Extract scene identifier
            output_filename = f"{scene_id}_8192.npy"  
            output_path = os.path.join(output_dir, output_filename)

            ply_path = os.path.join(root, file)
            convert_ply_to_npy(ply_path, output_path)

print("Conversion complete.")