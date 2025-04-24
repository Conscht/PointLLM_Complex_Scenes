import os

# This script downloads ScanNet data using the provided script and scene IDs.
# It iterates through a range of scene IDs and downloads the specified file types.

output_dir = r"C:\Users\const\Projects\SAP_intro"
scannet_script = "download-scannet.py"

scene_range = range(0, 151)  



for i in scene_range:
    scene_id = f"scene{i:04d}_00"  # Formats scene ID (e.g., scene0000_00)

    for file_type in file_types:
        command = f"python {scannet_script} -o {output_dir} --id {scene_id} --type {file_type}"
        
        print(f"Downloading {scene_id} - {file_type}...")
        os.system(command) 

print("All downloads completed.")
