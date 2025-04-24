import json
import os

# This script extracts unique object classes from ScanNet scene files and saves them into a JSON file.
# It iterates through a specified range of scene IDs, reads the corresponding JSON files, and collects unique object classes.

def extract_classes_per_scene(scene_id, scene_folder, scene_classes):
    """Extracts unique object classes for a scene and stores them in the dictionary."""
    scene_path = os.path.join(scene_folder, f"{scene_id}_00/{scene_id}_00.aggregation.json")

    if not os.path.exists(scene_path):
        print(f"[WARNING] Scene file {scene_path} not found. Skipping.")
        return

    with open(scene_path, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

        # Extract the unique object classes of the current scene
        unique_classes = set(obj["label"] for obj in scene_data.get("segGroups", []))
        
        # Store the classes in a dictionary
        scene_classes[scene_id] = list(unique_classes)

scene_folder = "/Users/const/Projects/SAP_intro/scans"

scene_classes = {}

# Iterate over all scenes
for i in range(0, 151):  
    scene_id = f"scene{i:04d}"
    extract_classes_per_scene(scene_id, scene_folder, scene_classes)

output_file = "ground_truth.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(scene_classes, f, indent=4)

print(f"[INFO] Ground truth saved to {output_file}")