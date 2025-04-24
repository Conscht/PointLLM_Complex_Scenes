import os
import json

SCENE_RANGE = 150  # We decided to use the first 150 scenes of ScanNet for evaluation.
context_folder = "/Users/const/Projects/SAP_intro/scans"
output_file = "/Users/const/OneDrive/Dokumente/Python Scripts/PointLLM_Eval"

# Script to combine all contexts files.

def load_scene_contexts():
    """Reads all scene contexts and stores them in a single JSON file."""
    scene_contexts = {}

    for scene_id in range(0, SCENE_RANGE):  
        scene_id_str = f"scene{scene_id:04d}"
        context_path = f"{context_folder}/{scene_id_str}_00/{scene_id_str}_context.txt"

        try:
            with open(context_path, "r", encoding="utf-8") as f:
                context = f.read().strip()
                scene_contexts[scene_id_str] = context  

        except FileNotFoundError:
            print(f"[WARNING] Context file not found: {context_path}")
            scene_contexts[scene_id_str] = "No context available."

    # Save all contexts to a single JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scene_contexts, f, indent=4)

    print(f"[SUCCESS] All scene contexts saved to {output_file}")

# Run the function
load_scene_contexts()
