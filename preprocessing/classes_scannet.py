import json 

# This script is used to extract and count the number of objects in a given scene.
# It reads a JSON file containing scene data and counts the occurrences of each object label.
def get_objects(scene_id):
    """Reads the corresponding 2D context description from a .txt file."""
 #/hpi/fs00/share/fg/doellner/constantin.auga/checkpoints/checkpoints/scans/scans/scene0000_00
    scenePath = f"{scene_folder}/{scene_id}_00/{scene_id}_00.aggregation.json"

    with open(scenePath, "r") as f:
        scene_data = json.load(f)
        
        occInCurrentScene = {}
        for obj in scene_data["segGroups"]:
            label = obj["label"]
            occTotal[label] = occTotal.get(label, 0) + 1

            if label not in occInCurrentScene:
                occPerScene[label] = occPerScene.get(label, 0) + 1
                occInCurrentScene[label] = 1


def get_objects(file_path):
    """Reads the corresponding 2D context description from a .txt file."""
 #/hpi/fs00/share/fg/doellner/constantin.auga/checkpoints/checkpoints/scans/scans/scene0000_00

    with open(file_path, "r") as f:
        scene_data = json.load(f)
        print(scene_data)
        
        unique_materials = set()
        for scene in scene_data:
            for obj in scene:
                unique_materials.add(obj)
        return unique_materials

    

# scene_folder = '/Users/const/Projects/SAP_intro/scans'
occTotal = {}
occPerScene = {}
for i in range(0, 151, 1):

    scene_id = f"scene{i:04d}"
    get_objects(scene_id)

print("[Total Sum]", sorted(occTotal.items(), key=lambda item: item[1]))
print("[Per Scene]", sorted(occPerScene.items(), key=lambda item: item[1]))

file = '/Users/const/OneDrive/Dokumente/Python Scripts/material_list_updated (1).json'

print(get_objects(file))
