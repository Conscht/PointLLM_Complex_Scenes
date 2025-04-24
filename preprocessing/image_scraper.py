import os
import requests

# This script downloads images from the ScanNet dataset using the provided base URL and scene IDs.

base_url = "https://kaldir.vc.in.tum.de/scannet_browse/scannet/v2/data/scans_extra/renders/{scene_id}/{scene_id}.color.png"

output_dir = r"C:\Users\const\Projects\SAP_intro\scans"

scene_range = range(0, 151)  # Scene IDs from 0000 to 0150

for i in scene_range:
    scene_id = f"scene{i:04d}_00"
    image_url = base_url.format(scene_id=scene_id)

    scene_folder = os.path.join(output_dir, scene_id)
    os.makedirs(scene_folder, exist_ok=True)

    image_path = os.path.join(scene_folder, f"{scene_id}.color.png")

    try: 
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {scene_id}.color.jpg")
        else:
            print(f"Failed to download {scene_id}.color.jpg")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {scene_id}.color.jpg: {e}")

    print("Doanload finished!")
