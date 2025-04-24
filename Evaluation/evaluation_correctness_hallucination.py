import re

# Script to extract correctness and hallucination scores from a text file

def extract_scores_per_scene(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split by each "Scene" section
    scenes = re.split(r"Scene\s+", text)


    total_c = 0
    total_h = 0

    print("Per-Scene Scores:\n")
    for scene in scenes:
        # Extract scene number
        scene_id_match = re.match(r"(\d+)", scene)
        scene_id = scene_id_match.group(1) if scene_id_match else "Unknown"

        # Extract scores (we follow the pattern "Correctness Score: X" and "Hallucination Score: Y")
        # Example: "Correctness Score: 5" and "Hallucination Score: 2"
        c_match = re.search(r"Correctness Score.*?:\s*(\d+)", scene)
        h_match = re.search(r"Hallucination Score.*?:\s*(\d+)", scene)

        if c_match and h_match:
            c_score = int(c_match.group(1))
            h_score = int(h_match.group(1))
            total_c += c_score
            total_h += h_score
            print(f"Scene {scene_id}: C = {c_score}, H = {h_score}")
        else:
            print(f"Scene {scene_id}: Scores not found.")

    print("\n🔢 Total Scenes Evaluated:", len(scenes))
    print(f"Total Correctness Score: {total_c}")
    print(f"Total Hallucination Score: {total_h}")

extract_scores_per_scene("C:/Users/const/OneDrive/Dokumente/Python Scripts/Project_Seminar/data/sample_correct_hallu_eval.txt")
