import json
import re

# This script evaluates the performance of PointLLM on a dataset by comparing its answers to ground truth data.
# It calculates object and material accuracy, and provides a summary of the results.
# The evaluation is done in a strict manner, where the answers must match the expected format and content.
# The script also includes a mapping for material aliases and fallback materials for certain objects.

# Load files
with open("/mnt/data/ground_truth.json") as f:
    ground_truth_data = json.load(f)

with open("/mnt/data/material_list_updated.json") as f:
    material_data = json.load(f)

with open("/mnt/data/No_Context_FINALFINAL.json") as f:
    no_context_data = json.load(f)

# Alias mapping for object normalization
material_aliases = {
    "desk": "table",
    "armchair": "chair",
    "tv": "monitor"
}

# Fallback materials for hardcoded cases
fallback_materials = {
    "monitor": "plastic",
    "tv": "plastic",
    "microwave": "metal",
    "refrigerator": "metal",
    "mirror": "glass",
    "window": "glass"
}


def evaluate_with_aliases_strict_yes(data, dataset_name):
    '''Evaluation function.

    This function evaluates the performance of PointLLM on a dataset by comparing its answers to ground truth data.
    It calculates object and material accuracy, and provides a summary of the results.

    Args:
        data (list): The dataset to evaluate, containing scene information and answers.
        dataset_name (str): The name of the dataset for reporting purposes.

    Outputs:
        dict: A dictionary containing the evaluation results, including object and material accuracy, total score,
              and the number of questions answered.
    '''

    object_score = 0
    material_correct = 0
    material_incorrect = 0
    total_material_questions = 0

    total_object_questions = len(data) * len(data[0]["objects_in_question"])

    for scene in data:
        scene_id = scene["scene_id"]
        gt_objects = set(ground_truth_data.get(scene_id, []))
        gt_materials = material_data.get(scene_id, {})

        answers = scene["answers"]
        objects = scene["objects_in_question"]

        obj_index = 0
        last_object = None

        # Loop through all logged answers and check if they are correct
        for ans in answers:
            # Check if the answer is related to material
            if "Material Q" in ans:
                obj = last_object
                if obj:
                    # Try to get material from scene, fallback if needed (fallback_materials are trivial cases) and checked for correctness

                    correct_material = gt_materials.get(obj)
                    if not correct_material:
                        alias_obj = material_aliases.get(obj, obj)
                        correct_material = fallback_materials.get(alias_obj)

                    # Striclty filter for yes/no answers and do the binary evaluation
                    if correct_material:

                        # Extract the material from the answer
                        # Example: "Material Q: Is the object made of plastic? => yes"
                        material_q_part = ans.split("=>")[0].lower()
                        match = re.search(r"made of\s+([a-zA-Z0-9\-\s]+)", material_q_part)
                        given_material = match.group(1).strip().lower() if match else None

                        yes_no_match = re.search(r"=>\s*(yes|no)", ans.lower())
                        is_yes = yes_no_match and yes_no_match.group(1) == "yes"

                        if given_material:
                            if is_yes and given_material == correct_material:
                                material_correct += 1
                            elif not is_yes and given_material != correct_material:
                                material_correct += 1
                            else:
                                material_incorrect += 1
                            total_material_questions += 1
            else:
                if obj_index < len(objects):
                    obj = objects[obj_index]
                    if "yes" in ans.lower() and obj in gt_objects:
                        object_score += 1
                    elif "yes" not in ans.lower() and obj not in gt_objects:
                        object_score += 1
                    last_object = obj
                    obj_index += 1

    total_score = object_score + material_correct
    total_questions = total_object_questions + total_material_questions
    return {
        "dataset": dataset_name,
        "total_object_score": object_score,
        "total_material_score": material_correct,
        "total_score": total_score,
        "object_accuracy": round(object_score / total_object_questions, 4),
        "material_accuracy": round(material_correct / total_material_questions, 4) if total_material_questions else 0,
        "overall_accuracy": round(total_score / total_questions, 4),
        "num_object_questions": total_object_questions,
        "num_material_questions": total_material_questions,
        "num_total_questions": total_questions
    }

# Run evaluation on the a dataset
evaluate_with_aliases_strict_yes(no_context_data, "NO_CONTEXT_FINALFINAL")
 