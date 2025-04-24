import argparse
from transformers import AutoTokenizer
import torch
import os
import json
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria
import datetime
from pointllm.data import load_objaverse_point_cloud
import random

SCENE_RANGE = 151
CONTEXT_JSON_PATH = 'reduced_context.json'
CONTEXT_JSON_PATH = 'optimized_natural_context.json'
object_list_path = '/hpi/fs00/share/fg/doellner/constantin.auga/checkpoints/checkpoints/scans/scans/sceneclouds/scene_metadata.json'
point_clouds = '/hpi/fs00/share/fg/doellner/constantin.auga/checkpoints/checkpoints/scenes/processed_npy/processed_npy'
#  '/hpi/fs00/share/fg/doellner/constantin.auga/checkpoints/checkpoints/scenes/processed_npy/scene0004_00_8192.npy'
#scene0435_00_8192.npy scene{i:04}_00_8192.npy

scene_ground_truth = {}

with open("ground_truth.json", "r") as f:
    scene_ground_truth = json.load(f)

material_ground_truth = {}
with open("material_list_updated.json", "r") as f:
    material_ground_truth = json.load(f)

# table == desk | tv == monitor
SELECTED_CLASSES = [
    "chair", "table", "sofa", "bed", "backpack", "bookshelf", "refrigerator",
    "microwave", "pillow", "window", "tv", "monitor", "lamp", "sink", "toilet",
    "bathtub", "mirror", "carpet", "trash can"
]

MATERIAL_CLASSES = ["ceramic", "fabric", "glass", "leather", "metal", "plastic", "stone", "wood"]









def load_point_cloud(args):
    object_id = args.object_id
    print(f"[INFO] Loading point clouds using object_id: {object_id}")
    point_cloud = load_objaverse_point_cloud(args.data_path, object_id, pointnum=8192, use_color=True)
    
    return object_id, torch.from_numpy(point_cloud).unsqueeze_(0).to(torch.float32)



def load_context(scene_id):
    """Reads the corresponding scene context from the JSON file."""
    try:
        with open(CONTEXT_JSON_PATH, "r", encoding="utf-8") as f:
            scene_data = json.load(f)  # Load the JSON file
        
        # Extract scene context if available
        scene_key = f"{scene_id}"  # Ensure correct key format
        return scene_data.get(scene_key, "No additional context available.")
    
    except FileNotFoundError:
        print(f"[WARNING] Context JSON file not found: {CONTEXT_JSON_PATH}")
        return "No additional context available."

def load_object_list_question_context(object_id, context, asked_objects):
    """Reads the corresponding object list description from a .txt file.
    
    We must make sure that the object questions are not repeated in the same scene.
    """
    
    available_objects = [obj for obj in SELECTED_CLASSES if obj not in context.lower() and obj not in asked_objects]
    obj = random.choice(available_objects)
    question = f"Is in this point cloud a {obj} present? You MUST strictly answer 'Yes' or 'No'."
    return question, obj



def load_object_material_question(scene_id, obj, prob_correct=0.125):
    """Generate a material-based question for an object in a scene."""
    
    material_list_list_path = "material_list_updated.json"
    
    try:
        with open(material_list_list_path, "r") as f:
            material_data = json.load(f)
        
        if obj == "mirror" or obj == "window":
            correct_material = "glass"
        if obj == "monitor" or obj == "tv":
            correct_material = "plastic"
        elif obj == "refrigerator" or obj == "microwave":
            correct_material = "metal"
        else:
            correct_material = material_data[scene_id][obj]
        
        # Decide whether to use the correct material or a wrong one
        if random.random() < prob_correct:
            selected_material = correct_material
        else:
            wrong_materials = [mat for mat in MATERIAL_CLASSES if mat != correct_material]
            selected_material = random.choice(wrong_materials)

        # Build the question
        output_string = (
            f"Is the {obj} made of {selected_material}? "
            "You MUST strictly answer 'Yes' or 'No'."
        )
        # Return ground truth material maybe for object
        return output_string

    except FileNotFoundError:
        print(f"[WARNING] Material file not found: {material_list_list_path}")
        return "No additional context available.", None




def save_conversation_log(log_filename, scene_id, prompt, context, response):
    """Append conversation logs to a JSON file. Create file if it doesnt exist."""

    log_entry = {
        "scene_id": scene_id,
        "prompt": prompt,
        "context": context,
        "response": response
    }

    # If file exists, load existing data and append
    if os.path.exists(log_filename):
        try:
            with open(log_filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [] 
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"[WARNING] Log file corrupted or missing. Creating a new one.")
            data = []
    else:
        data = [] 

    # Append new log entry
    data.append(log_entry)

    # Save updated JSON file
    with open(log_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"[SUCCESS] Log entry appended to {log_filename}")

def init_model(args):
    # Model
    disable_torch_init()

    model_path = args.model_path 
    print(f'[INFO] Model name: {model_path}')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, use_cache=True, torch_dtype=args.torch_dtype).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    model.eval()

    mm_use_point_start_end = getattr(model.config, "mm_use_point_start_end", False)
    # Add special tokens ind to model.point_config
    point_backbone_config = model.get_model().point_backbone_config
    print(f"[DEBUG] Loading Point Backbone Config: {point_backbone_config}")

    if mm_use_point_start_end:
        if "v1" in model_path.lower():
            conv_mode = "vicuna_v1_1"
        else:
            raise NotImplementedError

        conv = conv_templates[conv_mode].copy()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    return model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv

def start_conversation(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv):
    '''Original conversation function of PointLLM.'''

    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    # The while loop will keep running until the user decides to quit
    print("[INFO] Starting conversation... Enter 'q' to exit the program and enter 'exit' to exit the current conversation.")
    while True:
        print("-" * 80)
        # Prompt for object_id
        object_id = input("[INFO] Please enter the object_id or 'q' to quit: ")

        # Check if the user wants to quit
        if object_id.lower() == 'q':
            print("[INFO] Quitting...")
            break
        else:
            # print info
            print(f"[INFO] Chatting with object_id: {object_id}.")
        
        # Update args with new object_id
        args.object_id = object_id.strip()
        
        # Load the point cloud data
        try:
            id, point_clouds = load_point_cloud(args)
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
        point_clouds = point_clouds.cuda().to(args.torch_dtype)

        # Reset the conversation template
        conv.reset()

        print("-" * 80)

        # Start a loop for multiple rounds of dialogue
        for i in range(100):
            # This if-else block ensures the initial question from the user is included in the conversation
            qs = input(conv.roles[0] + ': ')
            if qs == 'exit':
                break
            
            if i == 0:
                if mm_use_point_start_end:
                    qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
                else:
                    qs = default_point_patch_token * point_token_len + '\n' + qs

            # Append the new message to the conversation history
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            stop_str = keywords[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    point_clouds=point_clouds,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    max_length=2048,
                    top_p=0.95,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            # Append the model's response to the conversation history
            conv.pop_last_none_message()
            conv.append_message(conv.roles[1], outputs)
            print(f'{conv.roles[1]}: {outputs}\n')

def evaluate_scenes_manual(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv):
    '''Playground evaluation function.'''

    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"results/evaluation_log_{timestamp}.json"

    eva_prompt = input("Please enter a prompt that should be used for evaluation")
    counter = 55
    
    while counter < SCENE_RANGE:
        print("-" * 80)
        object_id = f"scene{counter:04d}_00"  # Format to sceneXXXX_00_8192
        scene_id = f"scene{counter:04d}"
        context = load_context(scene_id)

        print(f"[INFO] Evaluating object {object_id} with context...")
        
        # Update args with new object_id
        args.object_id = object_id.strip()
        
        # Load the point cloud data
        try:
            id, point_clouds = load_point_cloud(args)
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
        point_clouds = point_clouds.cuda().to(args.torch_dtype)

        # Reset the conversation template
        conv.reset()
        system_context = (
            "I provide you with some information of the scene extracted by an image classification model. Use it as a slight assistanceto get a better understanding of the scene\n\n"
        )
        qs = f"USER: {eva_prompt}"

        print("-" * 80)

        # Start a loop for multiple rounds of dialogue
        for i in range(1):
            # This if-else block ensures the initial question from the user is included in the conversation

            if i == 0:
                if mm_use_point_start_end:
                    qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
                else:
                    qs = default_point_patch_token * point_token_len + '\n' + qs

            # Append the new message to the conversation history
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            stop_str = keywords[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    point_clouds=point_clouds,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    max_length=2048,
                    top_p=0.95,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            print(f'{conv.roles[1]}: {outputs}\n')



        print(f"[DEBUG] Log file should be saved at: {log_filename}")
        save_conversation_log(log_filename, scene_id, eva_prompt, context, outputs)
        counter += 1



def evaluate_scenes(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv):
    '''Captioning task evaluation prompt generation.'''

    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    # The while loop will keep running until the user decides to quit

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"results/Captioning_NO_CONTEXT_log_{timestamp}.json"
    results_log = []
    counter = 0

    while counter < SCENE_RANGE:
        print("-" * 80)
        object_id = f"scene{counter:04d}_00"  # Format to sceneXXXX_00_8192
        scene_id = f"scene{counter:04d}"


        print(f"[INFO] Evaluating object {object_id} without context...")

        scene_log = {
            "scene_id": scene_id,
            "answers": []  
        }

        
        # Update args with new object_id
        args.object_id = object_id.strip()
        
        # Load the point cloud data
        try:
            id, point_clouds = load_point_cloud(args)
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
        point_clouds = point_clouds.cuda().to(args.torch_dtype)

        for _ in range(1):  # 3 rounds per scene
            conv.reset()
            eva_prompt = "Caption this 3D scene in detail."

            qs =  f"{eva_prompt}."
            print("-" * 80)

            if mm_use_point_start_end:
                qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
            else:
                qs = default_point_patch_token * point_token_len + '\n' + qs

                # Append the new message to the conversation history#
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            stop_str = keywords[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    point_clouds=point_clouds,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    max_length=2048,
                    top_p=0.95,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
                
            print(f'{conv.roles[1]}: {outputs}\n')

            scene_log["answers"].append(outputs)

        results_log.append(scene_log)

        print(f"[DEBUG] Log file should be saved at: {log_filename}")

        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(results_log, f, indent=4)
        counter += 1




def evaluate_scenes_classification_no_context(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv):
    '''Classification evaluation generation (without context).'''

    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"results/Cationing_w_NO_CONTEXT{timestamp}.json"
    results_log = []

    counter = 0

    while counter < SCENE_RANGE:
        print("-" * 80)
        object_id = f"scene{counter:04d}_00"  # Format to sceneXXXX_00_8192
        scene_id = f"scene{counter:04d}"
        context = load_context(scene_id)


        print(f"[INFO] Evaluating object {object_id} without...")

        scene_log = {
            "scene_id": scene_id,
            "objects_in_question": [],  
            "answers": []  
        }
        
        args.object_id = object_id.strip()
        
        # Load the point cloud data
        try:
            id, point_clouds = load_point_cloud(args)
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
        point_clouds = point_clouds.cuda().to(args.torch_dtype)
        asked_objects = []

        for _ in range(6):  # 6 Questions per scene

            print("-" * 80)

            # Start a loop for multiple rounds of dialogue. Double loop, as otherwise me might run into (llm dependent) memory issues
            for i in range(1):
                conv.reset()

                eva_prompt, obje = load_object_list_question_context(object_id=object_id, context=context, asked_objects=asked_objects)

                scene_log["objects_in_question"].append(obje)

                qs =  f"{eva_prompt}."
                asked_objects.append(obje)

                # This if-else block ensures the initial question from the user is included in the conversation
                if i == 0:
                    if mm_use_point_start_end:
                        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
                    else:
                        qs = default_point_patch_token * point_token_len + '\n' + qs

                # Append the new message to the conversation history#
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)

                prompt = conv.get_prompt()
                inputs = tokenizer([prompt])

                input_ids = torch.as_tensor(inputs.input_ids).cuda()

                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                stop_str = keywords[0]

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        point_clouds=point_clouds,
                        do_sample=True,
                        temperature=1.0,
                        top_k=50,
                        max_length=2048,
                        top_p=0.95,
                        stopping_criteria=[stopping_criteria])

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()

                print(f'{conv.roles[1]}: {outputs}\n')

                scene_log["answers"].append(outputs)

                object_in_scene = obje in scene_ground_truth.get(scene_id, [])

                if object_in_scene and 'yes' in outputs.lower():
                    material_question = load_object_material_question(scene_id, obje)
                    print("Mat question:", material_question)
                    matq = material_question
                    material_question = default_point_patch_token * point_token_len + '\n' + material_question

                    # Ask material question
                    conv.append_message(conv.roles[0], material_question)
                    conv.append_message(conv.roles[1], None)

                    material_prompt = conv.get_prompt()
                    material_inputs = tokenizer([material_prompt])
                    material_input_ids = torch.as_tensor(material_inputs.input_ids).cuda()

                    with torch.inference_mode():
                        material_output_ids = model.generate(
                            material_input_ids,
                            point_clouds=point_clouds,
                            do_sample=True,
                            temperature=1.0,
                            top_k=50,
                            max_length=2048,
                            top_p=0.95,
                            stopping_criteria=[stopping_criteria])

                    material_outputs = tokenizer.batch_decode(material_output_ids[:, material_input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
                    if material_outputs.endswith(stop_str):
                        material_outputs = material_outputs[:-len(stop_str)].strip()
                    print(f'{conv.roles[1]}:  {material_outputs}\n')
                    scene_log["answers"].append(f"Material Q: {matq} => {material_outputs}")





        results_log.append(scene_log)

        print(f"[DEBUG] Log file should be saved at: {log_filename}")

        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(results_log, f, indent=4)
        counter += 1


def evaluate_scenes_classification(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv):
    '''Classification evaluation generation (with context).'''

    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"results/evaluation_log_{timestamp}.json"
    results_log = []
    counter = 0

    while counter < SCENE_RANGE:
        print("-" * 80)
        object_id = f"scene{counter:04d}_00"  # Format to sceneXXXX_00_8192
        scene_id = f"scene{counter:04d}"
        context = load_context(scene_id)

        print(f"[INFO] Evaluating object {object_id} with context...")

        scene_log = {
            "scene_id": scene_id,
            "context": context,
            "objects_in_question": [],  
            "answers": []  
        }
        
        # Update args with new object_id
        args.object_id = object_id.strip()
        
        # Load the point cloud data
        try:
            id, point_clouds = load_point_cloud(args)
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
        point_clouds = point_clouds.cuda().to(args.torch_dtype)

        asked_objects = []

        for _ in range(6):  # 3 rounds per scene
            conv.reset()

 

            # zconv.append_message("SYSTEM_CONTEXT", system_context)
            print("-" * 80)

            # Start a loop for multiple rounds of dialogue
            for i in range(1):
                # This if-else block ensures the initial question from the user is included in the conversation
                conv.reset()

                eva_prompt, obje = load_object_list_question_context(object_id=object_id, context=context, asked_objects=asked_objects)

                scene_log["objects_in_question"].append(obje)

                qs =  f"{eva_prompt}."

                asked_objects.append(obje)

                if i == 0:
                    if mm_use_point_start_end:
                        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
                    else:
                        qs = default_point_patch_token * point_token_len + '\n' + qs

                # Append the new message to the conversation history#
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)

                prompt = conv.get_prompt()
                inputs = tokenizer([prompt])

                input_ids = torch.as_tensor(inputs.input_ids).cuda()

                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                stop_str = keywords[0]

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        point_clouds=point_clouds,
                        do_sample=True,
                        temperature=1.0,
                        top_k=50,
                        max_length=2048,
                        top_p=0.95,
                        stopping_criteria=[stopping_criteria])

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()

                print(f'{conv.roles[1]}: {outputs}\n')

                scene_log["answers"].append(outputs)
                object_in_scene = False
                if obje == "table" or obje == "desk":
                    if "table" in scene_ground_truth.get(scene_id, []) or "desk" in scene_ground_truth.get(scene_id, []):
                        object_in_scene = True
                elif obje == "monitor" or obje == "tv":
                    if "monitor" in scene_ground_truth.get(scene_id, []) or "tv" in scene_ground_truth.get(scene_id, []):
                        object_in_scene = True
                elif obje == "chair" or obje == "armchair":
                    if "chair" in scene_ground_truth.get(scene_id, []) or "armchair" in scene_ground_truth.get(scene_id, []):
                        object_in_scene = True
                elif obje in scene_ground_truth.get(scene_id, []):
                        object_in_scene = True



                if object_in_scene and 'yes' in outputs.lower():
                    material_question = load_object_material_question(scene_id, obje)

                    print("Mat question:", material_question)
                    matq = material_question
                    material_question = default_point_patch_token * point_token_len + '\n' + material_question

                    # Ask material question
                    conv.append_message(conv.roles[0], material_question)
                    conv.append_message(conv.roles[1], None)

                    material_prompt = conv.get_prompt()
                    material_inputs = tokenizer([material_prompt])
                    material_input_ids = torch.as_tensor(material_inputs.input_ids).cuda()

                    with torch.inference_mode():
                        material_output_ids = model.generate(
                            material_input_ids,
                            point_clouds=point_clouds,
                            do_sample=True,
                            temperature=1.0,
                            top_k=50,
                            max_length=2048,
                            top_p=0.95,
                            stopping_criteria=[stopping_criteria])

                    material_outputs = tokenizer.batch_decode(material_output_ids[:, material_input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
                    if material_outputs.endswith(stop_str):
                        material_outputs = material_outputs[:-len(stop_str)].strip()
                    print(f'{conv.roles[1]}:  {material_outputs}\n')
                    scene_log["answers"].append(f"Material Q: {matq} => {material_outputs}")





        results_log.append(scene_log)

        print(f"[DEBUG] Log file should be saved at: {log_filename}")

        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(results_log, f, indent=4)
        counter += 1


if __name__ == "__main__":
    random.seed(21)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, \
       default="RunsenXu/PointLLM_7B_v1.2")

    parser.add_argument("--data_path", type=str, default="data/objaverse_data")
    parser.add_argument("--torch_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    args = parser.parse_args()

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    args.torch_dtype = dtype_mapping[args.torch_dtype]

    model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv = init_model(args)
    
    evaluate_scenes_classification(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv)
    # evaluate_scenes(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv) 