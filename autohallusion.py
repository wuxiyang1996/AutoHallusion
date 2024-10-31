import numpy as np
import time
import logging, sys, os, json, shutil
from tqdm import tqdm
import traceback 
import random
from PIL import Image

from utils.utils_merge import import_functions_given_model_type, load_cfg_given_model_type
from utils.utils_eval import evaluation_existence_removed_pred_model_simple, evaluation_spatial_relation_removed_pred_model_simple
from utils.utils import (OwlViTProcessor, OwlViTForObjectDetection, scene_thinking, object_detection, spatial_gt_generation,
                         resize_img_n_store, target_obj_decide, vanilla_scene_img_generation)


verbose = False

# Experiment tag to store data
exp_tag = 'corr_obj_removal'
exp_name = "exi_exp_{}".format(exp_tag)
use_dataset = False # Using Synthetic Data

scene_constrain = None # Add constraints on the scene themes

obj_count = 5 # Number of correlated objects generation within the scene image
list_obj_count = 5 # Number of object to be detected from the given image
max_attempt = 5 # Attempt number to remove objects from the scene

# Specify the model type among:
# 'gemini', 'claude', 'gpt4v', 'llava', 'minigpt4'
obj_think_model_type, img_caption_model_type = 'gpt4v', 'gpt4v'

# Folder to store the experiment data and logs
exp_dir = "./exp_{}_{}_{}_use_dataset-{}".format(exp_tag, obj_think_model_type, img_caption_model_type, use_dataset)

save_dir = "{}/{}/".format(exp_dir, exp_name)

# Load functions from the utils
(
    generate_noun_given_scene_aimodel,
    random_obj_thinking_aimodel,
    irrelevant_obj_thinking_aimodel,
    gt_generation_aimodel,
    gt_generation_multi_obj_removal_aimodel,
    image_caption_aimodel,
    vqa_aimodel,
    filter_remove_obj_under_scene_aimodel,
    filter_most_irrelevant_aimodel,
    list_objects_given_img_aimodel,
    correlated_obj_thinking_aimodel,
    correlated_example_create_aimodel,
    safe_remove_dir,
    close_logger
) = import_functions_given_model_type(obj_think_model_type, img_caption_model_type)

(temp_generate_noun_given_scene, temp_filter_remove_obj_under_scene, temp_filter_most_irrelevant,
 temp_random_obj_thinking, temp_irrelevant_obj_thinking, temp_correlated_obj_thinking) = load_cfg_given_model_type(
    obj_think_model_type)

# Initialize the timestamps and storage path for images generated
case_name = str(int(time.time()))
save_loc = save_dir + "{}/".format(case_name)

init_img_path = "init.png"
result_img_path = "results.png"
target_obj_path = "target_obj.png"

if not os.path.exists(save_loc):
    os.makedirs(save_loc)

# Initialize the images (initial and AutoHallusion) for storage
log_path = os.path.join(save_loc, "output.log")
init_img_path = os.path.join(save_loc, init_img_path)
result_img_path = os.path.join(save_loc, result_img_path)
target_obj_path = os.path.join(save_loc, target_obj_path)

# Initialize the logger
logger = logging.getLogger(case_name)
fileHandler = logging.FileHandler(log_path, mode='w')
fileHandler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(message)s')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.CRITICAL)
logger.addHandler(sh)

# Load the object detection model (Owl-ViT)
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Generate scene image, allowing some level of diversity
scene_name = scene_thinking(constraint=scene_constrain, temperature=1.5)

# Generate several objects based on the scene
target_obj = None

while target_obj is None:
    vanilla_scene_img_generation(init_img_path, scene_name, num_obj=obj_count)
    found_obj_list = list_objects_given_img_aimodel(init_img_path, list_obj_count)

    if len(found_obj_list) >= list_obj_count:
        # Choose one object for removal based on the scene
        target_obj, target_obj_bbox = target_obj_decide(init_img_path, result_img_path,
                                                        found_obj_list,
                                                        processor, model, save_loc,
                                                        max_attempt=max_attempt)
        
# generate the image based on the provided objects
scene_key_char = list([val for val in scene_name if val.isalpha() or val.isnumeric()])
scene_key = "".join(scene_key_char).lower()

target_obj_key_char = list([val for val in target_obj if val.isalpha() or val.isnumeric()])
target_obj_key = "".join(target_obj_key_char).lower()

scene_details = {
    "scene_name": scene_name,
    "found_obj_list": found_obj_list,
    "init_path": init_img_path,
    "result_path": result_img_path,
    "target_obj": target_obj,
    "target_obj_bbox": target_obj_bbox,
    "target_obj_path": target_obj_path
}


if verbose:
    # Declare hallucination case information through warning messages
    print("[Input] scene constrain: " + str(scene_constrain))
    print("[Generated] scene name: " + scene_name)
    print("[Target Model Generated] Found objects in the initial image: " + str(found_obj_list))
    print("[Target Model Generated] Target object (Removed, for attack): " + target_obj)

# Re-do the image ground after object removal
mask_img, mask_bbox = None, None
for word in found_obj_list:
    if word != target_obj:
        text_input = "a photo of " + word
        mask_img, mask_bbox = object_detection(init_img_path, text_input, processor, model, mask_img,
                                               mask_bbox, save_prefix=save_loc)
        
# Generate the ground truth of existence questions
ground_truth = gt_generation_multi_obj_removal_aimodel(result_img_path, mask_bbox, scene_name,
                                                     target_obj,  save_prefix=save_loc)

if verbose:
    # List detected object name and captions
    print("[Detection Model] detected objects: " + str(ground_truth["object_name"]))
    print("[Target Model Generated] detected object captions: " + str(ground_truth["object_description"]))

    print(
        "[Target Model Generated] Target object caption (Removed, for attack): " + ground_truth[
            "non_exist_target_object_description"])

# image level caption
result_caption = image_caption_aimodel(result_img_path)
ground_truth["result_description"] = result_caption

if verbose:
    print("[Target Model Generated] image-level caption: " + ground_truth["result_description"])

# Generate the ground truth of spatial relations
ground_truth = spatial_gt_generation(ground_truth, target_obj_bbox, mask_bbox, enable=True)

spatial_relation_str = ""
for i in range(len(ground_truth["spatial_relation"])):
    spatial_relation_str += str(ground_truth["spatial_relation"][i]) + "; "

if verbose:
    print(
        "[Target Model Generated] Spatial Relations (w.r.t. detected objects): " + spatial_relation_str)
    print("[Target Model Generated] Irrelevant-detected Object Distances): " + str(
        ground_truth["spatial_distance"]))

# Evaluate answers for existence questions based on the given evaluation models
success1, question1, pred_answer_simple1, queried_info1 = evaluation_existence_removed_pred_model_simple(result_img_path,
                                                                             ground_truth,
                                                                             vqa_model_func=vqa_aimodel,
                                                                             logger=logger, debug=True, notebook_print=True)


# Evaluate answers for spatial relation questions based on the given evaluation models
success2, question2, pred_answer_simple2, queried_info2 = evaluation_spatial_relation_removed_pred_model_simple(
                    result_img_path, ground_truth, vqa_model_func=vqa_aimodel, logger=logger, debug=True, notebook_print=True)


print("")
print("")
print("#####")
print("")
print("")
print("generated image path:" + result_img_path)
print("generated question1:" + question1)
print("generated question2:" + question2)