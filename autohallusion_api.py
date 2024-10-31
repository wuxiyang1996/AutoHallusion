import numpy as np
import time
import logging, sys, os, json, shutil
from tqdm import tqdm
import traceback 
import random
from PIL import Image

from utils.utils_merge import import_functions_given_model_type, load_cfg_given_model_type
from utils.utils_eval import (evaluation_existence_given_pred_model_simple, evaluation_spatial_relation_give_pred_model_simple,
                            evaluation_existence_removed_pred_model_simple, evaluation_spatial_relation_removed_pred_model_simple,
                            evaluation_correlated_existence_pred_model_simple)

from utils.utils import (OwlViTProcessor, OwlViTForObjectDetection, scene_thinking, generate_image_given_scene, object_detection, spatial_gt_generation,
                         Addition_Image_Operation_VD_stitch, Addition_Image_Operation_VD_stitch_correlation, resize_img_n_store, target_obj_decide, vanilla_scene_img_generation, convert_into_sequare)


def generate_data(output_image_dir, count, mtype_ratio = [1, 1, 1], qtype_ratio=[1,1], max_attempt_ratio = 10, verbose = False, remove_log = False):
    
    if mtype_ratio is None:
        ab_insert_ct = int(count / 3)
        paired_insert_ct = int(count / 3)
        corr_remove_ct = count - ab_insert_ct - paired_insert_ct
    else:
        ab_insert_ct = int(count * mtype_ratio[0] / sum(mtype_ratio))
        paired_insert_ct = int(count * mtype_ratio[1] / sum(mtype_ratio))
        corr_remove_ct = count - ab_insert_ct - paired_insert_ct

    count_lst = [ab_insert_ct, paired_insert_ct, corr_remove_ct]
    method_lst = [abnormal_insert, paired_insert, correlated_removal]
    # method_lst = [correlated_removal, correlated_removal, correlated_removal]
    method_name = ["abnormal_insert", "paired_insert", "correlated_removal"]

    final_vqa_list = []

    for i in range(len(method_lst)):

        function = method_lst[i]

        if qtype_ratio is None:

            vqa_sublist = []

            counter = 0
            while len(vqa_sublist) < count_lst[i] and counter < count_lst[i] * max_attempt_ratio:
                try:
                    final_image_path, question1, gt1, question2, gt2 = function(output_image_dir)
                    if question1 is not None:
                        vqa_sublist.append((final_image_path, question1, gt1))

                    if question2 is not None:
                        vqa_sublist.append((final_image_path, question2, gt2))
                except:
                    continue

                print("{}: {} / {}".format(method_name[i], counter, count_lst[i] * max_attempt_ratio))

                counter += 1

            final_vqa_list.extend(vqa_sublist)

        else:

            vqa_sublist_exi = []
            vqa_sublist_sp = []

            exi_count = int(count_lst[i] * qtype_ratio[0] / sum(qtype_ratio))
            sp_count = count_lst[i] - exi_count
            # from IPython import embed;embed()

            counter = 0
            while len(vqa_sublist_exi) < exi_count and counter < exi_count * max_attempt_ratio:
                try:
                    final_image_path, question1, gt1, question2, gt2 = function(output_image_dir)
                    if question1 is not None:
                        vqa_sublist_exi.append((final_image_path, question1, gt1))

                    if question2 is not None:
                        vqa_sublist_sp.append((final_image_path, question2, gt2))
                except:
                    continue

                print("{} - exi: {} / {}".format(method_name[i], counter, exi_count * max_attempt_ratio))

                counter += 1

            counter = 0
            while len(vqa_sublist_sp) < sp_count and counter < sp_count * max_attempt_ratio:
                try:
                    final_image_path, question1, gt1, question2, gt2 = function(output_image_dir)
                    if question1 is not None:
                        vqa_sublist_exi.append((final_image_path, question1, gt1))

                    if question2 is not None:
                        vqa_sublist_sp.append((final_image_path, question2, gt2))
                except:
                    continue

                print("{} - sp: {} / {}".format(method_name[i], counter, sp_count * max_attempt_ratio))

                counter += 1

            if len(vqa_sublist_exi) < exi_count and len(vqa_sublist_sp) < sp_count:

                final_vqa_list.extend(vqa_sublist_exi)
                final_vqa_list.extend(vqa_sublist_sp)
            elif len(vqa_sublist_exi) < exi_count:
                final_vqa_list.extend(vqa_sublist_exi)
                final_vqa_list.extend(vqa_sublist_sp[:min(len(vqa_sublist_sp), count_lst[i] - len(vqa_sublist_exi))])
            elif len(vqa_sublist_sp) < sp_count:
                final_vqa_list.extend(vqa_sublist_sp)
                final_vqa_list.extend(vqa_sublist_exi[:min(len(vqa_sublist_exi), count_lst[i] - len(vqa_sublist_sp))])

            else:
                final_vqa_list.extend(vqa_sublist_exi[:exi_count])
                final_vqa_list.extend(vqa_sublist_sp[:sp_count])

    return final_vqa_list


def abnormal_insert(output_image_dir, scene_constrain = None, verbose = False, remove_log = False):

    # Experiment tag to store data
    exp_tag = 'abnormal_obj_insertion'
    exp_name = "exi_exp_{}".format(exp_tag)
    use_dataset = False # Using Synthetic Data

    # scene_constrain = None # Add constraints on the scene themes
    irrelevant_obj_category = None # Add constraints on the object to be inserted

    obj_count = 5 # Number of correlated objects generation within the scene image

    # Specify the model type among:
    # 'gemini', 'claude', 'gpt4v', 'llava', 'minigpt4'
    obj_think_model_type, img_caption_model_type = 'gpt4v', 'gpt4v'

    # Folder to store the experiment data and logs
    exp_dir = "./exp_{}_{}_{}_use_dataset-{}".format(exp_tag, obj_think_model_type, img_caption_model_type, use_dataset)

    save_dir = "{}/{}/".format(exp_dir, exp_name)

    attribute_category_list = []

    object_size = (200, 200) # Object size (Abnormal / Paired Object Insertion)
    scene_img_raw_size = None

    # Ablation study: Object-Scene Alignment (for Abnormal Object Insertion)
    # Initial: Intentionally choose abnormal object to insert into the scene
    obj_random = False # Using random object to insert
    scene_ramdom = False # Randomly shuffle the scene images
    same = False # Using objects within the same context to insert

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

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    # Initialize the images (initial and AutoHallusion) for storage
    log_path = os.path.join(save_loc, "output.log")
    init_img_path = os.path.join(save_loc, init_img_path)
    result_img_path = os.path.join(save_loc, result_img_path)

    # Initialize the logger
    logger = logging.getLogger(case_name)
    fileHandler = logging.FileHandler(log_path, mode='w')
    fileHandler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    # sh = logging.StreamHandler(sys.stdout)
    # sh.setLevel(logging.CRITICAL)
    # logger.addHandler(sh)

    # Generate scene image, allowing some level of diversity
    scene_name = scene_thinking(constraint=scene_constrain, temperature=1.5)

    # Retrieve words aligned with the scene image
    if same:
        word_list, _ = generate_noun_given_scene_aimodel(num=obj_count + 1, scene=scene_name,
                                                        temperature=temp_generate_noun_given_scene)
    else:
        word_list, _ = generate_noun_given_scene_aimodel(num=obj_count, scene=scene_name,
                                                        temperature=temp_generate_noun_given_scene)
    generate_image_given_scene(word_list, scene_name, image_name=init_img_path)

    # generate the image based on the provided objects
    scene_key_char = list([val for val in scene_name if val.isalpha() or val.isnumeric()])
    scene_key = "".join(scene_key_char).lower()

    scene_details = {
        "scene_name": scene_name,
        "word_list": word_list,
        "path": init_img_path
    }

    # Generate several objects based on the scene
    # Constraints to eliminate some objects already created
    diversity_cond = ""

    if obj_random:
        if scene_ramdom:
            irrelevant_obj = random_obj_thinking_aimodel(None, temperature=temp_random_obj_thinking,
                                                        cond=diversity_cond)
        else:
            irrelevant_obj = random_obj_thinking_aimodel(scene_name,
                                                        temperature=temp_random_obj_thinking,
                                                        cond=diversity_cond)
    else:
        # Generate one irrelevant object based on the scene
        irrelevant_obj = irrelevant_obj_thinking_aimodel(scene_name, word_list,
                                                        category=irrelevant_obj_category,
                                                        temperature=temp_irrelevant_obj_thinking,
                                                        cond=diversity_cond)

    if verbose:
        # Declare hallucination case information through warning messages
        print("[Input] scene constrain: " + str(scene_constrain))
        print("[Generated] scene name: " + scene_name)
        print("[Target Model Generated] relevant objects: " + str(word_list))
        print("[Input] irrelevant object category: " + str(irrelevant_obj_category))
        print("[Target Model Generated] irrelevant object: " + irrelevant_obj)

    # Load the object detection model (Owl-ViT)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    
    # generate list of detected objects
    mask_img, mask_bbox = None, None
    for word in word_list:
        text_input = "a photo of " + word
        mask_img, mask_bbox = object_detection(init_img_path, text_input, processor, model, mask_img,
                                            mask_bbox, save_prefix=save_loc)

    # add new item -- need to prepare mask_region_img.png, obj.png, pure_obj.png and obj_mask.png
    result_img, irrelevant_obj_bbox, irrelevant_obj_attribute = \
        Addition_Image_Operation_VD_stitch(init_img_path=init_img_path,
                                        existing_bbox=mask_bbox,
                                        attribute_category_list=attribute_category_list,
                                        add_object=irrelevant_obj, path_prefix=save_loc,
                                        out_image_name=result_img_path,
                                        add_object_size=object_size, overlapped_ratio=0.5,
                                        scene_img_raw_size=scene_img_raw_size)

    # Generate the ground truth of existence questions
    ground_truth = gt_generation_aimodel(init_img_path, mask_bbox, scene_name, irrelevant_obj,
                                        irrelevant_obj_attribute, save_prefix=save_loc)

    if verbose:
        # List detected object name and captions
        print("[Detection Model] detected objects: " + str(ground_truth["object_name"]))
        print(
            "[Target Model Generated] detected object captions: " + str(ground_truth["object_description"]))

        print("[Target Model Generated] irrelevant object caption: " + ground_truth[
            "irrelevant_object_description"])

    # image level caption
    result_caption = image_caption_aimodel(result_img_path)
    ground_truth["result_description"] = result_caption

    if verbose:
        print("[Target Model Generated] image-level caption: " + ground_truth["result_description"])

    # Generate the ground truth of spatial relations
    ground_truth = spatial_gt_generation(ground_truth, irrelevant_obj_bbox, mask_bbox, enable=True)

    spatial_relation_str = ""
    for i in range(len(ground_truth["spatial_relation"])):
        spatial_relation_str += str(ground_truth["spatial_relation"][i]) + "; "

    if verbose:
        print(
            "[Target Model Generated] Spatial Relations (w.r.t. detected objects): " + spatial_relation_str)
        print("[Target Model Generated] Irrelevant-detected Object Distances): " + str(
            ground_truth["spatial_distance"]))

    # Evaluate answers for existence questions based on the given evaluation models
    success1, question1, pred_answer_simple1, queried_info1 = evaluation_existence_given_pred_model_simple(result_img_path,
                                                                            ground_truth, result_caption,
                                                                            vqa_model_func=vqa_aimodel,
                                                                            logger=logger, debug=True, 
                                                                            notebook_print=False)

    # Evaluate answers for spatial relation questions based on the given evaluation models
    success2, question2, pred_answer_simple2, queried_info2 = evaluation_spatial_relation_give_pred_model_simple(result_img_path,
                                                                                            ground_truth,
                                                                                            vqa_model_func=vqa_aimodel,
                                                                                            logger=logger,
                                                                                            debug=True,
                                                                                            notebook_print=False)

    final_image_path = os.path.join(output_image_dir, case_name + ".png")
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    shutil.copy(result_img_path, final_image_path)

    if remove_log:
        shutil.rmtree(save_loc)
    if verbose:
        print("")
        print("")
        print("##### abnormal_insert #####")
        print("")
        print("")
        print("generated image path:" + final_image_path)
        print("generated question1:" + question1)
        print("generated question2:" + question2)

    return final_image_path, question1, queried_info1, question2, queried_info2


def paired_insert(output_image_dir, scene_constrain = None, verbose = False, remove_log = False):


    # Experiment tag to store data
    exp_tag = 'paired_obj_insertion'
    exp_name = "exi_exp_{}".format(exp_tag)
    use_dataset = False # Using Synthetic Data

    irrelevant_obj_category = None # Add constraints on the object to be inserted

    obj_count = 5 # Number of correlated objects generation within the scene image

    # Specify the model type among:
    # 'gemini', 'claude', 'gpt4v', 'llava', 'minigpt4'
    obj_think_model_type, img_caption_model_type = 'gpt4v', 'gpt4v'

    # Folder to store the experiment data and logs
    exp_dir = "./exp_{}_{}_{}_use_dataset-{}".format(exp_tag, obj_think_model_type, img_caption_model_type, use_dataset)

    save_dir = "{}/{}/".format(exp_dir, exp_name)

    attribute_category_list = []

    object_size = (200, 200) # Object size (Abnormal / Paired Object Insertion)
    scene_img_raw_size = None

    # Ablation study: Object-Scene Alignment (for Abnormal Object Insertion)
    # Initial: Intentionally choose abnormal object to insert into the scene
    obj_random = False # Using random object to insert
    scene_ramdom = False # Randomly shuffle the scene images
    same = False # Using objects within the same context to insert

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


    temp_generate_noun_given_scene, temp_filter_remove_obj_under_scene, temp_filter_most_irrelevant, \
    temp_random_obj_thinking, temp_irrelevant_obj_thinking, temp_correlated_obj_thinking = load_cfg_given_model_type(obj_think_model_type)

    # Initialize the timestamps and storage path for images generated
    case_name = str(int(time.time()))
    save_loc = save_dir + "{}/".format(case_name)

    init_img_path = "init.png"
    result_img_path = "results.png"

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    # Initialize the images (initial and AutoHallusion) for storage
    log_path = os.path.join(save_loc, "output.log")
    init_img_path = os.path.join(save_loc, init_img_path)
    result_img_path = os.path.join(save_loc, result_img_path)

    # Initialize the logger
    logger = logging.getLogger(case_name)
    fileHandler = logging.FileHandler(log_path, mode='w')
    fileHandler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    # sh = logging.StreamHandler(sys.stdout)
    # sh.setLevel(logging.CRITICAL)
    # logger.addHandler(sh)

    # Generate scene image, allowing some level of diversity
    scene_name = scene_thinking(constraint=scene_constrain, temperature=1.5)

    # Retrieve words aligned with the scene image
    if same:
        word_list, _ = generate_noun_given_scene_aimodel(num=obj_count + 1, scene=scene_name,
                                                        temperature=temp_generate_noun_given_scene)
    else:
        word_list, _ = generate_noun_given_scene_aimodel(num=obj_count, scene=scene_name,
                                                        temperature=temp_generate_noun_given_scene)
    generate_image_given_scene(word_list, scene_name, image_name=init_img_path)

    # generate the image based on the provided objects
    scene_key_char = list([val for val in scene_name if val.isalpha() or val.isnumeric()])
    scene_key = "".join(scene_key_char).lower()

    scene_details = {
        "scene_name": scene_name,
        "word_list": word_list,
        "path": init_img_path
    }

    # Constraints to eliminate some objects already created
    diversity_cond = ""

    # Generate one irrelevant object based on the scene
    correlated_obj = correlated_obj_thinking_aimodel(temperature=temp_correlated_obj_thinking, cond=diversity_cond)

    # Generate the target object and disturbing object from the generated object pair
    target_obj, disturbing_obj, target_attributes, disturbing_attributes \
        = correlated_example_create_aimodel(correlated_obj, attribute_category_list,
                                        save_prefix=save_loc)

    if verbose:
        # Declare hallucination case information through warning messages
        print("[Input] scene constrain: " + str(scene_constrain))
        print("[Generated] scene name: " + scene_name)
        print("[Target Model Generated] relevant objects: " + str(word_list))
        print("[Target Model Generated] target object (for addition attack): " + target_obj)
        print("[Target Model Generated] disturbance object (not exist, for correlation attack): " + disturbing_obj)

    # Load the object detection model (Owl-ViT)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # generate list of detected objects
    mask_img, mask_bbox = None, None
    for word in word_list:
        text_input = "a photo of " + word
        mask_img, mask_bbox = object_detection(init_img_path, text_input, processor, model, mask_img,
                                            mask_bbox, save_prefix=save_loc)


    # add new item -- need to prepare mask_region_img.png, obj.png, pure_obj.png and obj_mask.png
    result_img, irrelevant_obj_bbox = Addition_Image_Operation_VD_stitch_correlation(
        init_img_path=init_img_path, existing_bbox=mask_bbox, path_prefix=save_loc,
        out_image_name=result_img_path, add_object_size=object_size, overlapped_ratio=0.5,
        scene_img_raw_size=scene_img_raw_size)

    # Generate the ground truth of existence questions
    ground_truth = gt_generation_aimodel(init_img_path, mask_bbox, scene_name, target_obj,
                                    target_attributes, save_prefix=save_loc)

    if verbose:
        # List detected object name and captions
        print("[Detection Model] detected objects: " + str(ground_truth["object_name"]))
        print(
            "[Target Model Generated] detected object captions: " + str(ground_truth["object_description"]))

        print("[Target Model Generated] irrelevant object caption: " + ground_truth[
            "irrelevant_object_description"])

    # image level caption
    result_caption = image_caption_aimodel(result_img_path)
    ground_truth["result_description"] = result_caption
    ground_truth["disturbing_object"] = disturbing_obj

    if verbose:
        print("[Target Model Generated] image-level caption: " + ground_truth["result_description"])

    # Generate the ground truth of spatial relations
    ground_truth = spatial_gt_generation(ground_truth, irrelevant_obj_bbox, mask_bbox, enable=True)

    spatial_relation_str = ""
    for i in range(len(ground_truth["spatial_relation"])):
        spatial_relation_str += str(ground_truth["spatial_relation"][i]) + "; "

    if verbose:
        print(
            "[Target Model Generated] Spatial Relations (w.r.t. detected objects): " + spatial_relation_str)
        print("[Target Model Generated] Irrelevant-detected Object Distances): " + str(
            ground_truth["spatial_distance"]))

    # Evaluate answers for existence questions based on the given evaluation models
    success1, question1, pred_answer_simple1, queried_info1 = evaluation_existence_given_pred_model_simple(result_img_path, ground_truth,
                                                                    result_caption, vqa_model_func=vqa_aimodel, 
                                                                logger=logger, debug=True, notebook_print=False)

    if not success1:
        # Evaluate answers for correlated objects existence questions based on the given evaluation models
        success1, question1, pred_answer_simple1, queried_info1 = evaluation_correlated_existence_pred_model_simple(result_img_path,
                                                                                    ground_truth,
                                                                                    vqa_model_func=vqa_aimodel,
                                                                                    logger=logger, debug=True, notebook_print=False)

    # Evaluate answers for spatial relation questions based on the given evaluation models
    success2, question2, pred_answer_simple2, queried_info2 = evaluation_spatial_relation_give_pred_model_simple(result_img_path,
                                                                                    ground_truth, vqa_model_func=vqa_aimodel,
                                                                                    logger=logger, debug=True, notebook_print=False)

    final_image_path = os.path.join(output_image_dir, case_name + ".png")
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    shutil.copy(result_img_path, final_image_path)

    if remove_log:
        shutil.rmtree(save_loc)
    if verbose:
        print("")
        print("")
        print("##### paired_insert #####")
        print("")
        print("")
        print("generated image path:" + final_image_path)
        print("generated question1:" + question1)
        print("generated question2:" + question2)

    return final_image_path, question1, queried_info1, question2, queried_info2



def correlated_removal(output_image_dir, scene_constrain = None, verbose = False, remove_log = False):

    # Experiment tag to store data
    exp_tag = 'corr_obj_removal'
    exp_name = "exi_exp_{}".format(exp_tag)
    use_dataset = False # Using Synthetic Data

    # scene_constrain = None # Add constraints on the scene themes

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
    # sh = logging.StreamHandler(sys.stdout)
    # sh.setLevel(logging.CRITICAL)
    # logger.addHandler(sh)

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
                                                                                logger=logger, debug=True, notebook_print=False)


    # Evaluate answers for spatial relation questions based on the given evaluation models
    success2, question2, pred_answer_simple2, queried_info2 = evaluation_spatial_relation_removed_pred_model_simple(
                        result_img_path, ground_truth, vqa_model_func=vqa_aimodel, logger=logger, debug=True, notebook_print=False)


    final_image_path = os.path.join(output_image_dir, case_name + ".png")
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    shutil.copy(result_img_path, final_image_path)

    if remove_log:
        shutil.rmtree(save_loc)
    if verbose:
        print("")
        print("")
        print("##### correlated_removal #####")
        print("")
        print("")
        print("generated image path:" + final_image_path)
        print("generated question1:" + question1)
        print("generated question2:" + question2)

    return final_image_path, question1, queried_info1, question2, queried_info2


# final_image_path, question1, gt1, question2, gt2 = correlated_removal("./test/")
# print(final_image_path)
# print(question1)
# print(gt1)
# print(question2)
# print(gt2)

print(generate_data("./test/", 9, [1, 1, 1]))