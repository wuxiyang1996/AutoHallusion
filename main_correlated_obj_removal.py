#from utils_gemini import *
import numpy as np
import time
import logging, sys, os, json, shutil
from tqdm import tqdm
import traceback 
import random


from utils.utils_merge import import_functions_given_model_type, load_cfg_given_model_type
from utils.utils_eval import evaluation_existence_removed_pred_model, evaluation_spatial_relation_removed_pred_model
from utils.utils import (OwlViTProcessor, OwlViTForObjectDetection, scene_thinking, object_detection, spatial_gt_generation,
                         resize_img_n_store, target_obj_decide, vanilla_scene_img_generation)

# This is a helper function to replace the initial head path
def path_header_replace(path, new_header, case_id):
    file_name = os.path.split(path)[1]
    return os.path.join(new_header, case_id, file_name)

# Main function for abnormal object insertion
def run_exp(args, meta_log = "exp_results.json"):
    # Declare the hyper-parameters from the running file
    exp_name = args["exp_name"]
    save_dir = args["save_dir"]
    total = args["total"]

    diffusion = args["diffusion"]

    # Generate scene name
    scene_constrain = args["scene_constrain"]

    # Generate scene img
    new_scene_img = args["new_scene_img"]
    obj_count = args["obj_count"]
    list_obj_count = args["list_obj_count"]

    # Remove object
    new_removal_obj = args["new_removal_obj"]
    max_attempt = args["max_attempt"]

    database_scene_ref_path = args["database_scene_ref_path"]
    scene_gen_prob = args["scene_gen_prob"]
    diversity_prob = args["diversity_prob"]
    diversity_count = args["diversity_count"]

    # Reuse previous dataset hyperparameters
    reuse_dataset_scene_ref_path = args["dataset_scene_ref_path"]
    reuse_dataset_raw_data_path = args["dataset_raw_data_path"]

    reuse_dataset_scene_query_path = args["dataset_scene_query_path"]

    reuse_scene = args["reuse_scene"]
    reuse_obj_removal = args["reuse_obj_removal"]

    obj_think_model_type = args["obj_think_model_type"]
    img_caption_model_type = args["img_caption_model_type"]

    resize_img = args["resize_img"]

    # Load the model for retrieving objects and image caption
    print('verbose...merge_reuse obj_think_model_type {}, img_caption_model_type {}, reuse_scene {}, reuse_dataset_scene_ref_path {}'.format(obj_think_model_type, img_caption_model_type, reuse_scene, reuse_dataset_scene_ref_path))

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

    cur = 0

    # Declare the data storage path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result = {}

    # Restore the checkpoints once interrupted
    if os.path.exists(os.path.join(save_dir, meta_log)):
        with open(os.path.join(save_dir, meta_log), "r") as f:
            old_result = json.load(f)
        for folders in os.listdir(save_dir):  # loop over all files
            if os.path.isdir(os.path.join(save_dir, folders)):  # if it's a directory
                if folders in old_result:
                    cur += 1  # increment counter
                    result[folders] = old_result[folders]
                else:
                    shutil.rmtree(save_dir + "{}/".format(folders))
            
        assert len(result.values()) == cur

    with open(os.path.join(save_dir, meta_log), 'w') as f:
        # indent=2 is not needed but makes the file human-readable 
        # if the data is nested
        json.dump(result, f, indent=2)

    # Load the scene image path
    if os.path.exists(database_scene_ref_path):
        with open(database_scene_ref_path, "r") as f:
            database_scene_ref = json.load(f)
    else:
        database_scene_ref = {}
        with open(database_scene_ref_path, 'w') as f:
            json.dump(database_scene_ref, f, indent=2)

    # Create the helper json file to query scene and object generated if set the reuse tag
    if reuse_scene:

        all_files = os.listdir(reuse_dataset_raw_data_path)
        with open(reuse_dataset_scene_ref_path, 'r') as f:
            # indent=2 is not needed but makes the file human-readable
            # if the data is nested
            scene_db = json.load(f)

            print('verbose..load scene_db from {}'.format(reuse_dataset_scene_ref_path))

        if not os.path.exists(reuse_dataset_scene_query_path):
            scene_query_db = {}
            for key in list(scene_db.keys()):
                for i in range(len(scene_db[key])):
                    id = scene_db[key][i]["path"].split("/")[-2]
                    if id in all_files:
                        scene_query_db[id] = {'scene_name': key, "position": str(i)}

            with open(reuse_dataset_scene_query_path, 'w') as f:
                json.dump(scene_query_db, f, indent=2)
        else:
            with open(reuse_dataset_scene_query_path, 'r') as f:
                # indent=2 is not needed but makes the file human-readable
                # if the data is nested
                scene_query_db = json.load(f)

        reuse_scene_id = list(scene_query_db.keys())

    # Load the object detection model (Owl-ViT)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # Pointer to restore the process
    ptr = cur
    if reuse_scene:
        if total > len(list(scene_query_db.keys())):
            total = len(list(scene_query_db.keys()))

    # Main loop to run the hallucination cases generation
    for _ in tqdm(range(total - cur)):

        completed = False

        while not completed:
            
            try:
                # Initialize the timestamps and storage path for images generated
                case_name = str(int(time.time()))

                save_loc = save_dir + "{}/".format(case_name)

                init_img_path = "init.png"
                result_img_path = "results.png"
                target_obj_path = "target_obj.png"

                attribute_category_list = []

                # scene_constrain = "Indoor Scene"
                # irrelevant_obj_category = "animals"

                if not os.path.exists(save_loc):
                    os.makedirs(save_loc)

                log_path = os.path.join(save_loc, "output.log")
                init_img_path = os.path.join(save_loc, init_img_path)
                result_img_path = os.path.join(save_loc, result_img_path)
                target_obj_path = os.path.join(save_loc, target_obj_path)

                # Initialize the logger
                logger = logging.getLogger(case_name)
                # logging.basicConfig(encoding='utf-8', level=logging.CRITICAL)
                # logging.basicConfig(encoding='utf-8', level=logging.WARNING)
                # logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.INFO)
                fileHandler = logging.FileHandler(log_path, mode='w')
                fileHandler.setLevel(logging.WARNING)
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                fileHandler.setFormatter(formatter)
                logger.addHandler(fileHandler)
                sh = logging.StreamHandler(sys.stdout)
                sh.setLevel(logging.CRITICAL)
                logger.addHandler(sh)

                # generate the name of the scene
                gen_new_scene = np.random.random() < scene_gen_prob | len(list(database_scene_ref.values())) > 0
                scene_diversity = np.random.random() < diversity_prob

                # Load the scene from dataset when using existing scene images
                # Need to adjust the image if necessary to fit the image editing tools
                if reuse_scene:
                    scene_load_id = reuse_scene_id[ptr]

                    scene_name_queried = scene_query_db[scene_load_id]['scene_name']
                    scene_name_queried_pos = int(scene_query_db[scene_load_id]['position'])

                    scene_info_loaded = scene_db[scene_name_queried][scene_name_queried_pos]

                    # Align two datasets
                    if "init_path" in scene_info_loaded:
                        path_key = "init_path"
                    else:
                        path_key = "path"

                    scene_name = scene_info_loaded["scene_name"]
                    shutil.copyfile(
                        path_header_replace(scene_info_loaded[path_key], reuse_dataset_raw_data_path, scene_load_id),
                        init_img_path)

                    if resize_img:
                        resize_img_n_store(init_img_path)

                    # Align two datasets
                    if "found_obj_list" in scene_info_loaded:
                        word_key = "found_obj_list"
                    else:
                        word_key = "word_list"

                    found_obj_list = scene_info_loaded[word_key]

                    if not reuse_obj_removal:
                        target_obj = scene_info_loaded["target_obj"]
                        target_obj_bbox = scene_info_loaded["target_obj_bbox"]

                        # There's a bug in the initial path storage
                        temp_path = os.path.join(os.path.split(scene_info_loaded["init_path"])[0],
                                                 scene_info_loaded["target_obj_path"])
                        shutil.copyfile(path_header_replace(temp_path, reuse_dataset_raw_data_path,
                                                            scene_load_id), target_obj_path)
                        shutil.copyfile(path_header_replace(scene_info_loaded["result_path"], reuse_dataset_raw_data_path,
                                                            scene_load_id), result_img_path)

                        if resize_img:
                            resize_img_n_store(target_obj_path)
                            resize_img_n_store(result_img_path)

                    else:
                        # Generate several objects based on the scene
                        target_obj = None

                        while target_obj is None:
                            found_obj_list = list_objects_given_img_aimodel(init_img_path, list_obj_count)

                            if len(found_obj_list) >= list_obj_count:
                                # Choose one object for removal based on the scene
                                target_obj, target_obj_bbox = target_obj_decide(init_img_path, result_img_path,
                                                                                found_obj_list,
                                                                                processor, model, save_loc,
                                                                                max_attempt=max_attempt)

                    gen_new_scene = True

                else:
                    # Generate scene image, allowing some level of diversity
                    if gen_new_scene:
                        if scene_diversity:
                            scene_lst = [sce[0]["scene_name"] for sce in list(database_scene_ref.values())]
                            if len(scene_lst) > diversity_count:
                                scene_lst = random.sample(scene_lst, diversity_count)
                            if scene_constrain is None:
                                scene_constrain = "Try to generate a different scene from the following: {}.".format(
                                    ",".join(scene_lst))
                            else:
                                scene_constrain += "; and try to generate a different scene from the following: {}.".format(
                                    ",".join(scene_lst))

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
                    else:
                        # Reuse scene image from previous results
                        load = random.choice(list(database_scene_ref.values()))
                        scene_name = load["scene_name"]

                        if new_scene_img:
                            target_obj = None

                            while target_obj is None:
                                vanilla_scene_img_generation(init_img_path, scene_name, num_obj=obj_count)
                                found_obj_list = list_objects_given_img_aimodel(init_img_path, list_obj_count)

                                # Choose one object for removal based on the scene
                                if len(found_obj_list) >= list_obj_count:
                                    target_obj, target_obj_bbox = \
                                        target_obj_decide(init_img_path, result_img_path, found_obj_list, processor,
                                                          model, save_loc,
                                                          max_attempt=max_attempt)

                            gen_new_scene = True
                        else:
                            found_obj_list = load["found_obj_list"][:obj_count]
                            shutil.copyfile(load["init_path"], init_img_path)

                            if new_removal_obj or load["target_obj"] is None:
                                # Choose one object for removal based on the scene
                                target_obj, target_obj_bbox = \
                                    target_obj_decide(init_img_path, result_img_path, found_obj_list, processor, model,
                                                      save_loc, max_attempt=max_attempt)
                                gen_new_scene = True
                            else:
                                target_obj = load["target_obj"]
                                target_obj_bbox = load["target_obj_bbox"]
                                shutil.copyfile(load["result_path"], result_img_path)
                                shutil.copyfile(load["target_obj_path"], target_obj_path)

                # generate the image based on the provided objects
                if gen_new_scene:

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

                    concat_key = scene_key + "_" + target_obj_key

                    if concat_key in database_scene_ref:
                        database_scene_ref[concat_key].append(scene_details)
                    else:
                        database_scene_ref[concat_key] = [scene_details]

                # Declare hallucination case information through warning messages
                logger.warning("[Input] scene constrain: " + str(scene_constrain))
                logger.warning("[Generated] scene name: " + scene_name)
                logger.warning(
                    "[Target Model Generated] Found objects in the initial image: " + str(found_obj_list))
                logger.warning("[Target Model Generated] Target object (Removed, for attack): " + target_obj)
                logger.warning("[Generated] new generated scene: " + str(gen_new_scene))
                logger.warning("[Generated] scene diversity: " + str(scene_diversity))
                logger.warning("[Generated] new scene img: " + str(new_scene_img))
                logger.warning("[Generated] new object removal: " + str(new_removal_obj))

                # generate list of detected objects
                mask_img, mask_bbox = None, None
                for word in found_obj_list:
                    if word != target_obj:
                        text_input = "a photo of " + word
                        mask_img, mask_bbox = object_detection(init_img_path, text_input, processor, model, mask_img,
                                                               mask_bbox, save_prefix=save_loc)

                # Generate the ground truth of existence questions
                ground_truth = gt_generation_multi_obj_removal_aimodel(result_img_path, mask_bbox, scene_name,
                                                                     target_obj,  save_prefix=save_loc)

                # List detected object name and captions
                logger.warning("[Detection Model] detected objects: " + str(ground_truth["object_name"]))
                logger.warning("[Target Model Generated] detected object captions: " + str(ground_truth["object_description"]))

                logger.warning(
                    "[Target Model Generated] Target object caption (Removed, for attack): " + ground_truth[
                        "non_exist_target_object_description"])

                # image level caption
                result_caption = image_caption_aimodel(result_img_path)

                ground_truth["result_description"] = result_caption
                logger.warning("[Target Model Generated] image-level caption: " + ground_truth["result_description"])

                # Generate the ground truth of spatial relations
                ground_truth = spatial_gt_generation(ground_truth, target_obj_bbox, mask_bbox, enable=True)

                spatial_relation_str = ""
                for i in range(len(ground_truth["spatial_relation"])):
                    spatial_relation_str += str(ground_truth["spatial_relation"][i]) + "; "

                logger.warning(
                    "[Target Model Generated] Spatial Relations (w.r.t. detected objects): " + spatial_relation_str)
                logger.warning("[Target Model Generated] Non-existing Target-detected Object Distances: " + str(
                    ground_truth["spatial_distance"]))

                # Evaluate answers for existence questions based on the given evaluation models
                existence_results, exi_case_result = evaluation_existence_removed_pred_model(result_img_path,
                                                                                             ground_truth,
                                                                                             vqa_model_func=vqa_aimodel,
                                                                                             logger=logger, debug=True)
                logger.warning("[Evaluation Model] Existence Eval Results: " + existence_results)

                # Evaluate answers for spatial relation questions based on the given evaluation models
                spatial_relation_results, spa_case_result = evaluation_spatial_relation_removed_pred_model(
                    result_img_path, ground_truth, vqa_model_func=vqa_aimodel, logger=logger, debug=True)

                logger.warning("[Evaluation Model] Spatial Relation Eval Results: " + spatial_relation_results)

                # Store the result file for metric computing
                result[case_name] = {}
                result[case_name]["scene"] = scene_name
                result[case_name]["result_path"] = os.path.join(save_loc, result_img_path)
                result[case_name]["found_obj"] = found_obj_list
                result[case_name]["target_obj"] = target_obj
                result[case_name]["target_obj_bbox"] = target_obj_bbox
                result[case_name]["target_obj_img"] = os.path.join(save_loc, target_obj_path)
                result[case_name]["existence_results"] = exi_case_result
                result[case_name]["spatial_objects"] = ground_truth["object_name"]
                result[case_name]["spatial_results"] = spa_case_result

                # Store the log file, and query file for scene/object images generated
                with open(os.path.join(save_dir, meta_log), 'w') as f:
                    # indent=2 is not needed but makes the file human-readable 
                    # if the data is nested
                    json.dump(result, f, indent=2) 
                
                with open(database_scene_ref_path, 'w') as f:
                    json.dump(database_scene_ref, f, indent=2)

                completed = True

                ptr += 1

            except Exception as error:
                close_logger(logger)

                print("An exception occurred:", error) # An exception occurred: division by zero
                traceback.print_exc() 
                print("generation error, doing it again...")

                safe_remove_dir(save_loc) # handle nfs
                print('removed save_loc {}'.format(save_loc))
                time.sleep(5)