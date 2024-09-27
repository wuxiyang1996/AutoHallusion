# from utils_gemini import *
import numpy as np
import time
import logging, sys, os, json, shutil
from tqdm import tqdm
import traceback
import random
from PIL import Image

from utils.utils_merge import import_functions_given_model_type, load_cfg_given_model_type
from utils.utils_eval import evaluation_existence_given_pred_model, evaluation_spatial_relation_give_pred_model
from utils.utils import (OwlViTProcessor, OwlViTForObjectDetection, scene_thinking, generate_image_given_scene,
                         Addition_Image_Operation_VD_stitch, object_detection, spatial_gt_generation,
                         resize_img_n_store, convert_into_sequare)


# This is a helper function to replace the initial head path
def path_header_replace(path, new_header, case_id):
    file_name = os.path.split(path)[1]
    return os.path.join(new_header, case_id, file_name)


# Main function for abnormal object insertion
def run_exp(args, meta_log="exp_results.json"):
    # Declare the hyper-parameters from the running file
    exp_name = args["exp_name"]
    save_dir = args["save_dir"]
    total = args["total"]
    object_size = args["object_size"]
    obj_count = args["obj_count"]
    diffusion = args["diffusion"]
    obj_random = args["random"]
    scene_ramdom = args["scene_ramdom"]
    same = args["same"]

    # Generate scene name
    scene_constrain = args["scene_constrain"]
    irrelevant_obj_category = args["irrelevant_obj_category"]

    database_scene_ref_path = args["database_scene_ref_path"]
    database_obj_ref_path = args["database_obj_ref_path"]
    scene_gen_prob = args["scene_gen_prob"]
    obj_gen_prob = args["obj_gen_prob"]
    diversity_prob = args["diversity_prob"]
    diversity_count = args["diversity_count"]

    # Reuse previous dataset hyperparameters
    reuse_dataset_scene_ref_path = args["dataset_scene_ref_path"]
    reuse_dataset_obj_ref_path = args["dataset_obj_ref_path"]
    reuse_dataset_raw_data_path = args["dataset_raw_data_path"]

    reuse_dataset_scene_query_path = args["dataset_scene_query_path"]
    reuse_dataset_obj_query_path = args["dataset_obj_query_path"]

    reuse_scene = args["reuse_scene"]
    reuse_scene_obj_align = args["reuse_scene_obj_align"]
    reuse_obj = args["reuse_obj"]
    reuse_obj_partial_random = args["reuse_obj_partial_random"]
    reuse_obj_complete_random = args["reuse_obj_complete_random"]

    obj_think_model_type = args["obj_think_model_type"]
    img_caption_model_type = args["img_caption_model_type"]

    resize_img = args["resize_img"]

    # Load the model for retrieving objects and image caption
    print(
        'verbose...merge_reuse obj_think_model_type {}, img_caption_model_type {}, reuse_scene {}, reuse_dataset_scene_ref_path {}'.format(
            obj_think_model_type, img_caption_model_type, reuse_scene, reuse_dataset_scene_ref_path))

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

    # Load the object image path
    if os.path.exists(database_obj_ref_path):
        with open(database_obj_ref_path, "r") as f:
            database_obj_ref = json.load(f)
    else:
        database_obj_ref = {}
        with open(database_obj_ref_path, 'w') as f:
            json.dump(database_obj_ref, f, indent=2)

    # Create the helper json file to query scene generated if set the reuse tag
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

    # Create the helper json file to query object generated if set the reuse tag
    if reuse_obj:
        all_files = os.listdir(reuse_dataset_raw_data_path)

        with open(reuse_dataset_obj_ref_path, 'r') as f:
            # indent=2 is not needed but makes the file human-readable
            # if the data is nested
            obj_db = json.load(f)

        if not os.path.exists(reuse_dataset_obj_query_path):
            obj_query_db = {}
            for key in list(obj_db.keys()):
                for i in range(len(obj_db[key])):
                    id = obj_db[key][i]["path"].split("/")[-2]
                    if id in all_files:
                        obj_query_db[id] = {'obj_name': key, "position": str(i)}

            with open(reuse_dataset_obj_query_path, 'w') as f:
                json.dump(obj_query_db, f, indent=2)
        else:
            with open(reuse_dataset_obj_query_path, 'r') as f:
                # indent=2 is not needed but makes the file human-readable
                # if the data is nested
                obj_query_db = json.load(f)

        raw_list = []
        for key in list(obj_db.keys()):
            raw_list += obj_db[key]

        if reuse_obj_complete_random:
            reuse_obj_id = list(np.random.choice(list(obj_query_db.keys()), total))
        else:
            reuse_obj_id = list(obj_query_db.keys())

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

                scene_img_raw_size = None

                attribute_category_list = []

                # scene_constrain = "Indoor Scene"
                # irrelevant_obj_category = "animals"

                if not os.path.exists(save_loc):
                    os.makedirs(save_loc)

                # Initialize the images (initial and AutoHallusion) for storage
                log_path = os.path.join(save_loc, "output.log")
                init_img_path = os.path.join(save_loc, init_img_path)
                result_img_path = os.path.join(save_loc, result_img_path)

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
                gen_new_obj = np.random.random() < obj_gen_prob | len(list(database_obj_ref.values())) > 0
                scene_diversity = np.random.random() < diversity_prob
                obj_diversity = np.random.random() < diversity_prob

                # Load the scene from dataset when using existing scene images
                # Need to adjust the image if necessary to fit the image editing tools
                if reuse_scene:
                    scene_load_id = reuse_scene_id[ptr]

                    scene_name_queried = scene_query_db[scene_load_id]['scene_name']
                    scene_name_queried_pos = int(scene_query_db[scene_load_id]['position'])

                    scene_info_loaded = scene_db[scene_name_queried][scene_name_queried_pos]

                    scene_name = scene_info_loaded["scene_name"]
                    word_list = scene_info_loaded["word_list"][:obj_count]

                    shutil.copyfile(
                        path_header_replace(scene_info_loaded["path"], reuse_dataset_raw_data_path, scene_load_id),
                        init_img_path)

                    if resize_img:
                        init_img = Image.open(init_img_path)
                        scene_img_raw_size = init_img.size

                        convert_into_sequare(init_img_path, init_img_path)
                        resize_img_n_store(init_img_path)

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

                        # Retrieve words aligned with the scene image
                        if same:
                            word_list, _ = generate_noun_given_scene_aimodel(num=obj_count + 1, scene=scene_name,
                                                                             temperature=temp_generate_noun_given_scene)
                        else:
                            word_list, _ = generate_noun_given_scene_aimodel(num=obj_count, scene=scene_name,
                                                                             temperature=temp_generate_noun_given_scene)
                        generate_image_given_scene(word_list, scene_name, image_name=init_img_path)

                    else:
                        # Reuse scene image from previous results
                        load = random.choice(list(database_scene_ref.values()))
                        if same:
                            if len(word_list) >= obj_count + 1:
                                word_list = load["word_list"][:obj_count + 1]
                                shutil.copyfile(load["path"], init_img_path)
                            else:
                                word_list, _ = generate_noun_given_scene_aimodel(num=obj_count + 1, scene=scene_name,
                                                                                 temperature=temp_generate_noun_given_scene)
                                generate_image_given_scene(word_list, scene_name, image_name=init_img_path)
                                gen_new_scene = True
                        else:
                            if len(word_list) >= obj_count:
                                word_list = load["word_list"][:obj_count]
                                shutil.copyfile(load["path"], init_img_path)
                            else:
                                word_list, _ = generate_noun_given_scene_aimodel(num=obj_count, scene=scene_name,
                                                                                 temperature=temp_generate_noun_given_scene)
                                generate_image_given_scene(word_list, scene_name, image_name=init_img_path)
                                gen_new_scene = True

                # generate the image based on the provided objects
                if gen_new_scene:

                    scene_key_char = list([val for val in scene_name if val.isalpha() or val.isnumeric()])
                    scene_key = "".join(scene_key_char).lower()

                    scene_details = {
                        "scene_name": scene_name,
                        "word_list": word_list,
                        "path": init_img_path
                    }

                    if scene_key in database_scene_ref:
                        database_scene_ref[scene_key].append(scene_details)
                    else:
                        database_scene_ref[scene_key] = [scene_details]

                # Reuse the previous object generation results or not
                if reuse_obj:
                    # Decide if using the previous scene-object alignment
                    if reuse_scene_obj_align:
                        if reuse_obj_complete_random:
                            load_id = reuse_obj_id[ptr]
                        else:
                            load_id = reuse_scene_id[ptr]
                        obj_name_queried = obj_query_db[load_id]['obj_name']

                        if reuse_obj_partial_random:
                            idx = np.random.randint(len(obj_db[obj_name_queried]))
                            obj_info_loaded = obj_db[obj_name_queried][idx]
                        else:
                            obj_name_queried_pos = int(obj_query_db[load_id]['position'])
                            obj_info_loaded = obj_db[obj_name_queried][obj_name_queried_pos]

                        irrelevant_obj = obj_info_loaded["obj_name"]
                        shutil.copyfile(
                            path_header_replace(obj_info_loaded["path"], reuse_dataset_raw_data_path, load_id),
                            os.path.join(save_loc, "obj.png"))
                        shutil.copyfile(
                            path_header_replace(obj_info_loaded["pure_obj_path"], reuse_dataset_raw_data_path, load_id),
                            os.path.join(save_loc, "pure_obj.png"))
                        shutil.copyfile(
                            path_header_replace(obj_info_loaded["mask_path"], reuse_dataset_raw_data_path, load_id),
                            os.path.join(save_loc, "mask_obj.png"))

                        # Resize the image to fit the image editing requirements
                        if resize_img:
                            convert_into_sequare(os.path.join(save_loc, "obj.png"), os.path.join(save_loc, "obj.png"))
                            resize_img_n_store(os.path.join(save_loc, "obj.png"))
                            convert_into_sequare(os.path.join(save_loc, "pure_obj.png"),
                                                 os.path.join(save_loc, "pure_obj.png"))
                            resize_img_n_store(os.path.join(save_loc, "pure_obj.png"))
                            convert_into_sequare(os.path.join(save_loc, "mask_obj.png"),
                                                 os.path.join(save_loc, "mask_obj.png"))
                            resize_img_n_store(os.path.join(save_loc, "mask_obj.png"))

                        gen_new_obj = True

                    else:

                        irrelevant_obj_dict_lst = []
                        for key in list(obj_db.keys()):
                            irrelevant_obj_dict_lst += obj_db[key]

                        if len(irrelevant_obj_dict_lst) > diversity_count:
                            irrelevant_obj_dict_lst = random.sample(irrelevant_obj_dict_lst, diversity_count)

                        irrelevant_obj_dict_lst = filter_remove_obj_under_scene_aimodel(scene_name,
                                                                                        irrelevant_obj_dict_lst,
                                                                                        temperature=temp_filter_remove_obj_under_scene)
                        if len(irrelevant_obj_dict_lst) == 0:
                            irrelevant_obj_dict_lst = []
                            for key in list(obj_db.keys()):
                                irrelevant_obj_dict_lst += obj_db[key]
                        irrelevant_obj_dict = filter_most_irrelevant_aimodel(scene_name, word_list,
                                                                             irrelevant_obj_dict_lst,
                                                                             temperature=temp_filter_most_irrelevant)

                        irrelevant_obj = irrelevant_obj_dict["obj_name"]

                        load_id = irrelevant_obj_dict["path"].split("/")[-2]

                        shutil.copyfile(
                            path_header_replace(irrelevant_obj_dict["path"], reuse_dataset_raw_data_path, load_id),
                            os.path.join(save_loc, "obj.png"))
                        shutil.copyfile(
                            path_header_replace(irrelevant_obj_dict["pure_obj_path"], reuse_dataset_raw_data_path,
                                                load_id),
                            os.path.join(save_loc, "pure_obj.png"))
                        shutil.copyfile(
                            path_header_replace(irrelevant_obj_dict["mask_path"], reuse_dataset_raw_data_path, load_id),
                            os.path.join(save_loc, "mask_obj.png"))

                        if resize_img:
                            convert_into_sequare(os.path.join(save_loc, "obj.png"), os.path.join(save_loc, "obj.png"))
                            resize_img_n_store(os.path.join(save_loc, "obj.png"))
                            convert_into_sequare(os.path.join(save_loc, "pure_obj.png"),
                                                 os.path.join(save_loc, "pure_obj.png"))
                            resize_img_n_store(os.path.join(save_loc, "pure_obj.png"))
                            convert_into_sequare(os.path.join(save_loc, "mask_obj.png"),
                                                 os.path.join(save_loc, "mask_obj.png"))
                            resize_img_n_store(os.path.join(save_loc, "mask_obj.png"))

                        gen_new_obj = True

                elif same:

                    irrelevant_obj = word_list[0]
                    word_list = word_list[1:]
                    gen_new_obj = True
                else:
                    # Generate several objects based on the scene
                    if gen_new_obj:
                        if obj_diversity:
                            objs_lst = [obj[0]["obj_name"] for obj in list(database_obj_ref.values())]
                            if len(objs_lst) > diversity_count:
                                objs_lst = random.sample(objs_lst, diversity_count)
                            if len(objs_lst) > 0:
                                diversity_cond = " Try to generate a different object from the following: {}.".format(
                                    ",".join(objs_lst))
                            else:
                                diversity_cond = ""
                        else:
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
                    else:

                        # Determine if using random object to insert, or using random scene image
                        if obj_random:
                            if scene_ramdom:
                                irrelevant_obj_dict = random.choice(list(database_obj_ref.values()))
                                irrelevant_obj = irrelevant_obj_dict["obj_name"]
                            else:
                                irrelevant_obj_dict_lst = list(database_obj_ref.values())
                                if len(irrelevant_obj_dict_lst) > diversity_count:
                                    irrelevant_obj_dict_lst = random.sample(irrelevant_obj_dict_lst, diversity_count)
                                irrelevant_obj_dict_lst = filter_remove_obj_under_scene_aimodel(scene_name,
                                                                                                irrelevant_obj_dict_lst,
                                                                                                temperature=temp_filter_remove_obj_under_scene)
                                if len(irrelevant_obj_dict_lst) == 0:
                                    irrelevant_obj_dict_lst = list(database_obj_ref.values())
                                irrelevant_obj_dict = random.choice(irrelevant_obj_dict_lst)
                                irrelevant_obj = irrelevant_obj_dict["obj_name"]
                        else:
                            # Generate one irrelevant object based on the scene
                            irrelevant_obj_dict_lst = list(database_obj_ref.values())
                            if len(irrelevant_obj_dict_lst) > diversity_count:
                                irrelevant_obj_dict_lst = random.sample(irrelevant_obj_dict_lst, diversity_count)
                            irrelevant_obj_dict_lst = filter_remove_obj_under_scene_aimodel(scene_name,
                                                                                            irrelevant_obj_dict_lst,
                                                                                            temperature=temp_filter_remove_obj_under_scene)
                            if len(irrelevant_obj_dict_lst) == 0:
                                irrelevant_obj_dict_lst = list(database_obj_ref.values())
                            irrelevant_obj_dict = filter_most_irrelevant_aimodel(scene_name, word_list,
                                                                                 irrelevant_obj_dict_lst,
                                                                                 temperature=temp_filter_most_irrelevant)
                            irrelevant_obj = irrelevant_obj_dict["obj_name"]

                        shutil.copyfile(irrelevant_obj_dict["path"], os.path.join(save_loc, "obj.png"))
                        shutil.copyfile(irrelevant_obj_dict["pure_obj_path"], os.path.join(save_loc, "pure_obj.png"))
                        shutil.copyfile(irrelevant_obj_dict["mask_path"], os.path.join(save_loc, "mask_obj.png"))

                # Declare hallucination case information through warning messages
                logger.warning("[Input] scene constrain: " + str(scene_constrain))
                logger.warning("[Generated] scene name: " + scene_name)
                logger.warning("[Target Model Generated] relevant objects: " + str(word_list))
                logger.warning("[Input] irrelevant object category: " + str(irrelevant_obj_category))
                logger.warning("[Target Model Generated] irrelevant object: " + irrelevant_obj)
                logger.warning("[Generated] new generated scene: " + str(gen_new_scene))
                logger.warning("[Generated] new generated obj: " + str(gen_new_obj))
                logger.warning("[Generated] scene diversity: " + str(scene_diversity))
                logger.warning("[Generated] obj diversity: " + str(obj_diversity))

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

                # List detected object name and captions
                logger.warning("[Detection Model] detected objects: " + str(ground_truth["object_name"]))
                logger.warning(
                    "[Target Model Generated] detected object captions: " + str(ground_truth["object_description"]))

                logger.warning("[Target Model Generated] irrelevant object caption: " + ground_truth[
                    "irrelevant_object_description"])

                # image level caption
                result_caption = image_caption_aimodel(result_img_path)

                ground_truth["result_description"] = result_caption
                logger.warning("[Target Model Generated] image-level caption: " + ground_truth["result_description"])

                # Generate the ground truth of spatial relations
                ground_truth = spatial_gt_generation(ground_truth, irrelevant_obj_bbox, mask_bbox, enable=True)


                spatial_relation_str = ""
                for i in range(len(ground_truth["spatial_relation"])):
                    spatial_relation_str += str(ground_truth["spatial_relation"][i]) + "; "

                logger.warning(
                    "[Target Model Generated] Spatial Relations (w.r.t. detected objects): " + spatial_relation_str)
                logger.warning("[Target Model Generated] Irrelevant-detected Object Distances): " + str(
                    ground_truth["spatial_distance"]))


                # Evaluate answers for existence questions based on the given evaluation models
                existence_results, exi_case_result = evaluation_existence_given_pred_model(result_img_path,
                                                                                           ground_truth, result_caption,
                                                                                           vqa_model_func=vqa_aimodel,
                                                                                           logger=logger, debug=True)
                logger.warning("[Evaluation Model] Existence Eval Results: " + existence_results)

                # Evaluate answers for spatial relation questions based on the given evaluation models
                spatial_relation_results, spa_case_result = evaluation_spatial_relation_give_pred_model(result_img_path,
                                                                                                        ground_truth,
                                                                                                        vqa_model_func=vqa_aimodel,
                                                                                                        logger=logger,
                                                                                                        debug=True)

                logger.warning("[Evaluation Model] Spatial Relation Eval Results: " + spatial_relation_results)

                # Store the generated objects into the database file
                if gen_new_obj:
                    obj_key_char = list([val for val in irrelevant_obj if val.isalpha() or val.isnumeric()])
                    obj_key = "".join(obj_key_char).lower()

                    obj_details = {
                        "obj_name": irrelevant_obj,
                        "path": os.path.join(save_loc, "obj.png"),
                        "pure_obj_path": os.path.join(save_loc, "pure_obj.png"),
                        "mask_path": os.path.join(save_loc, "mask_obj.png")
                    }

                    if obj_key in database_obj_ref:
                        database_obj_ref[obj_key].append(obj_details)
                    else:
                        database_obj_ref[obj_key] = [obj_details]

                # Store the result file for metric computing
                result[case_name] = {}
                result[case_name]["scene"] = scene_name
                result[case_name]["obj"] = word_list
                result[case_name]["irr_obj"] = irrelevant_obj
                result[case_name]["irr_obj_img"] = os.path.join(save_loc, "obj.png")
                result[case_name]["irr_obj_mask"] = os.path.join(save_loc, "mask_obj.png")
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

                with open(database_obj_ref_path, 'w') as f:
                    json.dump(database_obj_ref, f, indent=2)

                completed = True

                ptr += 1

            except Exception as error:
                close_logger(logger)

                print("An exception occurred:", error)  # An exception occurred: division by zero
                traceback.print_exc()
                print("generation error, doing it again...")

                safe_remove_dir(save_loc)  # handle nfs
                print('removed save_loc {}'.format(save_loc))
                time.sleep(5)