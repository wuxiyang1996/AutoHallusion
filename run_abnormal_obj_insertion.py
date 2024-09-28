from main_abnormal_obj_insertion import run_exp
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Run the experimental setup for Claude.")

# Experiment tag to store data
parser.add_argument('-t', '--exp_tag', type=str, required=False, default='abnormal_obj_insertion', help='Experiment tag to uniquely identify the setup. Default is "base".')
# Parse arguments from the command line
# Determine if reuse the dataset stored previously
parser.add_argument('--use_dataset', action='store_true', default=False,
                    help='Flag to indicate if using datasets generated before. No argument needed, presence of flag sets it to True.')
# The path to store the dataset
parser.add_argument('--dataset_dir', type=str, default="./datasets/coco_dataset", help='dataset dir')
# The model used to retrieve the objects
parser.add_argument('--obj_think_model_type', type=str, choices=['gemini', 'claude', 'gpt4v', 'llava', 'minigpt4'], default='gpt4v', help='Specify the model type.')
# The model used to generate the image captions
parser.add_argument('--img_caption_model_type', type=str, choices=['gemini', 'claude', 'gpt4v', 'llava', 'minigpt4'], default='gpt4v', help='Specify the model type.')

args_cmd = parser.parse_args()

print('debug...reuse_scene is {}'.format(args_cmd.reuse_scene))

args = {
    "exp_tag": args_cmd.exp_tag, # Experiment tag to store data
    # Folder to store the experiment data and logs
    "exp_dir": "./exp_{}_{}_{}_use_dataset-{}".format(args_cmd.exp_tag, args_cmd.obj_think_model_type, args_cmd.img_caption_model_type, args_cmd.use_dataset),
    "total": 200, # Number of data generated
    "object_size": (200, 200), # Object size (Abnormal / Paired Object Insertion)
    "obj_count": 5, # Number of correlated objects generation within the scene image
    "diffusion": False, # Diffusion model usage flag for object insertion, leaving for further implementation

    # Ablation study: Object-Scene Alignment (for Abnormal Object Insertion)
    # Initial: Intentionally choose abnormal object to insert into the scene
    "random": False, # Using random object to insert
    "scene_ramdom": False, # Randomly shuffle the scene images
    "same": False, # Using objects within the same context to insert

    "scene_constrain": None, # Add constraints on the scene themes
    "irrelevant_obj_category": None, # Add cnstraints on the object to be inserted

    # Reuse dataset flags
    "dataset_dir": args_cmd.dataset_dir, # Path to load the dataset
    "dataset_raw_data": "coco_val_sets", # Name of the dataset for usage
    "dataset_obj_db": "obj_db_clean", # query file to store the objects within the dataset
    "dataset_scene_db": "scene_db_clean", # query file to store the scenes within the dataset

    # hsy 2024.05.20 
    "reuse_scene": args_cmd.use_dataset, # Determine if using the existing scene images from the dataset
    "reuse_scene_obj_align": False, # Decide if using the previous obj-scene alignment results or re-do alignment
    "reuse_obj": True, # Determine if using the existing object images from the dataset
    "reuse_obj_partial_random": False, # Shuffle the object image under the same image
    "reuse_obj_complete_random": False, # Decide if using the previous obj-scene alignment results or randomly assign
    "resize_img": True, # Resize the input image to 1024 * 1024

    "obj_think_model_type": args_cmd.obj_think_model_type, # Model type to retrieve the object
    "img_caption_model_type": args_cmd.img_caption_model_type, # Model type to caption the image
}

# Path to store generated images
args["exp_name"] = "exi_exp_{}".format(args["exp_tag"])
args["save_dir"] = "{}/{}/".format(args["exp_dir"], args["exp_name"])

# Load the database files that queries scene and object images from the dataset
args["database_scene_ref_path"] = "{}/scene_db_{}.json".format(args["exp_dir"], args["exp_tag"])
args["database_obj_ref_path"] = "{}/obj_db_{}.json".format(args["exp_dir"], args["exp_tag"])

# Hyper-parameters in generating scene and objects
args["scene_gen_prob"] = 1
args["obj_gen_prob"] = 1

args["diversity_prob"] = 0.5
args["diversity_count"] = 25

# When reusing previous datasets
args["dataset_raw_data_path"] = "{}/{}/".format(args["dataset_dir"], args["dataset_raw_data"])
args["dataset_scene_ref_path"] = "{}/{}.json".format(args["dataset_dir"], args["dataset_scene_db"])
args["dataset_obj_ref_path"] = "{}/{}.json".format(args["dataset_dir"], args["dataset_obj_db"])

args["dataset_scene_query_path"] = "{}/{}.json".format(args["dataset_dir"], "query_" + args["dataset_scene_db"])
args["dataset_obj_query_path"] = "{}/{}.json".format(args["dataset_dir"], "query_" + args["dataset_obj_db"])

# scene_details = {
#     "scene_name": scene_name,
#     "word_list": word_list,
#     "path": init_img_path
# }
# obj_details = {
#     "obj_name": irrelevant_obj,
#     "path": os.path.join(save_loc, "obj.png"),
#     "pure_obj_path": os.path.join(save_loc, "pure_obj.png"),
#     "mask_path": os.path.join(save_loc, "mask_obj.png")
# }

run_exp(args)
        