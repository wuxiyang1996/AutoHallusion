#

# llava related
from llava.model.builder import load_pretrained_model

from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from llava.constants import (
    IMAGE_TOKEN_INDEX,
)

from PIL import Image
import torch
import time
import os
import numpy as np
import shutil
import random
from utils.utils_eval import evaluate_by_chatgpt_quick_test
from utils.utils import correlated_object_segment, correlated_img_generate

MODEL_PATH="your path"



########### common utils
# Extract the single object from the scene with masks
def single_obj_extract(img_path, mask_bbox, idx, save_prefix):
    raw_img = Image.open(img_path)
    bbox = mask_bbox[idx]
    obj_img = np.array(raw_img)[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    obj_img = Image.fromarray(obj_img.astype(np.uint8)).convert('RGBA')
    obj_img.save(os.path.join(save_prefix, "obj_" + str(idx) + ".png"))
    return obj_img

def safe_remove_dir(path, retries=5, delay=3):
    """Attempt to remove a directory with retries for stubborn files like .nfs files on NFS mounts."""
    for attempt in range(retries):
        try:
            # Try to remove the directory with all its contents
            shutil.rmtree(path)
            print(f"Successfully removed directory: {path}")
            break
        except OSError as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(delay)  # Wait before trying again
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
    else:
        # Final attempt to list remaining files if any
        remaining_files = os.listdir(path)
        if remaining_files:
            print(f"Failed to remove all files, remaining: {remaining_files}")
        else:
            print(f"Directory already empty and will now be removed.")
            os.rmdir(path)  # Final removal if somehow it's empty now

def close_logger(logger):
    """Closes all handlers of the logger to free up resources."""
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
        
        
        
################### LLaVA MODEL ####################

def setup_llava(model_path="/fs/cml-projects/Pretrain_MBRL/hallusion/llava-v1.5-7b"):
    """
    set up local llava model from a local path
    
    """
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    return tokenizer,model,image_processor,context_len


def qa_llava(question, tokenizer=None, model=None):
    
    if not tokenizer:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH,
        model_base=None,
        model_name=get_model_name_from_path(MODEL_PATH)
        )
        
    while True:
        try:
            input_ids = (
            tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=None,
                    image_sizes=None,
                    do_sample=False,
                    temperature=1,
                    #top_p=3,
                    num_beams=3,
                    max_new_tokens=200,
                    use_cache=True,
                )
            answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            break
        except:
            print("llava Timeout qa_llava, retrying... Q: {}".format(question))
            time.sleep(5)  # Wait for 5 seconds before retrying


    # print('qa_llava  question {}'.format(question))
    # print('qa_llava  answer {}'.format(answer))

    return answer


def vqa_llava(masked_image_path, questions, image_processor=None, model=None, tokenizer=None, temp=1, max_try=5):
    
    if not tokenizer:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH,
        model_base=None,
        model_name=get_model_name_from_path(MODEL_PATH)
        )
    
    counter = 0

    while True:
        try:
            image_file = Image.open(masked_image_path).convert("RGB")
            image_size = [image_file.size]

            images_tensor = process_images(
                                            [image_file],
                                            image_processor,
                                            model.config
                                            ).to(model.device, dtype=torch.float16)
            input_ids = (tokenizer_image_token(questions, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                         .unsqueeze(0)
                         .cuda()
                        )
            
            with torch.inference_mode():
                output_ids = model.generate(
                                            input_ids,
                                            images=images_tensor,
                                            image_sizes=image_size,
                                            do_sample=True,
                                            temperature=temp,
                                            #top_p=3,
                                            num_beams=3,
                                            max_new_tokens=200,
                                            use_cache=True,
                                            )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            answer = outputs
            break
        except:
            counter += 1
            print("vqa_llava image query timeout,  try {} / {}...".format(str(counter), str(max_try)))
            if counter > max_try:
                raise Exception("generate_image reaches max attempt of {}, regenerate case".format(str(max_try)))
            time.sleep(5)  # Wait for 5 seconds before retrying

    # print('vqa_llava question {}'.format(questions))

    # print('vqa_llava masked_image_path {}'.format(masked_image_path))

    # print('vqa_llava answer {}'.format(answer))

    return answer



# General image caption function, create a caption with image provided
# Input: Masked image path
# Output: Object name
def image_caption_llava(input_img_path, image_processor=None, model=None, tokenizer=None):
    msg = "USER: <image>\nDescribe what you see in this image.\nASSISTANT:"
    caption = vqa_llava(input_img_path, msg, image_processor, model, tokenizer)
    print('image_caption_llava done')
    return caption


def generate_noun_given_scene_llava(num, scene, tokenizer=None, model=None, examples=None, temperature=1):

    if examples is None:
        example_prompt = ""
        example_prompt2 = ""
    else:
        example_prompt = "like {}, all ".format(",".join(examples))
        example_prompt2 = "Do not generate the same items as the ones mentioned in the example."

    prompt = '''USER: Generate {} words that are noun representing different physical objects and identities that are the most likely to exist in the scene of {} ({}with very high similarity score){}.
    Output format should be the list of nouns separated by comma. The output should only contain a string of {} words and commas.
    \nASSISTANT:
    '''.format(num, scene, example_prompt, example_prompt2, num)

    
    if not tokenizer:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=MODEL_PATH,
            model_base=None,
            model_name=get_model_name_from_path(MODEL_PATH)
            )

    while True:
        try:
            answer = qa_llava(prompt,tokenizer,model)
            break
        except:
            print("llava Timeout generate_noun_given_scene_llava, retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying

    words = answer

    print('generate_noun_given_scene_llava: prompt: {}'.format(prompt))
    print('llava answer: {}'.format(words))

    word_lst = words.split(",")
    word_lst = [w.strip() for w in word_lst]
    
    print(len(word_lst), '\n')

    assert len(word_lst) == num

    print('output word_lst: {}'.format(word_lst))

    return word_lst, None
    

def random_obj_thinking_llava(scene_name, tokenizer=None, model=None, cond=""):
    print('random_obj_thinking_gemini:')

    if scene_name is None:
        scene_desc = '''
                Think about one random object. Use a one or two words to describe this object.
                '''
    else:
        scene_desc = '''
                Think about one random object that is unlikely to exist in the {}. Use a one or two words to describe this object.
                '''.format(scene_name)
        
    prompt = "User: " + scene_desc + cond + " \nASSISTANT:"

    obj_name = qa_llava(prompt, tokenizer, model)

    print("obj_name: ", obj_name)
    return obj_name


# # Find an irrelevant object given the scene and the existing objects
# # Input: Masked image path
# # Output: Object name
def irrelevant_obj_thinking_llava(scene_name, word_list, category=None, tokenizer=None, model=None, cond=""):
    print('irrelevant_obj_thinking_llava:')
    words = ",".join(word_list)
    if category is not None:
        scene_desc = '''
                Think about one commonly seen physical object from the category of {} that is irrelevant to the existing physical objects: "{}" and is unlikely to exist in the {}. Use a one or two words to describe this object.
                This object should not be a concept or too abstract. For example, Ocean, or Space is too abstract to describe by a concrete identity, while fish and space ship are good examples under those concepts.
                Output only the name of this irrelevant object.
                '''.format(category, words, scene_name)
    else:
        scene_desc = '''
                Think about one commonly seen physical object that is irrelevant to the existing physical objects: "{}" and is unlikely to exist in the {}. Use a one or two words to describe this object.
                This object should not be a concept or too abstract. For example, Ocean, or Space is too abstract to describe by a concrete identity, while fish and space ship are good examples under those concepts.
                Output only the name of this irrelevant object.
                '''.format(words, scene_name)
        
    prompt = "User: " + scene_desc + cond + " \nASSISTANT:"

    obj_name = qa_llava(prompt, tokenizer, model)

    # print(obj_name)

    print('irrelevant_obj_thinking_llava done')

    

    return obj_name


# Given the image of the object, generate its name and describe using no more than 3 words
def single_obj_naming_llava(single_obj_img_path, image_processor=None, model=None, tokenizer=None, temp=1, max_try=5):
    msg = "USER: <image>\nDescribe the object in this image with no more than 3 words.\nASSISTANT:"
    object_name = vqa_llava(single_obj_img_path, msg, image_processor, model, tokenizer, temp, max_try)
    
    # print(object_name)
    
    print('single_obj_naming_llava done.')
    return object_name


# Given the image of the object and its name, use 1 around-10-word sentence to describe this object
def single_obj_caption_llava(single_obj_img_path, single_obj_name, image_processor=None, model=None, tokenizer=None, temp=1, max_try=5):
    prompt = "USER: <image>\nGiven the image of {}, describe this object using a single sentence within 10 words.\nASSISTANT:".format(single_obj_name)
    answer = vqa_llava(single_obj_img_path, prompt, image_processor, model, tokenizer, temp, max_try)
    # print(answer)
    print('single_obj_caption_llava done.')
    return answer


# General image caption function, create a caption with image provided
# Input: Masked image path
# Output: Object name
def object_image_caption_llava(input_img_path, obj_name, irrelevant_obj_attribute,  image_processor=None, model=None, tokenizer=None, temp=1, max_try=5):
    attribute_category_str = ""
    attribute_category_list = list(irrelevant_obj_attribute.keys())

    for attribute_category in attribute_category_list:
        attribute_category_str += attribute_category + ", "
    msg = "Describe the {} in this image with a single paragraph. Focus on the following attributes: {}".format(obj_name, attribute_category_str)
    
    prompt = "User: " + msg + " \nASSISTANT:"
    
    caption = vqa_llava(input_img_path, prompt, image_processor, model, tokenizer, temp, max_try)
    print('object_image_caption_llava done')
    return caption


# Generate the ground truth of the image after editing
# Content: Scene (Given), Object name and description one-by-one, Background, Added Object
def gt_generation_llava(init_img_path, mask_bbox, scene_name, irrelevant_obj, irrelevant_obj_attribute,
                          save_prefix="./", image_processor=None, model=None, tokenizer=None):
    obj_img = []
    obj_name = []
    obj_description = []
    for i in range(len(mask_bbox)):
        obj_img.append(single_obj_extract(init_img_path, mask_bbox, i, save_prefix))
        obj_temp = single_obj_naming_llava(os.path.join(save_prefix, "obj_" + str(i) + ".png"),
                                           image_processor, model, tokenizer)
        obj_name.append(obj_temp)
        obj_description.append(single_obj_caption_llava(os.path.join(save_prefix, "obj_" + str(i) + ".png")
                                                        , obj_temp , image_processor, model, tokenizer)
                                )

    # background_extract(init_img_path, mask_bbox, save_prefix)

    irrelevant_obj_description = object_image_caption_llava(os.path.join(save_prefix, "pure_obj.png"),
                                                            irrelevant_obj, irrelevant_obj_attribute,
                                                            image_processor, model, tokenizer
                                                            )

    ground_truth = {
        "scene": scene_name,
        # "background": background_caption_gpt4v(os.path.join(save_prefix, "background.png"), scene_name),
        "object_name": obj_name,
        "object_description": obj_description,
        "irrelevant_object_name": irrelevant_obj,
        "irrelevant_obj_attribute": irrelevant_obj_attribute,
        "irrelevant_object_description": irrelevant_obj_description
    }
    return ground_truth

# Generate the ground truth of the image before object removal editing
# Content: Scene (Given), Object name and description one-by-one, Removed Object and description
# Output: Ground truth, target/removed/preserved object index (For image editing and spatial relation)
def gt_generation_multi_obj_removal_llava(init_img_path, mask_bbox, scene_name, target_obj, 
                                           save_prefix="./", image_processor=None, model=None, tokenizer=None):
    obj_img = []
    obj_name = []
    obj_description = []
    for i in range(len(mask_bbox)):
        obj_img.append(single_obj_extract(init_img_path, mask_bbox, i, save_prefix))
        obj_temp = single_obj_naming_llava(os.path.join(save_prefix, "obj_" + str(i) + ".png"),
                                           image_processor, model, tokenizer)
        obj_name.append(obj_temp)
        obj_description.append(single_obj_caption_llava(os.path.join(save_prefix, "obj_" + str(i) + ".png"), 
                                                        obj_temp, image_processor, model, tokenizer
                                                        )
                               )

    target_obj_description = object_image_caption_llava(os.path.join(save_prefix, "target_obj.png"), target_obj, {},
                                                        image_processor, model, tokenizer
                                                        )

    ground_truth = {
        "scene": scene_name,
        "object_name": obj_name,
        "object_description": obj_description,
        "non_exist_target_object_name": target_obj,
        "non_exist_target_object_description": target_obj_description
    }
    return ground_truth


def filter_remove_obj_under_scene_llava(scene_name, irrelevant_obj_dict_lst, tokenizer=None, model=None):
    objs_prompt = ""
    for i, o in enumerate(irrelevant_obj_dict_lst):
        objs_prompt += "{}. {};".format(str(i), o["obj_name"])

    prompt = '''
        Given a list of objects, pick all objects that are relevant and related to the scene {}, or it is common to see those objects in this scene of {}.
        Each object is given an index, and return the number index only separated by comma. 
        The AutoHallusion output should only consists of numbers and comma. The list of objects are as follows: {}
    '''.format(scene_name, scene_name, objs_prompt)


    prompt = "User: " + prompt + " \nASSISTANT:"

    filtered_objs_idx = qa_llava(prompt, tokenizer, model)

    

    obj_idx_rev = filtered_objs_idx.split(",")

    obj_idx_rev = [int(i.strip()) for i in obj_idx_rev]

    obj_idx = []
    for i in range(len(irrelevant_obj_dict_lst)):
        if i not in obj_idx_rev:
            obj_idx.append(i)

    ret_dict_lst = []

    for idx in obj_idx:
        if idx >= 0 and idx < len(irrelevant_obj_dict_lst):
            ret_dict_lst.append(irrelevant_obj_dict_lst[idx])
        else:
            print("invalid index {} in filter_remove_obj_under_scene_llava from llava.".format(str(idx)))

    return ret_dict_lst


def filter_most_irrelevant_llava(scene_name, word_lst, irrelevant_obj_dict_lst, tokenizer=None, model=None):
    objs_prompt = ""
    for i, o in enumerate(irrelevant_obj_dict_lst):
        objs_prompt += "{}. {};".format(str(i), o["obj_name"])

    if isinstance(word_lst, str):
        words = word_lst
    elif isinstance(word_lst, list):
        words = ", ".join(word_lst)
    else:
        words = word_lst

    prompt = '''
        Given a list of objects, pick the most irrelevant object that is the most irrelevant and non-related to the scene {} and objects like {}, or it is extremely uncommon to see such object in this scene of {}.
        Each object is given an index, and return a single index number only. 
        The AutoHallusion output should only consists of a single number. The list of objects are as follows: {}
    '''.format(scene_name, words, scene_name, objs_prompt)

    prompt = "User: " + prompt + " \nASSISTANT:"

    filtered_obj_idx = qa_llava(prompt, tokenizer, model)

    obj_idx = int(filtered_obj_idx.strip())

    if obj_idx >= 0 and obj_idx < len(irrelevant_obj_dict_lst):
        return irrelevant_obj_dict_lst[obj_idx]
    else:
        filtered_obj_idx = list([val for val in filtered_obj_idx if val.isnumeric()])
        obj_idx = int(filtered_obj_idx)
        if obj_idx >= 0 and obj_idx < len(irrelevant_obj_dict_lst):
            return irrelevant_obj_dict_lst[obj_idx]
        else:
            print("invalid index {} in filter_most_irrelevant_gpt4v from gpt.".format(filtered_obj_idx))
            return random.choice(irrelevant_obj_dict_lst)
        

        
def list_objects_given_img_llava(img_path, num_list_obj, image_processor=None, model=None, tokenizer=None):
    prompt = "List {} physical objects in the given image. The output should be a list contains object names separated by commas.".format(
        str(num_list_obj))
    
    prompt = "User: " + prompt + " \nASSISTANT:"

    out_list = []
    while len(out_list) != num_list_obj:
        answer = vqa_llava(img_path, prompt, image_processor, model, tokenizer)
        out_list = answer.split(",")

    final_list = []
    for i in range(len(out_list)):
        temp = list([val for val in out_list[i] if val.isalpha() or val.isnumeric() or val == " "])
        final_list.append("".join(temp).lower())
    return final_list

# Brutally think about two closely related objects without any other constraints
def correlated_obj_thinking_llava(cond="", tokenizer=None, model=None):
    prompt = """
    Can you generate two objects that are strongly correlated? If one thing appears, it often appears with the other 
    objects. For example, [fish tank, fish]. Please only generate two objects separated with commas.
    """

    prompt = "User: " + prompt +  cond + " \nASSISTANT:"

    answer = qa_llava(prompt, tokenizer, model)
    answer = answer.replace("[", "").replace("]", "")

    out_list = answer.split(",")

    final_list = []
    for i in range(len(out_list)):
        temp = list([val for val in out_list[i] if val.isalpha() or val.isnumeric() or val == " "])
        final_list.append("".join(temp).lower())
    return final_list


# Validate code (advanced): See if the generate image only contain the target object and not disturbing obj
def correlated_img_validate_advanced_llava(img_path, target_obj, disturbing_obj, image_processor=None, model=None, tokenizer=None):
    question = "Does this image contain {} and not contain {}?".format(target_obj, disturbing_obj)
    pred_answer = vqa_llava(img_path, question, image_processor, model, tokenizer)
    gt = "This image contains {} and does not contain {}.".format(target_obj, disturbing_obj)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, gt)

    if eval_result == "0" or eval_result == "2":
        return False
    else:
        return True
    

def correlated_example_create_llava(correlated_obj, attribute_category_list, save_prefix, advanced=False, image_processor=None, model=None, tokenizer=None):
    success_flag = False
    target_obj = None
    disturbing_obj = None

    target_attributes, disturbing_attributes = {}, {}

    while not success_flag:
        for i in range(len(correlated_obj)):
            if i == 0:
                not_i = 1
            else:
                not_i = 0

            raw_obj_path = save_prefix + "raw_" + correlated_obj[i] + ".png"
            target_attributes, disturbing_attributes = \
                correlated_img_generate([correlated_obj[i], correlated_obj[not_i]], attribute_category_list,
                                        raw_obj_path)
            correlated_object_segment(raw_obj_path, correlated_obj[i], save_prefix)

            img_path = save_prefix + "extracted_" + correlated_obj[i] + ".png"
            if advanced:
                valid_result = correlated_img_validate_advanced_llava(img_path, correlated_obj[i], correlated_obj[not_i], image_processor, model, tokenizer)
            else:
                valid_result = correlated_img_validate_llava(img_path, correlated_obj[i], correlated_obj[not_i], image_processor, model, tokenizer)

            if valid_result:
                disturbing_obj = correlated_obj[not_i]
                target_obj = correlated_obj[i]
                success_flag = True
                break

        if success_flag:
            break

    input_path = save_prefix + "raw_" + target_obj + ".png"
    img = Image.open(input_path).convert("RGBA")
    img.save(save_prefix + "obj.png")

    input_path = save_prefix + "extracted_" + target_obj + ".png"
    img = Image.open(input_path).convert("RGBA")
    img.save(save_prefix + "pure_obj.png")
    return target_obj, disturbing_obj, target_attributes, disturbing_attributes



# Validate code: See if the generate image only contain the target object and not disturbing obj
def correlated_img_validate_llava(img_path, target_obj, disturbing_obj, image_processor=None, model=None, tokenizer=None):
    question = "Does this image contain {} and not contain {}? Only answer with yes or no.".format(target_obj,
                                                                                                   disturbing_obj)
    pred_answer = vqa_llava(img_path, question, image_processor, model, tokenizer)

    if "yes" in pred_answer.lower():
        return True
    else:
        return False
    



if __name__ == "__main__":
    
    # question = "USER: <image>\nWrite a description for the given image sequence in a single paragraph, what is happening in this episode?\nASSISTANT:"
    image_dir = '/fs/cml-projects/Pretrain_MBRL/Mementos_dataset/hallucination_image/rw_6346290362.png'
    # vqa_llava(image_dir, question)
    # generate_noun_given_scene_llava(5, "snow")
    # random_obj_thinking_llava("snow")
    # irrelevant_obj_thinking_llava("snow", ["ski", "gloves", "snowboard"], "food")
    # single_obj_naming_llava(image_dir)
    # single_obj_caption_llava(image_dir, "statue")
    # filter_remove_obj_under_scene_llava("",None)




    
   