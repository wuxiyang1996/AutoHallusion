from openai import OpenAI
from api_key import OPENAI_API_KEY
import requests
import os
from PIL import Image
import numpy as np
import base64
import time

import random
import shutil

from utils.utils_eval import evaluate_by_chatgpt_quick_test
from utils.utils import correlated_object_segment, correlated_img_generate

########### common utils
# Extract the single object from the scene with masks
def single_obj_extract(img_path, mask_bbox, idx, save_prefix):
    raw_img = Image.open(img_path)
    bbox = mask_bbox[idx]
    obj_img = np.array(raw_img)[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    obj_img = Image.fromarray(obj_img.astype(np.uint8)).convert('RGBA')
    obj_img.save(os.path.join(save_prefix, "obj_" + str(idx) + ".png"))
    return obj_img

def generate_noun_given_scene_gpt4v(num, scene, examples=None, temperature=0, log=False):
    if examples is None:
        example_prompt = ""
        example_prompt2 = ""
    else:
        example_prompt = "like {}, all ".format(",".join(examples))
        example_prompt2 = "Do not generate the same items as the ones mentioned in the example."

    prompt = '''
    Generate {} words that are noun representing different physical objects and identities that are the most likely to exist in the scene of {} ({}with very high similarity score){}

    Output format should be the list of nouns separated by comma. The output should be 
    a string with {} words and comma only.
    '''.format(num, scene, example_prompt, example_prompt2, num)

    client = OpenAI(api_key=OPENAI_API_KEY)

    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4-turbo",
                # model="gpt-3.5-turbo",
                temperature=temperature,
                messages=[
                    {"role": "user",
                     "content": prompt}
                ],
                logprobs=True,
                top_logprobs=5,
                #   max_tokens=1000,
                #   frequency_penalty=0.0,
            )
            break
        except:
            print("GPT4V Timeout generate_noun_given_scene_gpt4v, retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying

    words = completion.choices[0].message.content

    logprobs = completion.choices[0].logprobs.content
    logprobs_out = []
    acc = 1
    count = 1
    for lprobs in logprobs:
        if "," in lprobs.token:
            logprobs_out.append(acc / count)
            acc = 1
            count = 1
        else:
            acc += np.exp(lprobs.logprob) * 100
            count += 1


    logprobs_out.append(acc / count)

    word_lst = words.split(",")
    word_lst = [w.strip() for w in word_lst]

    assert len(word_lst) == num
    assert len(logprobs_out) == num

    return word_lst, logprobs_out

def random_obj_thinking_gpt4v(scene_name, temperature, cond=""):
    if scene_name is None:
        scene_desc = '''
                Think about one random object. Use a one or two words to describe this object.
                '''
    else:
        scene_desc = '''
                Think about one random object that is unlikely to exist in the {}. Use a one or two words to describe this object.
                '''.format(scene_name)
    obj_name = qa_gpt4v(scene_desc + cond, temperature)

    temp = list([val for val in obj_name if val.isalpha() or val.isnumeric() or val == " "])
    object_name = "".join(temp).lower()
    return object_name

def list_objects_given_img_gpt4v(img_path, num_list_obj):
    prompt = "List {} physical objects in the given image. The output should be a list contains object names separated by commas.".format(
        str(num_list_obj))

    out_list = []
    while len(out_list) != num_list_obj:
        answer = vqa_gpt4v(img_path, prompt)
        out_list = answer.split(",")

    final_list = []
    for i in range(len(out_list)):
        temp = list([val for val in out_list[i] if val.isalpha() or val.isnumeric() or val == " "])
        final_list.append("".join(temp).lower())
    return final_list


# # Find an irrelevant object given the scene and the existing objects
# # Input: Masked image path
# # Output: Object name
def irrelevant_obj_thinking_gpt4v(scene_name, word_list, category, temperature, cond=""):
    words = ",".join(word_list)
    if category is not None:
        scene_desc = '''
                Think about one commonly seen physical object from the category of {} that is irrelevant to the existing physical objects: "{}" and is unlikely to exist in the {}. Use a one or two words to describe this object.
                This object should not be a concept or too abstract. For example, Ocean, or Space is too abstract to describe by a concrete identity, while fish and space ship are good examples under those concepts.
                '''.format(category, words, scene_name)
    else:
        scene_desc = '''
                Think about one commonly seen physical object that is irrelevant to the existing physical objects: "{}" and is unlikely to exist in the {}. Use a one or two words to describe this object.
                This object should not be a concept or too abstract. For example, Ocean, or Space is too abstract to describe by a concrete identity, while fish and space ship are good examples under those concepts.
                '''.format(words, scene_name)
    obj_name = qa_gpt4v(scene_desc + cond, temperature)

    temp = list([val for val in obj_name if val.isalpha() or val.isnumeric() or val == " "])
    object_name = "".join(temp).lower()
    return object_name

def filter_remove_obj_under_scene_gpt4v(scene_name, irrelevant_obj_dict_lst, temperature=1.5):
    objs_prompt = ""
    for i, o in enumerate(irrelevant_obj_dict_lst):
        objs_prompt += "{}. {};".format(str(i), o["obj_name"])

    prompt = '''
        Given a list of objects, pick all objects that are relevant and related to the scene {}, or it is common to see those objects in this scene of {}.
        Each object is given an index, and return the number index only separated by comma. 
        The AutoHallusion output should only consists of numbers and comma. The list of objects are as follows: {}
    '''.format(scene_name, scene_name, objs_prompt)

    filtered_objs_idx = qa_gpt4v(prompt, temperature)

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
            print("invalid index {} in filter_remove_obj_under_scene_gpt4v from gpt.".format(str(idx)))

    return ret_dict_lst

def filter_most_irrelevant_gpt4v(scene_name, word_lst, irrelevant_obj_dict_lst, temperature=0):
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

    filtered_obj_idx = qa_gpt4v(prompt, temperature)

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

# Given the image of the object, generate its name and describe using no more than 3 words
def single_obj_naming_gpt4v(single_obj_img_path):
    msg = "Describe the object in this image with no more than 3 words"
    object_name = vqa_gpt4v(single_obj_img_path, msg)
    return object_name


# Given the image of the object and its name, use 1 around-10-word sentence to describe this object
def single_obj_caption_gpt4v(single_obj_img_path, single_obj_name):
    prompt = "Given the image of {}, describe this object using a single sentence within 10 words.".format(
        single_obj_name)
    answer = vqa_gpt4v(single_obj_img_path, prompt)
    return answer

# Generate the ground truth of the image after editing
# Content: Scene (Given), Object name and description one-by-one, Background, Added Object
def gt_generation_gpt4v(init_img_path, mask_bbox, scene_name, irrelevant_obj, irrelevant_obj_attribute,
                        save_prefix="./"):
    obj_img = []
    obj_name = []
    obj_description = []
    for i in range(len(mask_bbox)):
        obj_img.append(single_obj_extract(init_img_path, mask_bbox, i, save_prefix))
        obj_temp = single_obj_naming_gpt4v(os.path.join(save_prefix, "obj_" + str(i) + ".png"))
        obj_name.append(obj_temp)
        obj_description.append(single_obj_caption_gpt4v(os.path.join(save_prefix, "obj_" + str(i) + ".png"), obj_temp))

    # background_extract(init_img_path, mask_bbox, save_prefix)

    irrelevant_obj_description = object_image_caption_gpt4v(os.path.join(save_prefix, "pure_obj.png"),
                                                            irrelevant_obj, irrelevant_obj_attribute)

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

# General image caption function, create a caption with image provided
# Input: Masked image path
# Output: Object name
def object_image_caption_gpt4v(input_img_path, obj_name, irrelevant_obj_attribute):
    attribute_category_str = ""
    attribute_category_list = list(irrelevant_obj_attribute.keys())

    for attribute_category in attribute_category_list:
        attribute_category_str += attribute_category + ", "
    msg = "Describe the {} in this image with a single paragraph. Focus on the following attributes: {}".format(
        obj_name, attribute_category_str)
    caption = vqa_gpt4v(input_img_path, msg)
    return caption

# Generate the ground truth of the image before object removal editing
# Content: Scene (Given), Object name and description one-by-one, Removed Object and description
# Output: Ground truth, target/removed/preserved object index (For image editing and spatial relation)
def gt_generation_multi_obj_removal_gpt4v(init_img_path, mask_bbox, scene_name, target_obj, save_prefix="./"):
    obj_img = []
    obj_name = []
    obj_description = []
    for i in range(len(mask_bbox)):
        obj_img.append(single_obj_extract(init_img_path, mask_bbox, i, save_prefix))
        obj_temp = single_obj_naming_gpt4v(os.path.join(save_prefix, "obj_" + str(i) + ".png"))
        obj_name.append(obj_temp)
        obj_description.append(single_obj_caption_gpt4v(os.path.join(save_prefix, "obj_" + str(i) + ".png"), obj_temp))

    target_obj_description = object_image_caption_gpt4v(os.path.join(save_prefix, "target_obj.png"), target_obj, {})

    ground_truth = {
        "scene": scene_name,
        "object_name": obj_name,
        "object_description": obj_description,
        "non_exist_target_object_name": target_obj,
        "non_exist_target_object_description": target_obj_description
    }
    return ground_truth

# Brutally think about two closely related objects without any other constraints
def correlated_obj_thinking_gpt4v(temperature=1.5, cond=""):
    prompt = """
    Can you generate two objects that are strongly correlated? If one thing appears, it often appears with the other 
    objects. For example, [fish tank, fish]. Please only generate two objects separated with commas.
    """

    answer = qa_gpt4v(prompt + cond, temp=temperature)
    answer = answer.replace("[", "").replace("]", "")

    out_list = answer.split(",")

    final_list = []
    for i in range(len(out_list)):
        temp = list([val for val in out_list[i] if val.isalpha() or val.isnumeric() or val == " "])
        final_list.append("".join(temp).lower())
    return final_list

# Validate code (advanced): See if the generate image only contain the target object and not disturbing obj
def correlated_img_validate_advanced_gpt4v(img_path, target_obj, disturbing_obj):
    question = "Does this image contain {} and not contain {}?".format(target_obj, disturbing_obj)
    pred_answer = vqa_gpt4v(img_path, question)
    gt = "This image contains {} and does not contain {}.".format(target_obj, disturbing_obj)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, gt)

    if eval_result == "0" or eval_result == "2":
        return False
    else:
        return True


# Validate code: See if the generate image only contain the target object and not disturbing obj
def correlated_img_validate_gpt4v(img_path, target_obj, disturbing_obj):
    question = "Does this image contain {} and not contain {}? Only answer with yes or no.".format(target_obj,
                                                                                                   disturbing_obj)
    pred_answer = vqa_gpt4v(img_path, question)

    if "yes" in pred_answer.lower():
        return True
    else:
        return False

def correlated_example_create_gpt4v(correlated_obj, attribute_category_list, save_prefix, advanced=False):
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
                valid_result = correlated_img_validate_advanced_gpt4v(img_path, correlated_obj[i], correlated_obj[not_i])
            else:
                valid_result = correlated_img_validate_gpt4v(img_path, correlated_obj[i], correlated_obj[not_i])

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

# General image caption function, create a caption with image provided
# Input: Masked image path
# Output: Object name
def image_caption_gpt4v(input_img_path):
    msg = "Describe what you see in this image."
    caption = vqa_gpt4v(input_img_path, msg)
    return caption

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Ask GPT with questions and images, get answers
# Input: Masked image path
# Output: Object name
def vqa_gpt4v(masked_image_path, questions, temp=1):
    base64_image = encode_image(masked_image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": questions
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": temp,
        "max_tokens": 300
    }

    while True:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            # print(response.json())
            answer = response.json()["choices"][0]["message"]["content"]
            break
        except:
            print("GPT4V image query timeout, retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying

    return answer

# Easy QA function to interact with GPT
def qa_gpt4v(question, temp=1.5):
    client = OpenAI(api_key=OPENAI_API_KEY)

    question += "Keep the outcome within 200 words."

    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "user", "content": question}
                ],
                temperature=temp,
            )
            break
        except:
            print("GPT4V Timeout qa_gpt4v, retrying... Q: {}".format(question))
            time.sleep(5)  # Wait for 5 seconds before retrying

    answer = response.choices[0].message.content

    return answer

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