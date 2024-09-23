# hsy 2024.05.14 clean utils for claude
# setup
from api_key_claude import CLAUDE_API_KEY

import anthropic
import os
from PIL import Image
import time
import numpy as np
import base64
import random

import shutil
from utils_eval import evaluate_by_chatgpt_quick_test
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

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


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


############ Related to Claude

def debug2():
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    # Prompt = "Write the Go code for the simple data analysis."
    # message = client.messages.create(
    #     model="claude-3-opus-20240229",
    #     max_tokens=1024,
    #     messages=[
    #         {"role": "user", "content": Prompt}
    #     ]
    # )
    
    img_path = './test2.jpg'
    base64_image = encode_image(img_path)

    #image_encode_1 = base64.b64encode(base64_image).decode("utf-8")

    img_suffix = img_path.split('.')[-1].lower()
    if img_suffix in ['jpg', 'jpeg']:
        media_type = "image/jpeg"
    elif img_suffix in ['png']:
        media_type = "image/png"
    elif img_suffix in ['gif']:
        media_type = "image/gif"
    elif img_suffix in ['webp']:
        media_type = "image/webp"
    else:
        raise NotImplementedError('img suffix {} not recognized!'.format(img_suffix))

    print('debug media type', media_type, 'img_suffix', img_suffix)

    # url_1 = "https://images.pexels.com/photos/20433278/pexels-photo-20433278/free-photo-of-indian-blue-jay.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"

    # image_encode_1 = base64.b64encode(httpx.get(url_1).content).decode("utf-8")

    # url_2 = "https://images.pexels.com/photos/772429/pexels-photo-772429.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"

    # image_encode_2 = base64.b64encode(httpx.get(url_2).content).decode("utf-8")

    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type":  media_type,
                            "data": base64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe the image."
                    }
                ],
                
            }
        ],
    )

    print(message.content[0].text)
    return 

def debug():
    print('debug claude key {}'.format(CLAUDE_API_KEY))

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Claude, explain what is AI"}
        ]
    )
    print('answer claude',message.content[0].text)

    return 

# Easy QA function to interact with claude
def qa_claude(question, temp=0.75):
    # claude temperature ranges [0,1], 0 for more deterministic output, and 1 for more creative ones
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    question += "Keep the outcome within 200 words."

    while True:
        try:
            response = client.messages.create(
                            model="claude-3-opus-20240229",
                            messages=[
                                {"role": "user", "content": question}
                            ],
                            max_tokens=1024,
                            #temperature=temp,
            )
            break
        except:
            print("Claude Timeout qa_claude, retrying... Q: {}".format(question))
            time.sleep(5)  # Wait for 5 seconds before retrying

    answer = response.content[0].text

    print('qa_claude question {}'.format(question))
    print('qa_claude answer {}'.format(answer))

    return answer


# Ask Claude with questions and images, get answers
# Input: Masked image path
# Output: Object name
# limit max tokens within 300
def vqa_claude(masked_image_path, questions, temp=0.5, max_try=5):
    counter = 0

    img_suffix = masked_image_path.split('.')[-1].lower()
    if img_suffix in ['jpg', 'jpeg']:
        media_type = "image/jpeg"
    elif img_suffix in ['png']:
        media_type = "image/png"
    elif img_suffix in ['gif']:
        media_type = "image/gif"
    elif img_suffix in ['webp']:
        media_type = "image/webp"
    else:
        raise NotImplementedError('img suffix {} not recognized!'.format(img_suffix))

    #print('vqa_claude media type', media_type, 'img_suffix', img_suffix)

    while True:
        try:
            client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

            base64_image = encode_image(masked_image_path)
            
            message = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=300,
                temperature=temp,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type":  media_type,
                                    "data": base64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": questions,
                            }
                        ],
                        
                    }
                ],
            )

            answer = message.content[0].text
            break
        except:
            counter += 1
            print("vqa_claude image query timeout,  try {} / {}...".format(str(counter), str(max_try)))
            if counter > max_try:
                raise Exception("vqa_claude reaches max attempt of {}, regenerate case".format(str(max_try)))
            time.sleep(5)  # Wait for 5 seconds before retrying

    print('vqa_claude question {}'.format(questions))
    print('vqa_claude image_path {}'.format(masked_image_path))
    print('vqa_claude answer {}'.format(answer))

    return answer

# General image caption function, create a caption with image provided
# Input: Masked image path
# Output: Object name
def image_caption_claude(input_img_path):
    msg = "Describe what you see in this image."
    caption = vqa_claude(input_img_path, msg)
    print('image_caption_claude done')
    return caption

def generate_noun_given_scene_claude(num, scene, examples=None, temperature=0.0):

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

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    while True:
        try:
            response = client.messages.create(
                            model="claude-3-opus-20240229",
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=1024,
                            temperature=temperature,
            )

            break
        except:
            print("Claude Timeout generate_noun_given_scene_claude, retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying

    words = response.content[0].text

    print('generate_noun_given_scene_claude: prompt: {}'.format(prompt))
    print('claude answer: {}'.format(words))

    word_lst = words.split(",")
    word_lst = [w.strip() for w in word_lst]

    assert len(word_lst) == num

    print('output word_lst: {}'.format(word_lst))

    return word_lst, None


# def generate_noun_given_scene_claude(num, scene, examples=None, temperature=0.5):

#     if examples is None:
#         example_prompt = ""
#         example_prompt2 = ""
#     else:
#         example_prompt = "like {}, all ".format(",".join(examples))
#         example_prompt2 = "Do not generate the same items as the ones mentioned in the example."

#     prompt = '''
#     Generate {} words that are noun representing different physical objects and identities that are the most likely to exist in the scene of {} ({}with very high similarity score){}

#     Output format should be the list of nouns separated by comma. The output should be 
#     a string with {} words and comma only.
#     '''.format(num, scene, example_prompt, example_prompt2, num)

#     client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
#     response = client.messages.create(
#                             model="claude-3-opus-20240229",
#                             messages=[
#                                 {"role": "user", "content": prompt}
#                             ],
#                             max_tokens=1024,
#                             temperature=temperature,
#             )
    
#     words = response.content[0].text

#     print('generate_noun_given_scene_claude: prompt: {}'.format(prompt))
#     print('claude answer: {}'.format(words))

#     word_lst = words.split(",")
#     word_lst = [w.strip() for w in word_lst]

#     assert len(word_lst) == num

#     print('output word_lst: {}'.format(word_lst))

#     return word_lst, None


# not sure whether it is used
def random_obj_thinking_claude(scene_name, temperature, cond=""):
    print('random_obj_thinking_claude:')

    if scene_name is None:
        scene_desc = '''
                Think about one random object. Use a one or two words to describe this object.
                '''
    else:
        scene_desc = '''
                Think about one random object that is unlikely to exist in the {}. Use a one or two words to describe this object.
                '''.format(scene_name)

    obj_name = qa_claude(scene_desc + cond, temperature)
    return obj_name


# # Find an irrelevant object given the scene and the existing objects
# # Input: Masked image path
# # Output: Object name
def irrelevant_obj_thinking_claude(scene_name, word_list, category, temperature, cond=""):
    print('irrelevant_obj_thinking_claude:')
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
    obj_name = qa_claude(scene_desc + cond, temperature)

    print('irrelevant_obj_thinking_claude done')

    return obj_name

# Given the image of the object, generate its name and describe using no more than 3 words
def single_obj_naming_claude(single_obj_img_path):
    msg = "Describe the object in this image with no more than 3 words"
    object_name = vqa_claude(single_obj_img_path, msg)
    print('single_obj_naming_claude done.')
    return object_name


# Given the image of the object and its name, use 1 around-10-word sentence to describe this object
def single_obj_caption_claude(single_obj_img_path, single_obj_name):
    prompt = "Given the image of {}, describe this object using a single sentence within 10 words.".format(single_obj_name)
    answer = vqa_claude(single_obj_img_path, prompt)
    print('single_obj_caption_claude done.')
    return answer

# General image caption function, create a caption with image provided
# Input: Masked image path
# Output: Object name
def object_image_caption_claude(input_img_path, obj_name, irrelevant_obj_attribute):
    attribute_category_str = ""
    attribute_category_list = list(irrelevant_obj_attribute.keys())

    for attribute_category in attribute_category_list:
        attribute_category_str += attribute_category + ", "
    msg = "Describe the {} in this image with a single paragraph. Focus on the following attributes: {}".format(obj_name, attribute_category_str)
    caption = vqa_claude(input_img_path, msg)
    print('object_image_caption_claude done')
    return caption

# Generate the ground truth of the image after editing
# Content: Scene (Given), Object name and description one-by-one, Background, Added Object
def gt_generation_claude(init_img_path, mask_bbox, scene_name, irrelevant_obj, irrelevant_obj_attribute, save_prefix="./"):
    obj_img = []
    obj_name = []
    obj_description = []
    for i in range(len(mask_bbox)):
        obj_img.append(single_obj_extract(init_img_path, mask_bbox, i, save_prefix))
        obj_temp = single_obj_naming_claude(os.path.join(save_prefix, "obj_" + str(i) + ".png"))
        obj_name.append(obj_temp)
        obj_description.append(single_obj_caption_claude(os.path.join(save_prefix, "obj_" + str(i) + ".png"), obj_temp))

    # background_extract(init_img_path, mask_bbox, save_prefix)

    irrelevant_obj_description = object_image_caption_claude(os.path.join(save_prefix, "pure_obj.png"),
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

# Generate the ground truth of the image before object removal editing
# Content: Scene (Given), Object name and description one-by-one, Removed Object and description
# Output: Ground truth, target/removed/preserved object index (For image editing and spatial relation)
def gt_generation_multi_obj_removal_claude(init_img_path, mask_bbox, scene_name, target_obj, save_prefix="./"):
    obj_img = []
    obj_name = []
    obj_description = []
    for i in range(len(mask_bbox)):
        obj_img.append(single_obj_extract(init_img_path, mask_bbox, i, save_prefix))
        obj_temp = single_obj_naming_claude(os.path.join(save_prefix, "obj_" + str(i) + ".png"))
        obj_name.append(obj_temp)
        obj_description.append(single_obj_caption_claude(os.path.join(save_prefix, "obj_" + str(i) + ".png"), obj_temp))

    target_obj_description = object_image_caption_claude(os.path.join(save_prefix, "target_obj.png"), target_obj, {})

    ground_truth = {
        "scene": scene_name,
        "object_name": obj_name,
        "object_description": obj_description,
        "non_exist_target_object_name": target_obj,
        "non_exist_target_object_description": target_obj_description
    }
    return ground_truth

def filter_remove_obj_under_scene_claude(scene_name, irrelevant_obj_dict_lst, temperature=0.75):
    objs_prompt = ""
    for i, o in enumerate(irrelevant_obj_dict_lst):
        objs_prompt += "{}. {};".format(str(i), o["obj_name"])

    prompt = '''
        Given a list of objects, pick all objects that are relevant and related to the scene {}, or it is common to see those objects in this scene of {}.
        Each object is given an index, and return the number index only separated by comma. 
        The AutoHallusion output should only consists of numbers and comma. The list of objects are as follows: {}
    '''.format(scene_name, scene_name, objs_prompt)

    filtered_objs_idx = qa_claude(prompt, temperature)

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
            print("invalid index {} in filter_remove_obj_under_scene_claude from claude.".format(str(idx)))

    return ret_dict_lst


def filter_most_irrelevant_claude(scene_name, word_lst, irrelevant_obj_dict_lst, temperature=0):
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

    filtered_obj_idx = qa_claude(prompt, temperature)

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

def list_objects_given_img_claude(img_path, num_list_obj):
    prompt = "List {} physical objects in the given image. The output should be a list contains object names separated by commas.".format(
        str(num_list_obj))

    out_list = []
    while len(out_list) != num_list_obj:
        answer = vqa_claude(img_path, prompt)
        if "," in answer:
            out_list = answer.split(",")
        elif "\n" in answer:
            out_list = answer.split("\n")

    final_list = []
    for i in range(len(out_list)):
        temp = list([val for val in out_list[i] if val.isalpha() or val.isnumeric() or val == " "])
        final_list.append("".join(temp).lower())
    return final_list

# Brutally think about two closely related objects without any other constraints
def correlated_obj_thinking_claude(temperature=0.75, cond=""):
    prompt = """
    Can you generate two objects that are strongly correlated? If one thing appears, it often appears with the other 
    objects. For example, [fish tank, fish]. Please only generate two objects separated with commas.
    """

    answer = qa_claude(prompt + cond, temp=temperature)
    answer = answer.replace("[", "").replace("]", "")

    out_list = answer.split(",")

    final_list = []
    for i in range(len(out_list)):
        temp = list([val for val in out_list[i] if val.isalpha() or val.isnumeric() or val == " "])
        final_list.append("".join(temp).lower())
    return final_list

# Validate code (advanced): See if the generate image only contain the target object and not disturbing obj
def correlated_img_validate_advanced_claude(img_path, target_obj, disturbing_obj):
    question = "Does this image contain {} and not contain {}?".format(target_obj, disturbing_obj)
    pred_answer = vqa_claude(img_path, question)
    gt = "This image contains {} and does not contain {}.".format(target_obj, disturbing_obj)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, gt)

    if eval_result == "0" or eval_result == "2":
        return False
    else:
        return True


# Validate code: See if the generate image only contain the target object and not disturbing obj
def correlated_img_validate_claude(img_path, target_obj, disturbing_obj):
    question = "Does this image contain {} and not contain {}? Only answer with yes or no.".format(target_obj,
                                                                                                   disturbing_obj)
    pred_answer = vqa_claude(img_path, question)

    if "yes" in pred_answer.lower():
        return True
    else:
        return False


def correlated_example_create_claude(correlated_obj, attribute_category_list, save_prefix, advanced=False):
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
                valid_result = correlated_img_validate_advanced_claude(img_path, correlated_obj[i], correlated_obj[not_i])
            else:
                valid_result = correlated_img_validate_claude(img_path, correlated_obj[i], correlated_obj[not_i])

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

if __name__ == "__main__":
    debug()
    #generate_noun_given_scene_claude(num=5, scene='sunrise')
