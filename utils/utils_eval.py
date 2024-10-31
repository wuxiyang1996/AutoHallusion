import openai
from openai import OpenAI
from api_key import OPENAI_API_KEY
import requests
import os
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
# openai.api_key = OPENAI_API_KEY
from diffusers import StableDiffusionInpaintPipeline
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import base64
import torch
import json
from tqdm import tqdm
import time

from rembg import remove
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import random


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


# Evaluate if the input prediction answer is correct or not given the image caption
# Modified from the same function in HallusionBench v1.0
# Quickly given the answer without storage
def evaluate_by_chatgpt_quick_test(question, pred_answer, gt_image_caption_queried):

    prompt = 'Imagine you are an intelligent teacher. Thoroughly read the question, reference ground truth text and the prediction answer to ensure a clear understanding of the information provided. Assess the correctness of the predictions. '
    prompt += 'If the prediction answer does not conflict with the reference ground truth text, please generate “correct”. If the prediction answer conflict with the reference ground truth text, please generate “incorrect”. If the prediction answer is unclear about the answer, please generate "unclear". \n\n Question:'
    prompt += question
    prompt += '\nReference Ground Truth Text: '
    prompt += gt_image_caption_queried
    prompt += '\nPrediction answer:'
    prompt += pred_answer
    prompt += '\nOutput:'

    output_text = qa_gpt4v(prompt)

    if 'incorrect' in output_text.lower():
        gpt_correctness = "0"

    elif 'correct' in output_text.lower():
        gpt_correctness = "1"
    else:
        gpt_correctness = "2"

    return gpt_correctness



def extract_obj_from_caption(caption):
    lst_str = qa_gpt4v('''Base on the following caption, list all objects mentioned in this caption. '{}' 
                                Return the list of objects only, separated by comma.
                               '''.format(caption))

    lst = lst_str.split(",")
    lst = [w.strip() for w in lst]

    return lst

# Evaluate if the given object exists within the reference list of objects
def evaluate_if_given_within(reference, generated):
    prompt = 'Imagine you are an intelligent teacher. Thoroughly read the given object and reference list of objects. Determine if the given object exists within the reference list. Note duplicated words with a similar meaning are taken as the same.'
    prompt += 'If the given object exists in the reference list of objects, please generate a single word “yes”, otherwise please generate a single word “no”. \n\n The given object:'
    prompt += generated
    prompt += '\nThe reference list of objects: '
    prompt += ','.join(reference)
    prompt += '\nOutput:'

    output_text = qa_gpt4v(prompt)

    if 'yes' in output_text.lower():
        gpt_correctness = "1"
    else:
        gpt_correctness = "0"

    return gpt_correctness


# Evaluation function, only detect if the object exists in this given image
# vqa_model_func should implement vqa_gpt4v/gemini/claude
def evaluation_existence_given_pred_model(result_img_path, ground_truth, result_caption, vqa_model_func, logger, debug=False, notebook_print=False):
    # background = ground_truth["background"]
    # scene_name = ground_truth["scene"]
    obj_name = ground_truth["object_name"]
    # object_description = ground_truth["object_description"]
    irrelevant_object_name = ground_truth["irrelevant_object_name"]
    irrelevant_object_description = ground_truth["irrelevant_object_description"]

    success = False

    if debug:
        # case_result = [False, False, False]
        case_result = {"irr_obj_correct_stitch1": False, 
                       "irr_obj_correct_stitch2": False, 
                       "irr_obj_response_conflict": False, 
                       "irr_obj_local_global_conflict1": False, 
                       "irr_obj_local_global_conflict2": False, 
                       "other_obj": False}

    # gt_concat_text = gt_concat(ground_truth)

    # background_info = "Given the scene of {} with a background of {}. ".format(scene_name, background)
    # background_info = "Given the scene of {}. ".format(scene_name)

    ########## Existence Detection: Added, Irrelevant Object ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning("[ATTACK] Adding new irrelevant object '{}' -- existence".format(irrelevant_object_name))

    if notebook_print:
        print("######")
        print("[ATTACK] Adding new irrelevant object '{}' -- existence".format(irrelevant_object_name))

    question = "Is there a {} in this image?".format(irrelevant_object_name)
    #pred_answer = vqa_gpt4v(result_img_path, question)
    pred_answer = vqa_model_func(result_img_path, question)

    # queried_info = "There is a {}. We find that this object is {} ".format(irrelevant_object_name,
    #                                                                        irrelevant_object_description)
    queried_info = "There is a {}.".format(irrelevant_object_name)

    # eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, background_info + queried_info)
    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, queried_info)
    logger.warning("[Obj] {}".format(irrelevant_object_name))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer))
    # logger.warning("[GT] {}".format(background_info + queried_info))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Obj] {}".format(irrelevant_object_name))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    # print("Object: ", irrelevant_object_name, ", Question: ", question)
    # print("Answer: ", pred_answer, "Eval Result: ", eval_result)
    if eval_result == "0":
        success = True

        if debug:
            case_result["irr_obj_correct_stitch1"] = True

        logger.warning("[Attack Success] Added irrelevant object not detected.")
        if notebook_print:
            print("[Attack Success] Added irrelevant object not detected.")
    else:
        logger.warning("[Attack Failed] Added irrelevant object detected.")
        if notebook_print:
            print("[Attack Failed] Added irrelevant object detected.")

    ########## Existence Detection: Added, Irrelevant Object, with caption ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning("[ATTACK] Adding new irrelevant object '{}' with image-level caption -- existence".format(
        irrelevant_object_name))

    if notebook_print:
        print("######")
        print("[ATTACK] Adding new irrelevant object '{}' with image-level caption -- existence".format(
            irrelevant_object_name))

    question = "We have an image depicting {}. Is there a {} in this image?".format(result_caption,
                                                                                    irrelevant_object_name)
    #pred_answer2 = vqa_gpt4v(result_img_path, question)
    pred_answer2 = vqa_model_func(result_img_path, question)

    # eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer2, background_info + queried_info)
    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer2, queried_info)
    logger.warning("[Obj] {}".format(irrelevant_object_name))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer2))
    # logger.warning("[GT] {}".format(background_info + queried_info))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Obj] {}".format(irrelevant_object_name))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer2))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    # print("Object: ", irrelevant_object_name, ", Question: ", question)
    # print("Answer: ", pred_answer2, "Eval Result: ", eval_result)
    if eval_result == "0":
        success = True

        if debug:
            case_result["irr_obj_correct_stitch2"] = True

        logger.warning(
            "[Attack Success] Added irrelevant object not detected, with image-level caption in the question.")

        if notebook_print:
            print("[Attack Success] Added irrelevant object not detected, with image-level caption in the question.")

    else:
        logger.warning("[Attack Failed] Added irrelevant object detected, with image-level caption in the question.")
        if notebook_print:
            print(
                "[Attack Failed] Added irrelevant object detected, with image-level caption in the question.")

    ########## Existence Detection: irrelevant obj conflict ###########
    eval_result = evaluate_by_chatgpt_quick_test("Is there a {} in this image?".format(irrelevant_object_name),
                                                 pred_answer2,
                                                 pred_answer)

    if eval_result == "0":
        success = True

        if debug:
            case_result["irr_obj_response_conflict"] = True

        logger.warning("[Attack Success] Added irrelevant object inconsistent.")
        if notebook_print:
            print("[Attack Success] Added irrelevant object inconsistent.")
    else:
        logger.warning("[Attack Failed] Added irrelevant object consistent")
        if notebook_print:
            print("[Attack Failed] Added irrelevant object consistent")


    ########## Existence Detection: irrelevant obj local global conflict ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a {} in this image?".format(irrelevant_object_name),
                                                 pred_answer,
                                                 "We find that this object is {}.".format(irrelevant_object_description))

    if eval_result == "0":
        success = True

        if debug:
            case_result["irr_obj_local_global_conflict1"] = True

        logger.warning("[Attack Success] Added irrelevant object local global inconsistent.")
        if notebook_print:
            print("[Attack Success] Added irrelevant object local global inconsistent.")
    else:
        logger.warning("[Attack Failed] Added irrelevant object local global consistent")
        if notebook_print:
            print("[Attack Failed] Added irrelevant object local global consistent")

    eval_result = evaluate_by_chatgpt_quick_test("Is there a {} in this image?".format(irrelevant_object_name),
                                                 pred_answer2,
                                                 "We find that this object is {}.".format(irrelevant_object_description))

    if eval_result == "0":
        success = True

        if debug:
            case_result["irr_obj_local_global_conflict2"] = True
        logger.warning("[Attack Success] Added irrelevant object local global inconsistent.")
        if notebook_print:
            print("[Attack Success] Added irrelevant object local global inconsistent.")
    else:
        logger.warning("[Attack Failed] Added irrelevant object local global consistent")
        if notebook_print:
            print("[Attack Failed] Added irrelevant object local global consistent")

    ########## Existence Detection: other objects, in caption but not in detection ###########
    logger.warning("######")
    logger.warning("[ATTACK] image-level caption conflict with objects -- existence")
    if notebook_print:
        print("######")
        print("[ATTACK] image-level caption conflict with objects -- existence")

    caption_obj_lst = extract_obj_from_caption(result_caption)
    logger.warning(
        "[Target Model Generated] objects described in the generated caption: {}".format(",".join(caption_obj_lst)))
    det_obj_lst = obj_name + [irrelevant_object_name]
    non_exist = []

    if notebook_print:
        print(
            "[Target Model Generated] objects described in the generated caption: {}".format(",".join(caption_obj_lst)))

    for i in range(len(caption_obj_lst)):
        if evaluate_if_given_within(det_obj_lst, caption_obj_lst[i]) == "0":
            non_exist.append(caption_obj_lst[i])
    logger.warning(
        "[Detection Model] objects described in the image-level caption, but not detected by the detection model: {}".format(
            ",".join(caption_obj_lst)))

    if notebook_print:
        print(
            "[Detection Model] objects described in the image-level caption, but not detected by the detection model: {}".format(
                ",".join(caption_obj_lst)))

    for i in range(len(non_exist)):
        logger.warning("*****")
        if notebook_print:
            print("*****")

        question = "Is there a {} in this image?".format(non_exist[i])
        #pred_answer = vqa_gpt4v(result_img_path, question)
        pred_answer = vqa_model_func(result_img_path, question)

        # eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, result_caption)
        eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer,
                                                     "There is a {} in this image.".format(non_exist[i]))

        logger.warning("[Obj] {}".format(non_exist[i]))
        logger.warning("[Q] {}".format(question))
        logger.warning("[Pred] {}".format(pred_answer))
        logger.warning("[GT] {}".format(result_caption))
        logger.warning("[Eval (same or not)] {}".format(eval_result))

        if notebook_print:
            print("[Obj] {}".format(non_exist[i]))
            print("[Q] {}".format(question))
            print("[Pred] {}".format(pred_answer))
            print("[GT] {}".format(result_caption))
            print("[Eval (same or not)] {}".format(eval_result))

        if eval_result == "0":
            success = True

            if debug:
                case_result["other_obj"] = True
            logger.warning("[Attack Success] {} not detected, but it is mentioned in the image-level caption.".format(
                non_exist[i]))

            if notebook_print:
                print("[Attack Success] {} not detected, but it is mentioned in the image-level caption.".format(
                non_exist[i]))

        else:
            logger.warning("[Attack Failed] {} mentioned in the image-level caption exists.".format(non_exist[i]))
            if notebook_print:
                print("[Attack Failed] {} mentioned in the image-level caption exists.".format(non_exist[i]))

    ########## Existence Detection (other): relevant Objects ###########

    # # detected object
    # reference = obj_name + [irrelevant_object_name]

    # non_exist = []
    # non_exist_description = []

    # for i in range(len(obj_name)):
    #     if evaluate_if_given_within(reference, obj_name[i]) == "0":
    #         non_exist.append(obj_name[i])
    #         non_exist_description.append(object_description[i])

    # ########## Existence Detection: Existing Object in the caption, not in the ground truth ###########
    # for i in range(len(non_exist)):
    #     question = "Is there a {} in this image?".format(non_exist[i])
    #     pred_answer = vqa_gpt4v(result_img_path, question)

    if success:
        if debug:
            return "Existence Attack Success", case_result
        else:
            return "Existence Attack Success"
    else:
        if debug:
            return "Existence Attack Failed", case_result
        else:
            return "Existence Attack Failed"
        
# Evaluation function, only detect if the object exists in this given image
# vqa_model_func should implement vqa_gpt4v/gemini/claude
def evaluation_existence_given_pred_model_simple(result_img_path, ground_truth, result_caption, vqa_model_func, logger, debug=False, notebook_print=False):
    # background = ground_truth["background"]
    # scene_name = ground_truth["scene"]
    obj_name = ground_truth["object_name"]
    # object_description = ground_truth["object_description"]
    irrelevant_object_name = ground_truth["irrelevant_object_name"]
    irrelevant_object_description = ground_truth["irrelevant_object_description"]

    success = False

    if debug:
        # case_result = [False, False, False]
        case_result = {"irr_obj_correct_stitch1": False, 
                       "irr_obj_correct_stitch2": False, 
                       "irr_obj_response_conflict": False, 
                       "irr_obj_local_global_conflict1": False, 
                       "irr_obj_local_global_conflict2": False, 
                       "other_obj": False}

    # gt_concat_text = gt_concat(ground_truth)

    # background_info = "Given the scene of {} with a background of {}. ".format(scene_name, background)
    # background_info = "Given the scene of {}. ".format(scene_name)

    ########## Existence Detection: Added, Irrelevant Object ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning("[ATTACK] Adding new irrelevant object '{}' -- existence".format(irrelevant_object_name))

    if notebook_print:
        print("######")
        print("[ATTACK] Adding new irrelevant object '{}' -- existence".format(irrelevant_object_name))

    question = "Is there a {} in this image?".format(irrelevant_object_name)
    #pred_answer = vqa_gpt4v(result_img_path, question)
    pred_answer = vqa_model_func(result_img_path, question)

    # queried_info = "There is a {}. We find that this object is {} ".format(irrelevant_object_name,
    #                                                                        irrelevant_object_description)
    queried_info = "There is a {}.".format(irrelevant_object_name)

    # eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, background_info + queried_info)
    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, queried_info)
    logger.warning("[Obj] {}".format(irrelevant_object_name))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer))
    # logger.warning("[GT] {}".format(background_info + queried_info))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Obj] {}".format(irrelevant_object_name))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    # print("Object: ", irrelevant_object_name, ", Question: ", question)
    # print("Answer: ", pred_answer, "Eval Result: ", eval_result)
    if eval_result == "0":
        success = True

        if debug:
            case_result["irr_obj_correct_stitch1"] = True

        logger.warning("[Attack Success] Added irrelevant object not detected.")
        if notebook_print:
            print("[Attack Success] Added irrelevant object not detected.")
        return success, question, pred_answer, queried_info

    else:
        logger.warning("[Attack Failed] Added irrelevant object detected.")
        if notebook_print:
            print("[Attack Failed] Added irrelevant object detected.")

    ########## Existence Detection: Added, Irrelevant Object, with caption ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning("[ATTACK] Adding new irrelevant object '{}' with image-level caption -- existence".format(
        irrelevant_object_name))

    if notebook_print:
        print("######")
        print("[ATTACK] Adding new irrelevant object '{}' with image-level caption -- existence".format(
            irrelevant_object_name))

    question = "We have an image depicting {}. Is there a {} in this image?".format(result_caption,
                                                                                    irrelevant_object_name)
    #pred_answer2 = vqa_gpt4v(result_img_path, question)
    pred_answer2 = vqa_model_func(result_img_path, question)

    # eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer2, background_info + queried_info)
    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer2, queried_info)
    logger.warning("[Obj] {}".format(irrelevant_object_name))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer2))
    # logger.warning("[GT] {}".format(background_info + queried_info))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Obj] {}".format(irrelevant_object_name))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer2))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    # print("Object: ", irrelevant_object_name, ", Question: ", question)
    # print("Answer: ", pred_answer2, "Eval Result: ", eval_result)
    if eval_result == "0":
        success = True

        if debug:
            case_result["irr_obj_correct_stitch2"] = True

        logger.warning(
            "[Attack Success] Added irrelevant object not detected, with image-level caption in the question.")

        if notebook_print:
            print("[Attack Success] Added irrelevant object not detected, with image-level caption in the question.")
        return success, question, pred_answer2, queried_info

    else:
        logger.warning("[Attack Failed] Added irrelevant object detected, with image-level caption in the question.")
        if notebook_print:
            print(
                "[Attack Failed] Added irrelevant object detected, with image-level caption in the question.")

    ########## Existence Detection: irrelevant obj conflict ###########
    eval_result = evaluate_by_chatgpt_quick_test("Is there a {} in this image?".format(irrelevant_object_name),
                                                 pred_answer2,
                                                 pred_answer)

    if eval_result == "0":
        success = True

        if debug:
            case_result["irr_obj_response_conflict"] = True

        logger.warning("[Attack Success] Added irrelevant object inconsistent.")
        if notebook_print:
            print("[Attack Success] Added irrelevant object inconsistent.")
        return success, "Is there a {} in this image?".format(irrelevant_object_name), pred_answer, queried_info

    else:
        logger.warning("[Attack Failed] Added irrelevant object consistent")
        if notebook_print:
            print("[Attack Failed] Added irrelevant object consistent")


    ########## Existence Detection: irrelevant obj local global conflict ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a {} in this image?".format(irrelevant_object_name),
                                                 pred_answer,
                                                 "We find that this object is {}.".format(irrelevant_object_description))

    if eval_result == "0":
        success = True

        if debug:
            case_result["irr_obj_local_global_conflict1"] = True

        logger.warning("[Attack Success] Added irrelevant object local global inconsistent.")
        if notebook_print:
            print("[Attack Success] Added irrelevant object local global inconsistent.")
        return success, "Is there a {} in this image?".format(irrelevant_object_name), pred_answer, "We find that this object is {}.".format(irrelevant_object_description)

    else:
        logger.warning("[Attack Failed] Added irrelevant object local global consistent")
        if notebook_print:
            print("[Attack Failed] Added irrelevant object local global consistent")

    eval_result = evaluate_by_chatgpt_quick_test("Is there a {} in this image?".format(irrelevant_object_name),
                                                 pred_answer2,
                                                 "We find that this object is {}.".format(irrelevant_object_description))

    if eval_result == "0":
        success = True

        if debug:
            case_result["irr_obj_local_global_conflict2"] = True
        logger.warning("[Attack Success] Added irrelevant object local global inconsistent.")
        if notebook_print:
            print("[Attack Success] Added irrelevant object local global inconsistent.")
        return success, "Is there a {} in this image?".format(irrelevant_object_name), pred_answer, "We find that this object is {}.".format(irrelevant_object_description)

    else:
        logger.warning("[Attack Failed] Added irrelevant object local global consistent")
        if notebook_print:
            print("[Attack Failed] Added irrelevant object local global consistent")

    ########## Existence Detection: other objects, in caption but not in detection ###########
    logger.warning("######")
    logger.warning("[ATTACK] image-level caption conflict with objects -- existence")
    if notebook_print:
        print("######")
        print("[ATTACK] image-level caption conflict with objects -- existence")

    caption_obj_lst = extract_obj_from_caption(result_caption)
    logger.warning(
        "[Target Model Generated] objects described in the generated caption: {}".format(",".join(caption_obj_lst)))
    det_obj_lst = obj_name + [irrelevant_object_name]
    non_exist = []

    if notebook_print:
        print(
            "[Target Model Generated] objects described in the generated caption: {}".format(",".join(caption_obj_lst)))

    for i in range(len(caption_obj_lst)):
        if evaluate_if_given_within(det_obj_lst, caption_obj_lst[i]) == "0":
            non_exist.append(caption_obj_lst[i])
    logger.warning(
        "[Detection Model] objects described in the image-level caption, but not detected by the detection model: {}".format(
            ",".join(caption_obj_lst)))

    if notebook_print:
        print(
            "[Detection Model] objects described in the image-level caption, but not detected by the detection model: {}".format(
                ",".join(caption_obj_lst)))

    for i in range(len(non_exist)):
        logger.warning("*****")
        if notebook_print:
            print("*****")

        question = "Is there a {} in this image?".format(non_exist[i])
        #pred_answer = vqa_gpt4v(result_img_path, question)
        pred_answer = vqa_model_func(result_img_path, question)

        # eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, result_caption)
        eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer,
                                                     "There is a {} in this image.".format(non_exist[i]))

        logger.warning("[Obj] {}".format(non_exist[i]))
        logger.warning("[Q] {}".format(question))
        logger.warning("[Pred] {}".format(pred_answer))
        logger.warning("[GT] {}".format(result_caption))
        logger.warning("[Eval (same or not)] {}".format(eval_result))

        if notebook_print:
            print("[Obj] {}".format(non_exist[i]))
            print("[Q] {}".format(question))
            print("[Pred] {}".format(pred_answer))
            print("[GT] {}".format(result_caption))
            print("[Eval (same or not)] {}".format(eval_result))

        if eval_result == "0":
            success = True

            if debug:
                case_result["other_obj"] = True
            logger.warning("[Attack Success] {} not detected, but it is mentioned in the image-level caption.".format(
                non_exist[i]))

            if notebook_print:
                print("[Attack Success] {} not detected, but it is mentioned in the image-level caption.".format(
                non_exist[i]))
            return success, question, pred_answer, result_caption

        else:
            logger.warning("[Attack Failed] {} mentioned in the image-level caption exists.".format(non_exist[i]))
            if notebook_print:
                print("[Attack Failed] {} mentioned in the image-level caption exists.".format(non_exist[i]))

    ########## Existence Detection (other): relevant Objects ###########

    # # detected object
    # reference = obj_name + [irrelevant_object_name]

    # non_exist = []
    # non_exist_description = []

    # for i in range(len(obj_name)):
    #     if evaluate_if_given_within(reference, obj_name[i]) == "0":
    #         non_exist.append(obj_name[i])
    #         non_exist_description.append(object_description[i])

    # ########## Existence Detection: Existing Object in the caption, not in the ground truth ###########
    # for i in range(len(non_exist)):
    #     question = "Is there a {} in this image?".format(non_exist[i])
    #     pred_answer = vqa_gpt4v(result_img_path, question)

    return success, None, None, None


# Evaluation function for spatial relation, detect if the model could detect the spatial relation between the attack
# object and the chosen existing object (with its description) as expected
# The questions are asked as Y/N questions, over Up/Down/Left/Right/Front.
# Results are compared with the prediction and the queried ground truth.
# Current version only consider 1 attack object
# vqa_model_func for gpt4v/gemini/claude
def evaluation_spatial_relation_give_pred_model(result_img_path, ground_truth, vqa_model_func, logger, debug=False, notebook_print=False):
    scene_name = ground_truth["scene"]
    # background = ground_truth["background"]
    obj_name = ground_truth["object_name"]
    object_description = ground_truth["object_description"]
    irrelevant_object_name = ground_truth["irrelevant_object_name"]
    irrelevant_obj_attribute = ground_truth["irrelevant_obj_attribute"]
    irrelevant_object_description = ground_truth["irrelevant_object_description"]

    spatial_relation_all = ground_truth["spatial_relation"]
    spatial_distance = ground_truth["spatial_distance"]

    # background_info = "Given the scene of {} with a background of {}. ".format(scene_name, background)
    # background_info = ""

    success = False
    if debug:
        # case_result = [False, False, False]
        case_result = {"irr_obj_spatial_correct_stitch": [], 
                       "irr_obj_spatial_response_conflict": []}
        
    spatial_distance_sorted = sorted(spatial_distance)

    ########## Spatial Relation Detection: Spatial Relation Between Added, Irrelevant Object and Existing Object, Y/N questions one-by-one over each object  ###########
    # Loop over all existing objects
    for i in range(len(obj_name)):
        existing_obj = obj_name[i]
        existing_obj_description = object_description[i]
        spatial_relation_curr = spatial_relation_all[i]
        spatial_distance_curr = spatial_distance[i]

        spatial_distance_rank = spatial_distance_sorted.index(spatial_distance_curr)

        # Determine if the spatial relation between the added adversarial object and the current existing object correct
        logger.warning("######")
        logger.warning("[ATTACK] Determine spatial relation between the irrelevant object '{}' and the existing object: {}".format(
            irrelevant_object_name, existing_obj))

        if notebook_print:
            print("######")
            print(
                "[ATTACK] Determine spatial relation between the irrelevant object '{}' and the existing object: {}".format(
                    irrelevant_object_name, existing_obj))

        if debug:
            case_result["irr_obj_spatial_correct_stitch"].append(list())
            case_result["irr_obj_spatial_response_conflict"].append(list())

        # Loop over all detected spatial relations in the ground truth to construct questions
        for spatial_relation_curr_single in spatial_relation_curr:
            # spatial_flag = "uncertain"

            if spatial_relation_curr_single == "left":
                spatial_flag = "on the left of"
            elif spatial_relation_curr_single == "right":
                spatial_flag = "on the right of"
            elif spatial_relation_curr_single == "up":
                spatial_flag = "above"
            elif spatial_relation_curr_single == "down":
                spatial_flag = "below"
            elif spatial_relation_curr_single == "front":
                spatial_flag = "on the front of"

            question = "Is the {} {} a/an {} in this image, given their center positions?".format(irrelevant_object_name, spatial_flag, existing_obj)
            # question = "Is the {} {} {} in this image?".format(irrelevant_object_name, spatial_flag, existing_obj)
            supplement = "(where {} is the one that {}.)".format(existing_obj, existing_obj_description)
            #pred_answer = vqa_gpt4v(result_img_path, question + supplement)
            pred_answer = vqa_model_func(result_img_path, question + supplement)

            # queried_info = "There is a {}. This object is {}, i.e. {}.".format(irrelevant_object_name, attributes[i], attribute_details[i])
            queried_info = "{} is {} a/an {} in this image".format(irrelevant_object_name, spatial_flag, existing_obj)

            # eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, background_info + queried_info)
            eval_result = evaluate_by_chatgpt_quick_test(question + supplement, pred_answer, queried_info + supplement)
            logger.warning("[Obj] {}".format(irrelevant_object_name))
            logger.warning("[Q] {}".format(question + supplement))
            logger.warning("[Pred] {}".format(pred_answer))
            # logger.warning("[GT] {}".format(background_info + queried_info))
            logger.warning("[GT] {}".format(queried_info + supplement))
            logger.warning("[Eval (same or not)] {}".format(eval_result))

            if notebook_print:
                print("[Obj] {}".format(irrelevant_object_name))
                print("[Q] {}".format(question + supplement))
                print("[Pred] {}".format(pred_answer))
                print("[GT] {}".format(queried_info + supplement))
                print("[Eval (same or not)] {}".format(eval_result))

            if eval_result == "0":
                success = True
                if debug:
                    case_result["irr_obj_spatial_correct_stitch"][-1].append(str(True))
                logger.warning(
                    "[Attack Success] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) is not correct.".format(existing_obj, str(spatial_distance_rank + 1)))
                if notebook_print:
                    print(
                        "[Attack Success] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) is not correct.".format(
                            existing_obj, str(spatial_distance_rank + 1)))

            else:
                if debug:
                    case_result["irr_obj_spatial_correct_stitch"][-1].append(str(False))
                logger.warning(
                    "[Attack Failed] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) is correct.".format(existing_obj, str(spatial_distance_rank + 1)))
                if notebook_print:
                    print(
                        "[Attack Failed] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) is correct.".format(
                            existing_obj, str(spatial_distance_rank + 1)))


            question = "Is the object({}) {} a/an {} in this image, given their center positions?".format(irrelevant_object_description, spatial_flag, existing_obj)
            # question = "Is the object({}) {} {} in this image?".format(irrelevant_object_description, spatial_flag, existing_obj)
            supplement = "(where {} is the one that {}.)".format(existing_obj, existing_obj_description)
            #pred_answer = vqa_gpt4v(result_img_path, question + supplement)
            pred_answer = vqa_model_func(result_img_path, question + supplement)

            # queried_info = "There is a {}. This object is {}, i.e. {}.".format(irrelevant_object_name, attributes[i], attribute_details[i])
            queried_info = "The object({}) is {} a/an {} in this image".format(irrelevant_object_description, spatial_flag, existing_obj)

            # eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, background_info + queried_info)
            eval_result = evaluate_by_chatgpt_quick_test(question + supplement, pred_answer, queried_info + supplement)
            logger.warning("[Obj] {}".format(irrelevant_object_description))
            logger.warning("[Q] {}".format(question + supplement))
            logger.warning("[Pred] {}".format(pred_answer))
            # logger.warning("[GT] {}".format(background_info + queried_info))
            logger.warning("[GT] {}".format(queried_info + supplement))
            logger.warning("[Eval (same or not)] {}".format(eval_result))

            if notebook_print:
                print("[Obj] {}".format(irrelevant_object_description))
                print("[Q] {}".format(question + supplement))
                print("[Pred] {}".format(pred_answer))
                print("[GT] {}".format(queried_info + supplement))
                print("[Eval (same or not)] {}".format(eval_result))


            if eval_result == "0":
                success = True
                if debug:
                    case_result["irr_obj_spatial_response_conflict"][-1].append(str(True))
                logger.warning(
                    "[Attack Success] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) has conflict.".format(existing_obj, str(spatial_distance_rank + 1)))

                if notebook_print:
                    print(
                        "[Attack Success] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) has conflict.".format(
                            existing_obj, str(spatial_distance_rank + 1)))

            else:
                if debug:
                    case_result["irr_obj_spatial_response_conflict"][-1].append(str(False))
                logger.warning(
                    "[Attack Failed] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) has no conflict.".format(existing_obj, str(spatial_distance_rank + 1)))

                if notebook_print:
                    print(
                        "[Attack Failed] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) has no conflict.".format(
                            existing_obj, str(spatial_distance_rank + 1)))

        if debug:
            case_result["irr_obj_spatial_correct_stitch"][-1] = ",".join(case_result["irr_obj_spatial_correct_stitch"][-1])
            case_result["irr_obj_spatial_response_conflict"][-1] = ",".join(case_result["irr_obj_spatial_response_conflict"][-1])

    if success:
        if debug:
            return "Spatial Relation Attack Success", case_result
        else:
            return "Spatial Relation Attack Success"
    else:
        if debug:
            return "Spatial Relation Attack Success", case_result
        else:
            return "Spatial Relation Attack Failed"


# Evaluation function for spatial relation, detect if the model could detect the spatial relation between the attack
# object and the chosen existing object (with its description) as expected
# The questions are asked as Y/N questions, over Up/Down/Left/Right/Front.
# Results are compared with the prediction and the queried ground truth.
# Current version only consider 1 attack object
# vqa_model_func for gpt4v/gemini/claude
def evaluation_spatial_relation_give_pred_model_simple(result_img_path, ground_truth, vqa_model_func, logger, debug=False, notebook_print=False):
    scene_name = ground_truth["scene"]
    # background = ground_truth["background"]
    obj_name = ground_truth["object_name"]
    object_description = ground_truth["object_description"]
    irrelevant_object_name = ground_truth["irrelevant_object_name"]
    irrelevant_obj_attribute = ground_truth["irrelevant_obj_attribute"]
    irrelevant_object_description = ground_truth["irrelevant_object_description"]

    spatial_relation_all = ground_truth["spatial_relation"]
    spatial_distance = ground_truth["spatial_distance"]

    # background_info = "Given the scene of {} with a background of {}. ".format(scene_name, background)
    # background_info = ""

    success = False
    if debug:
        # case_result = [False, False, False]
        case_result = {"irr_obj_spatial_correct_stitch": [], 
                       "irr_obj_spatial_response_conflict": []}
        
    spatial_distance_sorted = sorted(spatial_distance)

    ########## Spatial Relation Detection: Spatial Relation Between Added, Irrelevant Object and Existing Object, Y/N questions one-by-one over each object  ###########
    # Loop over all existing objects
    for i in range(len(obj_name)):
        existing_obj = obj_name[i]
        existing_obj_description = object_description[i]
        spatial_relation_curr = spatial_relation_all[i]
        spatial_distance_curr = spatial_distance[i]

        spatial_distance_rank = spatial_distance_sorted.index(spatial_distance_curr)

        # Determine if the spatial relation between the added adversarial object and the current existing object correct
        logger.warning("######")
        logger.warning("[ATTACK] Determine spatial relation between the irrelevant object '{}' and the existing object: {}".format(
            irrelevant_object_name, existing_obj))

        if notebook_print:
            print("######")
            print(
                "[ATTACK] Determine spatial relation between the irrelevant object '{}' and the existing object: {}".format(
                    irrelevant_object_name, existing_obj))

        if debug:
            case_result["irr_obj_spatial_correct_stitch"].append(list())
            case_result["irr_obj_spatial_response_conflict"].append(list())

        # Loop over all detected spatial relations in the ground truth to construct questions
        for spatial_relation_curr_single in spatial_relation_curr:
            # spatial_flag = "uncertain"

            if spatial_relation_curr_single == "left":
                spatial_flag = "on the left of"
            elif spatial_relation_curr_single == "right":
                spatial_flag = "on the right of"
            elif spatial_relation_curr_single == "up":
                spatial_flag = "above"
            elif spatial_relation_curr_single == "down":
                spatial_flag = "below"
            elif spatial_relation_curr_single == "front":
                spatial_flag = "on the front of"

            question = "Is the {} {} a/an {} in this image, given their center positions?".format(irrelevant_object_name, spatial_flag, existing_obj)
            # question = "Is the {} {} {} in this image?".format(irrelevant_object_name, spatial_flag, existing_obj)
            supplement = "(where {} is the one that {}.)".format(existing_obj, existing_obj_description)
            #pred_answer = vqa_gpt4v(result_img_path, question + supplement)
            pred_answer = vqa_model_func(result_img_path, question + supplement)

            # queried_info = "There is a {}. This object is {}, i.e. {}.".format(irrelevant_object_name, attributes[i], attribute_details[i])
            queried_info = "{} is {} a/an {} in this image".format(irrelevant_object_name, spatial_flag, existing_obj)

            # eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, background_info + queried_info)
            eval_result = evaluate_by_chatgpt_quick_test(question + supplement, pred_answer, queried_info + supplement)
            logger.warning("[Obj] {}".format(irrelevant_object_name))
            logger.warning("[Q] {}".format(question + supplement))
            logger.warning("[Pred] {}".format(pred_answer))
            # logger.warning("[GT] {}".format(background_info + queried_info))
            logger.warning("[GT] {}".format(queried_info + supplement))
            logger.warning("[Eval (same or not)] {}".format(eval_result))

            if notebook_print:
                print("[Obj] {}".format(irrelevant_object_name))
                print("[Q] {}".format(question + supplement))
                print("[Pred] {}".format(pred_answer))
                print("[GT] {}".format(queried_info + supplement))
                print("[Eval (same or not)] {}".format(eval_result))

            if eval_result == "0":
                success = True
                if debug:
                    case_result["irr_obj_spatial_correct_stitch"][-1].append(str(True))
                logger.warning(
                    "[Attack Success] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) is not correct.".format(existing_obj, str(spatial_distance_rank + 1)))
                if notebook_print:
                    print(
                        "[Attack Success] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) is not correct.".format(
                            existing_obj, str(spatial_distance_rank + 1)))
            
                return success, question + supplement, pred_answer, queried_info + supplement
            
            else:
                if debug:
                    case_result["irr_obj_spatial_correct_stitch"][-1].append(str(False))
                logger.warning(
                    "[Attack Failed] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) is correct.".format(existing_obj, str(spatial_distance_rank + 1)))
                if notebook_print:
                    print(
                        "[Attack Failed] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) is correct.".format(
                            existing_obj, str(spatial_distance_rank + 1)))


            question = "Is the object({}) {} a/an {} in this image, given their center positions?".format(irrelevant_object_description, spatial_flag, existing_obj)
            # question = "Is the object({}) {} {} in this image?".format(irrelevant_object_description, spatial_flag, existing_obj)
            supplement = "(where {} is the one that {}.)".format(existing_obj, existing_obj_description)
            #pred_answer = vqa_gpt4v(result_img_path, question + supplement)
            pred_answer = vqa_model_func(result_img_path, question + supplement)

            # queried_info = "There is a {}. This object is {}, i.e. {}.".format(irrelevant_object_name, attributes[i], attribute_details[i])
            queried_info = "The object({}) is {} a/an {} in this image".format(irrelevant_object_description, spatial_flag, existing_obj)

            # eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, background_info + queried_info)
            eval_result = evaluate_by_chatgpt_quick_test(question + supplement, pred_answer, queried_info + supplement)
            logger.warning("[Obj] {}".format(irrelevant_object_description))
            logger.warning("[Q] {}".format(question + supplement))
            logger.warning("[Pred] {}".format(pred_answer))
            # logger.warning("[GT] {}".format(background_info + queried_info))
            logger.warning("[GT] {}".format(queried_info + supplement))
            logger.warning("[Eval (same or not)] {}".format(eval_result))

            if notebook_print:
                print("[Obj] {}".format(irrelevant_object_description))
                print("[Q] {}".format(question + supplement))
                print("[Pred] {}".format(pred_answer))
                print("[GT] {}".format(queried_info + supplement))
                print("[Eval (same or not)] {}".format(eval_result))


            if eval_result == "0":
                success = True
                if debug:
                    case_result["irr_obj_spatial_response_conflict"][-1].append(str(True))
                logger.warning(
                    "[Attack Success] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) has conflict.".format(existing_obj, str(spatial_distance_rank + 1)))

                if notebook_print:
                    print(
                        "[Attack Success] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) has conflict.".format(
                            existing_obj, str(spatial_distance_rank + 1)))
                return success, question + supplement, pred_answer, queried_info + supplement

            else:
                if debug:
                    case_result["irr_obj_spatial_response_conflict"][-1].append(str(False))
                logger.warning(
                    "[Attack Failed] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) has no conflict.".format(existing_obj, str(spatial_distance_rank + 1)))

                if notebook_print:
                    print(
                        "[Attack Failed] The spatial relation between the added irrelevant object and the existing object {} (#{} closest) has no conflict.".format(
                            existing_obj, str(spatial_distance_rank + 1)))

        if debug:
            case_result["irr_obj_spatial_correct_stitch"][-1] = ",".join(case_result["irr_obj_spatial_correct_stitch"][-1])
            case_result["irr_obj_spatial_response_conflict"][-1] = ",".join(case_result["irr_obj_spatial_response_conflict"][-1])

    return success, None, None, None



# Evaluation function, only detect if the object exists in this given image
# For the object removal task
def evaluation_existence_removed_pred_model(result_img_path, ground_truth, vqa_model_func, logger, debug=False,
                                       object_wise=False, notebook_print=False):
    # background = ground_truth["background"]
    # scene_name = ground_truth["scene"]
    obj_name = ground_truth["object_name"]
    object_description = ground_truth["object_description"]
    target_object_name = ground_truth["non_exist_target_object_name"]
    target_object_description = ground_truth["non_exist_target_object_description"]
    result_caption = ground_truth["result_description"]

    success = False

    if debug:
        # case_result = [False, False, False]
        case_result = {"target_obj_correct_simple": False,
                       "target_obj_correct_desc": False,
                       "target_obj_correct_concat": False,
                       "target_obj_correct_caption": False,
                       "target_obj_response_simple_desc": False,
                       "target_obj_response_simple_concat": False,
                       "target_obj_response_simple_caption": False,
                       "target_obj_response_desc_concat": False,
                       "target_obj_response_desc_caption": False,
                       "target_obj_response_concat_caption": False}

    ########## Existence Detection: Remove Object ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning("[ATTACK] Remove an existing object '{}' -- existence".format(target_object_name))

    if notebook_print:
        print("######")
        print("[ATTACK] Remove an existing object '{}' -- existence".format(target_object_name))

    question = "Is there a/an {} in this image?".format(target_object_name)
    pred_answer_simple = vqa_model_func(result_img_path, question)

    queried_info = "There is no {}.".format(target_object_name)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_simple, queried_info)
    logger.warning("[Removed Obj] {}".format(target_object_name))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_simple))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Removed Obj] {}".format(target_object_name))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_simple))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_correct_simple"] = True

        logger.warning("[Attack Success] Removed object mistakenly detected.")

        if notebook_print:
            print("[Attack Success] Removed object mistakenly detected.")
    else:
        logger.warning("[Attack Failed] Removed object not detected.")
        if notebook_print:
            print("[Attack Failed] Removed object not detected.")

    ########## Existence Detection: Remove Object, with object-level caption ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning("[ATTACK] Remove an existing object '{}' with object-level caption -- existence".format(
        target_object_name))

    if notebook_print:
        print("######")
        print("[ATTACK] Remove an existing object '{}' with object-level caption -- existence".format(
            target_object_name))

    question = "Is there a/an {} in this image? This object is {}".format(target_object_name, target_object_description)
    pred_answer_desc = vqa_model_func(result_img_path, question)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_desc, queried_info)
    logger.warning("[Removed Obj] {}".format(target_object_name))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_desc))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Removed Obj] {}".format(target_object_name))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_desc))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_correct_desc"] = True

        logger.warning(
            "[Attack Success] Removed object mistakenly detected, with object-level caption in the question.")

        if notebook_print:
            print(
                "[Attack Success] Removed object mistakenly detected, with object-level caption in the question.")

    else:
        logger.warning("[Attack Failed] Removed object not detected, with object-level caption in the question.")

        if notebook_print:
            print("[Attack Failed] Removed object not detected, with object-level caption in the question.")

    ########## Existence Detection: Remove Object, with other existing object captions ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning("[ATTACK] Remove an existing object '{}' with other existing object captions -- existence".format(
        target_object_name))

    if notebook_print:
        print("######")
        print("[ATTACK] Remove an existing object '{}' with other existing object captions -- existence".format(
                target_object_name))

    other_obj_info = ""
    for i in range(len(object_description)):
        other_obj_info += object_description[i]

    question = "In this image, We have existing objects including: {}. Is there a/an {} in this image?".format(
        other_obj_info, target_object_name)
    pred_answer_concat = vqa_model_func(result_img_path, question)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_concat, queried_info)
    logger.warning("[Removed Obj] {}".format(target_object_name))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_concat))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Removed Obj] {}".format(target_object_name))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_concat))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_correct_concat"] = True

        logger.warning(
            "[Attack Success] Removed object mistakenly detected, with other existing object captions in the question.")

        if notebook_print:
            print(
                "[Attack Success] Removed object mistakenly detected, with other existing object captions in the question.")
    else:
        logger.warning(
            "[Attack Failed] Removed object not detected, with other existing object captions in the question.")

        if notebook_print:
            print("[Attack Failed] Removed object not detected, with other existing object captions in the question.")

    ########## Existence Detection: Remove Object, with image-level captions ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning("[ATTACK] Remove an existing object '{}' with image-level captions -- existence".format(
        target_object_name))

    if notebook_print:
        print("######")
        print("[ATTACK] Remove an existing object '{}' with image-level captions -- existence".format(
            target_object_name))

    other_obj_info = ""
    for i in range(len(object_description)):
        other_obj_info += object_description[i]

    question = "This image depicts {}. Is there a/an {} in this image?".format(result_caption, target_object_name)
    pred_answer_caption = vqa_model_func(result_img_path, question)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_caption, queried_info)
    logger.warning("[Removed Obj] {}".format(target_object_name))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_caption))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Removed Obj] {}".format(target_object_name))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_caption))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_correct_caption"] = True

        logger.warning(
            "[Attack Success] Removed object mistakenly detected, with image-level captions in the question.")

        if notebook_print:
            print(
                "[Attack Success] Removed object mistakenly detected, with image-level captions in the question.")

    else:
        logger.warning("[Attack Failed] Removed object not detected, with image-level captions in the question.")

        if notebook_print:
            print("[Attack Failed] Removed object not detected, with image-level captions in the question.")

    ########## Existence Detection: Remove obj conflict, simple vs. object-level ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(target_object_name),
                                                 pred_answer_simple,
                                                 pred_answer_desc)

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_response_simple_desc"] = True

        logger.warning(
            "[Attack Success] Removed object inconsistent between responses with and without object-level caption.")

        if notebook_print:
            print(
                "[Attack Success] Removed object inconsistent between responses with and without object-level caption.")

    else:
        logger.warning(
            "[Attack Failed] Removed object consistent between responses with and without object-level caption.")

        if notebook_print:
            print(
                "[Attack Failed] Removed object consistent between responses with and without object-level caption.")

    ########## Existence Detection: Remove obj conflict, simple vs. other object captions ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(target_object_name),
                                                 pred_answer_simple,
                                                 pred_answer_concat)

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_response_simple_concat"] = True

        logger.warning(
            "[Attack Success] Removed object inconsistent between responses with and without other object captions.")

        if notebook_print:
            print("[Attack Success] Removed object inconsistent between responses with and without other object captions.")

    else:
        logger.warning(
            "[Attack Failed] Removed object consistent between responses with and without other object captions.")

        if notebook_print:
            print("[Attack Failed] Removed object consistent between responses with and without other object captions.")

    ########## Existence Detection: Remove obj conflict, simple vs. image-level captions ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(target_object_name),
                                                 pred_answer_simple,
                                                 pred_answer_caption)

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_response_simple_caption"] = True

        logger.warning(
            "[Attack Success] Removed object inconsistent between responses with and without image-level captions.")

        if notebook_print:
            print("[Attack Success] Removed object inconsistent between responses with and without image-level captions.")
    else:
        logger.warning(
            "[Attack Failed] Removed object consistent between responses with and without image-level captions.")
        if notebook_print:
            print("[Attack Failed] Removed object consistent between responses with and without image-level captions.")

    ########## Existence Detection: Remove obj conflict, object-level vs. other object captions ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(target_object_name),
                                                 pred_answer_desc,
                                                 pred_answer_concat)

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_response_desc_concat"] = True

        logger.warning(
            "[Attack Success] Removed object inconsistent between responses with object-level and other object captions.")

        if notebook_print:
            print("[Attack Success] Removed object inconsistent between responses with object-level and other object captions.")

    else:
        logger.warning(
            "[Attack Failed] Removed object consistent between responses with object-level and other object captions.")

        if notebook_print:
            print("[Attack Failed] Removed object consistent between responses with object-level and other object captions.")

    ########## Existence Detection: Remove obj conflict, object-level vs. image-level captions ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(target_object_name),
                                                 pred_answer_desc,
                                                 pred_answer_caption)

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_response_desc_caption"] = True

        logger.warning(
            "[Attack Success] Removed object inconsistent between responses with object-level and image-level captions.")

        if notebook_print:
            print("[Attack Success] Removed object inconsistent between responses with object-level and image-level captions.")

    else:
        logger.warning(
            "[Attack Failed] Removed object consistent between responses with object-level and image-level captions.")

        if notebook_print:
            print("[Attack Failed] Removed object consistent between responses with object-level and image-level captions.")

    ########## Existence Detection: Remove obj conflict, other object vs. image-level captions ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(target_object_name),
                                                 pred_answer_concat,
                                                 pred_answer_caption)

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_response_concat_caption"] = True

        logger.warning(
            "[Attack Success] Removed object inconsistent between responses with other object and image-level captions.")

        if notebook_print:
            print("[Attack Success] Removed object inconsistent between responses with other object and image-level captions.")

    else:
        logger.warning(
            "[Attack Failed] Removed object consistent between responses with other object and image-level captions.")
        if notebook_print:
            print("[Attack Failed] Removed object consistent between responses with other object and image-level captions.")

    ########## (Optional, For Case Study) Existence Detection: Remove object given the existence of other objects ###########
    if object_wise:
        logger.warning("######")
        logger.warning(
            "[ATTACK] Remove an existing object '{}' given the existence of another object captions -- existence")

        if notebook_print:
            print("######")
            print("[ATTACK] Remove an existing object '{}' given the existence of another object captions -- existence")

        for i in range(len(object_description)):
            logger.warning("*****")

            if notebook_print:
                print("*****")

            question = "We have an object ({}) in this image. Is there a/an {} in this image?".format(
                object_description[i], target_object_name)
            pred_answer = vqa_model_func(result_img_path, question)
            queried_info = "There is no {} in this image.".format(target_object_name)

            # eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, result_caption)
            eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, queried_info)

            logger.warning("[Existing Obj] {}".format(obj_name[i]))
            logger.warning("[Removed Obj] {}".format(target_object_name))
            logger.warning("[Q] {}".format(question))
            logger.warning("[Pred] {}".format(pred_answer))
            logger.warning("[GT] {}".format(queried_info))
            logger.warning("[Eval (same or not)] {}".format(eval_result))

            if notebook_print:
                print("[Existing Obj] {}".format(obj_name[i]))
                print("[Removed Obj] {}".format(target_object_name))
                print("[Q] {}".format(question))
                print("[Pred] {}".format(pred_answer))
                print("[GT] {}".format(queried_info))
                print("[Eval (same or not)] {}".format(eval_result))

            if eval_result == "0":
                success = True
                key_name = "target_obj_correct_other_obj_" + str(i)

                if debug:
                    case_result[key_name] = True

                logger.warning(
                    "[Attack Success] Removed object mistakenly detected, with the existing object's caption ({}) in the question.".format(
                        obj_name[i]))

                if notebook_print:
                    print(
                        "[Attack Success] Removed object mistakenly detected, with the existing object's caption ({}) in the question.".format(
                            obj_name[i]))

            else:
                logger.warning(
                    "[Attack Failed] Removed object not detected, with the existing object's caption ({}) in the question.".format(
                        obj_name[i]))

                if notebook_print:
                    print(
                        "[Attack Failed] Removed object not detected, with the existing object's caption ({}) in the question.".format(
                            obj_name[i]))

    if success:
        if debug:
            return "Existence Attack Success (Removed object)", case_result
        else:
            return "Existence Attack Success (Removed object)"
    else:
        if debug:
            return "Existence Attack Failed (Removed object)", case_result
        else:
            return "Existence Attack Failed (Removed object)"


def evaluation_existence_removed_pred_model_simple(result_img_path, ground_truth, vqa_model_func, logger, debug=False,
                                       object_wise=False, notebook_print=False):
    # background = ground_truth["background"]
    # scene_name = ground_truth["scene"]
    obj_name = ground_truth["object_name"]
    object_description = ground_truth["object_description"]
    target_object_name = ground_truth["non_exist_target_object_name"]
    target_object_description = ground_truth["non_exist_target_object_description"]
    result_caption = ground_truth["result_description"]

    success = False

    if debug:
        # case_result = [False, False, False]
        case_result = {"target_obj_correct_simple": False,
                       "target_obj_correct_desc": False,
                       "target_obj_correct_concat": False,
                       "target_obj_correct_caption": False,
                       "target_obj_response_simple_desc": False,
                       "target_obj_response_simple_concat": False,
                       "target_obj_response_simple_caption": False,
                       "target_obj_response_desc_concat": False,
                       "target_obj_response_desc_caption": False,
                       "target_obj_response_concat_caption": False}

    ########## Existence Detection: Remove Object ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning("[ATTACK] Remove an existing object '{}' -- existence".format(target_object_name))

    if notebook_print:
        print("######")
        print("[ATTACK] Remove an existing object '{}' -- existence".format(target_object_name))

    question = "Is there a/an {} in this image?".format(target_object_name)
    pred_answer_simple = vqa_model_func(result_img_path, question)

    queried_info = "There is no {}.".format(target_object_name)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_simple, queried_info)
    logger.warning("[Removed Obj] {}".format(target_object_name))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_simple))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Removed Obj] {}".format(target_object_name))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_simple))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_correct_simple"] = True

        logger.warning("[Attack Success] Removed object mistakenly detected.")

        if notebook_print:
            print("[Attack Success] Removed object mistakenly detected.")
        return success, question, pred_answer_simple, queried_info
    else:
        logger.warning("[Attack Failed] Removed object not detected.")
        if notebook_print:
            print("[Attack Failed] Removed object not detected.")

    ########## Existence Detection: Remove Object, with object-level caption ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning("[ATTACK] Remove an existing object '{}' with object-level caption -- existence".format(
        target_object_name))

    if notebook_print:
        print("######")
        print("[ATTACK] Remove an existing object '{}' with object-level caption -- existence".format(
            target_object_name))

    question = "Is there a/an {} in this image? This object is {}".format(target_object_name, target_object_description)
    pred_answer_desc = vqa_model_func(result_img_path, question)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_desc, queried_info)
    logger.warning("[Removed Obj] {}".format(target_object_name))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_desc))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Removed Obj] {}".format(target_object_name))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_desc))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_correct_desc"] = True

        logger.warning(
            "[Attack Success] Removed object mistakenly detected, with object-level caption in the question.")

        if notebook_print:
            print(
                "[Attack Success] Removed object mistakenly detected, with object-level caption in the question.")
        return success, question, pred_answer_simple, queried_info

    else:
        logger.warning("[Attack Failed] Removed object not detected, with object-level caption in the question.")

        if notebook_print:
            print("[Attack Failed] Removed object not detected, with object-level caption in the question.")

    ########## Existence Detection: Remove Object, with other existing object captions ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning("[ATTACK] Remove an existing object '{}' with other existing object captions -- existence".format(
        target_object_name))

    if notebook_print:
        print("######")
        print("[ATTACK] Remove an existing object '{}' with other existing object captions -- existence".format(
                target_object_name))

    other_obj_info = ""
    for i in range(len(object_description)):
        other_obj_info += object_description[i]

    question = "In this image, We have existing objects including: {}. Is there a/an {} in this image?".format(
        other_obj_info, target_object_name)
    pred_answer_concat = vqa_model_func(result_img_path, question)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_concat, queried_info)
    logger.warning("[Removed Obj] {}".format(target_object_name))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_concat))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Removed Obj] {}".format(target_object_name))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_concat))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_correct_concat"] = True

        logger.warning(
            "[Attack Success] Removed object mistakenly detected, with other existing object captions in the question.")

        if notebook_print:
            print(
                "[Attack Success] Removed object mistakenly detected, with other existing object captions in the question.")
        return success, question, pred_answer_simple, queried_info
    
    else:
        logger.warning(
            "[Attack Failed] Removed object not detected, with other existing object captions in the question.")

        if notebook_print:
            print("[Attack Failed] Removed object not detected, with other existing object captions in the question.")

    ########## Existence Detection: Remove Object, with image-level captions ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning("[ATTACK] Remove an existing object '{}' with image-level captions -- existence".format(
        target_object_name))

    if notebook_print:
        print("######")
        print("[ATTACK] Remove an existing object '{}' with image-level captions -- existence".format(
            target_object_name))

    other_obj_info = ""
    for i in range(len(object_description)):
        other_obj_info += object_description[i]

    question = "This image depicts {}. Is there a/an {} in this image?".format(result_caption, target_object_name)
    pred_answer_caption = vqa_model_func(result_img_path, question)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_caption, queried_info)
    logger.warning("[Removed Obj] {}".format(target_object_name))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_caption))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Removed Obj] {}".format(target_object_name))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_caption))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_correct_caption"] = True

        logger.warning(
            "[Attack Success] Removed object mistakenly detected, with image-level captions in the question.")

        if notebook_print:
            print(
                "[Attack Success] Removed object mistakenly detected, with image-level captions in the question.")
        return success, question, pred_answer_simple, queried_info

    else:
        logger.warning("[Attack Failed] Removed object not detected, with image-level captions in the question.")

        if notebook_print:
            print("[Attack Failed] Removed object not detected, with image-level captions in the question.")

    ########## Existence Detection: Remove obj conflict, simple vs. object-level ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(target_object_name),
                                                 pred_answer_simple,
                                                 pred_answer_desc)

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_response_simple_desc"] = True

        logger.warning(
            "[Attack Success] Removed object inconsistent between responses with and without object-level caption.")

        if notebook_print:
            print(
                "[Attack Success] Removed object inconsistent between responses with and without object-level caption.")
        return success, question, pred_answer_simple, queried_info

    else:
        logger.warning(
            "[Attack Failed] Removed object consistent between responses with and without object-level caption.")

        if notebook_print:
            print(
                "[Attack Failed] Removed object consistent between responses with and without object-level caption.")

    ########## Existence Detection: Remove obj conflict, simple vs. other object captions ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(target_object_name),
                                                 pred_answer_simple,
                                                 pred_answer_concat)

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_response_simple_concat"] = True

        logger.warning(
            "[Attack Success] Removed object inconsistent between responses with and without other object captions.")

        if notebook_print:
            print("[Attack Success] Removed object inconsistent between responses with and without other object captions.")
        return success, question, pred_answer_simple, queried_info

    else:
        logger.warning(
            "[Attack Failed] Removed object consistent between responses with and without other object captions.")

        if notebook_print:
            print("[Attack Failed] Removed object consistent between responses with and without other object captions.")

    ########## Existence Detection: Remove obj conflict, simple vs. image-level captions ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(target_object_name),
                                                 pred_answer_simple,
                                                 pred_answer_caption)

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_response_simple_caption"] = True

        logger.warning(
            "[Attack Success] Removed object inconsistent between responses with and without image-level captions.")

        if notebook_print:
            print("[Attack Success] Removed object inconsistent between responses with and without image-level captions.")
        return success, question, pred_answer_simple, queried_info

    else:
        logger.warning(
            "[Attack Failed] Removed object consistent between responses with and without image-level captions.")
        if notebook_print:
            print("[Attack Failed] Removed object consistent between responses with and without image-level captions.")

    ########## Existence Detection: Remove obj conflict, object-level vs. other object captions ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(target_object_name),
                                                 pred_answer_desc,
                                                 pred_answer_concat)

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_response_desc_concat"] = True

        logger.warning(
            "[Attack Success] Removed object inconsistent between responses with object-level and other object captions.")

        if notebook_print:
            print("[Attack Success] Removed object inconsistent between responses with object-level and other object captions.")
        return success, question, pred_answer_simple, queried_info

    else:
        logger.warning(
            "[Attack Failed] Removed object consistent between responses with object-level and other object captions.")

        if notebook_print:
            print("[Attack Failed] Removed object consistent between responses with object-level and other object captions.")

    ########## Existence Detection: Remove obj conflict, object-level vs. image-level captions ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(target_object_name),
                                                 pred_answer_desc,
                                                 pred_answer_caption)

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_response_desc_caption"] = True

        logger.warning(
            "[Attack Success] Removed object inconsistent between responses with object-level and image-level captions.")

        if notebook_print:
            print("[Attack Success] Removed object inconsistent between responses with object-level and image-level captions.")
        return success, question, pred_answer_simple, queried_info

    else:
        logger.warning(
            "[Attack Failed] Removed object consistent between responses with object-level and image-level captions.")

        if notebook_print:
            print("[Attack Failed] Removed object consistent between responses with object-level and image-level captions.")

    ########## Existence Detection: Remove obj conflict, other object vs. image-level captions ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(target_object_name),
                                                 pred_answer_concat,
                                                 pred_answer_caption)

    if eval_result == "0":
        success = True

        if debug:
            case_result["target_obj_response_concat_caption"] = True

        logger.warning(
            "[Attack Success] Removed object inconsistent between responses with other object and image-level captions.")

        if notebook_print:
            print("[Attack Success] Removed object inconsistent between responses with other object and image-level captions.")
        return success, question, pred_answer_simple, queried_info

    else:
        logger.warning(
            "[Attack Failed] Removed object consistent between responses with other object and image-level captions.")
        if notebook_print:
            print("[Attack Failed] Removed object consistent between responses with other object and image-level captions.")

    ########## (Optional, For Case Study) Existence Detection: Remove object given the existence of other objects ###########
    if object_wise:
        logger.warning("######")
        logger.warning(
            "[ATTACK] Remove an existing object '{}' given the existence of another object captions -- existence")

        if notebook_print:
            print("######")
            print("[ATTACK] Remove an existing object '{}' given the existence of another object captions -- existence")

        for i in range(len(object_description)):
            logger.warning("*****")

            if notebook_print:
                print("*****")

            question = "We have an object ({}) in this image. Is there a/an {} in this image?".format(
                object_description[i], target_object_name)
            pred_answer = vqa_model_func(result_img_path, question)
            queried_info = "There is no {} in this image.".format(target_object_name)

            # eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, result_caption)
            eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, queried_info)

            logger.warning("[Existing Obj] {}".format(obj_name[i]))
            logger.warning("[Removed Obj] {}".format(target_object_name))
            logger.warning("[Q] {}".format(question))
            logger.warning("[Pred] {}".format(pred_answer))
            logger.warning("[GT] {}".format(queried_info))
            logger.warning("[Eval (same or not)] {}".format(eval_result))

            if notebook_print:
                print("[Existing Obj] {}".format(obj_name[i]))
                print("[Removed Obj] {}".format(target_object_name))
                print("[Q] {}".format(question))
                print("[Pred] {}".format(pred_answer))
                print("[GT] {}".format(queried_info))
                print("[Eval (same or not)] {}".format(eval_result))

            if eval_result == "0":
                success = True
                key_name = "target_obj_correct_other_obj_" + str(i)

                if debug:
                    case_result[key_name] = True

                logger.warning(
                    "[Attack Success] Removed object mistakenly detected, with the existing object's caption ({}) in the question.".format(
                        obj_name[i]))

                if notebook_print:
                    print(
                        "[Attack Success] Removed object mistakenly detected, with the existing object's caption ({}) in the question.".format(
                            obj_name[i]))
                return success, question, pred_answer_simple, queried_info

            else:
                logger.warning(
                    "[Attack Failed] Removed object not detected, with the existing object's caption ({}) in the question.".format(
                        obj_name[i]))

                if notebook_print:
                    print(
                        "[Attack Failed] Removed object not detected, with the existing object's caption ({}) in the question.".format(
                            obj_name[i]))

    return success, None, None, None


# Evaluation function for spatial relation, detect if the model could detect the spatial relation between the attack
# object and the chosen existing object (with its description) as expected
# The questions are asked as Y/N questions, over Up/Down/Left/Right/Front.
# Results are compared with the prediction and the queried ground truth.
# Current version only consider 1 attack object
# For object removal experiments
def evaluation_spatial_relation_removed_pred_model(result_img_path, ground_truth, vqa_model_func, logger, debug=False,
                                              three_branch_analysis=False, notebook_print=False):
    scene_name = ground_truth["scene"]
    # background = ground_truth["background"]
    obj_name = ground_truth["object_name"]
    object_description = ground_truth["object_description"]
    target_object_name = ground_truth["non_exist_target_object_name"]
    target_object_description = ground_truth["non_exist_target_object_description"]

    spatial_relation_all = ground_truth["spatial_relation"]
    spatial_distance = ground_truth["spatial_distance"]

    success = False
    if debug:
        # case_result = [False, False, False]
        case_result = {"target_obj_spatial_correct_remove": [],
                       "target_obj_spatial_response_conflict": []}

        if three_branch_analysis:
            case_result["target_obj_spatial_correct_remove_adv"] = []
            case_result["target_obj_spatial_response_conflict_adv"] = []

    spatial_distance_sorted = sorted(spatial_distance)

    ########## Spatial Relation Detection: Spatial Relation Between Added, Irrelevant Object and Existing Object, Y/N questions one-by-one over each object  ###########
    # Loop over all existing objects
    for i in range(len(obj_name)):
        existing_obj = obj_name[i]
        existing_obj_description = object_description[i]
        spatial_relation_curr = spatial_relation_all[i]
        spatial_distance_curr = spatial_distance[i]

        spatial_distance_rank = spatial_distance_sorted.index(spatial_distance_curr)

        # Determine if the spatial relation between the added adversarial object and the current existing object correct
        logger.warning("######")
        logger.warning(
            "[ATTACK] Determine spatial relation between the removed object '{}' and the existing object: {}. The model should insist no.".format(
                target_object_name, existing_obj))

        if notebook_print:
            print("######")
            print(
                "[ATTACK] Determine spatial relation between the removed object '{}' and the existing object: {}. The model should insist no.".format(
                    target_object_name, existing_obj))

        if debug:
            case_result["target_obj_spatial_correct_remove"].append(list())
            case_result["target_obj_spatial_response_conflict"].append(list())

            # We assume there are three possible cases: Removed obj not exist (Right), Exist but on wrong position
            # (Wrong), Exist but on right position (Wrong)
            if three_branch_analysis:
                case_result["target_obj_spatial_correct_remove_adv"].append(list())
                case_result["target_obj_spatial_response_conflict_adv"].append(list())

        # Loop over all detected spatial relations in the ground truth to construct questions

        if len(spatial_relation_curr) == 0:
            logger.warning(
                "[ATTACK] No feasible spatial relation detected between the removed object '{}' and the existing object: {}.".format(
                    target_object_name, existing_obj))

            if notebook_print:
                print(
                    "[ATTACK] No feasible spatial relation detected between the removed object '{}' and the existing object: {}.".format(
                        target_object_name, existing_obj))

        for spatial_relation_curr_single in spatial_relation_curr:
            # spatial_flag = "uncertain"

            if spatial_relation_curr_single == "left":
                spatial_flag = "on the left of"
            elif spatial_relation_curr_single == "right":
                spatial_flag = "on the right of"
            elif spatial_relation_curr_single == "up":
                spatial_flag = "above"
            elif spatial_relation_curr_single == "down":
                spatial_flag = "below"
            elif spatial_relation_curr_single == "front":
                spatial_flag = "on the front of"

            question = "Is the {} {} a/an {} in this image, given their center positions?".format(
                target_object_name, spatial_flag, existing_obj)
            supplement = "(where {} is the one that {}.)".format(existing_obj, existing_obj_description)
            pred_answer = vqa_model_func(result_img_path, question + supplement)

            existing_info = "There is no {}. ".format(target_object_name)
            queried_info = "{} is not {} a/an {} in this image".format(target_object_name, spatial_flag, existing_obj)

            eval_result = evaluate_by_chatgpt_quick_test(question + supplement, pred_answer,
                                                         existing_info + queried_info + supplement)

            if three_branch_analysis:
                existing_info_adv = "There is a/an {}.".format(target_object_name)
                eval_result_adv = evaluate_by_chatgpt_quick_test(question + supplement, pred_answer,
                                                                 existing_info_adv + queried_info + supplement)

            logger.warning("[Obj] {}".format(target_object_name))
            logger.warning("[Q] {}".format(question + supplement))
            logger.warning("[Pred] {}".format(pred_answer))
            # logger.warning("[GT] {}".format(background_info + queried_info))
            logger.warning("[GT] {}".format(existing_info + queried_info + supplement))
            logger.warning("[Eval (same or not)] {}".format(eval_result))

            if notebook_print:
                print("[Obj] {}".format(target_object_name))
                print("[Q] {}".format(question + supplement))
                print("[Pred] {}".format(pred_answer))
                print("[GT] {}".format(existing_info + queried_info + supplement))
                print("[Eval (same or not)] {}".format(eval_result))

            if three_branch_analysis:
                logger.warning("[Eval advanced (same or not)] {}".format(eval_result_adv))

                if notebook_print:
                    print("[Eval advanced (same or not)] {}".format(eval_result_adv))

            if eval_result == "0":
                success = True

                if debug:
                    case_result["target_obj_spatial_correct_remove"][-1].append(str(True))

                    # True if existence wrong, spatial relation determination is correct
                    if three_branch_analysis:
                        if eval_result == "1":
                            case_result["target_obj_spatial_correct_remove_adv"][-1].append(str(True))
                        else:
                            case_result["target_obj_spatial_correct_remove_adv"][-1].append(str(False))

                logger.warning(
                    "[Attack Success] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) is not correct.".format(
                        existing_obj, str(spatial_distance_rank + 1)))

                if notebook_print:
                    print(
                        "[Attack Success] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) is not correct.".format(
                            existing_obj, str(spatial_distance_rank + 1)))

            else:
                if debug:
                    case_result["target_obj_spatial_correct_remove"][-1].append(str(False))
                logger.warning(
                    "[Attack Failed] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) is correct.".format(
                        existing_obj, str(spatial_distance_rank + 1)))

                if notebook_print:
                    print(
                        "[Attack Failed] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) is correct.".format(
                        existing_obj, str(spatial_distance_rank + 1)))

            question = "Is the object({}) {} a/an {} in this image, given their center positions?".format(
                target_object_description, spatial_flag, existing_obj)
            supplement = "(where {} is the one that {}.)".format(existing_obj, existing_obj_description)
            pred_answer = vqa_model_func(result_img_path, question + supplement)

            existing_info = "There is no {}. ".format(target_object_name)
            queried_info = "The object({}) is not {} a/an {} in this image".format(target_object_description,
                                                                                   spatial_flag, existing_obj)

            eval_result = evaluate_by_chatgpt_quick_test(question + supplement, pred_answer,
                                                         existing_info + queried_info + supplement)

            if three_branch_analysis:
                existing_info_adv = "There is a/an {}.".format(target_object_name)
                eval_result_adv = evaluate_by_chatgpt_quick_test(question + supplement, pred_answer,
                                                                 existing_info_adv + queried_info + supplement)

            logger.warning("[Obj] {}".format(target_object_description))
            logger.warning("[Q] {}".format(question + supplement))
            logger.warning("[Pred] {}".format(pred_answer))
            logger.warning("[GT] {}".format(existing_info + queried_info + supplement))
            logger.warning("[Eval (same or not)] {}".format(eval_result))

            if notebook_print:
                print("[Obj] {}".format(target_object_description))
                print("[Q] {}".format(question + supplement))
                print("[Pred] {}".format(pred_answer))
                print("[GT] {}".format(existing_info + queried_info + supplement))
                print("[Eval (same or not)] {}".format(eval_result))

            if three_branch_analysis:
                logger.warning("[Eval advanced (same or not)] {}".format(eval_result_adv))

                if notebook_print:
                    print("[Eval advanced (same or not)] {}".format(eval_result_adv))

            if eval_result == "0":
                success = True
                if debug:
                    case_result["target_obj_spatial_response_conflict"][-1].append(str(True))

                    # True if existence wrong, spatial relation determination is correct
                    if three_branch_analysis:
                        if eval_result == "1":
                            case_result["target_obj_spatial_response_conflict_adv"][-1].append(str(True))
                        else:
                            case_result["target_obj_spatial_response_conflict_adv"][-1].append(str(False))

                logger.warning(
                    "[Attack Success] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) has conflict.".format(
                        existing_obj, str(spatial_distance_rank + 1)))

                if notebook_print:
                    print(
                        "[Attack Success] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) has conflict.".format(
                            existing_obj, str(spatial_distance_rank + 1)))

            else:
                if debug:
                    case_result["target_obj_spatial_response_conflict"][-1].append(str(False))

                logger.warning(
                    "[Attack Failed] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) has no conflict.".format(
                        existing_obj, str(spatial_distance_rank + 1)))

                if notebook_print:
                    print(
                        "[Attack Failed] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) has no conflict.".format(
                            existing_obj, str(spatial_distance_rank + 1)))

        if debug:
            case_result["target_obj_spatial_correct_remove"][-1] = ",".join(
                case_result["target_obj_spatial_correct_remove"][-1])
            case_result["target_obj_spatial_response_conflict"][-1] = ",".join(
                case_result["target_obj_spatial_response_conflict"][-1])

            if three_branch_analysis:
                case_result["target_obj_spatial_correct_remove_adv"][-1] = ",".join(
                    case_result["target_obj_spatial_correct_remove_adv"][-1])
                case_result["target_obj_spatial_response_conflict_adv"][-1] = ",".join(
                    case_result["target_obj_spatial_response_conflict_adv"][-1])

    if success:
        if debug:
            return "Spatial Relation Attack Success (Removed object)", case_result
        else:
            return "Spatial Relation Attack Success (Removed object)"
    else:
        if debug:
            return "Spatial Relation Attack Success (Removed object)", case_result
        else:
            return "Spatial Relation Attack Failed (Removed object)"


def evaluation_spatial_relation_removed_pred_model_simple(result_img_path, ground_truth, vqa_model_func, logger, debug=False,
                                              three_branch_analysis=False, notebook_print=False):
    scene_name = ground_truth["scene"]
    # background = ground_truth["background"]
    obj_name = ground_truth["object_name"]
    object_description = ground_truth["object_description"]
    target_object_name = ground_truth["non_exist_target_object_name"]
    target_object_description = ground_truth["non_exist_target_object_description"]

    spatial_relation_all = ground_truth["spatial_relation"]
    spatial_distance = ground_truth["spatial_distance"]

    success = False
    if debug:
        # case_result = [False, False, False]
        case_result = {"target_obj_spatial_correct_remove": [],
                       "target_obj_spatial_response_conflict": []}

        if three_branch_analysis:
            case_result["target_obj_spatial_correct_remove_adv"] = []
            case_result["target_obj_spatial_response_conflict_adv"] = []

    spatial_distance_sorted = sorted(spatial_distance)

    ########## Spatial Relation Detection: Spatial Relation Between Added, Irrelevant Object and Existing Object, Y/N questions one-by-one over each object  ###########
    # Loop over all existing objects
    for i in range(len(obj_name)):
        existing_obj = obj_name[i]
        existing_obj_description = object_description[i]
        spatial_relation_curr = spatial_relation_all[i]
        spatial_distance_curr = spatial_distance[i]

        spatial_distance_rank = spatial_distance_sorted.index(spatial_distance_curr)

        # Determine if the spatial relation between the added adversarial object and the current existing object correct
        logger.warning("######")
        logger.warning(
            "[ATTACK] Determine spatial relation between the removed object '{}' and the existing object: {}. The model should insist no.".format(
                target_object_name, existing_obj))

        if notebook_print:
            print("######")
            print(
                "[ATTACK] Determine spatial relation between the removed object '{}' and the existing object: {}. The model should insist no.".format(
                    target_object_name, existing_obj))

        if debug:
            case_result["target_obj_spatial_correct_remove"].append(list())
            case_result["target_obj_spatial_response_conflict"].append(list())

            # We assume there are three possible cases: Removed obj not exist (Right), Exist but on wrong position
            # (Wrong), Exist but on right position (Wrong)
            if three_branch_analysis:
                case_result["target_obj_spatial_correct_remove_adv"].append(list())
                case_result["target_obj_spatial_response_conflict_adv"].append(list())

        # Loop over all detected spatial relations in the ground truth to construct questions

        if len(spatial_relation_curr) == 0:
            logger.warning(
                "[ATTACK] No feasible spatial relation detected between the removed object '{}' and the existing object: {}.".format(
                    target_object_name, existing_obj))

            if notebook_print:
                print(
                    "[ATTACK] No feasible spatial relation detected between the removed object '{}' and the existing object: {}.".format(
                        target_object_name, existing_obj))

        for spatial_relation_curr_single in spatial_relation_curr:
            # spatial_flag = "uncertain"

            if spatial_relation_curr_single == "left":
                spatial_flag = "on the left of"
            elif spatial_relation_curr_single == "right":
                spatial_flag = "on the right of"
            elif spatial_relation_curr_single == "up":
                spatial_flag = "above"
            elif spatial_relation_curr_single == "down":
                spatial_flag = "below"
            elif spatial_relation_curr_single == "front":
                spatial_flag = "on the front of"

            question = "Is the {} {} a/an {} in this image, given their center positions?".format(
                target_object_name, spatial_flag, existing_obj)
            supplement = "(where {} is the one that {}.)".format(existing_obj, existing_obj_description)
            pred_answer = vqa_model_func(result_img_path, question + supplement)

            existing_info = "There is no {}. ".format(target_object_name)
            queried_info = "{} is not {} a/an {} in this image".format(target_object_name, spatial_flag, existing_obj)

            eval_result = evaluate_by_chatgpt_quick_test(question + supplement, pred_answer,
                                                         existing_info + queried_info + supplement)

            if three_branch_analysis:
                existing_info_adv = "There is a/an {}.".format(target_object_name)
                eval_result_adv = evaluate_by_chatgpt_quick_test(question + supplement, pred_answer,
                                                                 existing_info_adv + queried_info + supplement)

            logger.warning("[Obj] {}".format(target_object_name))
            logger.warning("[Q] {}".format(question + supplement))
            logger.warning("[Pred] {}".format(pred_answer))
            # logger.warning("[GT] {}".format(background_info + queried_info))
            logger.warning("[GT] {}".format(existing_info + queried_info + supplement))
            logger.warning("[Eval (same or not)] {}".format(eval_result))

            if notebook_print:
                print("[Obj] {}".format(target_object_name))
                print("[Q] {}".format(question + supplement))
                print("[Pred] {}".format(pred_answer))
                print("[GT] {}".format(existing_info + queried_info + supplement))
                print("[Eval (same or not)] {}".format(eval_result))

            if three_branch_analysis:
                logger.warning("[Eval advanced (same or not)] {}".format(eval_result_adv))

                if notebook_print:
                    print("[Eval advanced (same or not)] {}".format(eval_result_adv))

            if eval_result == "0":
                success = True

                if debug:
                    case_result["target_obj_spatial_correct_remove"][-1].append(str(True))

                    # True if existence wrong, spatial relation determination is correct
                    if three_branch_analysis:
                        if eval_result == "1":
                            case_result["target_obj_spatial_correct_remove_adv"][-1].append(str(True))
                        else:
                            case_result["target_obj_spatial_correct_remove_adv"][-1].append(str(False))

                logger.warning(
                    "[Attack Success] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) is not correct.".format(
                        existing_obj, str(spatial_distance_rank + 1)))

                if notebook_print:
                    print(
                        "[Attack Success] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) is not correct.".format(
                            existing_obj, str(spatial_distance_rank + 1)))
                
                return success, question, pred_answer, existing_info + queried_info + supplement

            else:
                if debug:
                    case_result["target_obj_spatial_correct_remove"][-1].append(str(False))
                logger.warning(
                    "[Attack Failed] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) is correct.".format(
                        existing_obj, str(spatial_distance_rank + 1)))

                if notebook_print:
                    print(
                        "[Attack Failed] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) is correct.".format(
                        existing_obj, str(spatial_distance_rank + 1)))

            question = "Is the object({}) {} a/an {} in this image, given their center positions?".format(
                target_object_description, spatial_flag, existing_obj)
            supplement = "(where {} is the one that {}.)".format(existing_obj, existing_obj_description)
            pred_answer = vqa_model_func(result_img_path, question + supplement)

            existing_info = "There is no {}. ".format(target_object_name)
            queried_info = "The object({}) is not {} a/an {} in this image".format(target_object_description,
                                                                                   spatial_flag, existing_obj)

            eval_result = evaluate_by_chatgpt_quick_test(question + supplement, pred_answer,
                                                         existing_info + queried_info + supplement)

            if three_branch_analysis:
                existing_info_adv = "There is a/an {}.".format(target_object_name)
                eval_result_adv = evaluate_by_chatgpt_quick_test(question + supplement, pred_answer,
                                                                 existing_info_adv + queried_info + supplement)

            logger.warning("[Obj] {}".format(target_object_description))
            logger.warning("[Q] {}".format(question + supplement))
            logger.warning("[Pred] {}".format(pred_answer))
            logger.warning("[GT] {}".format(existing_info + queried_info + supplement))
            logger.warning("[Eval (same or not)] {}".format(eval_result))

            if notebook_print:
                print("[Obj] {}".format(target_object_description))
                print("[Q] {}".format(question + supplement))
                print("[Pred] {}".format(pred_answer))
                print("[GT] {}".format(existing_info + queried_info + supplement))
                print("[Eval (same or not)] {}".format(eval_result))

            if three_branch_analysis:
                logger.warning("[Eval advanced (same or not)] {}".format(eval_result_adv))

                if notebook_print:
                    print("[Eval advanced (same or not)] {}".format(eval_result_adv))

            if eval_result == "0":
                success = True
                if debug:
                    case_result["target_obj_spatial_response_conflict"][-1].append(str(True))

                    # True if existence wrong, spatial relation determination is correct
                    if three_branch_analysis:
                        if eval_result == "1":
                            case_result["target_obj_spatial_response_conflict_adv"][-1].append(str(True))
                        else:
                            case_result["target_obj_spatial_response_conflict_adv"][-1].append(str(False))

                logger.warning(
                    "[Attack Success] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) has conflict.".format(
                        existing_obj, str(spatial_distance_rank + 1)))

                if notebook_print:
                    print(
                        "[Attack Success] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) has conflict.".format(
                            existing_obj, str(spatial_distance_rank + 1)))
                return success, question, pred_answer, existing_info + queried_info + supplement

            else:
                if debug:
                    case_result["target_obj_spatial_response_conflict"][-1].append(str(False))

                logger.warning(
                    "[Attack Failed] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) has no conflict.".format(
                        existing_obj, str(spatial_distance_rank + 1)))

                if notebook_print:
                    print(
                        "[Attack Failed] The spatial relation between the removed, non-existing object and the existing object {} (#{} closest) has no conflict.".format(
                            existing_obj, str(spatial_distance_rank + 1)))

        if debug:
            case_result["target_obj_spatial_correct_remove"][-1] = ",".join(
                case_result["target_obj_spatial_correct_remove"][-1])
            case_result["target_obj_spatial_response_conflict"][-1] = ",".join(
                case_result["target_obj_spatial_response_conflict"][-1])

            if three_branch_analysis:
                case_result["target_obj_spatial_correct_remove_adv"][-1] = ",".join(
                    case_result["target_obj_spatial_correct_remove_adv"][-1])
                case_result["target_obj_spatial_response_conflict_adv"][-1] = ",".join(
                    case_result["target_obj_spatial_response_conflict_adv"][-1])

    return success, None, None, None

# Evaluation function, only detect if the object exists in this given image
def evaluation_correlated_existence_pred_model(result_img_path, ground_truth, vqa_model_func, logger, debug=False, notebook_print=False):
    # background = ground_truth["background"]
    # scene_name = ground_truth["scene"]
    # obj_name = ground_truth["object_name"]
    # object_description = ground_truth["object_description"]
    irrelevant_object_name = ground_truth["irrelevant_object_name"]
    irrelevant_object_description = ground_truth["irrelevant_object_description"]
    disturbing_obj = ground_truth["disturbing_object"]
    result_caption = ground_truth["result_description"]

    success = False

    if debug:
        # case_result = [False, False, False]
        case_result = {"disturb_obj_not_detect_vanilla": False,
                       "disturb_obj_not_detect_correlated": False,
                       "disturb_obj_not_detect_caption": False,
                       "disturb_obj_response_conflict_vanilla_correlated": False,
                       "disturb_obj_response_conflict_vanilla_caption": False,
                       "disturb_obj_response_conflict_correlated_caption": False}

    ########## Correlated Object Existence Detection: Non-existing object only ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning(
        "[ATTACK] Asking about non-existing, correlated object '{}' only -- existence".format(
            disturbing_obj))

    if notebook_print:
        print("######")
        print(
            "[ATTACK] Asking about non-existing, correlated object '{}' only -- existence".format(
                disturbing_obj))

    question = "Is there a/an {} in this image?".format(disturbing_obj)
    pred_answer_vanilla = vqa_model_func(result_img_path, question)

    queried_info = "There is no {}.".format(disturbing_obj)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_vanilla, queried_info)
    logger.warning("[Existing Obj] {}".format(irrelevant_object_name))
    logger.warning("[Correlated Obj] {}".format(disturbing_obj))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_vanilla))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Existing Obj] {}".format(irrelevant_object_name))
        print("[Correlated Obj] {}".format(disturbing_obj))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_vanilla))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["disturb_obj_not_detect_vanilla"] = True

        logger.warning("[Attack Success] Non-existing correlated object mistakenly detected.")
    else:
        logger.warning("[Attack Failed] Non-existing correlated object not detected.")

    ########## Correlated Object Existence Detection: Non-existing, correlated object given existing object ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning(
        "[ATTACK] Asking about non-existing, correlated object '{}' given existing object {} -- existence".format(
            disturbing_obj, irrelevant_object_name))

    if notebook_print:
        print("######")
        print(
            "[ATTACK] Asking about non-existing, correlated object '{}' given existing object {} -- existence".format(
                disturbing_obj, irrelevant_object_name))

    question = "We have {} in this image. Is there a/an {} in this image as well?".format(irrelevant_object_name,
                                                                                          disturbing_obj)
    pred_answer_correlated = vqa_model_func(result_img_path, question)

    queried_info = "There is no {}.".format(disturbing_obj)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_correlated, queried_info)
    logger.warning("[Existing Obj] {}".format(irrelevant_object_name))
    logger.warning("[Correlated Obj] {}".format(disturbing_obj))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_correlated))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Existing Obj] {}".format(irrelevant_object_name))
        print("[Correlated Obj] {}".format(disturbing_obj))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_correlated))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["disturb_obj_not_detect_correlated"] = True

        logger.warning("[Attack Success] Non-existing correlated object mistakenly detected given existing object.")

        if notebook_print:
            print("[Attack Success] Non-existing correlated object mistakenly detected given existing object.")

    else:
        logger.warning("[Attack Failed] Non-existing correlated object not detected given existing object.")

        if notebook_print:
            print("[Attack Failed] Non-existing correlated object not detected given existing object.")

    ########## Correlated Object Existence Detection: Non-existing, correlated object, with caption ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning(
        "[ATTACK] Asking about non-existing, correlated object '{}' with image-level caption -- existence".format(
            disturbing_obj, irrelevant_object_name))

    if notebook_print:
        print("######")
        print(
            "[ATTACK] Asking about non-existing, correlated object '{}' with image-level caption -- existence".format(
                disturbing_obj, irrelevant_object_name))

    question = "We have an image depicting {}. Is there a/an {} in this image as well?".format(
        result_caption, irrelevant_object_name, disturbing_obj)
    pred_answer_caption = vqa_model_func(result_img_path, question)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_caption, queried_info)
    logger.warning("[Existing Obj] {}".format(irrelevant_object_name))
    logger.warning("[Correlated Obj] {}".format(disturbing_obj))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_caption))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Existing Obj] {}".format(irrelevant_object_name))
        print("[Correlated Obj] {}".format(disturbing_obj))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_caption))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["disturb_obj_not_detect_caption"] = True

        logger.warning(
            "[Attack Success] Non-existing correlated object mistakenly detected, with image-level caption in the question.")

        if notebook_print:
            print(
                "[Attack Success] Non-existing correlated object mistakenly detected, with image-level caption in the question.")

    else:
        logger.warning(
            "[Attack Failed] Non-existing correlated object not detected, with image-level caption in the question.")

        if notebook_print:
            print(
                "[Attack Failed] Non-existing correlated object not detected, with image-level caption in the question.")

    ########## Correlated Object Existence Detection: Response conflict between vanilla and correlated questions ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(disturbing_obj),
                                                 pred_answer_vanilla,
                                                 pred_answer_correlated)

    if eval_result == "0":
        success = True

        if debug:
            case_result["disturb_obj_response_conflict_vanilla_correlated"] = True

        logger.warning(
            "[Attack Success] Non-existing correlated object detection inconsistent between vanilla and correlated questions.")

        if notebook_print:
            print(
                "[Attack Success] Non-existing correlated object detection inconsistent between vanilla and correlated questions.")

    else:
        logger.warning(
            "[Attack Failed] Non-existing correlated object detection consistent between vanilla and correlated questions")

        if notebook_print:
            print("[Attack Failed] Non-existing correlated object detection consistent between vanilla and correlated questions")

    ########## Correlated Object Existence Detection: Response conflict between vanilla and caption questions ###########
    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(disturbing_obj),
                                                 pred_answer_vanilla,
                                                 pred_answer_caption)

    if eval_result == "0":
        success = True

        if debug:
            case_result["disturb_obj_response_conflict_vanilla_caption"] = True

        logger.warning(
            "[Attack Success] Non-existing correlated object detection inconsistent between vanilla and caption questions.")

        if notebook_print:
            print("[Attack Success] Non-existing correlated object detection inconsistent between vanilla and caption questions.")

    else:
        logger.warning(
            "[Attack Failed] Non-existing correlated object detection consistent between vanilla and caption questions")

        if notebook_print:
            print(
                "[Attack Failed] Non-existing correlated object detection consistent between vanilla and caption questions")

    ########## Correlated Object Existence Detection: Response conflict between vanilla and caption questions ###########
    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(disturbing_obj),
                                                 pred_answer_correlated,
                                                 pred_answer_caption)

    if eval_result == "0":
        success = True

        if debug:
            case_result["disturb_obj_response_conflict_correlated_caption"] = True

        logger.warning(
            "[Attack Success] Non-existing correlated object detection inconsistent between correlated and caption questions.")

        if notebook_print:
            print(
                "[Attack Success] Non-existing correlated object detection inconsistent between correlated and caption questions.")

    else:
        logger.warning(
            "[Attack Failed] Non-existing correlated object detection consistent between correlated and caption questions")

        if notebook_print:
            print(
                "[Attack Failed] Non-existing correlated object detection consistent between correlated and caption questions")

    if success:
        if debug:
            return "Correlated Object Existence Attack Success", case_result
        else:
            return "Correlated Object Existence Attack Success"
    else:
        if debug:
            return "Correlated Object Existence Attack Failed", case_result
        else:
            return "Correlated Object Existence Attack Failed"



# Evaluation function, only detect if the object exists in this given image
def evaluation_correlated_existence_pred_model_simple(result_img_path, ground_truth, vqa_model_func, logger, debug=False, notebook_print=False):
    # background = ground_truth["background"]
    # scene_name = ground_truth["scene"]
    # obj_name = ground_truth["object_name"]
    # object_description = ground_truth["object_description"]
    irrelevant_object_name = ground_truth["irrelevant_object_name"]
    irrelevant_object_description = ground_truth["irrelevant_object_description"]
    disturbing_obj = ground_truth["disturbing_object"]
    result_caption = ground_truth["result_description"]

    success = False

    if debug:
        # case_result = [False, False, False]
        case_result = {"disturb_obj_not_detect_vanilla": False,
                       "disturb_obj_not_detect_correlated": False,
                       "disturb_obj_not_detect_caption": False,
                       "disturb_obj_response_conflict_vanilla_correlated": False,
                       "disturb_obj_response_conflict_vanilla_caption": False,
                       "disturb_obj_response_conflict_correlated_caption": False}

    ########## Correlated Object Existence Detection: Non-existing object only ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning(
        "[ATTACK] Asking about non-existing, correlated object '{}' only -- existence".format(
            disturbing_obj))

    if notebook_print:
        print("######")
        print(
            "[ATTACK] Asking about non-existing, correlated object '{}' only -- existence".format(
                disturbing_obj))

    question = "Is there a/an {} in this image?".format(disturbing_obj)
    pred_answer_vanilla = vqa_model_func(result_img_path, question)

    queried_info = "There is no {}.".format(disturbing_obj)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_vanilla, queried_info)
    logger.warning("[Existing Obj] {}".format(irrelevant_object_name))
    logger.warning("[Correlated Obj] {}".format(disturbing_obj))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_vanilla))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Existing Obj] {}".format(irrelevant_object_name))
        print("[Correlated Obj] {}".format(disturbing_obj))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_vanilla))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["disturb_obj_not_detect_vanilla"] = True

        logger.warning("[Attack Success] Non-existing correlated object mistakenly detected.")

        return success, question, pred_answer_vanilla, queried_info

    else:
        logger.warning("[Attack Failed] Non-existing correlated object not detected.")

    ########## Correlated Object Existence Detection: Non-existing, correlated object given existing object ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning(
        "[ATTACK] Asking about non-existing, correlated object '{}' given existing object {} -- existence".format(
            disturbing_obj, irrelevant_object_name))

    if notebook_print:
        print("######")
        print(
            "[ATTACK] Asking about non-existing, correlated object '{}' given existing object {} -- existence".format(
                disturbing_obj, irrelevant_object_name))

    question = "We have {} in this image. Is there a/an {} in this image as well?".format(irrelevant_object_name,
                                                                                          disturbing_obj)
    pred_answer_correlated = vqa_model_func(result_img_path, question)

    queried_info = "There is no {}.".format(disturbing_obj)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_correlated, queried_info)
    logger.warning("[Existing Obj] {}".format(irrelevant_object_name))
    logger.warning("[Correlated Obj] {}".format(disturbing_obj))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_correlated))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Existing Obj] {}".format(irrelevant_object_name))
        print("[Correlated Obj] {}".format(disturbing_obj))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_correlated))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["disturb_obj_not_detect_correlated"] = True

        logger.warning("[Attack Success] Non-existing correlated object mistakenly detected given existing object.")

        if notebook_print:
            print("[Attack Success] Non-existing correlated object mistakenly detected given existing object.")

        return success, question, pred_answer_correlated, queried_info

    else:
        logger.warning("[Attack Failed] Non-existing correlated object not detected given existing object.")

        if notebook_print:
            print("[Attack Failed] Non-existing correlated object not detected given existing object.")

    ########## Correlated Object Existence Detection: Non-existing, correlated object, with caption ###########

    # Determine if the added adversarial object exists
    logger.warning("######")
    logger.warning(
        "[ATTACK] Asking about non-existing, correlated object '{}' with image-level caption -- existence".format(
            disturbing_obj, irrelevant_object_name))

    if notebook_print:
        print("######")
        print(
            "[ATTACK] Asking about non-existing, correlated object '{}' with image-level caption -- existence".format(
                disturbing_obj, irrelevant_object_name))

    question = "We have an image depicting {}. Is there a/an {} in this image as well?".format(
        result_caption, irrelevant_object_name, disturbing_obj)
    pred_answer_caption = vqa_model_func(result_img_path, question)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer_caption, queried_info)
    logger.warning("[Existing Obj] {}".format(irrelevant_object_name))
    logger.warning("[Correlated Obj] {}".format(disturbing_obj))
    logger.warning("[Q] {}".format(question))
    logger.warning("[Pred] {}".format(pred_answer_caption))
    logger.warning("[GT] {}".format(queried_info))
    logger.warning("[Eval (same or not)] {}".format(eval_result))

    if notebook_print:
        print("[Existing Obj] {}".format(irrelevant_object_name))
        print("[Correlated Obj] {}".format(disturbing_obj))
        print("[Q] {}".format(question))
        print("[Pred] {}".format(pred_answer_caption))
        print("[GT] {}".format(queried_info))
        print("[Eval (same or not)] {}".format(eval_result))

    if eval_result == "0":
        success = True

        if debug:
            case_result["disturb_obj_not_detect_caption"] = True

        logger.warning(
            "[Attack Success] Non-existing correlated object mistakenly detected, with image-level caption in the question.")

        if notebook_print:
            print(
                "[Attack Success] Non-existing correlated object mistakenly detected, with image-level caption in the question.")

        return success, question, pred_answer_caption, queried_info

    else:
        logger.warning(
            "[Attack Failed] Non-existing correlated object not detected, with image-level caption in the question.")

        if notebook_print:
            print(
                "[Attack Failed] Non-existing correlated object not detected, with image-level caption in the question.")

    ########## Correlated Object Existence Detection: Response conflict between vanilla and correlated questions ###########

    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(disturbing_obj),
                                                 pred_answer_vanilla,
                                                 pred_answer_correlated)

    if eval_result == "0":
        success = True

        if debug:
            case_result["disturb_obj_response_conflict_vanilla_correlated"] = True

        logger.warning(
            "[Attack Success] Non-existing correlated object detection inconsistent between vanilla and correlated questions.")

        if notebook_print:
            print(
                "[Attack Success] Non-existing correlated object detection inconsistent between vanilla and correlated questions.")

        return success, "Is there a/an {} in this image?".format(disturbing_obj), pred_answer_vanilla, queried_info

    else:
        logger.warning(
            "[Attack Failed] Non-existing correlated object detection consistent between vanilla and correlated questions")

        if notebook_print:
            print("[Attack Failed] Non-existing correlated object detection consistent between vanilla and correlated questions")

    ########## Correlated Object Existence Detection: Response conflict between vanilla and caption questions ###########
    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(disturbing_obj),
                                                 pred_answer_vanilla,
                                                 pred_answer_caption)

    if eval_result == "0":
        success = True

        if debug:
            case_result["disturb_obj_response_conflict_vanilla_caption"] = True

        logger.warning(
            "[Attack Success] Non-existing correlated object detection inconsistent between vanilla and caption questions.")

        if notebook_print:
            print("[Attack Success] Non-existing correlated object detection inconsistent between vanilla and caption questions.")

        return success, "Is there a/an {} in this image?".format(disturbing_obj), pred_answer_vanilla, queried_info

    else:
        logger.warning(
            "[Attack Failed] Non-existing correlated object detection consistent between vanilla and caption questions")

        if notebook_print:
            print(
                "[Attack Failed] Non-existing correlated object detection consistent between vanilla and caption questions")

    ########## Correlated Object Existence Detection: Response conflict between correlated and caption questions ###########
    eval_result = evaluate_by_chatgpt_quick_test("Is there a/an {} in this image?".format(disturbing_obj),
                                                 pred_answer_correlated,
                                                 pred_answer_caption)

    if eval_result == "0":
        success = True

        if debug:
            case_result["disturb_obj_response_conflict_correlated_caption"] = True

        logger.warning(
            "[Attack Success] Non-existing correlated object detection inconsistent between correlated and caption questions.")

        if notebook_print:
            print(
                "[Attack Success] Non-existing correlated object detection inconsistent between correlated and caption questions.")

        return success, "Is there a/an {} in this image?".format(disturbing_obj), pred_answer_correlated, queried_info

    else:
        logger.warning(
            "[Attack Failed] Non-existing correlated object detection consistent between correlated and caption questions")

        if notebook_print:
            print(
                "[Attack Failed] Non-existing correlated object detection consistent between correlated and caption questions")

    return success, None, None, None
