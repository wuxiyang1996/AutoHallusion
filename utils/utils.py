from openai import OpenAI
from api_key import OPENAI_API_KEY
import requests
import os
import cv2
from PIL import Image
import numpy as np
import base64
from diffusers import StableDiffusionInpaintPipeline

import torch
import time

from rembg import remove
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

################# GPT-4V for image generation prompt polish and evalution ####################
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

# Think about a scenario
# Input: Masked image path
# Output: Object name
def scene_thinking(constraint, temperature):
    msg = "Randomly think about a generic scene or place that can use a noun or phrase to describe. Only generate a single word or a short phrase."
    # msg = "Randomly think about a generic scene or place that can use a one or two words to describe."
    if constraint is not None:
        msg += "Constraint: "
        msg += constraint
    scene_name = qa_gpt4v(msg, temperature)

    temp = list([val for val in scene_name if val.isalpha() or val.isnumeric() or val == " "])
    scene_name = "".join(temp).lower()
    return scene_name

# Object_detection function, using Owl-ViT, text input needed.
# Output: Mask image, Bounding Box
def object_detection(img_path, text_input, processor, model, mask_img=None, mask_bbox=None, box_per_item=1, nms=False,
                     save_prefix="./"):
    raw_img = Image.open(img_path)
    img = raw_img.convert("RGB")
    texts = [text_input]
    inputs = processor(text=texts, images=img, return_tensors="pt")
    outputs = model(**inputs)

    H, W, C = np.array(img).shape

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([img.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries

    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    if nms:
        # if True:
        pick = non_max_suppression(boxes, scores)
        boxes = boxes[pick]
        scores = scores[pick]
        labels = labels[pick]

    if mask_img is None:
        mask_img = []
        mask_bbox = []

    keep_idx = torch.argsort(scores.detach(), descending=True)
    # from IPython import embed;embed()
    if len(labels) < box_per_item:
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]
    else:
        boxes = boxes[keep_idx[:box_per_item]]
        scores = scores[keep_idx[:box_per_item]]
        labels = labels[keep_idx[:box_per_item]]

    start_idx = len(mask_img)

    for i in range(len(labels)):
        box, score, label = boxes[i], scores[i], labels[i]
        box = [max(round(i), 0) for i in box.tolist()]

        box[0], box[2] = min(box[0], W), min(box[2], W)
        box[1], box[3] = min(box[1], H), min(box[3], H)

        mosaic_img = np.zeros((H, W))

        mosaic_img[box[1]:box[3], box[0]:box[2]] = 1
        masked_img = np.array(raw_img) * np.array(mosaic_img).reshape(H, W, 1)
        masked_img = Image.fromarray(masked_img.astype(np.uint8)).convert('RGBA')
        masked_img.save(os.path.join(save_prefix, "new_mask_" + str(start_idx + i) + ".png"))

        mask_img.append(masked_img)
        mask_bbox.append(box)

    return mask_img, mask_bbox

def non_max_suppression(boxes, scores, overlapThresh=0.5):
    # if there are no boxes, return an empty list
    boxes = np.array(boxes.detach())
    scores = np.array(scores.detach())
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0] - (boxes[:, 2] / 2)
    y1 = boxes[:, 1] - (boxes[:, 3] / 2)
    x2 = boxes[:, 0] + (boxes[:, 2] / 2)
    y2 = boxes[:, 1] + (boxes[:, 3] / 2)

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap_area = w * h

        # compute the ratio of overlap
        overlap = overlap_area / (area[i] + area[idxs[:last]] - overlap_area)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes and scores that were picked
    return pick

# Generate the relation in 5 cases: Up/Down/Left/Right/Front
# Up/Down/Left/Right:
# Center-Center distance on the horizontal/vertical - threshold * width/height of the reference box
# Front:
# Added object box smaller than the reference box: Center locates within the reference box region
# Added object box larger than the reference box: Skip
# Note: x is horizontal, y is vertical, in numpy array, the index is [y, x]
# All bounding boxes follow the form xyxy
# Return the relations detected and the center-Center distance
def relation_determine(target_bbox, ref_boox, thres=0):
    target_center_x, target_center_y = round((target_bbox[0] + target_bbox[2]) / 2), round(
        (target_bbox[1] + target_bbox[3]) / 2)
    target_size_x, target_size_y = abs(target_bbox[2] - target_bbox[0]), abs(target_bbox[3] - target_bbox[1])

    ref_center_x, ref_center_y = round((ref_boox[0] + ref_boox[2]) / 2), round((ref_boox[1] + ref_boox[3]) / 2)
    ref_size_x, ref_size_y = abs(ref_boox[2] - ref_boox[0]), abs(ref_boox[3] - ref_boox[1])

    # Center-Center distance
    distance = np.sqrt(pow(target_center_x - ref_center_x, 2) + pow(target_center_y - ref_center_y, 2))

    relation_all = []
    # Spatial Relation Determination Logic
    # Determine if on the left
    if (ref_center_x - target_center_x) > (target_size_x + ref_size_x) / 2 - thres * ref_size_x / 2:
        relation_all.append("left")

    # Determine if on the right
    if (target_center_x - ref_center_x) > (target_size_x + ref_size_x) / 2 - thres * ref_size_x / 2:
        relation_all.append("right")

    # Determine if on the upper side
    if (ref_center_y - target_center_y) > (target_size_y + ref_size_y) / 2 - thres * ref_size_y / 2:
        relation_all.append("up")

    # Determine if on the lower side
    if (target_center_y - ref_center_y) > (target_size_y + ref_size_y) / 2 - thres * ref_size_y / 2:
        relation_all.append("down")

    # Determine if on the front
    if target_size_x <= ref_size_x or target_size_y <= ref_size_y:
        if target_center_x >= ref_boox[0] and target_center_x <= ref_boox[2] \
                and target_center_y >= ref_boox[1] and target_center_y <= ref_boox[3]:
            relation_all.append("front")

    return relation_all, distance

# Generate all the spatial relations for all detected objects and the added object
# Now only consider the single added object case
def spatial_gt_generation(ground_truth, target_bbox, mask_boox, enable=True):
    if not enable:
        return ground_truth

    spatial_relation_all, spatial_distance = [], []
    for i in range(len(mask_boox)):
        relation_all, distance = relation_determine(target_bbox, mask_boox[i])
        spatial_relation_all.append(relation_all.copy())
        spatial_distance.append(distance)

    ground_truth["spatial_relation"] = spatial_relation_all
    ground_truth["spatial_distance"] = spatial_distance
    return ground_truth

def resize_img_n_store(input_img):
    img = Image.open(input_img)
    img.resize((1024, 1024)).save(input_img)

# Create the binary mask image for object removal
def remove_obj_mask(img_path, obj_mask_img_path, obj_bbox, enlarge_ratio=0.2):
    img = Image.open(img_path).convert('RGBA')
    H, W, C = np.array(img).shape

    mask_array = np.array(img)

    for i in range(len(obj_bbox)):
        bbox = obj_bbox[i]

        bbox_center_x = round((bbox[0] + bbox[2]) / 2)
        bbox_size_x = bbox[2] - bbox[0]
        bbox_center_y = round((bbox[1] + bbox[3]) / 2)
        bbox_size_y = bbox[3] - bbox[1]

        bbox_center_x_min = max(0, int(bbox_center_x - bbox_size_x * (1 + enlarge_ratio) / 2))
        bbox_center_x_max = min(W, int(bbox_center_x + bbox_size_x * (1 + enlarge_ratio) / 2))

        bbox_center_y_min = max(0, int(bbox_center_y - bbox_size_y * (1 + enlarge_ratio) / 2))
        bbox_center_y_max = min(H, int(bbox_center_y + bbox_size_y * (1 + enlarge_ratio) / 2))

        mask_array[bbox_center_y_min:bbox_center_y_max, bbox_center_x_min:bbox_center_x_max, :] = 0

    mask_img = Image.fromarray(mask_array)
    mask_img.save(obj_mask_img_path)

def generate_image_edit(input_image_name, input_image_mask, scene_desc, out_image_name):
    client = OpenAI(api_key=OPENAI_API_KEY)

    # or use
    # image = Image.open("image.png")
    # width, height = 256, 256
    # image = image.resize((width, height))

    # # Convert the image to a BytesIO object
    # byte_stream = BytesIO()
    # image.save(byte_stream, format='PNG')
    # byte_array = byte_stream.getvalue()
    ## image=byte_array,

    generated = False

    while not generated:
        try:

            response = client.images.edit(
                model="dall-e-2",
                image=open(input_image_name, "rb"),
                mask=open(input_image_mask, "rb"),
                prompt=scene_desc,
                n=1,
                size="1024x1024"
            )

            image_url = response.data[0].url

            url_parts = image_url.split('?')
            file_name = url_parts[0].split('/')[-1]

            print('file_name : ', file_name)

            ### Download the image
            response = requests.get(image_url)
            if response.status_code == 200:
                with open(out_image_name, "wb") as f:
                    f.write(response.content)
                # print("The image was downloaded and saved.")
                generated = True
            else:
                print("Image download failed.")

        except:
            print("regenerate generate_image_edit...")
            time.sleep(5)

    image = Image.open(out_image_name)

    return image

def prompt_engineering_dalle3(input_text):
    prefix = "What prompt would you give to dalle 3, if you are asked to generate an image based on this requirement:"

    suffix = "Please only generate the prompt you would use without any suffix or prefix."

    prompt = prefix + input_text + suffix
    caption = qa_gpt4v(prompt)
    return caption

def verify_generation(image_path, question, gt):
    pred_answer = vqa_gpt4v(image_path, question)

    eval_result = evaluate_by_chatgpt_quick_test(question, pred_answer, gt)

    if eval_result == "0" or eval_result == "2":
        return False
    else:
        return True

# https://platform.openai.com/docs/guides/images/usage
def generate_image(word_lst, image_name=None, scene_desc=None, advanced=True, verify=False, max_try=5):
    client = OpenAI(api_key=OPENAI_API_KEY)

    if isinstance(word_lst, str):
        words = word_lst
    elif isinstance(word_lst, list):
        words = ",".join(word_lst)
    else:
        words = word_lst

    if scene_desc is None:
        scene_desc = '''
        Generate a scene with those word/objects: "{}". This scene should have other elements as well to provide a strong context for those obejcts.
        '''.format(words)

    generated = False
    counter = 0
    while not generated:
        if advanced:
            scene_desc_new = prompt_engineering_dalle3(scene_desc)
        else:
            scene_desc_new = scene_desc
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=scene_desc_new,
                size="1024x1024",
                quality="hd",
                # quality="standard",
                n=1,
                # style="vivid",
                style="natural",
            )

            image_url = response.data[0].url

            url_parts = image_url.split('?')
            file_name = url_parts[0].split('/')[-1]

            # print('file_name : ', file_name)
            if image_name is None:
                image_name = "./" + file_name

            ### Download the image
            response = requests.get(image_url)
            if response.status_code == 200:
                with open(image_name, "wb") as f:
                    f.write(response.content)
                # print("The image was downloaded and saved to {}.".format(image_name))
                generated = True
            else:
                print("Image download failed.")

            if verify:
                generated = verify_generation(image_name, "Is there {} in this image?".format(words),
                                              "There is {} in this image.".format(words))
                if not generated:
                    raise Exception("bad generate_image, regenerate...")
        except:
            counter += 1
            #print("regenerate generate_image, try {} / {}...".format(str(counter)), str(max_try))
            print("regenerate generate_image, try {} / {}...".format(str(counter), str(max_try))) # fix bug
            if counter > max_try:
                raise Exception("generate_image reaches max attempt of {}, regenerate case", format(str(max_try)))
            time.sleep(5)

    image = Image.open(image_name)

    return image

# https://platform.openai.com/docs/guides/images/usage
def generate_image_given_scene(word_lst, scene_name, image_name=None, scene_desc=None):
    client = OpenAI(api_key=OPENAI_API_KEY)

    words = ",".join(word_lst)

    if scene_desc is None:
        scene_desc = '''
        Generate a scene of {} with those word/objects: "{}". This scene should have other elements as well to provide a strong context for those obejcts.
        '''.format(scene_name, words)

    generated = False

    while not generated:
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=scene_desc,
                size="1024x1024",
                quality="hd",
                # quality="standard",
                n=1,
                # style="vivid",
                style="natural",
            )

            image_url = response.data[0].url

            url_parts = image_url.split('?')
            file_name = url_parts[0].split('/')[-1]

            # print('file_name : ', file_name)
            if image_name is None:
                image_name = "./" + file_name

            ### Download the image
            response = requests.get(image_url)
            if response.status_code == 200:
                with open(image_name, "wb") as f:
                    f.write(response.content)
                # print("The image was downloaded and saved to {}.".format(image_name))
                generated = True
            else:
                print("Image download failed.")

        except:
            print("regenerate generate_image_given_scene...")
            time.sleep(5)

    image = Image.open(image_name)

    return image

# Remove Operation, for VD
# Provide the initial, a mask image, and the noun for replacing
# We assume the input mask are to extract objects, so we need to exclude them
def Remove_Image_Operation_VD_multi_obj(init_img_path, masked_img_path, out_image_name):
    # masked_img_path = masking_image_create(init_img_path, obj_mask_img_path, path_prefix)

    new_caption = "Fill the masked region and make it harmonious given the background when preserving all other context in this image"

    image_removed = generate_image_edit(init_img_path, masked_img_path, new_caption, out_image_name)
    return image_removed

# Extract the single object from the scene with masks
def target_obj_extract(img_path, bbox, save_prefix):
    raw_img = Image.open(img_path)
    obj_img = np.array(raw_img)[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    obj_img = Image.fromarray(obj_img.astype(np.uint8)).convert('RGBA')
    obj_img.save(os.path.join(save_prefix, "target_obj.png"))
    return obj_img

def target_obj_decide(img_path, result_img_path, obj_list, processor, model, save_prefix, max_attempt=5):
    idx_order = np.random.choice(len(obj_list), len(obj_list), replace=False)

    removal_flag = False

    obj_mask_img_path = save_prefix + "temp_removal_mask.png"
    out_image_name = save_prefix + "temp_removal.png"

    for i in list(idx_order):
        target_obj = obj_list[i]

        init_img_path = img_path

        target_obj_img, target_obj_bbox = None, None

        # Try max_attempt times to remove all target objects within
        for num_max_attempt in range(max_attempt):
            text_input = "a photo of " + target_obj

            mask_img, mask_bbox = object_detection(init_img_path, text_input, processor, model, None, None,
                                                   box_per_item=5, save_prefix=save_prefix)

            if len(mask_bbox) == 0 and num_max_attempt == 0:
                break

            if len(mask_bbox) == 0:
                removal_flag = True
                break
            else:
                target_obj_img, target_obj_bbox = mask_img[0], mask_bbox[0]
                remove_obj_mask(init_img_path, obj_mask_img_path, mask_bbox, enlarge_ratio=0.2)
                Remove_Image_Operation_VD_multi_obj(init_img_path, obj_mask_img_path, out_image_name)
                init_img_path = out_image_name


        if removal_flag:
            break

    if target_obj_img is not None:
        target_obj_img.save(save_prefix + "target_obj.png")
        target_obj_extract(save_prefix + "target_obj.png", target_obj_bbox, save_prefix)
        final_img = Image.open(out_image_name)

        final_img.save(result_img_path)
    return target_obj, target_obj_bbox

def vanilla_scene_img_generation(img_path, scene_name, num_obj):
    prompt = "Generate a high-quaily, realistic image of {} with at least {} distinct objects and other necessary context.".format(
        scene_name, str(num_obj))
    img = generate_image("", img_path, prompt, advanced=False)
    return img

def Addition_Image_Operation_VD_stitch_correlation(init_img_path, existing_bbox, path_prefix, out_image_name,
                                                   add_object_size=(300, 300), overlapped_ratio=0.5, max_attempt=5,
                                                   scene_img_raw_size=None):
    img = Image.open(init_img_path).convert('RGBA')
    H, W, C = np.array(img).shape

    if scene_img_raw_size is not None:
        longer = max(scene_img_raw_size[0], scene_img_raw_size[1])

        H = int(1024 / longer * scene_img_raw_size[0])
        W = int(1024 / longer * scene_img_raw_size[1])


    obj_x, obj_y = add_object_size

    no_overlap_flag = False
    count = 1
    while not no_overlap_flag and count < max_attempt:
        mask_x, mask_y = np.random.randint(H - obj_x), np.random.randint(W - obj_y)

        new_bbox = [mask_x, mask_y, mask_x + obj_x, mask_y + obj_y]
        no_overlap_flag = overlap_detection(init_img_path, new_bbox, existing_bbox, overlapped_ratio)
        count += 1

    mask_img = np.zeros((H, W))

    mask_img[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]] = 255
    mask_img = Image.fromarray(mask_img.astype(np.uint8))
    mask_img.save(os.path.join(path_prefix, "mask_region_img.png"))

    input_path, output_path, obj_mask_path = os.path.join(path_prefix, "obj.png"), \
                                             os.path.join(path_prefix, "pure_obj.png"), \
                                             os.path.join(path_prefix, "mask_obj.png")

    img = Image.open(input_path).convert("RGB")
    img.save(input_path)

    if not os.path.exists(obj_mask_path):
        obj_mask_img = create_mask_from_png(output_path)
        obj_mask_img = Image.fromarray(obj_mask_img.astype(np.uint8))
        obj_mask_img.save(obj_mask_path)

    out_img = object_stitch(init_img_path, input_path, obj_mask_path, out_image_name, new_bbox)
    return out_img, new_bbox

# Remove the background of the given images
def background_removal(input_path, output_path):
    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)
    return output


# Create the mask image from PNG image
def create_mask_from_png(input_img):
    im = cv2.imread(input_img, cv2.IMREAD_UNCHANGED)
    ret, mask = cv2.threshold(im[:, :, 3], 0, 255, cv2.THRESH_BINARY)

    return mask

# Stitch the object image onto the given image
def object_stitch(init_img_path, obj_img_path, obj_mask_path, out_image_name, new_bbox):
    img_size = (new_bbox[3] - new_bbox[1], new_bbox[2] - new_bbox[0])

    init_img = Image.open(init_img_path)
    obj_img = Image.open(obj_img_path)
    obj_mask = Image.open(obj_mask_path)

    init_img = np.array(init_img)

    obj_img, obj_mask = crop_img(obj_img, obj_mask)

    obj_img = np.array(obj_img.resize(img_size))
    obj_mask = (np.array(obj_mask.resize(img_size)) > 0).astype(np.uint)

    img_ws = init_img[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2], :]

    img_ws = img_ws * (1 - obj_mask.reshape((img_size[0], img_size[1], 1))) + obj_img * obj_mask.reshape(
        (img_size[0], img_size[1], 1))

    init_img[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2], :] = img_ws
    out_img = Image.fromarray(init_img.astype(np.uint8)).convert("RGB")
    out_img.save(out_image_name)

    return out_img

def crop_img(obj_img, obj_mask):
    y_coords, x_coords = np.where(np.array(obj_mask) > 0)

    # Extract minimum and maximum x and y coordinates
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)

    if max_x - min_x > max_y - min_y:
        max_y = min_y + (max_x - min_x)
        if max_y >= obj_mask.size[0]:
            max_y = obj_mask.size[0] - 1
            min_y = max_y - (max_x - min_x)
    else:
        max_x = min_x + (max_y - min_y)
        if max_x >= obj_mask.size[1]:
            max_x = obj_mask.size[1] - 1
            min_x = max_x - (max_y - min_y)

    assert max_x - min_x == max_y - min_y
    # obj_img = obj_img[min_y: max_y, min_x: max_x]
    # obj_mask = obj_mask[min_y: max_y, min_x: max_x]

    obj_img = obj_img.crop((min_x, min_y, max_x, max_y))
    obj_mask = obj_mask.crop((min_x, min_y, max_x, max_y))

    return obj_img, obj_mask

# Detect if the given masked region has a large overlapped region with existing mask regions
# BBox follows the xyxy format, existing_bbox is a list
# If exceeds the allowed overlapped ratio, then reject
def overlap_detection(img_path, new_bbox, existing_bbox, overlapped_ratio):
    img = Image.open(img_path)
    H, W, C = np.array(img).shape

    masked_img = np.zeros((H, W))

    for i in range(len(existing_bbox)):
        box = existing_bbox[i]

        masked_img[box[1]:box[3], box[0]:box[2]] = 1

    existing_masked_number = len(np.where(masked_img == 1)[0])
    masked_img[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]] = 2

    proposed_masked_number = len(np.where(masked_img >= 1)[0])
    new_masked_number = len(np.where(masked_img == 2)[0])

    overlapped = existing_masked_number + new_masked_number - proposed_masked_number

    if overlapped > new_masked_number * overlapped_ratio:
        return False

    return True

def convert_into_sequare(img_name, output_name):
    raw_img = Image.open(img_name)
    img = np.array(raw_img)
    img_size = img.shape

    long_bound = max(img_size[0], img_size[1])

    if len(img_size) == 2:
        square_img = np.zeros((long_bound, long_bound))

        for i in range(img_size[0]):
            for j in range(img_size[1]):
                # if img[i, j] > 0: print(img[i, j])
                square_img[i, j] = img[i, j]
        out_img = Image.fromarray(square_img.astype(np.uint8))
        out_img.save(output_name)
    elif img_size[2] == 3:
        square_img = np.zeros((long_bound, long_bound, 4))
        square_img[:img_size[0], :img_size[1], :3] = img
        square_img[:img_size[0], :img_size[1], 3] = 255

        out_img = Image.fromarray(square_img.astype(np.uint8), 'RGBA')
        out_img.save(output_name)

    else:
        square_img = np.zeros((long_bound, long_bound, 4))
        square_img[:img_size[0], :img_size[1], :] = img

        out_img = Image.fromarray(square_img.astype(np.uint8), 'RGBA')
        out_img.save(output_name)

# Randomly think about an attribute, not used in the current EMNLP findings release
def attribute_thinking(category, temp=1):
    prompt = "Randomly think about a {}. The output must be less than 2 words".format(category)
    attribute = qa_gpt4v(prompt, temp=temp)
    return attribute

# Generate one irrelevant object based on the scene
def irrelevant_obj_generation(irrelevant_obj, attribute_category_list, obj_path):
    irrelevant_obj_attribute = {}

    for attribute_category in attribute_category_list:
        attribute_generated = attribute_thinking(attribute_category)
        irrelevant_obj_attribute[attribute_category] = attribute_generated

    attribute_string = ""
    for attribute_category in attribute_category_list:
        attribute_string += "a " + attribute_category + " of " + irrelevant_obj_attribute[attribute_category] + ", "

    if attribute_string != "":
        attribute_string = "with " + attribute_string

    scene_desc = '''
        Generate a high-quality image of a {} {} in a pure-color background that has a contrasting color from the object, so that this object is much more salient and obvious.
        This object should be realistic and should show the entire object, not just part of the object.
        '''.format(irrelevant_obj, attribute_string)

    obj_img = generate_image(irrelevant_obj, obj_path, scene_desc, verify=True)

    return irrelevant_obj_attribute

def Addition_Image_Operation_VD_stitch(init_img_path, existing_bbox, add_object, path_prefix, out_image_name,
                                       attribute_category_list, add_object_size=(300, 300), overlapped_ratio=0.5,
                                       max_attempt=5, scene_img_raw_size=None):
    img = Image.open(init_img_path).convert('RGBA')
    H, W, C = np.array(img).shape

    if scene_img_raw_size is not None:
        longer = max(H, W)

        H = int(1024 / longer * H)
        W = int(1024 / longer * W)

    obj_x, obj_y = add_object_size

    no_overlap_flag = False
    count = 1
    while not no_overlap_flag and count < max_attempt:
        mask_x, mask_y = np.random.randint(H - obj_x), np.random.randint(W - obj_y)

        new_bbox = [mask_x, mask_y, mask_x + obj_x, mask_y + obj_y]
        no_overlap_flag = overlap_detection(init_img_path, new_bbox, existing_bbox, overlapped_ratio)
        count += 1

    mask_img = np.zeros((H, W))

    mask_img[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]] = 255
    mask_img = Image.fromarray(mask_img.astype(np.uint8))
    mask_img.save(os.path.join(path_prefix, "mask_region_img.png"))

    input_path, output_path, obj_mask_path = os.path.join(path_prefix, "obj.png"), os.path.join(path_prefix,
                                                                                                "pure_obj.png"), os.path.join(
        path_prefix, "mask_obj.png")

    if not os.path.exists(input_path):
        # irrelevant_obj_attribute = irrelevant_obj_generation(add_object, attribute_category_list, input_path)
        irrelevant_obj_generation(add_object, attribute_category_list, input_path)

    if not os.path.exists(output_path):
        background_removal(input_path, output_path)

    if not os.path.exists(obj_mask_path):
        obj_mask_img = create_mask_from_png(output_path)
        obj_mask_img = Image.fromarray(obj_mask_img.astype(np.uint8))
        obj_mask_img.save(obj_mask_path)

    # out_img = object_stitch(init_img_path, output_path, obj_mask_path, out_image_name, new_bbox)
    out_img = object_stitch(init_img_path, input_path, obj_mask_path, out_image_name, new_bbox)
    # return out_img, new_bbox, irrelevant_obj_attribute
    return out_img, new_bbox, {}

# Generate the image with the correlated object pair
def correlated_img_generate(correlated_obj, attribute_category_list, output_path, temperature=1.5,
                            attribute_enable=False):
    correlated_obj_0, correlated_obj_1 = correlated_obj[0], correlated_obj[1]

    obj_1_attribute, obj_2_attribute = {}, {}

    attribute_string_1, attribute_string_2 = "", ""

    # Generate the object attributes
    if attribute_enable:
        for attribute_category in attribute_category_list:
            attribute_generated = attribute_thinking(attribute_category)
            obj_1_attribute[attribute_category] = attribute_generated

        for attribute_category in attribute_category_list:
            attribute_generated = attribute_thinking(attribute_category)
            obj_2_attribute[attribute_category] = attribute_generated

        for attribute_category in attribute_category_list:
            attribute_string_1 += "a " + attribute_category + " of " + obj_1_attribute[attribute_category] + ", "

        if attribute_string_1 != "":
            attribute_string_1 = "with " + attribute_string_1

        for attribute_category in attribute_category_list:
            attribute_string_2 += "a " + attribute_category + " of " + obj_1_attribute[attribute_category] + ", "

        if attribute_string_2 != "":
            attribute_string_2 = "with " + attribute_string_2

    prompt = '''
    Generate an realistic image of a {} {} without any {} {} in a pure-color background that has a contrasting color from the object.
    '''.format(correlated_obj_0, attribute_string_1, correlated_obj_1, attribute_string_2)

    # prompt = None

    generate_image(correlated_obj, output_path, prompt, advanced=False)
    return obj_1_attribute, obj_2_attribute

# Object_detection function, using Owl-ViT, text input needed.
# Output: Mask image, Bounding Box
def correlated_object_segment(img_path, target_obj, save_prefix="./"):
    raw_img = Image.open(img_path)
    masked_img = remove(raw_img)
    masked_img.save(os.path.join(save_prefix, "extracted_" + target_obj + ".png"))
    return masked_img

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
