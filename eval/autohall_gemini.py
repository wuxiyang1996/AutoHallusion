import google.generativeai as genai
# from google.colab import userdata

from IPython.display import display
from IPython.display import Markdown
from PIL import Image
from tqdm import tqdm

import json
import os
import time

GOOGLE_API_KEY="Your Key"

genai.configure(api_key=GOOGLE_API_KEY)

for m in genai.list_models():
    print(m.name)
    print(m.supported_generation_methods)


model = genai.GenerativeModel('gemini-1.5-flash-latest')


root = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data"
json_in = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data/autohallusion_data.json"
json_out = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data/autohallusion_data_gemini_res.json"
json_tmp = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data/autohallusion_data_gemini_res_resume.json"

if os.path.exists(json_tmp):
    json_in = json_tmp


with open(json_in, 'r') as f:
    data = json.load(f)
# print(len(data))
count = 0
attempt = 10

for i, element in enumerate(tqdm(data)):
    # print(i)
    if "res" in element and element['res'] != 'null':
        continue
    attm_count = 0
    while attm_count < attempt:
        try:

            question = data[i]['prompt']
            # print(question)

            if "image_urls" not in element:
                response = model.generate_content([question])
                result = response.text

                element['res'] = result
            else:
                img_path = os.path.join(root, element["image_urls"][0])
                # img_path = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data/" + element['image_urls'][0]
                raw_image = Image.open(img_path)

                response = model.generate_content([question, raw_image])
                result = response.text

                element['res'] = result

            # print(result)
            count += 1
            break
        except:
            print("Timeout, Retrying...")
            print(response.prompt_feedback)
            time.sleep(5)
            attm_count += 1
            # if attm_count >= attempt:
            element['res'] = 'null'
            # try:
            #     question = data[i]['prompt']
            #     response = model.generate_content([question])
            #     result = response.text

            #     element['res'] = result
            #     print(result)
            # except:
            #     element['res'] = 'null'
            #     print('null')
            #     count += 1
    if i % 10 == 0:
        save_dict_tmp = json.dumps(data, indent=4)
        with open(json_tmp, "w") as out_f:
            out_f.write(save_dict_tmp)
            print("Progress saved.")
print(len(data) - count)

    # print(i)

output_json = json.dumps(data, indent=4)

with open(json_out, 'w') as f:
    f.write(output_json)
