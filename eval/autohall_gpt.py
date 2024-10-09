import os
import openai
import base64
import httpx
import json
import tqdm
import time
from tqdm import tqdm 
import requests
from openai import OpenAI


api_key = "Your Key"

openai.api_key = api_key

client = OpenAI(
    api_key = api_key
)

models = client.models.list()

for model in models.data:
    # if model.id.startswith("gpt-"):
    print(model.id)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def qa_gpt4v(question, temp=0.5):
    client = OpenAI(api_key=api_key)

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

def vqa_gpt4v(masked_image_path, questions, temp=0.5):
    base64_image = encode_image(masked_image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
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

root = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data"
json_in = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data/autohallusion_data.json"
json_out = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data/autohallusion_data_gpt_res.json"
json_tmp = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data/autohallusion_data_gpt_res_resume.json"

if os.path.exists(json_tmp):
    json_in = json_tmp

with open(json_in, "r") as f:
    data = json.load(f)
    
for i, element in enumerate(tqdm(data)):
    # from IPython import embed;embed()
    if "res" in element:
        continue
    while True:
        # from IPython import embed;embed()
        try:
            
            prompt = element["prompt"]
            
            if "image_urls" not in element:
                
                answer = qa_gpt4v(prompt)

            else:
                # from IPython import embed;embed()
                # image exceeds 5 MB maximum: 5334124 bytes > 5242880 bytes

                answer = vqa_gpt4v(os.path.join(root, element["image_urls"][0]), prompt)
                        
            element["res"] = answer
            
            
            break
        except:
            print("Timeout, Retrying...")
            time.sleep(5)
    
    if i % 10 == 0:
        save_dict_tmp = json.dumps(data, indent=4)
        with open(json_tmp, "w") as out_f:
            out_f.write(save_dict_tmp)
            print("Progress saved.")
        
save_dict = json.dumps(data, indent=4)

with open(json_out, "w") as out_f:
    out_f.write(save_dict)


