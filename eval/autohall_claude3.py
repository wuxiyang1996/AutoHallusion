import os
import anthropic
import base64
import httpx
import json
import tqdm
import time
from tqdm import tqdm 

claude_key = "Your Key"

root = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data"
json_in = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data/autohallusion_data.json"
json_out = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data/autohallusion_data_claude_res.json"
json_tmp = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data/autohallusion_data_claude_res_resume.json"

if os.path.exists(json_tmp):
    json_in = json_tmp


client = anthropic.Anthropic()

for model in client.models.list():
    print(model.name)

attemp = 10

with open(json_in, "r") as f:
    data = json.load(f)
    
for i, element in enumerate(tqdm(data)):
    # from IPython import embed;embed()
    if "res" in element:
        continue
    while True:
        # from IPython import embed;embed()
        try:
            client = anthropic.Anthropic(
                api_key=claude_key,
                # api_key=os.environ["ANTHROPIC_API_KEY"],
            )
            
            prompt = element["prompt"]
            
            if "image_urls" not in element:
                
                message = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    timeout = 30
                )
                
            else:
                # from IPython import embed;embed()
                # image exceeds 5 MB maximum: 5334124 bytes > 5242880 bytes
                image = open(os.path.join(root, element["image_urls"][0]), "rb").read()
                image_encode = base64.b64encode(image).decode("utf-8")

                
                message = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_encode,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ],
                        }
                    ],
                    timeout = 30
                )
                        
            element["res"] = message.content[0].text
            
            
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