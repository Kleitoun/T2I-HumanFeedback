import json
import os
from PIL import Image
import matplotlib.pyplot as plt


plt.ion()
meta_data_set = open("gen_data/train/0/metadata.jsonl").readlines()
for meta_data in meta_data_set:
    meta_data = json.loads(meta_data)
    meta_data_name = meta_data["file_name"]
    meta_data_text = meta_data["text"]
    img = Image.open(os.path.join("gen_data/train/0",meta_data_name))
    plt.clf()
    plt.imshow(img)
    plt.axis('off') 
    plt.title(meta_data_text) 
    plt.show()
    rate = input()
    if not rate or rate == "0":
        rate = 0.0
    else:
        rate = 1.0
    meta_data["rate"] = rate
    with open("gen_data/train/0/metadata_rate.jsonl","a") as fp:
        fp.write(json.dumps(meta_data)+"\n")
    
