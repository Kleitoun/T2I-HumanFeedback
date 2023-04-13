import os
import json

from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything
import torch

def gen_data():
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    prompts = open("gen_prompts.txt","r").readlines()
    n_samples = 3
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to(device)

    for seed in [0,1]:
        _gen_data(pipe,prompts,n_samples,seed)

def _gen_data(pipe, prompts, n_samples, seed):
    seed_everything(seed)
    directory_name = f"gen_data/train/{seed}/"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    if seed == 0:
        n_imgs = 0
    else:
        n_imgs = n_samples * len(prompts)
    for prompt in prompts:
        prompt = prompt.strip()
        for _ in range(n_samples):
            img = pipe(prompt).images[0]
            img.save(directory_name + f"{n_imgs}.png")
            meta_data = {"file_name": f"{n_imgs}.png", "text": prompt}
            meta_data = json.dumps(meta_data)

            with open(directory_name+"metadata.jsonl","a") as f:
                f.write(meta_data+"\n")
            n_imgs+=1

if __name__ == "__main__":
    gen_data()