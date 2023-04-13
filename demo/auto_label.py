from train_reward import RewardModel
import json
import os
from PIL import Image
from tqdm import tqdm


def label_directory(labeler,subdirectory):
    directory = os.path.join("gen_data/train/",str(subdirectory))

    metafiles = open(os.path.join(directory,"metadata.jsonl"),"r").readlines()
    
    if os.path.exists(os.path.join(directory,"metadata_rate.jsonl")):
        manual_files = open(os.path.join(directory,"metadata_rate.jsonl"),"r").readlines()
    else:
        manual_files = None

    with open(os.path.join(directory,"metadata_reward.jsonl"),"w") as fp:
        lines = []
        for id, metafile in tqdm(enumerate(metafiles)):
            metafile = json.loads(metafile)
            file_name = metafile["file_name"]
            img = Image.open(os.path.join(directory,file_name)).convert("RGB")
            prompt = [metafile["text"]]
            new_rate = labeler(img,prompt).detach().cpu().item()
            metafile["rate"] = new_rate
            
            if manual_files is not None:
                manual_file = json.loads(manual_files[id])
                original_rate = manual_file["rate"]
                print(f"{prompt}, original_rate: {original_rate:.3f}, new_rate: {new_rate:.3f}")
            metafile = json.dumps(metafile)
            lines.append(metafile)
        fp.write("\n".join(lines))    

if __name__ == "__main__":
    
    labeler = RewardModel.load_from_checkpoint("reward_model.pth")
    labeler.to("cuda")
    label_directory(labeler, "0")
    label_directory(labeler, "1")

