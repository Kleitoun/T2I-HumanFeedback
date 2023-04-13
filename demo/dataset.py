import torch
from PIL import Image
from torch.utils.data import Dataset
import json
import os
class DummyDataset(Dataset):
    def __init__(self, data_path, json_file, trsf, use_path=True):
        self.data_path = data_path
        self.json_file = json_file
        self.trsf = trsf
        self.use_path = use_path
        self.data = open(os.path.join(data_path,json_file)).readlines()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = json.loads(data)
        image = os.path.join(self.data_path,data["file_name"])
        prompt = data["text"].strip()
        if self.use_path:
            image = self.trsf(pil_loader(image))
        else:
            image = self.trsf(Image.fromarray(image))

        output = {"image":image, "prompt":prompt}

        if "rate" in data.keys():
            output.update({"rate":torch.tensor(data["rate"]).float()})
            if data["rate"] < 0.6:
                output["prompt"] += " weird image"
        return output

        


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
