import open_clip
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from gen_prompt import get_perturbated_prompts
import os
from tqdm import tqdm
import json
from PIL import Image
from dataset import DummyDataset

class RewardModel(pl.LightningModule):
    def __init__(self, arch = "ViT-H-14", version = "laion2b_s32b_b79k"):
        super().__init__()
        self.clip, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(arch, pretrained=version)
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.mlp = nn.Sequential(nn.Linear(2048,1024),
                                 nn.ReLU(),
                                 nn.Linear(1024,1),
                                 nn.Sigmoid())
        

    def forward(self, image, text):
        image = self.preprocess_val(image)
        text = self.tokenizer(text)  
        image = image.to("cuda")
        text = text.to("cuda")
        image = image.unsqueeze(0)
        with torch.no_grad():
            image_features = self.clip.encode_image(image)
            text_features = self.clip.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            features = torch.concat([text_features,image_features],dim=-1)
            score = self.mlp(features)   
            return score    
        
    def training_step(self, batch, batch_idx):
        images, texts, rates = batch["image"], batch["prompt"], batch["rate"]
        images = images.to("cuda")
        rates = rates.to("cuda")

        image_features = self.clip.encode_image(images)
        texts_tokenized = self.tokenizer(texts)
        texts_tokenized = texts_tokenized.to("cuda")
        text_features = self.clip.encode_text(texts_tokenized)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        features = torch.concat([text_features,image_features],dim=-1)
        pred_rates = self.mlp(features)
        loss_mse = F.mse_loss(pred_rates,rates.unsqueeze(1))
            
        loss = loss_mse

        
        self.log("train_loss_mse",loss_mse)
        self.log("train_loss_clip",loss_clip)
        self.log("train_loss", loss)
        return loss

            
    def configure_optimizers(self):
        optimizer = optim.Adam(self.mlp.parameters(), lr=5e-4)
        return optimizer



if __name__ == "__main__":
    reward_model = RewardModel()
    train_transforms = reward_model.preprocess_train
    tokenize_captions = reward_model.tokenizer

    train_dataset = DummyDataset("gen_data/train/0/","metadata_rate.jsonl",train_transforms)

    batch_size = 1
    dataloader_num_workers = 4
    N = 10
    lamda = 0.5

    seed_everything(0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=dataloader_num_workers,
    )

    trainer = pl.Trainer(accelerator="gpu",strategy="ddp",devices=4,max_epochs=5,enable_checkpointing=False)
    trainer.fit(model=reward_model, train_dataloaders=train_loader)
    trainer.save_checkpoint("reward_model.pth")
    
