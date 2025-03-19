# EXAMPLE USAGE:
# python run_autoencoder.py configs/default.yaml
import numpy as np
import numpy as np
import sys
import os
import yaml
import gc
import torch
import lightning as L

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb

from autoencoder import Autoencoder
from patchdataset import PatchDataset
from data import make_data

print("loading config file")
config_path = "default.yaml"
assert os.path.exists(config_path), f"Config file {config_path} not found"
config = yaml.safe_load(open(config_path, "r"))

gc.collect()
torch.cuda.empty_cache()

print("making the patch data")
# get the patches
images, patches = make_data(patch_size=config["data"]["patch_size"])

all_patches = []
image_ids = [] 
for i, image_patches in enumerate(patches):
    for patch in image_patches:
        all_patches.append(patch)
        image_ids.append(i)

image_ids = np.array(image_ids, dtype=int)

unique_images = np.unique(image_ids)
np.random.shuffle(unique_images) 

train_ratio = 0.8
split_idx = int(len(unique_images) * train_ratio)

train_image_ids = unique_images[:split_idx]
val_image_ids   = unique_images[split_idx:]

train_idx = np.where(np.isin(image_ids, train_image_ids))[0]
val_idx   = np.where(np.isin(image_ids, val_image_ids))[0]

print(f"Number of images: {len(unique_images)}")
print(f"Train images: {len(train_image_ids)}, Val images: {len(val_image_ids)}")
print(f"Train patches: {len(train_idx)}, Val patches: {len(val_idx)}")

train_patches = [all_patches[i] for i in train_idx]
val_patches   = [all_patches[i] for i in val_idx]
train_dataset = PatchDataset(train_patches)
val_dataset   = PatchDataset(val_patches)

dataloader_train = DataLoader(train_dataset, **config["dataloader_train"])
dataloader_val   = DataLoader(val_dataset, **config["dataloader_val"])

print(f"Train batches per epoch: {len(dataloader_train)}")
print(f"Validation batches per epoch: {len(dataloader_val)}")

print("initializing model")
model = Autoencoder(
    optimizer_config=config["optimizer"],
    patch_size=config["data"]["patch_size"],
    **config["autoencoder"],
)
print(model)

checkpoint_callback = ModelCheckpoint(**config["checkpoint"])
if "SLURM_JOB_ID" in os.environ:
    config["slurm_job_id"] = os.environ["SLURM_JOB_ID"]

wandb.login(key="6cb8e3b26b723f0706cf4c23c2e45ba69746ed8a")
wandb_logger = WandbLogger(config=config, **config["wandb"])

trainer = L.Trainer(
    logger=wandb_logger, callbacks=[checkpoint_callback], **config["trainer"]
)

print("training")
trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

gc.collect()
torch.cuda.empty_cache()
