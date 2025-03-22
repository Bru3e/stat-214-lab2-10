# EXAMPLE USAGE:
# python get_embedding.py configs/default.yaml checkpoints/cnn-default-epoch=009-v4.ckpt
import sys
import torch
import pandas as pd
import numpy as np
import yaml
import gc
from tqdm import tqdm

from autoencoder import Autoencoder
from data import make_data

config_path = sys.argv[1]
checkpoint_path = sys.argv[2]

config = yaml.safe_load(open(config_path, "r"))

print("Loading the saved model")
# initialize the autoencoder class
model = Autoencoder(patch_size=config["data"]["patch_size"], **config["autoencoder"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
# load the checkpoint's state_dict into the model
model.load_state_dict(checkpoint["state_dict"])
# put the model in evaluation mode
model.eval()

print("Making the patch data")
images_long, patches = make_data(patch_size=config["data"]["patch_size"])

print("Obtaining embeddings")
# get the embedding for each patch
embeddings = [] 
images_embedded = []  
batch_size = 64

for i in tqdm(range(len(images_long))):
    ys = images_long[i][:, 0]
    xs = images_long[i][:, 1]

    miny, minx = min(ys), min(xs)
    height, width = int(max(ys) - miny + 1), int(max(xs) - minx + 1)

    patch_array = np.array(patches[i]) 
    
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(patch_array.astype(np.float32)).clone().to(device)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0]  
            emb = model.embed(batch)
            batch_embeddings.append(emb.detach().cpu().numpy())  

    emb = np.vstack(batch_embeddings)  
    embeddings.append(emb)

    del dataloader
    gc.collect()
    torch.cuda.empty_cache()

    # represent the embedding as an image
    img_embedded = np.zeros((emb.shape[1], height, width))
    img_embedded[:, (ys - miny).astype(int), (xs - minx).astype(int)] = emb.T
    images_embedded.append(img_embedded)
    
print("Saving the embeddings")
# save the embeddings as csv
for i in tqdm(range(len(images_long))):
    embedding_df = pd.DataFrame(embeddings[i], columns=[f"ae{i}" for i in range(8)])
    embedding_df["y"] = images_long[i][:, 0]
    embedding_df["x"] = images_long[i][:, 1]
    # move y and x to front
    cols = embedding_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    embedding_df = embedding_df[cols]
    # save to csv
    embedding_df.to_csv(f"data/image{i+1}_ae.csv", index=False)


# here is some code to take a look at the embeddings.
# but you should probably just load the csv files in a jupyter notebook
# and visualize there.

# import matplotlib.pyplot as plt
# plt.imshow(images_embedded[0][0])
# plt.show()