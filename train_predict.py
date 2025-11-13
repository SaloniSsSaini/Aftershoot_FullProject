"""
Run with:
python train_predict.py --data_dir "dataset"

This script trains a small model and creates submission.csv in Validation folder.
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import tifffile as tiff
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import WBNet
from utils import read_image


# ---------------- Dataset Classes ----------------

def resolve_image_path(images_dir, img_id):
    """Try both .tif and .tiff, return whichever exists."""
    tif_path = os.path.join(images_dir, img_id + ".tif")
    tiff_path = os.path.join(images_dir, img_id + ".tiff")

    if os.path.exists(tif_path):
        return tif_path
    elif os.path.exists(tiff_path):
        return tiff_path
    else:
        raise FileNotFoundError(f"Image {img_id} not found as .tif or .tiff")


class WBTrainDataset(Dataset):
    def __init__(self, df, images_dir, meta_cols, scaler, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.meta_cols = meta_cols
        self.scaler = scaler
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        img_id = str(row['id_global'])
        img_path = resolve_image_path(self.images_dir, img_id)

        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)

        meta = row[self.meta_cols].values.astype(np.float32)
        meta = self.scaler.transform(meta.reshape(1, -1)).squeeze(0)

        as_temp = float(row["currTemp"])
        as_tint = float(row["currTint"])

        target_temp = float(row["Temperature"]) - as_temp
        target_tint = float(row["Tint"]) - as_tint

        return img, meta, np.array([target_temp, target_tint], dtype=np.float32)


class WBInferenceDataset(Dataset):
    def __init__(self, df, images_dir, meta_cols, scaler, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.meta_cols = meta_cols
        self.scaler = scaler
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = str(row['id_global'])
        img_path = resolve_image_path(self.images_dir, img_id)

        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)

        meta = row[self.meta_cols].values.astype(np.float32)
        meta = self.scaler.transform(meta.reshape(1, -1)).squeeze(0)

        return img, meta, row["id_global"], float(row["currTemp"]), float(row["currTint"])


# ---------------- Main Training + Prediction ----------------

def main(data_dir, epochs=3, batch_size=16, lr=1e-4):

    train_csv = os.path.join(data_dir, "Train", "sliders.csv")
    train_images = os.path.join(data_dir, "Train", "Images")

    val_csv = os.path.join(data_dir, "Validation", "sliders_inputs.csv")
    val_images = os.path.join(data_dir, "Validation", "Images")

    df = pd.read_csv(train_csv)

    meta_cols = ["grayscale", "aperture", "flashFired",
                 "focalLength", "isoSpeedRating", "shutterSpeed"]
    meta_cols = [c for c in meta_cols if c in df.columns]

    df[meta_cols] = df[meta_cols].fillna(0)

    if "flashFired" in df.columns:
        le = LabelEncoder()
        df["flashFired"] = le.fit_transform(df["flashFired"].astype(str))

    scaler = StandardScaler()
    scaler.fit(df[meta_cols].values.astype(float))

    train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)

    train_tf = T.Compose([
        T.RandomResizedCrop(256, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    val_tf = T.Compose([
        T.Resize((256,256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = WBTrainDataset(train_df, train_images, meta_cols, scaler, train_tf)
    val_ds = WBTrainDataset(valid_df, train_images, meta_cols, scaler, val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WBNet(meta_in=len(meta_cols)).to(device)

    loss_fn = nn.L1Loss()
    optimz = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, metas, tgts in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, metas, tgts = imgs.to(device), metas.to(device), tgts.to(device)

            out = model(imgs, metas)
            loss = loss_fn(out, tgts)

            optimz.zero_grad()
            loss.backward()
            optimz.step()

            running += loss.item()

        print(f"Epoch {epoch+1}, Loss = {running/len(train_loader):.4f}")

    torch.save(model.state_dict(), "best_wb.pth")

    val_df = pd.read_csv(val_csv)
    val_df[meta_cols] = val_df[meta_cols].fillna(0)

    if "flashFired" in val_df.columns:
        try:
            val_df["flashFired"] = le.transform(val_df["flashFired"].astype(str))
        except:
            pass

    infer_ds = WBInferenceDataset(val_df, val_images, meta_cols, scaler, val_tf)
    infer_loader = DataLoader(infer_ds, batch_size=16)

    model.eval()
    results = []

    with torch.no_grad():
        for imgs, metas, ids, as_temp, as_tint in infer_loader:
            imgs = imgs.to(device)
            metas = metas.to(device)

            preds = model(imgs, metas).cpu().numpy()

            for i in range(len(preds)):
                dT, dX = preds[i]
                final_temp = round(float(as_temp[i]) + dT)
                final_tint = round(float(as_tint[i]) + dX)
                results.append([ids[i], final_temp, final_tint])

    sub = pd.DataFrame(results, columns=["id_global", "Temperature", "Tint"])
    out_path = os.path.join(data_dir, "Validation", "submission.csv")
    sub.to_csv(out_path, index=False)

    print("Submission saved at:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    main(args.data_dir, epochs=args.epochs)
