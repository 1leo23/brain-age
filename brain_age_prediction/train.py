# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import numpy as np
import pandas as pd

from config import Config
from data.multi_patch_dataset import MultiPatchGlobalDataset
from models.brain_age_model import BrainAgeModel
from utils.metrics import mae_loss

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    count = 0

    for batch in loader:
        # 這裡只解包 4 個值
        global_img, local_img, age, sex = batch
        global_img = global_img.to(device)
        local_img  = local_img.to(device)
        age = age.to(device)

        optimizer.zero_grad()
        y_pred = model(global_img, local_img)  # (batch_size,)
        loss = mae_loss(y_pred, age)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(age)
        count += len(age)

    avg_loss = total_loss / count
    return avg_loss

def validate_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in loader:
            global_img, local_img, age, sex = batch
            global_img = global_img.to(device)
            local_img  = local_img.to(device)
            age = age.to(device)

            y_pred = model(global_img, local_img)
            loss = mae_loss(y_pred, age)

            total_loss += loss.item() * len(age)
            count += len(age)

    avg_loss = total_loss / count
    return avg_loss

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 讀取 train.csv 建立 DataFrame
    df = pd.read_csv(config.TRAIN_CSV)
    print(f"資料筆數: {len(df)}")

    # 2) 建立完整 Dataset
    #    訓練時不需要 subject_id => return_id=False
    full_dataset = MultiPatchGlobalDataset(
        df=df,
        global_dir=config.GLOBAL_IMG_DIR,
        local_dir=config.LOCAL_IMG_DIR,
        return_id=False  # 關鍵
    )

    total_len = len(full_dataset)
    val_len   = int(total_len * config.VAL_RATIO)
    train_len = total_len - val_len

    # 3) random_split 切成 train/val
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False)

    # 4) 建立模型
    model = BrainAgeModel(in_channels=6, hidden_dim=128)
    model.to(device)

    # 5) 定義優化器 (Adam)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 6) 訓練迴圈
    best_val_loss = float("inf")
    best_model_weights = None

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = validate_one_epoch(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{config.EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()

    # 7) 儲存最佳模型
    if best_model_weights is not None:
        print(f"Best Val Loss = {best_val_loss:.4f}, saving model to {config.SAVE_MODEL_PATH}")
        torch.save(best_model_weights, config.SAVE_MODEL_PATH)
    else:
        print("No improvement, not saving model.")

if __name__ == "__main__":
    main()
