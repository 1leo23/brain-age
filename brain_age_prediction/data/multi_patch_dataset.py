# multi_patch_dataset.py

import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F  
from PIL import Image
from collections import defaultdict

class MultiPatchGlobalDataset(Dataset):
    """
    同時讀取「全局投影影像」與「局部patch」的資料集：
      - df：包含 NIfTI_ID, AGE, SEX 等欄位的 DataFrame
      - global_dir：儲存全局投影影像 (e.g. Axial_Mean, Axial_Std...)
      - local_dir ：儲存局部patch影像 (local/{subject_id}/...)
      - return_id：若為 True，則 __getitem__ 回傳 5 個值 (含 subject_id)；否則回傳 4 個值
    """
    def __init__(self, df, global_dir, local_dir,
                 transform=None, local_transform=None,
                 return_id=False):
        self.df = df.reset_index(drop=True)
        self.global_dir = global_dir
        self.local_dir = local_dir
        self.transform = transform
        self.local_transform = local_transform if local_transform else transform
        self.return_id = return_id  # 關鍵：控制是否回傳 subject_id

        self.projections = [
            "Axial_Mean",
            "Axial_Std",
            "Coronal_Mean",
            "Coronal_Std",
            "Sagittal_Mean",
            "Sagittal_Std"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["NIfTI_ID"]
        age = row["AGE"]
        sex = row["SEX"]

        # 讀取全局影像 (6通道)
        global_img = self.load_global_image(subject_id)
        # 讀取局部patch (N, 6, patch_h, patch_w)
        local_imgs_list = self.load_all_local_patches(subject_id)

        # 依照 return_id 判斷是否回傳 subject_id
        if self.return_id:
            # 回傳 5 個值
            return global_img, local_imgs_list, torch.tensor(age, dtype=torch.float32), sex, subject_id
        else:
            # 回傳 4 個值
            return global_img, local_imgs_list, torch.tensor(age, dtype=torch.float32), sex

    def load_global_image(self, subject_id):
        imgs = []
        for proj in self.projections:
            filename = f"{subject_id}_{proj}.png"
            path = os.path.join(self.global_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"[Global] 無法讀取: {path}")
            pil_img = Image.fromarray(img)

            if self.transform:
                pil_img = self.transform(pil_img)
            else:
                pil_img = F.to_tensor(pil_img)

            imgs.append(pil_img)

        # 拼接成 (6, H, W)
        global_img = torch.cat(imgs, dim=0)
        return global_img

    def load_all_local_patches(self, subject_id):
        sub_dir = os.path.join(self.local_dir, subject_id)
        if not os.path.exists(sub_dir):
            raise FileNotFoundError(f"局部影像資料夾不存在: {sub_dir}")

        files = [f for f in os.listdir(sub_dir) if f.endswith(".png")]
        local_dict = defaultdict(dict)

        for f in files:
            base = os.path.splitext(f)[0]
            parts = base.split("_")
            if len(parts) < 5:
                print(f"[SKIP] 不符格式(underscore不足): {f}")
                continue

            if parts[0] != subject_id:
                print(f"[SKIP] ID不符: 檔案ID={parts[0]}, 資料夾ID={subject_id}, file={f}")
                continue

            proj = parts[1] + "_" + parts[2]
            try:
                row = int(parts[3])
                col = int(parts[4])
            except ValueError:
                print(f"[SKIP] row,col 不是數字: {f}")
                continue

            local_dict[(row, col)][proj] = f

        patch_list = []
        rowcols = sorted(local_dict.keys())
        for (r, c) in rowcols:
            proj_map = local_dict[(r, c)]
            if all(p in proj_map for p in self.projections):
                patch_imgs = []
                for proj in self.projections:
                    fname = proj_map[proj]
                    path = os.path.join(sub_dir, fname)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"[SKIP] 讀取失敗: {path}")
                        continue
                    pil_img = Image.fromarray(img)

                    if self.local_transform:
                        pil_img = self.local_transform(pil_img)
                    else:
                        pil_img = F.to_tensor(pil_img)

                    patch_imgs.append(pil_img)

                if len(patch_imgs) == 6:
                    patch_tensor = torch.cat(patch_imgs, dim=0)  # => (6, patch_h, patch_w)
                    patch_list.append(patch_tensor)
            else:
                print(f"[SKIP] 缺少某些投影, row={r}, col={c}, ID={subject_id}")

        if len(patch_list) == 0:
            raise ValueError(f"ID={subject_id} 無任何有效局部patch可用")

        local_imgs_list = torch.stack(patch_list, dim=0)  # (N, 6, patch_h, patch_w)
        return local_imgs_list
