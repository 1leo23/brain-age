# test.py

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from config import Config
from data.multi_patch_dataset import MultiPatchGlobalDataset
from models.brain_age_model import BrainAgeModel
from utils.metrics import mae_loss

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) 讀取 test.csv
    df_test = pd.read_csv(config.TEST_CSV)
    print(f"測試集資料筆數: {len(df_test)}")

    # 2) 建立測試集
    #    測試時需要 subject_id => return_id=True
    test_dataset = MultiPatchGlobalDataset(
        df=df_test,
        global_dir=config.TEST_GLOBAL_DIR,
        local_dir=config.TEST_LOCAL_DIR,
        return_id=True  # 關鍵
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3) 建立模型並載入訓練後權重
    model = BrainAgeModel(in_channels=6, hidden_dim=128)
    model.to(device)

    state_dict = torch.load(config.SAVE_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    total_loss = 0
    count = 0
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            # 這裡會解包 5 個值
            global_img, local_img, age, sex, subject_id = batch
            global_img = global_img.to(device)
            local_img  = local_img.to(device)
            age = age.to(device)

            pred = model(global_img, local_img)  # (batch_size,)
            loss = mae_loss(pred, age)

            total_loss += loss.item() * len(age)
            count += len(age)

            # 記錄每筆的 subject_id, true_age, pred_age
            for i in range(len(age)):
                sid = subject_id[i]
                true_age = age[i].item()
                pred_age = pred[i].item()

                predictions.append({
                    "subject_id": sid,
                    "true_age": true_age,
                    "pred_age": pred_age
                })

    test_mae = total_loss / count
    print(f"Test MAE = {test_mae:.4f}")

    # 輸出到指定資料夾
    output_dir = r"C:\ixi\IXI MNI\testset\prediction"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "predictions.csv")

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_csv, index=False)
    print(f"已輸出預測結果至: {output_csv}")

if __name__ == "__main__":
    main()
