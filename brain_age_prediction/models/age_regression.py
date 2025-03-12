import torch
import torch.nn as nn
import torch.nn.functional as F

class AgeRegressionHead(nn.Module):
    """
    完成腦年齡回歸的兩層 MLP：
      1) 第一層全連接 (FC) + ReLU
      2) 第二層全連接 (FC) (輸出 1 個值，用於回歸年齡)
    """
    def __init__(self, in_features=512, hidden_dim=128):
        """
        參數：
        --------
        in_features : 融合特徵的維度，預設 512
        hidden_dim  : 中間隱藏層大小，預設 128
        """
        super(AgeRegressionHead, self).__init__()
        
        # 第 1 層：全連接 + ReLU
        # F_fc = ReLU(W1 * F_reduced + b1)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # 第 2 層：全連接 (輸出 1 維，用於回歸)
        # y_hat = W2 * F_fc + b2
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: shape (batch_size, in_features)，通常是融合後的特徵向量
        回傳: shape (batch_size, 1)，表示每筆資料的預測年齡
        """
        # 通過第一層 FC + ReLU
        x = self.fc1(x)           # shape: (batch_size, hidden_dim)
        x = self.relu(x)          # shape: (batch_size, hidden_dim)

        # 通過第二層 FC，得到 (batch_size, 1)
        y_hat = self.fc2(x)
        return y_hat
