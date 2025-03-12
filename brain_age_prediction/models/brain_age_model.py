# brain_age_model.py

import torch
import torch.nn as nn

from models.global_net import GlobalNet
from models.local_net import LocalNet
from models.fusion_model import FeatureFusion
from models.age_regression import AgeRegressionHead

class BrainAgeModel(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=128):
        super(BrainAgeModel, self).__init__()
        # 全局與局部網路
        self.global_net = GlobalNet(in_channels)   # ex: 輸入 (B, 6, H, W)，輸出 (B, 512)
        self.local_net  = LocalNet(in_channels)    # ex: 輸入 (B*N, 6, patch_h, patch_w)，輸出 (B*N, 512)
        
        # 融合模組 (舉例)
        self.fusion = FeatureFusion(
            in_channels=512, 
            hidden_dim=hidden_dim, 
            out_channels=512
        )
        
        # 回歸頭
        self.regressor = AgeRegressionHead(in_features=512, hidden_dim=128)

    def forward(self, global_img, local_img):
        """
        global_img shape: (B, 6, H, W)
        local_img  shape: (B, N, 6, patch_h, patch_w)
          - B: batch_size
          - N: 每個樣本的patch數量
          - C=6: 通道數 (Axial_Mean, Axial_Std, Coronal_Mean, Coronal_Std, Sagittal_Mean, Sagittal_Std)
        """
        # 1) 取得全局特徵
        F_g = self.global_net(global_img)  # shape: (B, 512)

        # 2) reshape 局部影像 => (B*N, 6, patch_h, patch_w)，再丟給 local_net
        B, N, C, H, W = local_img.shape
        local_img_2d = local_img.view(B*N, C, H, W)  # => (B*N, 6, patch_h, patch_w)
        F_l_all = self.local_net(local_img_2d)       # => (B*N, 512)

        # 3) 再 reshape 回 (B, N, 512)，進行聚合 (例如平均)
        F_l_all = F_l_all.view(B, N, -1)             # => (B, N, 512)
        F_l = F_l_all.mean(dim=1)                    # => (B, 512)  (也可用注意力加權)

        # 4) 特徵融合
        F_fused = self.fusion(F_g, F_l)              # => (B, 512)

        # 5) 回歸
        y_pred = self.regressor(F_fused)             # => (B, 1)
        return y_pred.squeeze(-1)                    # => (B,) for convenience
