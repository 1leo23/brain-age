import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbam import CBAM

class FeatureFusion(nn.Module):
    """
    將全局特徵 F_g 與局部特徵 F_l 做加權融合，並依據圖示流程：
      1) dot product (計算相似度)
      2) MLP + ReLU + FC + softmax (學習到 alpha_g, alpha_l)
      3) Weighted Fusion (加權融合)
      4) Conv 1x1 (通道變換/降維)
      5) CBAM (通道與空間注意力加權)
    """

    def __init__(self, in_channels=512, hidden_dim=128, out_channels=512, reduction=16):
        """
        這裡定義該模組所需的子層與超參數。

        參數:
        --------
        in_channels : int
            F_g, F_l 的特徵維度 (例如 512)。表示每條路徑輸出512維的特徵向量。
        hidden_dim  : int
            MLP 隱藏層大小 (例如 128)。用於 dot product 得到的相似度 S 後，經過 MLP 時的中間維度。
        out_channels: int
            1×1 卷積後的輸出通道數，若要維持與輸入一樣可設 512；也可以選擇其它數字進行降維。
        reduction   : int
            給 CBAM 使用的通道壓縮比，預設 16；CBAM 在通道注意力時，會做 in_channels/reduction 的瓶頸。
        """
        super(FeatureFusion, self).__init__()
        
        # (A) MLP for alpha_g, alpha_l
        # ------------------------------------------------------------
        # dot product 產生一個 scalar S (batch_size, 1)
        # 我們需要把 S 映射到 2 維 (對應 [alpha_g, alpha_l])，
        # 這裡定義一個兩層的 MLP，輸出維度 = 2。
        # 中間加 ReLU 以增強非線性表達能力。
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),   # 輸入是 S => shape: (batch_size, 1)
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)    # 輸出 shape: (batch_size, 2) => [alpha_g, alpha_l]
        )

        # softmax 用於讓 [alpha_g, alpha_l] 相加 = 1
        # 這樣就能把它們當成加權係數
        self.softmax = nn.Softmax(dim=1)
        
        # (B) Conv 1x1 for channel transform
        # ------------------------------------------------------------
        # 融合完後的特徵仍是 (batch_size, in_channels)，
        # 但 CBAM 需要 4D (batch_size, channels, height, width)。
        # 我們會先把它 reshape 成 (B, in_channels, 1, 1)，
        # 再用一個 1×1 卷積進行通道變換 (out_channels)。
        self.conv_1x1 = nn.Conv2d(
            in_channels,    # 輸入通道
            out_channels,   # 輸出通道
            kernel_size=1,  # 1×1 卷積
            stride=1
        )
        
        # (C) CBAM for attention
        # ------------------------------------------------------------
        # CBAM 模組會對輸入的 4D 特徵做「通道注意力」與「空間注意力」的加權，
        # 以強調較重要的通道或空間區域。
        # reduction 決定了通道注意力中 MLP 的壓縮比。
        self.cbam = CBAM(out_channels, reduction=reduction)

    def forward(self, F_g, F_l):
        """
        前向傳播函式 (forward):
        
        輸入:
        --------
        F_g : torch.Tensor
            形狀 (batch_size, in_channels)，表示全局特徵向量
        F_l : torch.Tensor
            形狀 (batch_size, in_channels)，表示局部特徵向量

        輸出:
        --------
        F_out : torch.Tensor
            形狀 (batch_size, out_channels)，表示融合後再經過 CBAM 與 1×1 卷積的特徵
        """
        # B: batch_size
        B = F_g.size(0)

        # (1) Dot product => 相似度 S
        # ------------------------------------------------------------
        # 我們對應用在同一筆資料的 F_g[i] 與 F_l[i] 做內積，
        # 這可以衡量兩個向量的相似度大小。
        # F_g, F_l shape: (B, in_channels)
        # => F_g * F_l shape: (B, in_channels)
        # => sum(..., dim=1) shape: (B,) => 再 keepdim=True => (B,1)
        S = torch.sum(F_g * F_l, dim=1, keepdim=True)  # shape: (B, 1)
        
        # (2) MLP + softmax => [alpha_g, alpha_l]
        # ------------------------------------------------------------
        # 將 S 傳入我們定義好的 MLP => 輸出 shape: (B, 2)
        alpha = self.mlp(S)           # (B, 2)

        # 再透過 softmax，讓 alpha_g + alpha_l = 1
        alpha = self.softmax(alpha)   # (B, 2)
        
        # alpha[:, 0] 對應 alpha_g
        # alpha[:, 1] 對應 alpha_l
        # unsqueeze(-1) => 把 shape 變成 (B, 1)，以便和 F_g, F_l 相乘
        alpha_g = alpha[:, 0].unsqueeze(-1)  # (B, 1)
        alpha_l = alpha[:, 1].unsqueeze(-1)  # (B, 1)
        
        # (3) Weighted Fusion
        # ------------------------------------------------------------
        # F_fused = alpha_g * F_g + alpha_l * F_l
        # 代表以動態權重把全局特徵和局部特徵相加
        F_fused = alpha_g * F_g + alpha_l * F_l  # shape: (B, in_channels)
        
        # (4) Conv 1×1
        # ------------------------------------------------------------
        # CBAM 期望輸入是 4D (B, C, H, W)，
        # 但目前 F_fused 是 (B, in_channels) => (B, in_channels, 1, 1)
        F_fused_4d = F_fused.view(B, -1, 1, 1)         # shape: (B, in_channels, 1, 1)

        # 用 1×1 卷積做通道變換 (或降維)
        # => 輸出 shape: (B, out_channels, 1, 1)
        F_1x1 = self.conv_1x1(F_fused_4d)
        
        # (5) CBAM
        # ------------------------------------------------------------
        # CBAM 對 (B, out_channels, 1, 1) 做通道與空間注意力
        # 最後輸出同樣 shape: (B, out_channels, 1, 1)
        F_cbam = self.cbam(F_1x1)
        
        # reshape 回 2D => (B, out_channels)
        F_out = F_cbam.view(B, -1)
        return F_out
