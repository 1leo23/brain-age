# utils/metrics.py

import torch

def mae_loss(pred, target):
    """
    計算 Mean Absolute Error (MAE)
    pred, target shape: (batch_size,)
    回傳: scalar (float)
    """
    return torch.mean(torch.abs(pred - target))
