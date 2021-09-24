import torch
import torch.nn as nn
import torch.nn.functional as F


class DisparityRegressor(nn.Module):
    def __init__(self):
        """Final part of the architecture.\n
        Upsample the output feature map from the 3D CNN part by using trilinear interpolation.\n
        Finally, perform soft-regression on the disparity values"""
        super().__init__()
        self.disp = torch.arange(192).view(1, 192, 1, 1)

    def forward(self, x):
        _, _, D, H, W = x.shape
        x = F.interpolate(x, size=(D * 4, H * 4, W * 4), mode="trilinear")
        x = torch.squeeze(x, dim=1)
        x = F.softmax(x, dim=1)
        x = torch.sum(x * self.disp.to(x.device), dim=1, keepdim=True)
        return x
