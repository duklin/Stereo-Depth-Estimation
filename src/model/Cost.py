import torch
import torch.nn as nn


class Cost(nn.Module):
    def __init__(self):
        """Calculate the cost volume by concatenating the features across channel dimension for each disparity level"""
        super(Cost, self).__init__()
        self.max_disparity = 192

    def forward(self, left, right):
        B, C, H, W = left.shape
        cost = torch.zeros(B, C * 2, self.max_disparity // 4, H, W, device=left.device)
        for i in range(self.max_disparity // 4):
            if i == 0:
                cost[:, :C, i, :, :] = left
                cost[:, C:, i, :, :] = right
            else:
                cost[:, :C, i, :, i:] = left[:, :, :, i:]
                cost[:, C:, i, :, i:] = right[:, :, :, :-i]
        return cost
