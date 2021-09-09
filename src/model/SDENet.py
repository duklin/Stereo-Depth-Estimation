import torch.nn as nn
from src.model.ResNet2D import ResNet2D
from src.model.Cost import Cost
from src.model.ResNet3D import ResNet3D
from src.model.DisparityRegressor import DisparityRegressor


class SDENet(nn.Module):
    def __init__(self, resnet2d_inplanes, resnet3d_inplanes):
        super(SDENet, self).__init__()
        self.resnet18_2d = ResNet2D([*resnet2d_inplanes, 32])  # 64, 128, 64
        self.cost = Cost()
        self.resnet18_3d = ResNet3D([*resnet3d_inplanes, 1])  # 32, 64, 32
        self.disp_regressor = DisparityRegressor()

    def forward(self, left, right):
        left_ftrs = self.resnet18_2d(left)
        right_ftrs = self.resnet18_2d(right)

        cost_volume = self.cost(left_ftrs, right_ftrs)
        cost_features = self.resnet18_3d(cost_volume)

        disparity = self.disp_regressor(cost_features)

        return disparity
