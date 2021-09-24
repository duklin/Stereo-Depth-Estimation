import torch.nn as nn
from src.model.ResNet2D import ResNet2D
from src.model.Cost import Cost
from src.model.ResNet3D import ResNet3D
from src.model.DisparityRegressor import DisparityRegressor


class SDENet(nn.Module):
    def __init__(self, resnet2d_inplanes: list, resnet3d_inplanes: list):
        """The Stereo-Depth-Estimation Network contains 4 parts:
            * ResNet2D - Siamese Residual CNN extracting features from left and right images
            * Cost - The cost volume of the features
            * ResNet3D - Used for extracting joined features from the cost volume
            * Disparity regressor - Upsample and soft-regression
        
        Args:
            resnet2d_inplanes (list): The number of input planes for the first three blocks for ResNet2D
            resnet3d_inplanes (list): The number of input planes for the first three blocks for ResNet3D
        """
        super(SDENet, self).__init__()
        self.resnet18_2d = ResNet2D([*resnet2d_inplanes, 32])
        self.cost = Cost()
        self.resnet18_3d = ResNet3D([*resnet3d_inplanes, 1])
        self.disp_regressor = DisparityRegressor()

    def forward(self, left, right):
        left_ftrs = self.resnet18_2d(left)
        right_ftrs = self.resnet18_2d(right)

        cost_volume = self.cost(left_ftrs, right_ftrs)
        cost_features = self.resnet18_3d(cost_volume)

        disparity = self.disp_regressor(cost_features)

        return disparity
