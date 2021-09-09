import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image


class KITTI(Dataset):
    def __init__(
        self, root: str, train: bool, transform=None, batch_transform=None
    ) -> None:
        super().__init__()
        assert os.path.exists(root)
        self.root = root
        self.image_ids = [
            os.path.basename(file)
            for file in glob.glob(os.path.join(self.root, f"image_left/*png"))
        ]
        self.train = train
        self.transform = transform
        self.batch_transform = batch_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        left_img_path = os.path.join(self.root, "image_left", image_id)
        right_img_path = os.path.join(self.root, "image_right", image_id)

        left_img = Image.open(left_img_path)
        right_img = Image.open(right_img_path)
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)

        if self.train:
            left_disp_path = os.path.join(self.root, "disp_occ_left", image_id)
            left_disp_img = Image.open(left_disp_path)
            left_disp_img = F.to_tensor(left_disp_img).float()
            left_disp_img /= 256.0

            if self.batch_transform:
                concat = torch.cat([left_img, right_img, left_disp_img], dim=0)
                concat = self.batch_transform(concat)
                return torch.split(concat, [3, 3, 1])

            return left_img, right_img, left_disp_img

        return left_img, right_img


if __name__ == "__main__":
    dataset = KITTI("dataset/training", train=True, transform=F.pil_to_tensor)
    print(f"Train dataset length: {len(dataset)}")
    left, right, disp = dataset[0]
    print(f"Left: {left.shape} {left.dtype}")
    print(f"Right: {right.shape} {right.dtype}")
    print(f"Disparity: {disp.shape} {disp.dtype}")
