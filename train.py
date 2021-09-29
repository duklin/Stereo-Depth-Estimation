import datetime
from argparse import ArgumentParser

import torch
import torch.optim
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader

from src.model.SDENet import SDENet
from src.dataset.kitti import KITTI
from src.train.trainer import Trainer


DATASET_DIR = "dataset/training"

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

TRAIN_TRANSFORM = T.Compose(
    [
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ]
)
TRAIN_BATCH_TRANSFORM = T.RandomCrop((256, 512))
VAL_TRANSFORM = T.Compose(
    [T.ToTensor(), T.Normalize(mean=MEAN.tolist(), std=STD.tolist())]
)


def main(args):
    kitti = KITTI(
        root=DATASET_DIR,
        train=True,
        transform=TRAIN_TRANSFORM,
        batch_transform=TRAIN_BATCH_TRANSFORM,
    )
    kitti_train, kitti_val = random_split(kitti, [150, 50])
    kitti_val.transform = VAL_TRANSFORM
    train_loader = DataLoader(kitti_train, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(kitti_val, batch_size=args.batch_size, shuffle=True)

    log_dir = datetime.datetime.now().strftime("logs/%Y-%m-%d_%H-%M")
    writer = SummaryWriter(log_dir)

    # visualize the network
    left, right, _ = next(iter(train_loader))
    left, right = left.to(args.device), right.to(args.device)
    sde_net = SDENet(args.resnet2d_inplanes, args.resnet3d_inplanes).to(args.device)
    writer.add_graph(sde_net, (left, right))

    optimizer = getattr(torch.optim, args.optimizer)(
        sde_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    trainer = Trainer(
        model=sde_net,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=args.epochs,
        optimizer=optimizer,
        device=args.device,
        writer=writer,
    )
    trainer.train()

    torch.save(sde_net, "model.pt")

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a network for Stereo Depth Estimation")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument(
        "--resnet2d-inplanes",
        nargs="+",
        type=int,
        help="Set the number of input planes of the first three blocks of the ResNet-18 2D",
        required=True,
    )
    parser.add_argument(
        "--resnet3d-inplanes",
        nargs="+",
        type=int,
        help="Set the number of input planes of the first three blocks of the ResNet-18 3D",
        required=True,
    )
    parser.add_argument(
        "--batch-size", type=int, required=True,
    )
    parser.add_argument(
        "--optimizer", choices=["Adam", "SGD", "RMSprop"], required=True
    )
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--device", required=True)

    args = parser.parse_args()
    main(args)

