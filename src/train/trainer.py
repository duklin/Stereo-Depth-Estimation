import os
import torch
import numpy as np
from src.model.SDENet import SDENet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from src.train.loss import smooth_l1, three_pe


class Trainer:
    def __init__(
        self,
        model: SDENet,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int,
        optimizer: Optimizer,
        device: str,
        writer: SummaryWriter,
        ckp_dir: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.device = device
        self.writer = writer
        self.ckp_dir = ckp_dir

    @torch.no_grad()
    def valid_step(self):
        losses = []
        errors = []

        for left, right, target in self.valid_loader:
            left = left.to(self.device)
            right = right.to(self.device)
            target = target.to(self.device)

            prediction = self.model(left, right)
            loss = smooth_l1(target, prediction)
            error = three_pe(target, prediction)

            losses.append(loss.item())
            errors.append(error.item())

        loss = np.mean(losses)
        error = np.mean(errors)

        return loss, error

    def _train_step(self):
        losses = []
        errors = []

        for left, right, target in self.train_loader:
            left = left.to(self.device)
            right = right.to(self.device)
            target = target.to(self.device)

            prediction = self.model(left, right)
            loss = smooth_l1(target, prediction)
            error = three_pe(target, prediction)

            losses.append(loss.item())
            errors.append(error.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss = np.mean(losses)
        error = np.mean(errors)

        return loss, error

    def train(self):
        for i in range(self.num_epochs):
            self.model.train()
            train_loss, train_error = self._train_step()
            self.writer.add_scalar("Loss/train", train_loss, global_step=i)
            self.writer.add_scalar("3P Error/train", train_error, global_step=i)

            self.model.eval()
            valid_loss, valid_error = self.valid_step()
            self.writer.add_scalar("Loss/valid", valid_loss, global_step=i)
            self.writer.add_scalar("3P Error/valid", valid_error, global_step=i)

            file_path = os.path.join(self.ckp_dir, f"{i}.pt")
            torch.save(
                {
                    "state_dict": self.model.state_dict(),
                    "epoch": i,
                    "stats": {
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "train_error": train_error,
                        "valid_error": valid_error,
                    },
                },
                file_path,
            )

