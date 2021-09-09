import torch
import numpy as np
from src.train.loss import smooth_l1, three_pe


class Trainer:
    def __init__(
        self, model, train_loader, valid_loader, num_epochs, optimizer, device,
    ) -> None:
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.optimizer = optimizer

    @torch.no_grad()
    def valid_step(self):
        losses = []
        errors = []

        for i, (left, right, target) in enumerate(self.valid_loader):
            left = left.to(self.device)
            right = right.to(self.device)
            target = target.to(self.device)

            prediction = self.model(left, right)
            loss = smooth_l1(target, prediction)
            error = three_pe(target, prediction)

            losses.append(loss.item())
            errors.append(error.item())

        return losses, errors

    def _train_epoch(self):
        losses = []
        errors = []

        for i, (left, right, target) in enumerate(self.train_loader):
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

        return losses, errors

    def train(self):
        for i in range(self.num_epochs):
            self.model.train()
            self._train_epoch()

            self.model.eval()
            self.valid_step()

