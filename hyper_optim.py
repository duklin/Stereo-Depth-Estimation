import torch
import optuna
from optuna.trial import TrialState
import logging
import numpy as np
import torch.optim as optim
import torchvision.transforms as T
from tqdm import trange
from src.train.loss import smooth_l1, three_pe
from src.model.SDENet import SDENet
from src.dataset.kitti import KITTI
from torch.utils.data import random_split, DataLoader

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
MAX_TRAIN_SAMPLES_PER_EPOCH = 30
MAX_VALID_SAMPLES_PER_EPOCH = 10
DATASET_DIR = "dataset/training"


def define_model(trial):
    resnet2d_inplanes, resnet3d_inplanes = [], []
    for i in range(3):
        resnet2d_inplanes.append(trial.suggest_int(f"resnet2d_inplanes_{i}", 32, 64))
        resnet3d_inplanes.append(trial.suggest_int(f"resnet3d_inplanes_{i}", 16, 32))
    model = SDENet(resnet2d_inplanes, resnet3d_inplanes)
    return model


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


def get_dataloaders(trial):
    batch_size = trial.suggest_int("batch_size", 2, 5)
    kitti = KITTI(
        DATASET_DIR,
        train=True,
        transform=TRAIN_TRANSFORM,
        batch_transform=TRAIN_BATCH_TRANSFORM,
    )
    kitti_train, kitti_val = random_split(kitti, [150, 50])
    kitti_val.transform = VAL_TRANSFORM
    train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader


@torch.no_grad()
def eval_model(model, valid_loader):
    model.eval()
    losses, errors = [], []

    for i, (left, right, target) in enumerate(valid_loader):
        B, _, _, _ = left.shape
        if i * B > MAX_VALID_SAMPLES_PER_EPOCH:
            break

        left = left.to(DEVICE)
        right = right.to(DEVICE)
        target = target.to(DEVICE)

        prediction = model(left, right)
        loss = smooth_l1(target, prediction)
        error = three_pe(target, prediction)

        losses.append(loss.item())
        errors.append(error.item())

    loss = np.mean(losses)
    error = np.mean(errors)

    return loss, error


def objective(trial):
    model = define_model(trial).to(DEVICE)
    model.train()
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    if optimizer_name in ["RMSprop", "SGD"]:
        optimizer = getattr(optim, optimizer_name)(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        optimizer = getattr(optim, optimizer_name)(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    train_loader, valid_loader = get_dataloaders(trial)
    t = trange(EPOCHS)
    for epoch in t:
        for i, (left, right, target) in enumerate(train_loader):
            B, _, _, _ = left.shape
            if i * B > MAX_TRAIN_SAMPLES_PER_EPOCH:
                break

            left = left.to(DEVICE)
            right = right.to(DEVICE)
            target = target.to(DEVICE)

            prediction = model(left, right)
            loss = smooth_l1(target, prediction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_description(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item():.3}")

    loss, error = eval_model(model, valid_loader)
    return loss, error


if __name__ == "__main__":
    study_name = "hyper_optim"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, directions=["minimize", "minimize"]
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler("hyper_optim.log", mode="w"))
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
