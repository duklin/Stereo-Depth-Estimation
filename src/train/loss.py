import torch
import torch.nn.functional as F


def smooth_l1(target, prediction):
    mask = target > 0.0
    mask.detach_()
    return F.smooth_l1_loss(target[mask], prediction[mask])


def three_pe(target, prediction):
    mask = torch.logical_and(target > 0, target < 192)
    target = target[mask]
    prediction = prediction[mask]

    diff = torch.abs(target - prediction)
    less_than_3 = diff < 3
    less_than_5_perc = diff < (0.05 * target)

    correct = torch.logical_or(less_than_3, less_than_5_perc)
    N = torch.numel(target)
    N_correct = correct.sum()

    return 1 - (N_correct / N)
