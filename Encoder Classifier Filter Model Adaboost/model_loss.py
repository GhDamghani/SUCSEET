import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join


class CCE(nn.Module):
    def __init__(self, device="cpu", balancing_factor=1):
        super(CCE, self).__init__()
        self.nll_loss = nn.NLLLoss()
        self.device = device  # {'cpu', 'cuda:0', 'cuda:1', ...}
        self.balancing_factor = balancing_factor

    def forward(self, yHat, y):
        # Note: yHat.shape[1] <=> number of classes
        batch_size = len(y)
        # cross entropy
        cross_entropy = self.nll_loss(F.log_softmax(yHat, dim=1), y)
        # complement entropy
        yHat = F.softmax(yHat, dim=1)
        Yg = yHat.gather(dim=1, index=torch.unsqueeze(y, 1))
        Px = yHat / (1 - Yg) + 1e-7
        Px_log = torch.log(Px + 1e-10)
        y_zerohot = torch.ones(batch_size, yHat.shape[1]).scatter_(
            1, y.view(batch_size, 1).data.cpu(), 0
        )
        output = Px * Px_log * y_zerohot.to(device=self.device)
        complement_entropy = torch.sum(output) / (
            float(batch_size) * float(yHat.shape[1])
        )

        return cross_entropy - self.balancing_factor * complement_entropy


def CEL_weights_class_balanced(samples_per_class, num_classes, beta=0.999):
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_classes
    weights = torch.tensor(weights).float()
    return weights


""" def criterion(num_classes):
    def custom_loss(yHat, y):
        yHat = F.softmax(yHat, dim=1)
        y = torch.nn.functional.one_hot(y, num_classes=num_classes)
        cross_entropy = (-y * torch.log(yHat + 1e-8)).sum(dim=1).mean()
        return cross_entropy

    return custom_loss """


def criterion_cross_entropy_observation_weight(pred, y, w):
    y_onehot = torch.nn.functional.one_hot(y, num_classes=pred.shape[1]).float()
    log_probs = torch.nn.functional.log_softmax(pred, dim=1)
    weighted_loss = -torch.sum(y_onehot * log_probs, dim=1)
    loss = torch.sum(weighted_loss * w) / torch.sum(
        w
    )  # weighted average of weighted_loss with w
    return loss


def criterion(histogram_weights, num_classes, weights=True):
    histogram_weights = CEL_weights_class_balanced(histogram_weights, num_classes)
    return torch.nn.CrossEntropyLoss(
        histogram_weights,
        reduction="mean",
        label_smoothing=0.1,
    )  #


def criterion_NLLLoss(histogram_weights, num_classes, weights=True):
    if weights:
        histogram_weights = CEL_weights_class_balanced(histogram_weights, num_classes)
        return torch.nn.NLLLoss(
            histogram_weights,
            reduction="mean",
        )  #
    else:
        return torch.nn.NLLLoss(
            reduction="mean",
        )


""" def lossfn(kmeans_folder, num_classes):
    histogram_weights = np.load(join(kmeans_folder, "histogram.npy"))
    histogram_weights = CEL_weights_class_balanced(histogram_weights, num_classes)
    return torch.nn.CrossEntropyLoss(
        histogram_weights, reduction="mean", label_smoothing=0.1
    ) """
