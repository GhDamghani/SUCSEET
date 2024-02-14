import numpy as np
import torch
from os.path import join


def CEL_weights_class_balanced(samples_per_class, num_classes, beta=0.999):
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_classes
    weights = torch.tensor(weights).float()
    return weights


def lossfn(kmeans_folder, num_classes):
    histogram_weights = np.load(join(kmeans_folder, "histogram.npy"))
    histogram_weights = CEL_weights_class_balanced(histogram_weights, num_classes)
    return torch.nn.CrossEntropyLoss(
        histogram_weights, reduction="mean", label_smoothing=0.1
    )
