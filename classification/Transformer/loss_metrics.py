import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join


def calculate_mcd(mfccs1: np.ndarray, mfccs2: np.ndarray) -> float:
    """
    Calculate the Mel Cepstral Distortion (MCD) between two MFCC sets.

    Parameters:
    mfccs1 (numpy.ndarray): First set of MFCCs.
    mfccs2 (numpy.ndarray): Second set of MFCCs.

    Returns:
    float: The MCD value.

    Raises:
    AssertionError: If the dimensions of the input MFCCs do not match.
    """
    # Exclude the zeroth coefficient and ensure the dimensions match
    mfccs1, mfccs2 = mfccs1[:, 1:], mfccs2[:, 1:]
    assert mfccs1.shape == mfccs2.shape, "Dimensions do not match"

    print("     calculating mcd")

    # Calculate the squared differences
    diff = np.sum((mfccs1 - mfccs2) ** 2, axis=1)

    # Calculate the MCD
    mcd = np.mean(np.sqrt(diff)) * (10 / np.log(10))
    return mcd


def get_loss(train_dataset, num_classes, logger=None):
    _, cluster_train = next(train_dataset.generate_batch(-1))
    histogram_weights = np.unique(cluster_train, return_counts=True)[1]
    histogram_weights = histogram_weights / np.sum(histogram_weights)

    criterion = CrossEntropyLoss_class_balanced(histogram_weights, num_classes)
    return criterion, histogram_weights


def corrects(pred, y):
    return torch.sum(torch.argmax(pred, -1) == y).item()


total = lambda pred, y: len(y)

mse_loss = nn.MSELoss()


def accuracy(pred, y):
    return corrects(pred, y) / len(y)


def CEL_weights_class_balanced(samples_per_class, num_classes, beta=0.999):
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_classes
    weights = torch.tensor(weights).float()
    return weights


def CrossEntropyLoss_class_balanced(histogram_weights, num_classes):
    histogram_weights = CEL_weights_class_balanced(histogram_weights, num_classes)
    return torch.nn.CrossEntropyLoss(
        histogram_weights,
        reduction="mean",
        label_smoothing=0.1,
    )


def NLLLoss_class_balanced(histogram_weights, num_classes):
    histogram_weights = CEL_weights_class_balanced(histogram_weights, num_classes)
    return torch.nn.NLLLoss(
        histogram_weights,
        reduction="mean",
        label_smoothing=0.1,
    )
