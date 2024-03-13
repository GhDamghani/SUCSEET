import torch
import torch.nn as nn
import torch.nn.functional as F
import model_layers as layers
from torchinfo import summary
import numpy as np


def flatten_batches(x):
    return x.reshape(x.shape[0], -1)


def get_all_batches(dataset, flatten=False):
    X = list()
    y = list()
    res = list()
    for x, y1, res1 in dataset:
        X.append(x.cpu())
        y.append(y1)
        res.append(res1)
    X = torch.cat(X).numpy()
    y = torch.cat(y).numpy()
    res = torch.cat(res).numpy()
    if flatten:
        X = flatten_batches(X)
    return X, y, res


def get_QDA_accuracy(train_dataset, test_dataset):
    pred_QDA_test = test_dataset.logits_residual[
        np.array(test_dataset.indices) + test_dataset.timepoints - 1
    ]
    pred_QDA_train = train_dataset.logits_residual[
        np.array(train_dataset.indices) + train_dataset.timepoints - 1
    ]

    y_test = test_dataset.cluster[
        np.array(test_dataset.indices) + test_dataset.timepoints - 1
    ]
    y_train = train_dataset.cluster[
        np.array(train_dataset.indices) + train_dataset.timepoints - 1
    ]

    corrects_test = np.sum(y_test == np.argmax(pred_QDA_test, -1))
    corrects_train = np.sum(y_train == np.argmax(pred_QDA_train, -1))

    acc = corrects_test / y_test.size
    acc_train = corrects_train / y_train.size
    return acc, acc_train


class SpeechDecodingModel(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dim_feedforward,
        num_layers,
        num_classes,
        timepoints,
        num_eeg_channels,
        dropout_prenet,
        dropout_encoder,
        dropout_clf,
    ):
        super().__init__()
        self.timepoints = timepoints
        self.num_eeg_channels = num_eeg_channels
        self.num_classes = num_classes
        self.d_model = d_model

        self.encoder_prenet_mlp = nn.Sequential(
            # nn.Dropout(dropout_prenet),
            # nn.Linear(num_eeg_channels, num_eeg_channels),
            # nn.BatchNorm1d(timepoints),
            # nn.ReLU(),
            # nn.Linear(num_eeg_channels, num_eeg_channels),
            # nn.BatchNorm1d(timepoints),
            # nn.ReLU(),
            nn.Linear(num_eeg_channels, d_model),
            # nn.BatchNorm1d(timepoints),
            nn.ReLU(),
        )
        self.encoder = layers.EncoderModel(
            d_model, num_heads, dropout_encoder, num_layers, dim_feedforward, pos=False
        )
        self.clf = nn.Sequential(
            # nn.Dropout(dropout_clf),
            # nn.Linear(d_model, d_model),
            # nn.BatchNorm1d(1),
            # nn.ReLU(),
            nn.Linear(d_model, num_classes - 1),
        )

    def forward(self, x):
        x = self.encoder_prenet_mlp(x)
        x = self.encoder(x)
        x = x[:, -1:, :]
        return self.clf(x).squeeze(1)

    def __str__(self, batch_size=1):
        return summary(
            self,
            input_size=(batch_size, self.timepoints, self.num_eeg_channels),
            verbose=0,
            depth=4,
        ).__repr__()


if __name__ == "__main__":

    d_model = 128
    num_heads = 4
    dim_feedforward = 256
    num_layers = 2
    num_classes = 20
    timepoints = 96
    num_eeg_channels = 127
    dropout_prenet = 0
    dropout_encoder = 0
    model = SpeechDecodingModel(
        d_model,
        num_heads,
        dim_feedforward,
        num_layers,
        num_classes,
        timepoints,
        num_eeg_channels,
        dropout_prenet,
        dropout_encoder,
    )
    batch_size = 16
    print(model.__str__(batch_size))
