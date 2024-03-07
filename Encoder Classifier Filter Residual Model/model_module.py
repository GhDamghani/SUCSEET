import torch
import torch.nn as nn
import torch.nn.functional as F
import model_layers as layers
from torchinfo import summary
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


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


def get_LDA_accuracy(train_dataset, test_dataset):
    X_train, y_train, res_train = get_all_batches(train_dataset, True)
    X_test, y_test, res_test = get_all_batches(test_dataset, True)
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    acc_train = clf.score(X_train, y_train)
    return acc, acc_train


def get_QDA_accuracy(train_dataset, test_dataset):
    X_train, y_train, res_train = get_all_batches(train_dataset, True)
    X_test, y_test, res_test = get_all_batches(test_dataset, True)
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    acc_train = clf.score(X_train, y_train)
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
            nn.Dropout(dropout_prenet),
            nn.Linear(num_eeg_channels, num_eeg_channels),
            nn.BatchNorm1d(timepoints),
            nn.ReLU(),
            nn.Linear(num_eeg_channels, d_model),
            nn.BatchNorm1d(timepoints),
            nn.ReLU(),
        )
        self.encoder = layers.EncoderModel(
            d_model, num_heads, dropout_encoder, num_layers, dim_feedforward, pos=True
        )
        self.clf = nn.Sequential(
            nn.Dropout(dropout_clf),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )
        self.scale_factor = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self.encoder_prenet_mlp(x)
        x = self.encoder(x)
        x = x[:, -1:, :]
        return self.clf(x).squeeze(1) * self.scale_factor

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
