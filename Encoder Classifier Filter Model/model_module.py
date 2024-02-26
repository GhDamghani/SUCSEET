import torch
import torch.nn as nn
import torch.nn.functional as F
import model_layers as layers
from torchinfo import summary
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_LDA_accuracy(train_dataset, test_dataset):
    X_train = train_dataset.feat[train_dataset.indices]
    y_train = train_dataset.cluster[train_dataset.indices]
    X_test = test_dataset.feat[test_dataset.indices]
    y_test = test_dataset.cluster[test_dataset.indices]
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    return acc


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
        self.clf_factor = 2

        self.encoder_prenet_mlp = nn.Sequential(
            nn.Dropout(dropout_prenet),
            nn.Linear(timepoints, d_model * 8),
            nn.ReLU(),
            nn.BatchNorm1d(num_eeg_channels),
            nn.Linear(d_model * 8, d_model * 8),
            nn.ReLU(),
            nn.Linear(d_model * 8, d_model),
            nn.BatchNorm1d(num_eeg_channels),
        )
        self.encoder = layers.EncoderModel(
            d_model, num_heads, dropout_encoder, num_layers, dim_feedforward
        )
        self.channel_pool = nn.Sequential(
            nn.Dropout(dropout_clf),
            nn.Linear(num_eeg_channels, num_eeg_channels * 16),
            nn.ReLU(),
            nn.Linear(num_eeg_channels * 16, self.clf_factor),
            nn.BatchNorm1d(d_model),
        )
        self.clf = nn.Sequential(
            nn.Dropout(dropout_clf),
            nn.Linear(d_model * self.clf_factor, d_model * 8 * self.clf_factor),
            nn.ReLU(),
            nn.Linear(d_model * 8 * self.clf_factor, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder_prenet_mlp(x)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.channel_pool(x).view(-1, self.d_model * self.clf_factor)
        return self.clf(x)

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
