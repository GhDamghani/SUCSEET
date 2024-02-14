import torch.nn as nn
import torch.nn.functional as F
from .layers import EncoderPrenetMLP, EncoderModel
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
        encoder_prenet_out_d,
        d_model,
        num_heads,
        dim_feedforward,
        num_classes,
        dropout,
        enc_hidden_dim,
        timepoints,
        num_eeg_channels,
    ):
        super().__init__()
        self.timepoints = timepoints
        self.num_eeg_channels = num_eeg_channels
        self.num_classes = num_classes

        self.encoder_prenet_mlp = EncoderPrenetMLP(
            num_eeg_channels, encoder_prenet_out_d, timepoints, enc_hidden_dim, dropout
        )
        # self.pe = ScaledPositionalEncoding(d_model, dropout)
        self.encoder = EncoderModel(d_model, num_heads, dropout, 1, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Linear(encoder_prenet_out_d, num_classes)

    def forward(self, x):
        x = self.encoder_prenet_mlp(x)
        # x = self.pe(x)
        x = self.encoder(x)
        x = self.dropout(x)
        return self.clf(x)

    def __str__(self, batch_size=1):
        return summary(
            self,
            input_size=(batch_size, self.timepoints, self.num_eeg_channels),
            verbose=0,
        ).__repr__()


if __name__ == "__main__":

    encoder_prenet_out_d = 128
    num_classes = 20
    dropout = 0.1
    enc_hidden_dim = 256
    timepoints = 96
    num_eeg_channels = 127
    model = SpeechDecodingModel(
        encoder_prenet_out_d,
        num_classes,
        dropout,
        enc_hidden_dim,
        timepoints,
        num_eeg_channels,
    )
    batch_size = 16
    model.summary(batch_size)
