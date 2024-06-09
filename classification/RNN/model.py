import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from functools import partial
import math
import numpy as np


class SpeechDecodingModel_clf(nn.Module):
    def __init__(
        self,
        feat_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            feat_size,
            hidden_size,
            num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout,
        )
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.rnn(x)
        x = x[0][:, -1, :]
        return self.clf_head(x)

    def __str__(self, batch_size=1, window_size=40, feat_size=127):
        return summary(
            self,
            input_size=(batch_size, window_size, feat_size),
            verbose=0,
            depth=4,
        ).__repr__()


if __name__ == "__main__":

    feat_size = 127
    hidden_size = 128
    num_layers = 1
    num_classes = 5
    window_size = 40
    model = SpeechDecodingModel_clf(
        feat_size,
        hidden_size,
        num_layers,
        num_classes,
    )
    batch_size = 16

    input_arr = np.random.randn(batch_size, window_size, feat_size).astype(np.float32)
    model(torch.from_numpy(input_arr))
