import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from functools import partial
import math
import numpy as np


class SpeechDecodingModel_reg(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        window_size,
        num_eeg_channels,
        output_size,
        dropout=0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_eeg_channels = num_eeg_channels

        self.encoder_prenet_mlp = nn.Linear(num_eeg_channels, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.regressor = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.encoder_prenet_mlp(x)
        x = self.transformer_encoder(x)
        x = self.regressor(x)
        return x

    def __str__(self, batch_size=1):
        return summary(
            self,
            input_size=(batch_size, self.window_size, self.num_eeg_channels),
            verbose=0,
            depth=4,
        ).__repr__()


class SpeechDecodingModel_clf(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dim_feedforward,
        num_layers,
        num_classes,
        window_size,
        num_eeg_channels,
        dropout=0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_eeg_channels = num_eeg_channels
        self.num_classes = num_classes
        self.d_model = d_model

        self.encoder_prenet_mlp = nn.Sequential(
            nn.Linear(num_eeg_channels, num_eeg_channels),
            nn.BatchNorm1d(window_size),
            nn.ReLU(),
            nn.Linear(num_eeg_channels, d_model),
            nn.BatchNorm1d(window_size),
            nn.ReLU(),
        )
        self.encoder = EncoderModel(
            d_model, num_heads, dropout, num_layers, dim_feedforward, pos=False
        )
        self.clf = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
            nn.Softmax(-1),  # nn.Softmax(-1),
        )
        self.scale_factor = nn.Parameter(torch.ones(1) * 5, requires_grad=True)

    def forward(self, x):
        x = self.encoder_prenet_mlp(x)
        x = self.encoder(x)
        x = x[:, -1:, :]
        return self.clf(x).squeeze(1) * self.scale_factor

    def __str__(self, batch_size=1):
        return summary(
            self,
            input_size=(batch_size, self.window_size, self.num_eeg_channels),
            verbose=0,
            depth=4,
        ).__repr__()


class EncoderPrenet(nn.Module):
    def __init__(self, d_model, num_layers, window_size):
        super().__init__()
        kernel_size = (1, 4)
        stride = (1, 2)
        padding = (0, 1)
        nn.Conv2d = partial(
            nn.Conv2d, kernel_size=kernel_size, stride=stride, padding=padding
        )
        channels = []
        for i in range(num_layers):
            channels.insert(0, d_model // (2**i))
        channels.insert(0, 1)
        layers_list = []
        for i in range(num_layers):
            layers_list.append(nn.Conv2d(channels[i], channels[i + 1]))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.BatchNorm2d(channels[i + 1]))

        self.layers = nn.Sequential(*layers_list)
        self.linear = nn.Linear(window_size // (2**num_layers), 1, bias=False)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = x.unsqueeze(1)
        x = self.layers(x)
        x = self.linear(x).squeeze(-1)
        x = torch.permute(x, (0, 2, 1))
        return x


class EncoderPrenetMLP(nn.Module):
    def __init__(self, d_model, window_size, num_eeg_channels):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(window_size, d_model * 8),
            nn.ReLU(),
            nn.BatchNorm1d(num_eeg_channels),
            nn.Linear(d_model * 8, d_model * 8),
            nn.ReLU(),
            nn.Linear(d_model * 8, d_model),
            nn.BatchNorm1d(num_eeg_channels),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class EncoderModel(nn.Module):
    def __init__(
        self, d_model, num_heads, dropout, num_layers, dim_feedforward, pos=False
    ):
        super(EncoderModel, self).__init__()

        if pos:
            self.pe = PositionalEncoding(d_model, dropout)
        self.pos = pos

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        if self.pos:
            x = self.pe(x)
        x = self.encoder(x)
        return x


if __name__ == "__main__":

    d_model = 128
    num_heads = 4
    dim_feedforward = 256
    num_layers = 2
    num_classes = 20
    window_size = 96
    num_eeg_channels = 127
    dropout_prenet = 0
    dropout_encoder = 0
    model = SpeechDecodingModel_clf(
        d_model,
        num_heads,
        dim_feedforward,
        num_layers,
        num_classes,
        window_size,
        num_eeg_channels,
        dropout_prenet,
        dropout_encoder,
    )
    batch_size = 16
    print(model.__str__(batch_size))
