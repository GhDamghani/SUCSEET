import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


class EncoderPrenet(nn.Module):
    def __init__(self, d_model, num_layers, timepoints):
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
        self.linear = nn.Linear(timepoints // (2**num_layers), 1, bias=False)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = x.unsqueeze(1)
        x = self.layers(x)
        x = self.linear(x).squeeze(-1)
        x = torch.permute(x, (0, 2, 1))
        return x


class EncoderPrenetMLP(nn.Module):
    def __init__(self, d_model, timepoints, num_eeg_channels):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(timepoints, d_model * 8),
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
