import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


class EncoderPrenet(nn.Module):
    def __init__(self, channels=(32, 64, 128)):
        super().__init__()
        kernel_size = (1, 4)
        stride = (1, 2)
        padding = (0, 1)
        nn.Conv2d = partial(
            nn.Conv2d, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.layers = nn.Sequential(
            nn.Conv2d(1, channels[0]),
            nn.ReLU(),
            nn.BatchNorm2d(channels[0]),
            nn.Conv2d(channels[0], channels[1]),
            nn.ReLU(),
            nn.BatchNorm2d(channels[1]),
            nn.Conv2d(channels[1], channels[2]),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderPrenetMLP(nn.Module):
    def __init__(self, num_eeg_channels, out_d, timepoints, hidden_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_eeg_channels, out_d),
            nn.BatchNorm1d(timepoints),
            nn.LeakyReLU(),
        )

    def forward(self, x):
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
    def __init__(self, d_model, num_heads, dropout, num_layers, dim_feedforward):
        super(EncoderModel, self).__init__()

        self.pe = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.pe(x)
        x = self.encoder(x)
        return x
