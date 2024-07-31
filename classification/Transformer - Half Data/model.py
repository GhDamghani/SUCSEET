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
        d_model,
        num_heads,
        dim_feedforward,
        num_layers,
        num_classes,
        window_size,
        feat_size,
        output_indices=[-1],
        dropout=0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.feat_size = feat_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.output_indices = output_indices

        prenet_layers = 1
        prenet_nodes = np.geomspace(feat_size, d_model, num=prenet_layers + 1).astype(
            int
        )
        prenet_list = []
        for i in range(prenet_layers):
            prenet_list.append(nn.Linear(prenet_nodes[i], prenet_nodes[i + 1]))
            prenet_list.append(nn.ReLU())
            prenet_list.append(nn.BatchNorm1d(window_size))

        self.prenet = nn.Sequential(*prenet_list)
        self.encoder = EncoderModel(
            d_model, num_heads, dropout, num_layers, dim_feedforward, pos=False
        )
        self.clf_head = nn.Sequential(
            nn.Linear(d_model, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x):
        x = self.prenet(x)
        x = self.encoder(x)
        x = x[:, self.output_indices, :]
        return self.clf_head(x).squeeze(1).reshape(-1, self.num_classes)

    def __str__(self, batch_size=1):
        return summary(
            self,
            input_size=(batch_size, self.window_size, self.feat_size),
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
        return torch.nn.functional.softmax(x, -1)


class EncoderPrenetMLP(nn.Module):
    def __init__(self, d_model, window_size, feat_size):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(window_size, d_model * 8),
            nn.ReLU(),
            nn.BatchNorm1d(feat_size),
            nn.Linear(d_model * 8, d_model * 8),
            nn.ReLU(),
            nn.Linear(d_model * 8, d_model),
            nn.BatchNorm1d(feat_size),
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
    feat_size = 127
    dropout_prenet = 0
    dropout_encoder = 0
    model = SpeechDecodingModel_clf(
        d_model,
        num_heads,
        dim_feedforward,
        num_layers,
        num_classes,
        window_size,
        feat_size,
        dropout_prenet,
        dropout_encoder,
    )
    batch_size = 16
    print(model.__str__(batch_size))
