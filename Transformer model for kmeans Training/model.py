import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from functools import partial


class EncoderPrenet(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        channels = (32, 64, output_dim)
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

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = torch.permute(x, (0, 1, 3, 2))
        x = self.layers(x)
        x = torch.mean(x, -1)  # Averaging across time axis
        x = torch.permute(x, (0, 2, 1))
        return x


class DecoderPrenet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(
            self.layers[0].weight.device
        )  # Move input tensor to the device of the layers
        return self.layers(x)


class ScaledPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(ScaledPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale_param = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.scale_param * self.pe[: x.size(0), :]
        return self.dropout(x)


class EncoderModel(nn.Module):
    def __init__(self, d_model, num_heads, dropout, num_layers, dim_feedforward):
        super(EncoderModel, self).__init__()

        self.encoder_prenet = EncoderPrenet(d_model)
        self.scaled_positional_encoding = ScaledPositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.encoder_prenet(x)
        x = self.scaled_positional_encoding(x)
        x = self.encoder(x)
        return x


class DecoderModel(nn.Module):
    def __init__(
        self,
        d_model,
        num_classes,
        num_heads,
        dropout,
        num_layers,
        dim_feedforward,
        max_len=5000,
    ):
        super(DecoderModel, self).__init__()

        self.decoder_prenet = DecoderPrenet(num_classes, d_model)
        self.scaled_positional_encoding = ScaledPositionalEncoding(
            d_model, dropout, max_len
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.linear = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=-1)

        self.register_buffer(
            "mask", nn.Transformer.generate_square_subsequent_mask(d_model)
        )

    def forward(self, x, memory):
        x = self.decoder_prenet(x)
        x = self.scaled_positional_encoding(x)
        x = self.decoder(x, memory, memory_mask=self.mask, memory_is_causal=True)
        x = self.linear(x[:, -1:, :])
        x = F.sigmoid(x)
        x = self.softmax(x)
        return x


class SpeechDecodingModel(nn.Module):
    def __init__(
        self,
        d_model,
        num_classes,
        num_heads,
        dropout,
        num_layers,
        dim_feedforward,
        max_len=5000,
    ):
        super(SpeechDecodingModel, self).__init__()
        self.encoder = EncoderModel(
            d_model, num_heads, dropout, num_layers, dim_feedforward
        )
        self.decoder = DecoderModel(
            d_model,
            num_classes,
            num_heads,
            dropout,
            num_layers,
            dim_feedforward,
            max_len,
        )
        self.num_classes = num_classes

    def forward(self, x, init_class=0):
        timepoints = x.size(1)
        memory = self.encoder(x)
        audio = (
            (
                torch.zeros((x.size(0), 1, self.num_classes), dtype=torch.int64)
                + init_class
            )
            .to(dtype=torch.float32)
            .to(x.device)
        )
        for i in range(timepoints):
            x = self.decoder(audio, memory)
            audio_ = audio.clone()
            audio = torch.cat((audio, x), dim=1)
            del audio_
        return audio[:, 1:, :]


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Example usage
    batch_size = 16
    num_classes = 20

    d_model = 128
    num_heads = 4
    dim_feedforward = 256
    num_layers = 3
    dropout = 0.1

    model = SpeechDecodingModel(
        d_model, num_classes, num_heads, dropout, num_layers, dim_feedforward
    ).to(device)

    print(model)
    print(
        "Total number of trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    no_channels = 127
    no_timepoints = 96
    X = torch.rand((batch_size, no_timepoints, no_channels), dtype=torch.float32).to(
        device
    )
    y = model(X)
    print(X.shape, y.shape)
    pass
