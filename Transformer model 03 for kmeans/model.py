import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

ntokens = 100
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, d_model, d_out, num_heads=1, num_layers=6, dropout=0.1, dim_feedforward=None):
        super(TransformerModel, self).__init__()
        if dim_feedforward == None:
            dim_feedforward = 4*d_model

        # self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer =  nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer =  nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Fully connected layer with softmax
        self.lin = nn.Linear(d_model*(ntokens-1), d_out)
        self.emb = nn.Linear(d_out, d_model)

    def forward(self, x1, x2):
        
        # x = self.pos_encoder(x)
        x2 = self.emb(x2)
        x1 = self.transformer_encoder(x1)
        x = self.transformer_decoder(x2, x1)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)

        # Apply softmax to each vector in the output sequence
        x = nn.functional.softmax(x, dim=-1)

        return x


if __name__ == '__main__':
    # Example usage
    tokens_no = 100
    batch_size = 16
    d_model = 128
    d_out = 20

    model = TransformerModel(d_model, d_out).to(device)
    print(model)
    print('Total number of trainable parameters:', sum(p.numel()
          for p in model.parameters() if p.requires_grad))
    input_sequence_1 = torch.rand((batch_size, tokens_no, d_model)).to(device)
    input_sequence_2 = torch.rand((batch_size, tokens_no-1, d_out)).to(device)
    output_sequence = model(input_sequence_1, input_sequence_2)
    print(input_sequence_1.shape, output_sequence.shape)
