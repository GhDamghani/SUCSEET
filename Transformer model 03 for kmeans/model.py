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
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_model: int = 256, num_heads: int = 16, dim_feedforward: int = 2048, dropout: int = 0.1, num_layers: int = 6):
        super(TransformerModel, self).__init__()

        

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers)

        # self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.lin = nn.Linear(d_model*(ntokens-1), d_out)
        self.emb_in = nn.Linear(d_in, d_model)
        self.emb_out = nn.Linear(d_out, d_model)

    def forward(self, x1, x2):

        x1 = self.emb_in(x1)
        # x1 = self.pos_encoder(x1)
        x1 = self.transformer_encoder(x1)

        x2 = self.emb_out(x2)
        # x2 = self.pos_encoder(x2)
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(x2.size(1), device=device)
        
        x = self.transformer_decoder(x2, x1) # , tgt_mask=tgt_mask
        # x = x[:, -1:, :]
        x = x.view(x.size(0), -1)
        x = self.lin(x)

        # Apply softmax to each vector in the output sequence
        x = nn.functional.softmax(x, dim=-1)

        return x


if __name__ == '__main__':
    # Example usage
    tokens_no = 100
    batch_size = 16
    num_classes = 20

    d_model = 192
    d_in = 127
    d_out = num_classes
    num_heads = d_model//16
    dropout = 0.2
    num_layers = 4
    dim_feedforward = d_model*4

    model = TransformerModel(d_in, d_out, d_model, num_heads, dim_feedforward, dropout, num_layers).to(device)
    print(model)
    print('Total number of trainable parameters:', sum(p.numel()
          for p in model.parameters() if p.requires_grad))
    X1 = torch.rand((batch_size, tokens_no, d_in)).to(device)
    X2 = torch.rand((batch_size, tokens_no-1, d_out)).to(device)
    output_sequence = model(X1, X2)
    print(X1.shape, output_sequence.shape)
