import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, seq_length_in, seq_length_out, num_heads=1, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()

        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=seq_length_in,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout = dropout
        )

        # Fully connected layer with softmax
        self.fc_softmax = nn.Linear(seq_length_in, seq_length_out)

    def forward(self, x):

        # Transformer forward pass
        x = self.transformer(x, x)  # Pass the same sequence as src and tgt

        # Apply softmax to each vector in the output sequence
        x = self.fc_softmax(x)
        x = nn.functional.softmax(x, dim=-1)

        return x



if __name__ == '__main__':
    # Example usage
    seq_no = 100
    batch_size = 16
    seq_length_in = 127
    seq_length_out = 20

    model = TransformerModel(seq_length_in, seq_length_out)
    print(model)
    input_sequence = torch.rand((batch_size, seq_no, seq_length_in))
    output_sequence = model(input_sequence)
    print(input_sequence.shape, output_sequence.shape)
