import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, num_classes, dropout=0.5):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, mask=None):
        attention_output, _ = self.multi_head_attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(attention_output)
        x = self.layer_norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, d_ff, num_classes, dropout=0.5):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_head, d_ff, num_classes, dropout) for _ in range(num_layers)])
        self.classifier_head = nn.Sequential(
           nn.Linear(d_model, d_model),
           nn.LeakyReLU(),
           nn.Dropout(dropout),
           nn.Linear(d_model, d_model),
           nn.LeakyReLU(),
           nn.Linear(d_model, num_classes),
           nn.Softmax(dim=1),
       )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        # Global average pooling
        x = x.mean(dim=1)
        x = self.classifier_head(x)
        return x

if __name__ == '__main__':
    # Example usage
    num_layers = 8
    d_model = 100
    n_head = 10
    d_ff = 100
    dropout = 0.5

    transformer_encoder = TransformerEncoder(num_layers, d_model, n_head, d_ff, dropout)
    print(transformer_encoder)
