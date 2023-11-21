import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, d_ff, num_classes, dropout):
        super(TransformerEncoder, self).__init__()
        self.Encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout), num_layers)
        self.classifier_head = nn.Sequential(
           nn.Linear(d_model, num_classes),
           nn.Softmax(dim=1),
       )

    def forward(self, x):
        ba, ch, sa = x.shape
        # x = torch.reshape(x, (ba, sa, ch))
        # x = torch.transpose(x, 1, 2)
        x = self.Encoder(x)
        # torch.reshape(x, (ba, ch, sa))

        # Global average pooling
        x = x.mean(dim=1)
        
        x = self.classifier_head(x)
        return x
