import torch
import torch.nn as nn


class MathTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.output = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(
        self,
        src,
        tgt
    ):

        src = self.embedding(src)
        tgt = self.embedding(tgt)

        out = self.transformer(
            src,
            tgt
        )

        return self.output(out)