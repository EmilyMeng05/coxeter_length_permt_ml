import torch
import torch.nn as nn

class MathTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        num_layers=4,
        max_length=512
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            max_length,
            d_model
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(
        self,
        src,
        tgt
    ):
        batch_size = src.size(0)

        # Embed input sequence
        src_positions = torch.arange(
            src.size(1),
            device=src.device
        )

        tgt_positions = torch.arange(
            tgt.size(1),
            device=tgt.device
        )

        src_embedding = (
            self.embedding(src)
            +
            self.position_embedding(
                src_positions
            )
        )

        tgt_embedding = (
            self.embedding(tgt)
            +
            self.position_embedding(
                tgt_positions
            )
        )

        # Causal mask for autoregression
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(tgt.device)

        output = self.decoder(
            tgt_embedding,
            src_embedding,
            tgt_mask=tgt_mask
        )

        logits = self.output_layer(
            output
        )

        return logits