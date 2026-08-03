import math
import torch
import torch.nn as nn

# Positional Encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_length=500):

        super().__init__()

        pe = torch.zeros(max_length, d_model)

        position = torch.arange(
            0,
            max_length,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return x


# Transformer Model
class RSKTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1
    ):

        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )
        # Positional Encoding
        self.position = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output layer
        self.output = nn.Linear(
            d_model,
            vocab_size
        )

    # Decoder mask
    def generate_square_subsequent_mask(self, size):

        return torch.triu(
            torch.full(
                (size, size),
                float("-inf")
            ),
            diagonal=1
        )

    # Forward pass
    def forward(
        self,
        encoder_input,
        decoder_input
    ):
        # Embeddings
        encoder = self.embedding(encoder_input)

        decoder = self.embedding(decoder_input)

        # Positional Encoding
        encoder = self.position(encoder)

        decoder = self.position(decoder)

        # Decoder mask
        tgt_mask = self.generate_square_subsequent_mask(
            decoder.size(1)
        ).to(decoder.device)

        # Transformer
        output = self.transformer(
            src=encoder,
            tgt=decoder,
            tgt_mask=tgt_mask
        )

        # Predict next token
        output = self.output(output)

        return output

    # Generate a sequence
    @torch.no_grad()
    def generate(
        self,
        encoder_input,
        bos_token,
        eos_token,
        max_length=100
    ):
        self.eval()
        device = encoder_input.device

        # Start with <BOS>
        decoder_input = torch.full(
            (encoder_input.size(0), 1),
            bos_token,
            dtype=torch.long,
            device=device
        )

        for _ in range(max_length):
            prediction = self.forward(
                encoder_input,
                decoder_input
            )
            # Next predicted token
            next_token = prediction[:, -1, :].argmax(dim=-1)
            # Append prediction
            decoder_input = torch.cat(
                [
                    decoder_input,
                    next_token.unsqueeze(1)
                ],
                dim=1
            )
            # Stop if every sequence predicted <EOS>
            if torch.all(next_token == eos_token):
                break

        return decoder_input


# Example
if __name__ == "__main__":

    vocab_size = 20

    model = RSKTransformer(vocab_size)

    encoder = torch.randint(
        0,
        vocab_size,
        (4, 8)
    )

    decoder = torch.randint(
        0,
        vocab_size,
        (4, 12)
    )

    output = model(
        encoder,
        decoder
    )

    print(output.shape)