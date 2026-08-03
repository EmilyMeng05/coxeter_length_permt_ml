import ast
import pandas as pd
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from coxeter_length_transformer.tokenizer import RSKTokenizer


class RSKDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)
        self.tokenizer = RSKTokenizer()
        self.tokenizer.build_vocab(csv_file)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        permutation = row["permutation"]
        P = row["P"]
        Q = row["Q"]

        # Encode the permutation
        encoder_input = self.tokenizer.encode_permutation(permutation)

        # Encode the target sequence
        target = self.tokenizer.encode_target(P, Q)

        # Teacher forcing
        # shift the index by one
        decoder_input = target[:-1]

        target_output = target[1:]

        return (
            torch.tensor(encoder_input, dtype=torch.long),
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(target_output, dtype=torch.long)
        )


# Pad a batch
def collate_fn(batch):

    encoder_inputs = []
    decoder_inputs = []
    target_outputs = []

    for encoder, decoder, target in batch:

        encoder_inputs.append(encoder)
        decoder_inputs.append(decoder)
        target_outputs.append(target)

    encoder_inputs = pad_sequence(
        encoder_inputs,
        batch_first=True,
        padding_value=0
    )

    decoder_inputs = pad_sequence(
        decoder_inputs,
        batch_first=True,
        padding_value=0
    )

    target_outputs = pad_sequence(
        target_outputs,
        batch_first=True,
        padding_value=0
    )

    return (
        encoder_inputs,
        decoder_inputs,
        target_outputs
    )

# Example
if __name__ == "__main__":

    dataset = RSKDataset("rsk_data.csv")

    encoder, decoder, target = dataset[0]

    print("Encoder input")
    print(encoder)

    print()

    print("Decoder input")
    print(decoder)

    print()

    print("Target output")
    print(target)