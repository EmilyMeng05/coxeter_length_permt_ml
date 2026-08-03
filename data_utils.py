import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tokenizer import MathTokenizer


class PermutationDataset(Dataset):

    def __init__(
        self,
        csv_file,
        task="LENGTH",
        use_one_line=True,
        use_cycle=True,
        use_lehmer=True,
        use_inversion=True
    ):

        self.data = pd.read_csv(csv_file)
        self.tokenizer = MathTokenizer()
        self.task = task
        self.use_one_line = use_one_line
        self.use_cycle = use_cycle
        self.use_lehmer = use_lehmer
        self.use_inversion = use_inversion


    # return lens
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        row = self.data.iloc[index]
        # Encode the input sequence
        input_ids = self.tokenizer.encode_input(

            one_line=row["one_line"],
            cycle=row["cycle"],
            lehmer=row["lehmer"],
            inversion=row["inversion_vector"],

            task=self.task,
            use_one_line=self.use_one_line,
            use_cycle=self.use_cycle,
            use_lehmer=self.use_lehmer,
            use_inversion=self.use_inversion
        )

        # Encode the target sequence
        target = self.tokenizer.encode_target(
            row["coxeter_length"]
        )

        # Teacher forcing
        decoder_input_ids = target[:-1]
        labels = target[1:]

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),

            "decoder_input_ids": torch.tensor(
                decoder_input_ids,
                dtype=torch.long
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.long
            )
        }


# Pad a batch
def collate_fn(batch):

    input_ids = [
        item["input_ids"]
        for item in batch
    ]

    decoder_input_ids = [
        item["decoder_input_ids"]
        for item in batch
    ]

    labels = [
        item["labels"]
        for item in batch
    ]

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0

    )

    decoder_input_ids = pad_sequence(
        decoder_input_ids,
        batch_first=True,
        padding_value=0

    )

    # what the decoder should predict
    labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=0

    )

    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels

    }


# Example
if __name__ == "__main__":

    dataset = PermutationDataset(
        "multi_encoding_data.csv",
        task="LENGTH",
        use_one_line=True,
        use_cycle=True,
        use_lehmer=True,
        use_inversion=True

    )