import csv
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tokenizer import MathTokenizer


class MathDataset(Dataset):

    def __init__(self, csv_file):

        self.tokenizer = MathTokenizer()
        self.samples = []

        with open(csv_file, newline="") as f:

            reader = csv.DictReader(f)

            for row in reader:

                input_ids = self.tokenizer.tokenize(
                    row["input"]
                )

                target_ids = self.tokenizer.tokenize(
                    row["target"]
                )

                self.samples.append(
                    {
                        "input_ids": input_ids,
                        "target_ids": target_ids
                    }
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        input_ids = sample["input_ids"]
        target_ids = sample["target_ids"]

        bos = self.tokenizer.vocab["[BOS]"]
        eos = self.tokenizer.vocab["[EOS]"]

        decoder_input = [bos] + target_ids
        labels = target_ids + [eos]

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),
            "decoder_input_ids": torch.tensor(
                decoder_input,
                dtype=torch.long
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long
            )
        }


def collate_fn(batch):

    input_ids = pad_sequence(
        [x["input_ids"] for x in batch],
        batch_first=True,
        padding_value=0
    )

    decoder_input_ids = pad_sequence(
        [x["decoder_input_ids"] for x in batch],
        batch_first=True,
        padding_value=0
    )

    labels = pad_sequence(
        [x["labels"] for x in batch],
        batch_first=True,
        padding_value=0
    )

    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels
    }