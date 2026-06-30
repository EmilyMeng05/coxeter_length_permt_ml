import torch
import torch.nn as nn
from torch.utils.data import Dataset
import ast


class PermutationDataset(Dataset):
    def __init__(self, df):
        perms = df["permutation"]
        if isinstance(perms.iloc[0], str):
            perms = perms.apply(ast.literal_eval)
        self.perms = perms.tolist()
        self.lengths_target = df["coxeter_length"].tolist()

    def __len__(self):
        return len(self.perms)

    def __getitem__(self, idx):
        x = torch.tensor(self.perms[idx], dtype=torch.long)
        y = torch.tensor(self.lengths_target[idx], dtype=torch.float)
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    lens = [len(x) for x in xs]
    maxlen = max(lens)
    padded = torch.zeros(len(xs), maxlen, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, :len(x)] = x
    return padded, torch.tensor(lens), torch.stack(ys)


class LengthPredictor(nn.Module):
    def __init__(self, vocab_size=8, d_model=32, hidden=64):
        # vocab_size=8 covers values 1..7 + 0 for padding,
        # even though training data only contains values 1..6
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.LSTM(d_model, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, lengths):
        e = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.rnn(packed)
        return self.head(h[-1]).squeeze(-1)