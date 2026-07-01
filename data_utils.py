# This file contains how I choose to process the data (padd with 0)
# and the Model I used
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import ast


# converts csv lists into python list into torth tensor
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

# padd 0s to permutation with different lengths
# so they can be in the same length
def collate_fn(batch):
    xs, ys = zip(*batch)
    lens = [len(x) for x in xs]
    # here it will be 6
    maxlen = max(lens)
    padded = torch.zeros(len(xs), maxlen, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, :len(x)] = x
        # also record the original length of the permutation
    return padded, torch.tensor(lens), torch.stack(ys)

# Neural Network Model 
class LengthPredictor(nn.Module):
    # vocab size = 8 bc 
    # all the possible integers are 0,...,7
    def __init__(self, vocab_size=8, d_model=32, hidden=64):
        super().__init__()
        # creates table with integer (0,..,7) to the learned vector
        self.embed = nn.Embedding(vocab_size, d_model)
        # every time the model reads an integer, it will update its memory
        self.rnn = nn.LSTM(d_model, hidden, batch_first=True)
        # predicting stage
        # goes from 64 ->32 -> |32| -> 1
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, lengths):
        # change the permutations into (batch_size, sequence_length, 32)
        e = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.rnn(packed)
        # only curious to the predicted length
        return self.head(h[-1]).squeeze(-1)