# This file contains:
# 1. How I process permutation data
# 2. The LSTM model used to predict Coxeter length

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import ast


# Converts CSV permutation strings into tensors
class PermutationDataset(Dataset):

    def __init__(self, df):
        perms = df["permutation"]
        # CSV stores lists as strings, so convert them back to Python lists
        if isinstance(perms.iloc[0], str):
            perms = perms.apply(ast.literal_eval)
        self.perms = perms.tolist()
        # Target value: Coxeter length
        self.lengths_target = df["coxeter_length"].tolist()

    def __len__(self):
        return len(self.perms)

    def __getitem__(self, idx):
        # Example:
        # permutation = [3,1,4,2]
        perm = torch.tensor(self.perms[idx], dtype=torch.long)
        # Check that permutation values are valid
        if (perm <= 0).any():
            print("Found bad permutation!")
            print("Index:", idx)
            print("Permutation:", perm)
            raise ValueError("Permutation contains non-positive values")
        # Convert permutation into one-hot representation
        #
        # Example:
        # 1 -> [1,0,0,0,0,0,0]
        # 3 -> [0,0,1,0,0,0,0]
        #
        # num_classes=7 because we want to support S7 later
        x = F.one_hot(
            perm - 1,
            num_classes=7
        ).float()
        # Coxeter length target
        y = torch.tensor(
            self.lengths_target[idx],
            dtype=torch.float
        )
        return x, y

# Pads permutations so they have the same sequence length
#
# Input:
# [
#   permutation length 4,
#   permutation length 6
# ]
#
# Output:
# [
#   length 7 padded permutation,
#   length 7 padded permutation
# ]
#
# Also returns original lengths so LSTM can ignore padding
def collate_fn(batch):
    xs, ys = zip(*batch)
    # Original sequence lengths before padding
    lengths = [len(x) for x in xs]
    # We use 7 because we want the model to generalize to S7
    maxlen = 7
    # Each permutation element has a 7-dimensional one-hot vector
    feature_dim = xs[0].shape[1]
    # Shape:
    # (batch_size, maxlen, feature_dim)
    padded = torch.zeros(
        len(xs),
        maxlen,
        feature_dim
    )
    for i, x in enumerate(xs):
        # Put original permutation into padded tensor
        padded[i, :len(x)] = x
    return (
        padded,
        torch.tensor(lengths),
        torch.stack(ys)
    )

# LSTM model
#
# Input:
# permutation sequence
# represented as one-hot vectors
#
# Example:
# (batch, sequence_length, 7)
#
# Output:
# predicted Coxeter length
class LengthPredictor(nn.Module):

    def __init__(self, input_size=7, hidden=64):
        super().__init__()
        # LSTM directly consumes one-hot vectors
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            batch_first=True
        )
        # Prediction head
        #
        # hidden representation
        #       |
        #      64
        #       |
        #      32
        #       |
        #       1
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )



    def forward(self, x, lengths):
        # Ignore padded zeros
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        # h contains final hidden state
        _, (h, _) = self.rnn(packed)
        # Use final hidden state to predict Coxeter length
        prediction = self.head(h[-1]).squeeze(-1)
        
        return prediction