# Training details 
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import PermutationDataset, collate_fn, LengthPredictor

# I defined learning rate to be 0.001
def train_model(model, train_loader, val_loader, epochs=1000, lr=1e-3, device="cpu"):
    # optimizer that updates the parameters
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # use MSE to calculate the loss function
    loss_fn = nn.MSELoss()
    # document all the training and validation loss
    history = {"train_loss": [], "val_loss": []}

    # training up till 40 times
    for epoch in range(epochs):
        model.train()
        # reset total loss = 0
        total = 0
        for x, lens, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x, lens)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        train_loss = total / len(train_loader.dataset)
        # validation set
        model.eval()
        with torch.no_grad():
            val_total = sum(
                loss_fn(model(x.to(device), lens), y.to(device)).item() * len(y)
                for x, lens, y in val_loader
            )
        val_loss = val_total / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch+1:3d}: train={train_loss:.4f} val={val_loss:.4f}")

    return history


if __name__ == "__main__":
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")

    train_loader = DataLoader(PermutationDataset(train_df), batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PermutationDataset(val_df), batch_size=32, collate_fn=collate_fn)

    model = LengthPredictor()
    history = train_model(model, train_loader, val_loader)

    torch.save(model.state_dict(), "model.pt")
    with open("history.json", "w") as f:
        json.dump(history, f)
    print("Saved model.pt and history.json")