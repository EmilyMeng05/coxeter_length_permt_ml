# Training details 
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import PermutationDataset, collate_fn, LengthPredictor

# Train the model
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, device="cpu"):
    # optimizer that updates model parameters
    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    # Mean squared error because we are predicting a number
    loss_fn = nn.MSELoss()
    # Store losses for plotting later
    history = {
        "train_loss": [],
        "val_loss": []
    }
    for epoch in range(epochs):
        model.train()
        total = 0
        for x, lens, y in train_loader:
            x = x.to(device)
            lens = lens.to(device)
            y = y.to(device)
            opt.zero_grad()
            # LSTM prediction
            pred = model(x, lens)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        train_loss = total / len(train_loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_total = sum(
                loss_fn(
                    model(
                        x.to(device),
                        lens.to(device)
                    ),
                    y.to(device)
                ).item() * len(y)
                for x, lens, y in val_loader
            )

        val_loss = val_total / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"epoch {epoch+1:3d}: "
                f"train={train_loss:.4f} "
                f"val={val_loss:.4f}"
            )

    return history

if __name__ == "__main__":
    # Load training and validation data
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    train_loader = DataLoader(
        PermutationDataset(train_df),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        PermutationDataset(val_df),
        batch_size=32,
        collate_fn=collate_fn
    )
    # LSTM model
    #
    # Input:
    # one-hot vector of size 7
    #
    # Hidden:
    # 64-dimensional LSTM state
    model = LengthPredictor(
        input_size=7,
        hidden=64
    )
    #sanity check
    for x, lens, y in train_loader:
        print("Input shape:", x.shape)
        print("Lengths:", lens[:5])
        print("Target:", y[:5])
        print("Prediction shape:", model(x, lens).shape)
        break
    
    history = train_model(
        model,
        train_loader,
        val_loader
    )
    # Save model
    torch.save(
        model.state_dict(),
        "model_lstm_stratified.pt"
    )
    with open("history_lstm_stratified.json", "w") as f:
        json.dump(history, f)

    print("Saved model_lstm_stratified.pt and history_lstm_stratified.json")