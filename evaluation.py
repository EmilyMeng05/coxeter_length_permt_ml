# Evaluating using test.csv and length_7.csv
# Also plotting the results

import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from preprocess import load_data
from data_utils import PermutationDataset, collate_fn, LengthPredictor

def evaluate(model, loader, device="cpu"):
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for x, lens, y in loader:
            x = x.to(device)
            lens = lens.to(device)
            p = model(x, lens)
            preds.extend(p.tolist())
            actuals.extend(y.tolist())
    return preds, actuals

def main():
    # choose experiment
    experiment = "lstm_stratified"
    # experiment = "lstm_random"
    # Load LSTM model
    model = LengthPredictor(
        input_size=7,
        hidden=64
    )

    model.load_state_dict(
        torch.load(
            f"model_{experiment}.pt"
        )
    )
    model.eval()

    # sanity check:
    # Make sure this prints LSTM and not Embedding + LSTM
    print(model)

    with open(f"history_{experiment}.json") as f:
        history = json.load(f)

    # Test on S1-S6
    test_df = load_data(
        "test.csv"
    )
    test_loader = DataLoader(
        PermutationDataset(test_df),
        batch_size=64,
        collate_fn=collate_fn
    )
    # sanity check:
    # Check that test data has shape (batch, 7, 7)
    for x, lens, y in test_loader:
        print("Test input shape:", x.shape)
        print("Test lengths:", lens[:5])
        break

    test_preds, test_actuals = evaluate(
        model,
        test_loader
    )

    # Test generalization on S7
    n7_df = load_data(
        "coxeter_length_data_S7.csv"
    )
    n7_loader = DataLoader(
        PermutationDataset(n7_df),
        batch_size=64,
        collate_fn=collate_fn
    )
    n7_preds, n7_actuals = evaluate(
        model,
        n7_loader
    )

    # Calculate MSE
    test_mse = sum(
        (p-a)**2
        for p, a in zip(test_preds, test_actuals)
    ) / len(test_preds)
    n7_mse = sum(
        (p-a)**2
        for p, a in zip(n7_preds, n7_actuals)
    ) / len(n7_preds)
    print(
        f"{experiment} Test MSE = {test_mse:.4f}"
    )
    print(
        f"{experiment} S7 MSE = {n7_mse:.4f}"
    )

    # Figure 1
    # Plot training curves
    plt.figure(figsize=(6,5))
    plt.plot(
        history["train_loss"],
        label="Training"
    )
    plt.plot(
        history["val_loss"],
        label="Validation"
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(
        f"{experiment} Training Curves"
    )
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"training_curves_{experiment}_100.png",
        dpi=150
    )

    # Figure 4:
    # Prediction accuracy S1-S6
    plt.figure(figsize=(6,5))
    plt.scatter(
        test_actuals,
        test_preds,
        alpha=0.5,
        s=15
    )
    lims = [0,max(test_actuals)]
    plt.plot(
        lims,
        lims,
        "k--",
        linewidth=1,
        label="Perfect Prediction"
    )
    plt.xlabel(
        "Actual Coxeter Length"
    )
    plt.ylabel(
        "Predicted Coxeter Length"
    )
    plt.title(
        f"S1-S6 Test Prediction\n"
        f"MSE={test_mse:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"test_predictions_{experiment}_100.png",
        dpi=150
    )

    # Figure 5:
    # Generalization plot S7
    plt.figure(figsize=(6,5))
    plt.scatter(
        test_actuals,
        test_preds,
        alpha=0.45,
        s=15,
        label="S1-S6 Test"
    )
    plt.scatter(
        n7_actuals,
        n7_preds,
        alpha=0.45,
        s=15,
        label="S7 Unseen"
    )
    lims = [
        0,
        max(test_actuals + n7_actuals)
    ]
    plt.plot(
        lims,
        lims,
        "k--",
        linewidth=1
    )
    plt.xlabel(
        "Actual Coxeter Length"
    )
    plt.ylabel(
        "Predicted Coxeter Length"
    )
    plt.title(
        f"S7 Generalization\n"
        f"MSE={n7_mse:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"generalization_n7_{experiment}_100.png",
        dpi=150
    )


if __name__ == "__main__":
    main()