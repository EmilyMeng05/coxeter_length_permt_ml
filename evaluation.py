# Evaluating using test.csv and length_7.csv
# also plotting the plot 

import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_utils import PermutationDataset, collate_fn, LengthPredictor


def evaluate(model, loader, device="cpu"):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, lens, y in loader:
            p = model(x.to(device), lens)
            preds.extend(p.tolist())
            actuals.extend(y.tolist())
    return preds, actuals


def main():
    model = LengthPredictor()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    with open("history.json") as f:
        history = json.load(f)

    # testing for permutation with length 1-6
    test_df = pd.read_csv("test.csv")
    test_loader = DataLoader(PermutationDataset(test_df), batch_size=64, collate_fn=collate_fn)
    test_preds, test_actuals = evaluate(model, test_loader)

    # testing for permutation with length 7
    n7_df = pd.read_csv("length_7.csv") 
    n7_loader = DataLoader(PermutationDataset(n7_df), batch_size=64, collate_fn=collate_fn)
    n7_preds, n7_actuals = evaluate(model, n7_loader)

    test_mse = sum((p - a) ** 2 for p, a in zip(test_preds, test_actuals)) / len(test_preds)
    n7_mse = sum((p - a) ** 2 for p, a in zip(n7_preds, n7_actuals)) / len(n7_preds)
    print(f"In-distribution test MSE (n=1..6): {test_mse:.4f}")
    print(f"Zero-shot n=7 MSE:                 {n7_mse:.4f}")

    # First plot: 
    # depict training and validation set difference
    plt.figure(figsize=(6,5))

    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curves")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)

    # Second plot:
    # depict MSE trajectories across epoch for permutation with length 1-6
    plt.figure(figsize=(6,5))

    for n, group in test_df.assign(pred=test_preds, actual=test_actuals).groupby("n"):
        plt.scatter(group["actual"], group["pred"],
                    alpha=0.5,
                    s=12,
                    label=f"n={n}")

    lims = [0, max(test_actuals + n7_actuals)]
    plt.plot(lims, lims, "k--", linewidth=1)

    plt.xlabel("Actual Coxeter Length")
    plt.ylabel("Predicted Coxeter Length")
    plt.title(f"Test Set (n=1–6)\nMSE={test_mse:.3f}")
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("test_predictions.png", dpi=150)

    # Third plot: 
    # depict MSE trajectories for permutation with length 1-7
    plt.figure(figsize=(6,5))

    plt.scatter(test_actuals,
                test_preds,
                alpha=0.3,
                s=10,
                label="n=1–6",
                color="tab:blue")

    plt.scatter(n7_actuals,
                n7_preds,
                alpha=0.15,
                s=8,
                label="n=7",
                color="tab:red")

    plt.plot(lims, lims, "k--", linewidth=1)

    plt.xlabel("Actual Coxeter Length")
    plt.ylabel("Predicted Coxeter Length")
    plt.title(f"Generalization to n=7\nMSE={n7_mse:.3f}")
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("generalization_n7.png", dpi=150)


if __name__ == "__main__":
    main()