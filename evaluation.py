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

    test_df = pd.read_csv("test.csv")
    test_loader = DataLoader(PermutationDataset(test_df), batch_size=64, collate_fn=collate_fn)
    test_preds, test_actuals = evaluate(model, test_loader)

    n7_df = pd.read_csv("length_7.csv")   # <-- your Sage-generated file
    n7_loader = DataLoader(PermutationDataset(n7_df), batch_size=64, collate_fn=collate_fn)
    n7_preds, n7_actuals = evaluate(model, n7_loader)

    test_mse = sum((p - a) ** 2 for p, a in zip(test_preds, test_actuals)) / len(test_preds)
    n7_mse = sum((p - a) ** 2 for p, a in zip(n7_preds, n7_actuals)) / len(n7_preds)
    print(f"In-distribution test MSE (n=1..6): {test_mse:.4f}")
    print(f"Zero-shot n=7 MSE:                 {n7_mse:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE loss")
    axes[0].set_title("Training curves"); axes[0].legend(); axes[0].set_yscale("log")

    for n, group in test_df.assign(pred=test_preds, actual=test_actuals).groupby("n"):
        axes[1].scatter(group["actual"], group["pred"], alpha=0.5, s=12, label=f"n={n}")
    lims = [0, max(test_actuals + n7_actuals)]
    axes[1].plot(lims, lims, "k--", linewidth=1)
    axes[1].set_xlabel("actual length"); axes[1].set_ylabel("predicted length")
    axes[1].set_title(f"Test set (n=1..6)\nMSE={test_mse:.3f}"); axes[1].legend(fontsize=8)

    axes[2].scatter(test_actuals, test_preds, alpha=0.3, s=10, label="n=1..6 (test)", color="tab:blue")
    axes[2].scatter(n7_actuals, n7_preds, alpha=0.15, s=8, label="n=7 (zero-shot)", color="tab:red")
    axes[2].plot(lims, lims, "k--", linewidth=1)
    axes[2].set_xlabel("actual length"); axes[2].set_ylabel("predicted length")
    axes[2].set_title(f"Generalization to n=7\nMSE={n7_mse:.3f}"); axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("evaluation_plots.png", dpi=150)
    print("Saved evaluation_plots.png")


if __name__ == "__main__":
    main()