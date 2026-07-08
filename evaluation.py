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

    # choose experiment
    experiment = "stratified"
    # experiment = "random"

    model = LengthPredictor()
    model.load_state_dict(torch.load(f"model_{experiment}.pt"))
    model.eval()

    with open(f"history_{experiment}.json") as f:
        history = json.load(f)

    # testing for permutation with length 1-6
    test_df = pd.read_csv(f"test.csv")
    test_loader = DataLoader(
        PermutationDataset(test_df),
        batch_size=64,
        collate_fn=collate_fn
    )
    test_preds, test_actuals = evaluate(model, test_loader)

    # testing for permutation with length 7
    n7_df = pd.read_csv("coxeter_length_data_S7.csv")
    n7_loader = DataLoader(
        PermutationDataset(n7_df),
        batch_size=64,
        collate_fn=collate_fn
    )
    n7_preds, n7_actuals = evaluate(model, n7_loader)

    test_mse = sum((p-a)**2 for p, a in zip(test_preds, test_actuals))/len(test_preds)
    n7_mse = sum((p-a)**2 for p, a in zip(n7_preds, n7_actuals))/len(n7_preds)

    print(f"{experiment} Test MSE = {test_mse:.4f}")
    print(f"{experiment} n=7 MSE = {n7_mse:.4f}")

    # First plot: 
    # Plotting the training curves

    plt.figure(figsize=(6,5))

    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{experiment.capitalize()} Training Curves")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"training_curves_{experiment}.png", dpi=150)

    # Second plot:
    # Plotting the test predictions for permutation with length 1 - 6

    plt.figure(figsize=(6,5))

    for n, group in test_df.assign(
            pred=test_preds,
            actual=test_actuals
        ).groupby("n"):

        plt.scatter(
            group["actual"],
            group["pred"],
            alpha=0.5,
            s=12,
            label=f"n={n}"
        )

    lims = [0, max(test_actuals+n7_actuals)]
    plt.plot(lims, lims, "k--")

    plt.xlabel("Actual Coxeter Length")
    plt.ylabel("Predicted Coxeter Length")
    plt.title(f"{experiment.capitalize()} Test\nMSE={test_mse:.3f}")
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"test_predictions_{experiment}.png", dpi=150)

    # Third plot:
    # Generalizing to permutation with length 7
    plt.figure(figsize=(6,5))
    plt.scatter(
        test_actuals,
        test_preds,
        alpha=0.3,
        s=10,
        label="n=1-6"
    )
    plt.scatter(
        n7_actuals,
        n7_preds,
        alpha=0.15,
        s=8,
        label="n=7"
    )
    plt.plot(lims, lims, "k--")
    plt.xlabel("Actual Coxeter Length")
    plt.ylabel("Predicted Coxeter Length")
    plt.title(f"{experiment.capitalize()} Generalization\nMSE={n7_mse:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"generalization_n7_{experiment}.png", dpi=150)

    # Fourth plot:
    # Plotting prediction score for permutation with length 7 only
    plt.figure(figsize=(6,6))
    plt.scatter(
        n7_actuals,
        n7_preds,
        alpha=0.5,
        s=10
    )
    lims = [0, max(n7_actuals)]
    plt.plot(lims, lims, "k--")
    plt.xlabel("Actual Coxeter Length")
    plt.ylabel("Predicted Coxeter Length")
    plt.title(f"{experiment.capitalize()} Zero-shot n=7")
    plt.tight_layout()
    plt.savefig(f"n7_only_{experiment}.png", dpi=150)


    # Fifth plot: Plotting the residual errors
    errors = [p-a for p, a in zip(n7_preds, n7_actuals)]
    plt.figure(figsize=(6,5))
    plt.scatter(
        n7_actuals,
        errors,
        alpha=0.4
    )
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Actual Coxeter Length")
    plt.ylabel("Prediction Error")
    plt.title(f"{experiment.capitalize()} Residual Plot")
    plt.tight_layout()
    plt.savefig(f"n7_residuals_{experiment}.png", dpi=150)


if __name__ == "__main__":
    main()