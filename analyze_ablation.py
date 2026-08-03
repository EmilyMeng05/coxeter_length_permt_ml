import json
import glob
import pandas as pd
import matplotlib.pyplot as plt


def load_results():

    files = glob.glob("*_results.json")
    experiments = []

    for file in files:
        with open(file, "r") as f:
            data = json.load(f)

        name = data["experiment"]

        experiments.append({
            "experiment": name,
            "num_encoding": int(name.split("_")[0][1:]),
            "test_loss": data["test_loss"],
            "exact_accuracy": data["exact_accuracy"]
        })

    return pd.DataFrame(experiments)


def find_best_models(df):

    return (
        df.sort_values("test_loss")
        .groupby("num_encoding")
        .first()
        .reset_index()
    )


def plot_all(df):

    plt.figure(figsize=(10,6))

    plt.scatter(
        df["num_encoding"],
        df["exact_accuracy"]
    )

    for _, row in df.iterrows():

        plt.text(
            row["num_encoding"] + 0.03,
            row["exact_accuracy"],
            row["experiment"],
            fontsize=8
        )

    plt.xlabel("Number of Encodings")
    plt.ylabel("Exact Accuracy")
    plt.title("All Ablation Experiments")

    plt.tight_layout()
    plt.savefig("all_ablation_accuracy.png")
    plt.close()


def plot_best_accuracy(best):

    plt.figure(figsize=(8,5))

    plt.plot(
        best["num_encoding"],
        best["exact_accuracy"],
        marker="o"
    )

    for _, row in best.iterrows():

        plt.text(
            row["num_encoding"],
            row["exact_accuracy"],
            f'{row["experiment"]}\n{row["exact_accuracy"]:.4f}',
            fontsize=8
        )

    plt.xlabel("Number of Encodings")
    plt.ylabel("Exact Accuracy")
    plt.title("Best Model Accuracy by Encoding Count")

    plt.xticks(best["num_encoding"])

    plt.tight_layout()
    plt.savefig("best_ablation_accuracy.png")
    plt.close()


def plot_best_loss(best):

    plt.figure(figsize=(8,5))

    plt.plot(
        best["num_encoding"],
        best["test_loss"],
        marker="o"
    )

    for _, row in best.iterrows():

        plt.text(
            row["num_encoding"],
            row["test_loss"],
            f'{row["experiment"]}\n{row["test_loss"]:.4f}',
            fontsize=8
        )

    plt.xlabel("Number of Encodings")
    plt.ylabel("Test Loss")
    plt.title("Best Model Loss by Encoding Count")

    plt.xticks(best["num_encoding"])

    plt.tight_layout()
    plt.savefig("best_ablation_loss.png")
    plt.close()


def main():

    df = load_results()

    print("\nAblation Summary")
    print("----------------")
    print(df)

    df.to_csv(
        "ablation_summary.csv",
        index=False
    )

    best = find_best_models(df)

    print("\nBest Models")
    print("----------------")
    print(best)

    plot_all(df)
    plot_best_accuracy(best)
    plot_best_loss(best)

    print("\nSaved:")
    print("ablation_summary.csv")
    print("all_ablation_accuracy.png")
    print("best_ablation_accuracy.png")
    print("best_ablation_loss.png")


if __name__ == "__main__":
    main()