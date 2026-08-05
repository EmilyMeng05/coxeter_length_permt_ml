import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json

from tokenizer import MathTokenizer
from data_utils import MathDataset, collate_fn
from model import MathTransformer


def generate(
    model,
    src,
    tokenizer,
    device,
    max_length=20
):

    model.eval()

    bos = tokenizer.vocab["[BOS]"]

    generated = torch.tensor(
        [[bos]],
        dtype=torch.long
    ).to(device)


    for _ in range(max_length):

        output = model(
            src,
            generated
        )

        next_token_logits = output[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1
        ).unsqueeze(1)


        generated = torch.cat(
            [
                generated,
                next_token
            ],
            dim=1
        )


        if (
            next_token.item()
            ==
            tokenizer.vocab["[EOS]"]
        ):
            break


    return generated



def evaluate_loss(
    model,
    dataloader,
    criterion,
    vocab_size,
    device
):

    model.eval()

    total_loss = 0


    with torch.no_grad():

        for batch in dataloader:

            src = batch["input_ids"].to(device)
            tgt = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)


            output = model(
                src,
                tgt
            )


            loss = criterion(
                output.reshape(-1, vocab_size),
                labels.reshape(-1)
            )


            total_loss += loss.item()


    return total_loss / len(dataloader)



def evaluate_accuracy(
    model,
    dataloader,
    tokenizer,
    device
):

    correct_tokens = 0
    total_tokens = 0

    exact_correct = 0
    total_examples = 0


    model.eval()


    with torch.no_grad():

        for batch in dataloader:

            src = batch["input_ids"].to(device)
            labels = batch["labels"]


            for i in range(src.size(0)):

                prediction = generate(
                    model,
                    src[i].unsqueeze(0),
                    tokenizer,
                    device
                )


                predicted = (
                    prediction[0][1:]
                    .tolist()
                )


                actual = (
                    labels[i]
                    .tolist()
                )


                # remove padding
                actual = [
                    x for x in actual
                    if x != 0
                ]

                predicted = [
                    x for x in predicted
                    if x != 0
                ]


                length = min(
                    len(predicted),
                    len(actual)
                )


                for j in range(length):

                    if predicted[j] == actual[j]:
                        correct_tokens += 1

                    total_tokens += 1


                if predicted == actual:
                    exact_correct += 1


                total_examples += 1



    token_accuracy = (
        correct_tokens / total_tokens
    )

    exact_accuracy = (
        exact_correct / total_examples
    )


    return token_accuracy, exact_accuracy



def plot_history(history):

    plt.figure(
        figsize=(8,5)
    )


    plt.plot(
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Addition Transformer Training"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "training_curve.png"
    )

    plt.close()



def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    tokenizer = MathTokenizer()

    vocab_size = len(
        tokenizer.vocab
    )


    test_dataset = MathDataset(
        "test.csv"
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )


    model = MathTransformer(
        vocab_size=vocab_size
    )


    model.load_state_dict(
        torch.load(
            "addition_model.pt",
            map_location=device
        )
    )


    model = model.to(device)


    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )


    test_loss = evaluate_loss(
        model,
        test_loader,
        criterion,
        vocab_size,
        device
    )


    token_acc, exact_acc = evaluate_accuracy(
        model,
        test_loader,
        tokenizer,
        device
    )


    print("\nEvaluation Results")
    print("--------------------")

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Token Accuracy: {token_acc:.4f}"
    )

    print(
        f"Exact Accuracy: {exact_acc:.4f}"
    )


    with open(
        "history.json",
        "r"
    ) as f:

        history = json.load(f)


    plot_history(
        history
    )


    results = {
        "test_loss": test_loss,
        "token_accuracy": token_acc,
        "exact_accuracy": exact_acc
    }


    with open(
        "evaluation_results.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

if __name__ == "__main__":
    main()