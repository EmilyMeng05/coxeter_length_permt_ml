import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt

from data_utils import PermutationDataset, collate_fn
from tokenizer import MathTokenizer
from model import MathTransformer
import config



def generate(
    model,
    src,
    tokenizer,
    max_length=10,
    device="cpu"
):

    model.eval()


    generated = torch.tensor(
        [[tokenizer.vocab["<BOS>"]]],
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



        if next_token.item() == tokenizer.vocab["<EOS>"]:
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

    total_tokens = 0
    correct_tokens = 0

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
                    device=device
                )


                prediction_tokens = (
                    prediction[0][1:]
                    .tolist()
                )


                true_tokens = (
                    labels[i]
                    .tolist()
                )


                true_tokens = [
                    x for x in true_tokens
                    if x != 0
                ]


                prediction_tokens = [
                    x for x in prediction_tokens
                    if x != 0
                ]



                length = min(
                    len(prediction_tokens),
                    len(true_tokens)
                )


                for j in range(length):

                    if (
                        prediction_tokens[j]
                        ==
                        true_tokens[j]
                    ):
                        correct_tokens += 1

                    total_tokens += 1



                if prediction_tokens == true_tokens:

                    exact_correct += 1



                total_examples += 1



    token_accuracy = (
        correct_tokens / total_tokens
        if total_tokens > 0
        else 0
    )


    exact_accuracy = (
        exact_correct / total_examples
        if total_examples > 0
        else 0
    )


    return token_accuracy, exact_accuracy




def plot_history(
    history,
    experiment_name,
    exact_accuracy
):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )


    plt.figure(
        figsize=(8,5)
    )


    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )


    plt.plot(
        epochs,
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
        experiment_name
    )


    final_val_loss = (
        history["val_loss"][-1]
    )


    plt.text(
        0.55,
        0.8,
        f"Final Val Loss: {final_val_loss:.4f}\n"
        f"Exact Accuracy: {exact_accuracy:.4f}",
        transform=plt.gca().transAxes
    )


    plt.legend()

    plt.tight_layout()



    plt.savefig(
        f"{experiment_name}_results.png"
    )


    plt.close()




def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    experiment_name = config.create_experiment_name(
        config.ENCODING_CONFIG
    )


    print(
        "Evaluating:",
        experiment_name
    )


    tokenizer = MathTokenizer()

    vocab_size = len(
        tokenizer.vocab
    )



    test_dataset = PermutationDataset(
        "test.csv",
        task="LENGTH",
        **config.ENCODING_CONFIG
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )



    model = MathTransformer(
        vocab_size=vocab_size
    )



    model.load_state_dict(
        torch.load(
            f"{experiment_name}_model.pt",
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
    print("----------------------")

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
        f"{experiment_name}_history.json",
        "r"
    ) as f:
        history = json.load(f)

    plot_history(
        history,
        experiment_name,
        exact_acc
    )

    print(
        "Saved graph:",
        f"{experiment_name}_results.png"
    )

    results = {
        "experiment": experiment_name,
        "test_loss": test_loss,
        "token_accuracy": token_acc,
        "exact_accuracy": exact_acc
    }


    with open(
        f"{experiment_name}_results.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

if __name__ == "__main__":
    main()