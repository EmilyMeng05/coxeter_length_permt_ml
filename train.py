import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from tokenizer import MathTokenizer
from data_utils import MathDataset, collate_fn
from model import MathTransformer


def evaluate(
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

            output = model(src, tgt)

            loss = criterion(
                output.reshape(-1, vocab_size),
                labels.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = MathTokenizer()

    vocab_size = len(tokenizer.vocab)

    train_dataset = MathDataset("train.csv")
    val_dataset = MathDataset("val.csv")

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = MathTransformer(
        vocab_size=vocab_size
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )

    optimizer = Adam(
        model.parameters(),
        lr=1e-4
    )

    epochs = 50

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for batch in train_loader:

            src = batch["input_ids"].to(device)
            tgt = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            output = model(
                src,
                tgt
            )

            loss = criterion(
                output.reshape(-1, vocab_size),
                labels.reshape(-1)
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            vocab_size,
            device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch+1:2d} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f}"
        )

    torch.save(
        model.state_dict(),
        "addition_model.pt"
    )

    with open(
        "history.json",
        "w"
    ) as f:
        json.dump(
            history,
            f,
            indent=4
        )


if __name__ == "__main__":
    train()