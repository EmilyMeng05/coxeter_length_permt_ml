import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import json

from data_utils import PermutationDataset, collate_fn
from tokenizer import MathTokenizer
from model import MathTransformer
import config

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

def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    experiment_name = config.create_experiment_name(
        config.ENCODING_CONFIG
    )

    print(
        "Running:",
        experiment_name
    )

    # Tokenizer
    tokenizer = MathTokenizer()
    vocab_size = len(
        tokenizer.vocab
    )

    # Dataset
    train_dataset = PermutationDataset(
        "train.csv",
        task="LENGTH",
        **config.ENCODING_CONFIG
    )

    val_dataset = PermutationDataset(
        "val.csv",
        task="LENGTH",
        **config.ENCODING_CONFIG
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Model
    model = MathTransformer(
        vocab_size=vocab_size
    )

    model = model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=1e-4
    )

    epochs = 50

    history = {
        "experiment": experiment_name,
        "encoding": config.ENCODING_CONFIG,
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0

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
            total_loss += loss.item()

        train_loss = (
            total_loss / len(train_loader)
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            vocab_size,
            device
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

    # Save model
    torch.save(
        model.state_dict(),
        f"{experiment_name}_model.pt"
    )

    # Save history
    with open(
        f"{experiment_name}_history.json",
        "w"
    ) as f:
        json.dump(
            history,
            f,
            indent=4
        )

if __name__ == "__main__":
    train()