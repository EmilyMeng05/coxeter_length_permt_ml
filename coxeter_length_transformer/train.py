import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from coxeter_length_transformer.data_utils import RSKDataset, collate_fn
from coxeter_length_transformer.model import RSKTransformer


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)

# Dataset
train_dataset = RSKDataset("train.csv")

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# Model
vocab_size = len(train_dataset.tokenizer.vocab)
model = RSKTransformer(
    vocab_size=vocab_size
).to(device)


# Loss
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# Training
epochs = 500

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for encoder_input, decoder_input, target_output in train_loader:

        encoder_input = encoder_input.to(device)
        decoder_input = decoder_input.to(device)
        target_output = target_output.to(device)

        optimizer.zero_grad()

        # Forward
        prediction = model(
            encoder_input,
            decoder_input
        )

        # Cross entropy
        loss = criterion(

            prediction.reshape(
                -1,
                vocab_size
            ),

            target_output.reshape(-1)
        )

        # Backprop
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        predicted_tokens = prediction.argmax(dim=-1)

        mask = target_output != 0

        total_correct += (
            (predicted_tokens == target_output)
            & mask
        ).sum().item()

        total_tokens += mask.sum().item()

    accuracy = total_correct / total_tokens

    print(
        f"Epoch {epoch+1:3d}"
        f" | Loss = {total_loss:.4f}"
        f" | Accuracy = {accuracy:.4f}"
    )

# Save
torch.save(
    model.state_dict(),
    "model.pt"
)

print("Training complete!")