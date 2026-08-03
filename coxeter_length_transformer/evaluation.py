import torch
from torch.utils.data import DataLoader
from coxeter_length_transformer.data_utils import RSKDataset, collate_fn
from coxeter_length_transformer.model import RSKTransformer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load dataset
test_dataset = RSKDataset("test.csv")
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)


# Load model
vocab_size = len(test_dataset.tokenizer.vocab)
model = RSKTransformer(vocab_size)
model.load_state_dict(
    torch.load("model.pt", map_location=device)
)
model.to(device)
model.eval()


# Evaluate
correct = 0
total = 0

with torch.no_grad():

    for encoder_input, decoder_input, target_output in test_loader:

        encoder_input = encoder_input.to(device)
        target_output = target_output.to(device)

        prediction = model.generate(
            encoder_input,
            bos_token=test_dataset.tokenizer.vocab["<BOS>"],
            eos_token=test_dataset.tokenizer.vocab["<EOS>"]
        )

        prediction = prediction[:, 1:]

        prediction = prediction[:, :target_output.size(1)]

        predicted_tokens = prediction

        mask = target_output != 0

        correct += (
            (predicted_tokens == target_output)
            & mask
        ).sum().item()

        total += mask.sum().item()


print()
print("Token Accuracy")
print(correct / total)