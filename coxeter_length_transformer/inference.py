import torch
from coxeter_length_transformer.tokenizer import RSKTokenizer
from coxeter_length_transformer.model import RSKTransformer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = RSKTokenizer()
tokenizer.build_vocab("rsk_data.csv")
vocab_size = len(tokenizer.vocab)
model = RSKTransformer(vocab_size)

model.load_state_dict(
    torch.load("model.pt", map_location=device)
)

model.to(device)
model.eval()

permutation = [3, 1, 4, 2]
encoder = tokenizer.encode_permutation(permutation)

encoder = torch.tensor(
    encoder,
    dtype=torch.long
).unsqueeze(0)

encoder = encoder.to(device)

prediction = model.generate(
    encoder,
    bos_token=tokenizer.vocab["<BOS>"],
    eos_token=tokenizer.vocab["<EOS>"]
)


prediction = prediction.squeeze(0).tolist()


print()
print("Permutation")
print(permutation)
print()
print("Prediction")
print(tokenizer.decode(prediction))