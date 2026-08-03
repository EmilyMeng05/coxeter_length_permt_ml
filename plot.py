import json
import matplotlib.pyplot as plt


with open("history.json", "r") as f:
    history = json.load(f)


epochs = range(
    1,
    len(history["train_loss"]) + 1
)


plt.plot(
    epochs,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)


plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.savefig(
    "training_curve.png"
)

plt.show()