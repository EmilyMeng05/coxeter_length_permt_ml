import pandas as pd

from sklearn.model_selection import train_test_split


# Load dataset
data = pd.read_csv("rsk_data.csv")
print("Dataset size:", len(data))


# Train / Validation / Test split
train_data, temp_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    shuffle=True
)
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=42,
    shuffle=True
)


# Save files
train_data.to_csv(
    "train.csv",
    index=False
)
val_data.to_csv(
    "val.csv",
    index=False
)
test_data.to_csv(
    "test.csv",
    index=False
)


# Print sizes
print()
print("Training:", len(train_data))
print("Validation:", len(val_data))
print("Testing:", len(test_data))
print()
print("Finished preprocessing!")