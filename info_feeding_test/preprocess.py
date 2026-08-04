import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    input_file="multi_encoding_data.csv",
    train_file="train.csv",
    val_file="val.csv",
    test_file="test.csv",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42
):

    # Load dataset
    data = pd.read_csv(input_file)
    print("Total samples:", len(data))


    # First split:
    # train + temporary set
    train_data, temp_data = train_test_split(
        data,
        test_size=(1 - train_ratio),
        random_state=random_state,
        shuffle=True
    )

    # Split temporary set into validation and test
    val_size_adjusted = test_ratio / (val_ratio + test_ratio)

    val_data, test_data = train_test_split(
        temp_data,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True
    )


    # Save files
    train_data.to_csv(
        train_file,
        index=False
    )

    val_data.to_csv(
        val_file,
        index=False
    )

    test_data.to_csv(
        test_file,
        index=False
    )

if __name__ == "__main__":
    split_dataset()