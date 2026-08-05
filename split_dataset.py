import csv
import random


INPUT_FILE = "addition_data.csv"
TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
TEST_FILE = "test.csv"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

def main():

    data = []

    with open(
        INPUT_FILE,
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            data.append(row)


    random.shuffle(data)
    
    total = len(data)

    train_end = int(
        TRAIN_RATIO * total
    )

    val_end = int(
        (TRAIN_RATIO + VAL_RATIO)
        * total
    )


    train_data = data[:train_end]

    val_data = data[train_end:val_end]

    test_data = data[val_end:]


    save_csv(
        TRAIN_FILE,
        train_data
    )

    save_csv(
        VAL_FILE,
        val_data
    )

    save_csv(
        TEST_FILE,
        test_data
    )


    print(
        "Dataset split complete"
    )

    print(
        f"Train: {len(train_data)}"
    )

    print(
        f"Validation: {len(val_data)}"
    )

    print(
        f"Test: {len(test_data)}"
    )



def save_csv(
    filename,
    data
):

    with open(
        filename,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input",
                "target"
            ]
        )

        writer.writeheader()

        writer.writerows(data)



if __name__ == "__main__":
    main()