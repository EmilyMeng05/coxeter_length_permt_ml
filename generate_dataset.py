import csv

MAX_NUMBER = 200

# encoding the numbers
def encode_number(number):
    digits = list(str(number))
    return (
        "[LARGE_NUMBER BEGIN] "
        + " ".join(f"n{d}" for d in digits)
        + " [LARGE_NUMBER END]"
    )

with open("addition_data.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "input",
        "target"
    ])

    for a in range(MAX_NUMBER + 1):
        for b in range(MAX_NUMBER + 1):

            input_sentence = (
                "[TASK] [ADD] "
                + encode_number(a)
                + " "
                + encode_number(b)
                + " [EQUALS]"
            )

            target_sentence = encode_number(a + b)

            writer.writerow([
                input_sentence,
                target_sentence
            ])