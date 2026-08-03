import re


class MathTokenizer:

    def __init__(self):

        # Special tokens
        self.vocab = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<TASK>": 3,
            "<LENGTH>": 4,
            "<ONE>": 5,
            "<CYCLE>": 6,
            "<LEHMER>": 7,
            "<INV>": 8,
            "(": 9,
            ")": 10,
            ",": 11
        }

        # convert the tokens into integers 
        for i in range(0, 50):
            self.vocab[str(i)] = len(self.vocab)

        # reverse integers back to tokens
        self.id_to_token = {
            value: key for key, value in self.vocab.items()
        }


    # Split a mathematical string into tokens.
    # Example:
    # "(1,2)(3,4)" -> ["(", "1", ",", "2", ")", "(", "3", ",", "4", ")"]
    def split_math_string(self, text):
        return re.findall(r"\d+|[(),]", text)


    # Encode the input sequence
    def encode_input(
        self,
        one_line,
        cycle,
        lehmer,
        inversion,
        task="LENGTH",
        use_one_line=True,
        use_cycle=True,
        use_lehmer=True,
        use_inversion=True
    ):

        tokens = []
        tokens.append("<BOS>")
        tokens.append("<TASK>")
        tokens.append(f"<{task}>")

        if use_one_line:
            tokens.append("<ONE>")
            tokens.extend(one_line.split())

        if use_cycle:
            tokens.append("<CYCLE>")
            tokens.extend(self.split_math_string(cycle))

        if use_lehmer:
            tokens.append("<LEHMER>")
            tokens.extend(lehmer.split())

        if use_inversion:
            tokens.append("<INV>")
            tokens.extend(inversion.split())

        tokens.append("<EOS>")

        return [self.vocab[token] for token in tokens]


    # Encode the target sequence
    # Example:
    # Coxeter length = 7
    # <BOS> 7 <EOS>
    def encode_target(self, target):

        tokens = [
            "<BOS>",
            str(target),
            "<EOS>"
        ]

        return [self.vocab[token] for token in tokens]


    # Decode integer ids back into tokens
    def decode(self, ids):

        return [
            self.id_to_token[index]
            for index in ids
        ]


    # Sanity check: Print the vocabulary
    # def print_vocab(self):
    #     print("Vocabulary\n")
    #     for token, idx in self.vocab.items():
    #         print(f"{token:10} -> {idx}")


# Example
if __name__ == "__main__":
    tokenizer = MathTokenizer()