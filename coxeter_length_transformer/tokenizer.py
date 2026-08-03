import ast
import pandas as pd

class RSKTokenizer:

    def __init__(self):

        self.vocab = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<P>": 3,
            "</P>": 4,
            "<Q>": 5,
            "</Q>": 6,
            "/": 7
        }

        # we don't want to hard code the number of tokens
        self.id_to_token = {}

    # this file will automatically know the number of digits we want
    # by reading through the file
    def build_vocab(self, csv_file):

        data = pd.read_csv(csv_file)

        numbers = set()

        for _, row in data.iterrows():

            permutation = ast.literal_eval(row["permutation"])
            P = ast.literal_eval(row["P"])
            Q = ast.literal_eval(row["Q"])

            # permutation
            numbers.update(permutation)

            # P tableau
            for tableau_row in P:
                numbers.update(tableau_row)

            # Q tableau
            for tableau_row in Q:
                numbers.update(tableau_row)

        for number in sorted(numbers):
            self.vocab[str(number)] = len(self.vocab)

        self.id_to_token = {
            value: key for key, value in self.vocab.items()
        }

    # Encode a permutation
    def encode_permutation(self, permutation):
        # If permutation comes from the CSV, convert it first.
        if isinstance(permutation, str):
            permutation = ast.literal_eval(permutation)
        tokens = ["<BOS>"]
        for value in permutation:
            tokens.append(str(value))
        tokens.append("<EOS>")
        return [self.vocab[token] for token in tokens]
    
    # Convert a tableau into tokens
    def tableau_to_tokens(self, tableau):
        tokens = []
        for row_index, row in enumerate(tableau):
            for value in row:
                tokens.append(str(value))
            if row_index != len(tableau) - 1:
                tokens.append("/")
        return tokens

    # Encode the target (P,Q)
    def encode_target(self, P, Q):
        if isinstance(P, str):
            P = ast.literal_eval(P)
        if isinstance(Q, str):
            Q = ast.literal_eval(Q)

        tokens = ["<BOS>"]

        tokens.append("<P>")
        tokens.extend(self.tableau_to_tokens(P))
        tokens.append("</P>")

        tokens.append("<Q>")
        tokens.extend(self.tableau_to_tokens(Q))
        tokens.append("</Q>")

        tokens.append("<EOS>")

        return [self.vocab[token] for token in tokens]

    # Decode integers back into tokens
    def decode(self, ids):

        tokens = []

        for index in ids:
            tokens.append(self.id_to_token[index])

        return tokens

    # Print the vocabulary
    def print_vocab(self):

        print("Vocabulary")

        for token, idx in self.vocab.items():
            print(f"{token:6} -> {idx}")

# # Example
# if __name__ == "__main__":

#     tokenizer = RSKTokenizer()

#     tokenizer.print_vocab()

#     permutation = [3, 1, 4, 2]

#     P = [[1, 2], [3, 4]]
#     Q = [[1, 3], [2, 4]]

#     print()

#     print("Permutation:")
#     print(tokenizer.encode_permutation(permutation))

#     print()

#     print("Target:")
#     encoded = tokenizer.encode_target(P, Q)
#     print(encoded)

#     print()

#     print("Decoded:")
#     print(tokenizer.decode(encoded))