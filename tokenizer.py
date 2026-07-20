import ast

class RSKTokenizer:

    def __init__(self):

        # Special tokens
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

        # Numbers 1 through 6
        for i in range(1, 7):
            self.vocab[str(i)] = len(self.vocab)

        # Reverse dictionary
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