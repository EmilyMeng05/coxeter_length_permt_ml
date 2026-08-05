import re


class MathTokenizer:

    def __init__(self):

        special_tokens = [
            "[PAD]",
            "[BOS]",
            "[EOS]",
            "[TASK]",
            "[ADD]",
            "[LARGE_NUMBER]",
            "[BEGIN]",
            "[END]",
            "[EQUALS]"
        ]

        number_tokens = [
            f"n{i}"
            for i in range(10)
        ]

        tokens = (
            special_tokens
            + number_tokens
        )

        self.vocab = {
            token: i
            for i, token in enumerate(tokens)
        }

        self.inverse_vocab = {
            i: token
            for token, i in self.vocab.items()
        }


    def tokenize(self, sentence):
        # Find bracket groups, n0-n9, and equals
        raw_tokens = re.findall(
            r"\[[^\]]+\]|n\d",
            sentence
        )

        tokens = []

        for token in raw_tokens:

            if token.startswith("["):
                # Remove brackets
                content = token[1:-1]
                # Split inside:
                # LARGE_NUM BEGIN
                words = content.split()
                for word in words:
                    tokens.append(
                        f"[{word}]"
                    )

            else:
                tokens.append(token)


        return [
            self.vocab[token]
            for token in tokens
        ]