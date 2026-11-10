from collections import Counter, deque
from functools import lru_cache
import json
import os


# https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
# This is exact implimentation of sebstianraschka.
# Only the train method has been changed to include progress of training of the tokenizer
class BPETokenizerSimple:
    def __init__(self):
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}

        # For the official OpenAI GPT-2 merges, use a rank dict:
        #  of form {(string_A, string_B): rank}, where lower rank = higher priority
        self.bpe_ranks = {}

    # Train the BPE tokenizer from scratch.
    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}):
        # Preprocess: Replace spaces with "Ġ"
        # Note that Ġ is a particularity of the GPT-2 BPE implementation
        # E.g., "Hello world" might be tokenized as ["Hello", "Ġworld"]
        # (GPT-4 BPE would tokenize it as ["Hello", " world"])
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initialize vocab with unique characters, including "Ġ" if present
        # Start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            char for char in sorted(set(processed_text)) if char not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Add allowed special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize the processed_text into token IDs
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # BPE steps 1-3: Repeatedly find and replace frequent pairs
        total_merges = max(0, vocab_size - len(self.vocab))

        for step, new_id in enumerate(range(len(self.vocab), vocab_size), start=1):
            pair_id = self.find_freq_pair(token_ids, mode="most")

            if pair_id is None:
                break

            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

            # Display progress
            percentage = (step / total_merges) * 100

            print(
                f"\rTraining BPE: {step}/{total_merges} " f"[{percentage:.2f}%]",
                end="",
                flush=True,
            )

        print("\nBPE training completed!")

        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    # Encode the input text into a list of token IDs, with tiktoken-style handling of special tokens.
    def encode(self, text, allowed_special=None):
        import re

        token_ids = []

        # If special token handling is enabled
        if allowed_special is not None and len(allowed_special) > 0:
            # Build regex to match allowed special tokens
            special_pattern = (
                "("
                + "|".join(
                    re.escape(tok)
                    for tok in sorted(allowed_special, key=len, reverse=True)
                )
                + ")"
            )

            last_index = 0
            for match in re.finditer(special_pattern, text):
                prefix = text[last_index : match.start()]
                token_ids.extend(
                    self.encode(prefix, allowed_special=None)
                )  # Encode prefix without special handling

                special_token = match.group(0)
                if special_token in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[special_token])
                else:
                    raise ValueError(
                        f"Special token {special_token} not found in vocabulary."
                    )
                last_index = match.end()

            text = text[last_index:]  # Remaining part to process normally

            # Check if any disallowed special tokens are in the remainder
            disallowed = [
                tok
                for tok in self.inverse_vocab
                if tok.startswith("<|")
                and tok.endswith("|>")
                and tok in text
                and tok not in allowed_special
            ]
            if disallowed:
                raise ValueError(
                    f"Disallowed special tokens encountered in text: {disallowed}"
                )

        # If no special tokens, or remaining text after special token split:
        tokens = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                tokens.append("\n")
            words = line.split()
            for j, word in enumerate(words):
                if j == 0 and i > 0:
                    tokens.append("Ġ" + word)
                elif j == 0:
                    tokens.append(word)
                else:
                    tokens.append("Ġ" + word)

        for token in tokens:
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))

        return token_ids

    # Tokenize a single token using BPE merges.
    def tokenize_with_bpe(self, token):
        # Tokenize the token into individual characters (as initial token IDs)
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        # If we haven't loaded OpenAI's GPT-2 merges, use my approach
        if not self.bpe_ranks:
            can_merge = True
            while can_merge and len(token_ids) > 1:
                can_merge = False
                new_tokens = []
                i = 0
                while i < len(token_ids) - 1:
                    pair = (token_ids[i], token_ids[i + 1])
                    if pair in self.bpe_merges:
                        merged_token_id = self.bpe_merges[pair]
                        new_tokens.append(merged_token_id)
                        # Uncomment for educational purposes:
                        # print(f"Merged pair {pair} -> {merged_token_id} ('{self.vocab[merged_token_id]}')")
                        i += 2  # Skip the next token as it's merged
                        can_merge = True
                    else:
                        new_tokens.append(token_ids[i])
                        i += 1
                if i < len(token_ids):
                    new_tokens.append(token_ids[i])
                token_ids = new_tokens
            return token_ids

        # Otherwise, do GPT-2-style merging with the ranks:
        # 1) Convert token_ids back to string "symbols" for each ID
        symbols = [self.vocab[id_num] for id_num in token_ids]

        # Repeatedly merge all occurrences of the lowest-rank pair
        while True:
            # Collect all adjacent pairs
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break

            # Find the pair with the best (lowest) rank
            min_rank = float("inf")
            bigram = None
            for p in pairs:
                r = self.bpe_ranks.get(p, float("inf"))
                if r < min_rank:
                    min_rank = r
                    bigram = p

            # If no valid ranked pair is present, we're done
            if bigram is None or bigram not in self.bpe_ranks:
                break

            # Merge all occurrences of that pair
            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                # If we see (first, second) at position i, merge them
                if (
                    i < len(symbols) - 1
                    and symbols[i] == first
                    and symbols[i + 1] == second
                ):
                    new_symbols.append(first + second)  # merged symbol
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

            if len(symbols) == 1:
                break

        # Finally, convert merged symbols back to IDs
        merged_ids = [self.inverse_vocab[sym] for sym in symbols]
        return merged_ids

    # Decode a list of token IDs back into a string.
    def decode(self, token_ids):
        decoded_string = ""
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token == "\n":
                if decoded_string and not decoded_string.endswith(" "):
                    decoded_string += " "  # Add space if not present before a newline
                decoded_string += token
            elif token.startswith("Ġ"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string

    # Save the vocabulary and BPE merges to JSON files.
    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        # Save BPE merges as a list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.bpe_merges.items()
            ]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    # Load the vocabulary and BPE merges from JSON files.
    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))

        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                # Remove the 2nd token of the pair, 1st was already removed
                dq.popleft()
            else:
                replaced.append(current)

        return replaced


from pathlib import Path

# Tokenizer instance
tokenizer = BPETokenizerSimple()

# Paths
BASE_PATH = Path(r"C:\Users\Bharath Andukuri\Desktop\Code\SrikariGPT\data\shakespeare")

RAW_DATA = BASE_PATH / "raw_data.txt"
VOCAB_PATH = BASE_PATH / "vocab.json"
MERGES_PATH = BASE_PATH / "bpe_merges.json"


def show_menu():
    print("\n========== Tokenizer ==========")
    print("1. Train tokenizer")
    print("2. Load tokenizer")
    print("3. Save tokenizer")
    print("4. Encode text")
    print("5. Decode token IDs")
    print("6. Show vocabulary information")
    print("0. Exit")
    print("==========================================")


while True:
    show_menu()

    choice = input("Enter your choice: ").strip()

    if choice == "1":
        print("\nLoading training corpus...")

        with open(RAW_DATA, "r", encoding="utf-8") as file:
            text = file.read()

        vocab_size = int(input("Enter vocabulary size: "))

        print("\nStarting BPE training...\n")
        tokenizer.train(text, vocab_size)

    elif choice == "2":
        print("\nLoading tokenizer...")

        tokenizer.load_vocab_and_merges(VOCAB_PATH, MERGES_PATH)

        print("Tokenizer loaded successfully!")

    elif choice == "3":
        tokenizer.save_vocab_and_merges(VOCAB_PATH, MERGES_PATH)

        print("Tokenizer saved successfully!")

    elif choice == "4":
        text = input("\nEnter text to encode: ")

        encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        print("\nEncoded token IDs:")
        print(encoded)

        print(f"\nToken count: {len(encoded)}")

    elif choice == "5":
        raw_ids = input("\nEnter token IDs separated by spaces: ")

        token_ids = [int(token.strip()) for token in raw_ids.split()]

        decoded = tokenizer.decode(token_ids)

        print("\nDecoded text:")
        print(decoded)

    elif choice == "6":
        print("\n========== Vocabulary Information ==========")
        print(f"Vocabulary size : {len(tokenizer.vocab)}")
        print(f"Merge rules     : {len(tokenizer.bpe_merges)}")
        print(f"Special tokens  : {tokenizer.get_special_token_id('<|endoftext|>')}")
        print("============================================")

    elif choice == "0":
        print("\nExiting Tokenizer. Goodbye!")
        break

    else:
        print("\nInvalid choice. Please try again.")
