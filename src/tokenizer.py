"""A GPT-2-style byte-pair encoding (BPE) tokenizer trained from scratch."""

import gc
import json
import os
import regex as re
import numpy as np

from collections import defaultdict
from multiprocessing import Process, Queue
from typing import BinaryIO

from tqdm import tqdm


# GPT-2 pre-tokenization pattern. The `regex` package is required because
# this pattern uses Unicode character properties such as \p{L} and \p{N}.
GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+(?!\S)|\s+"""
)


def count_pre_tokenize(
    file_path: str,
    start: int,
    end: int,
    compiled_regex,
    special_tokens: list[str],
    q: Queue,
) -> None:
    """Count pre-token frequencies in one independently processable file chunk."""
    process_dict: dict[str, int] = {}

    with open(file_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    # Special tokens must be isolated before ordinary text is pre-tokenized.
    if special_tokens:
        special_pattern = "(" + "|".join(
            re.escape(token) for token in special_tokens
        ) + ")"
        split_special_tokens = re.split(special_pattern, chunk)
    else:
        split_special_tokens = [chunk]

    del chunk
    gc.collect()

    for split in split_special_tokens:
        # Do not include special tokens in the ordinary pre-token counts.
        if split in special_tokens:
            continue

        for match in compiled_regex.finditer(split):
            pre_token = match.group()
            process_dict[pre_token] = process_dict.get(pre_token, 0) + 1

    del split_special_tokens
    gc.collect()

    process_dict_binary = {
        tuple(pre_token.encode("utf-8")): frequency
        for pre_token, frequency in process_dict.items()
    }
    q.put(process_dict_binary)


def merge_dictionaries(dict_list: list[dict]) -> dict:
    """Merge frequency dictionaries produced by the worker processes."""
    merged = defaultdict(int)

    for dictionary in dict_list:
        for key, value in dictionary.items():
            merged[key] += value

    return dict(merged)


def get_all_pairs(token_tuple: tuple) -> list[tuple]:
    """Return every adjacent pair in a token sequence."""
    return list(zip(token_tuple[:-1], token_tuple[1:]))


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Find byte offsets that split a file at occurrences of a special token.

    Chunks can then be counted independently without splitting a document at
    an arbitrary byte offset. Fewer chunks may be returned if boundaries
    overlap or the special token does not occur often enough.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )
    if desired_num_chunks < 1:
        raise ValueError("desired_num_chunks must be at least 1")

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0]

    chunk_size = max(1, file_size // desired_num_chunks)

    # Initial guesses for chunk boundary locations, uniformly spaced.
    chunk_boundaries = [
        index * chunk_size for index in range(desired_num_chunks + 1)
    ]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for boundary_index in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[boundary_index]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[boundary_index] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[boundary_index] = initial_position + found_at
                break

            initial_position += len(mini_chunk)

    # Boundaries must be unique and in ascending order.
    return sorted(set(chunk_boundaries))


def find_max(pair_counts: dict, vocab: dict[int, bytes]) -> tuple:
    """Find the most frequent pair, using token bytes to break ties."""
    max_freq = -1
    max_pair = (-1, -1)

    for pair, frequency in pair_counts.items():
        if frequency > max_freq:
            max_freq = frequency
            max_pair = pair
        elif frequency == max_freq:
            if vocab[pair[0]] > vocab[max_pair[0]]:
                max_pair = pair
            elif vocab[pair[0]] == vocab[max_pair[0]]:
                if vocab[pair[1]] > vocab[max_pair[1]]:
                    max_pair = pair

    return max_pair


def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE vocabulary and return it with its merge list."""
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must fit all 256 byte tokens and special tokens")
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")

    compiled_regex = re.compile(GPT2_PATTERN)

    # Parallel pre-tokenization.
    processes = []
    dictionaries = Queue()
    dictionaries_list = []

    with open(input_path, "rb") as file:
        # This token marks document boundaries in the training corpus.
        split_token = b"<|endoftext|>"
        boundaries = find_chunk_boundaries(file, num_processes, split_token)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        process = Process(
            target=count_pre_tokenize,
            args=(
                input_path,
                start,
                end,
                compiled_regex,
                special_tokens,
                dictionaries,
            ),
        )
        process.start()
        processes.append(process)

    # Read exactly one result for each process actually started. The number
    # of usable chunks can be smaller than num_processes.
    for _ in processes:
        dictionaries_list.append(dictionaries.get())

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Pre-tokenization worker exited with code {process.exitcode}"
            )

    dictionaries.close()
    dictionaries.join_thread()
    gc.collect()

    # Merge the dictionaries from all worker processes.
    pre_tokens = merge_dictionaries(dictionaries_list)
    pair_counts: dict[tuple, int] = {}
    pair_to_pre_token: dict[tuple, set[tuple]] = {}

    # Count adjacent-pair frequencies in the initial pre-token sequences.
    for pre_token, count in pre_tokens.items():
        for pair in get_all_pairs(pre_token):
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_to_pre_token.setdefault(pair, set()).add(pre_token)

    # The initial vocabulary contains all 256 possible byte values.
    vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}

    # Add special tokens to the vocabulary. They are never BPE-merged.
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    print("Starting BPE merges!")
    merges: list[tuple[bytes, bytes]] = []

    for _ in tqdm(range(vocab_size - len(vocab))):
        if not pair_counts:
            break

        max_pair = find_max(pair_counts, vocab)
        if max_pair == (-1, -1):
            break

        token_bytes1 = vocab[max_pair[0]]
        token_bytes2 = vocab[max_pair[1]]
        merges.append((token_bytes1, token_bytes2))

        new_token_id = len(vocab)
        vocab[new_token_id] = token_bytes1 + token_bytes2

        # Only pre-token sequences containing the selected pair need updating.
        for old_pre_token in list(pair_to_pre_token.get(max_pair, set())):
            old_pre_token_frequency = pre_tokens[old_pre_token]
            new_pre_token = []

            # Remove the old sequence's contribution from pair counts.
            for pair in get_all_pairs(old_pre_token):
                pair_counts[pair] -= old_pre_token_frequency

                if pair != max_pair:
                    pair_to_pre_token[pair].discard(old_pre_token)
                    if not pair_to_pre_token[pair]:
                        del pair_to_pre_token[pair]

            # Replace every non-overlapping occurrence of max_pair.
            index = 0
            while index < len(old_pre_token):
                if index == len(old_pre_token) - 1:
                    new_pre_token.append(old_pre_token[index])
                    index += 1
                elif (old_pre_token[index], old_pre_token[index + 1]) == max_pair:
                    new_pre_token.append(new_token_id)
                    index += 2
                else:
                    new_pre_token.append(old_pre_token[index])
                    index += 1

            new_pre_token = tuple(new_pre_token)
            pre_tokens[new_pre_token] = old_pre_token_frequency
            del pre_tokens[old_pre_token]

            # Add the new sequence's contribution to pair counts.
            for pair in get_all_pairs(new_pre_token):
                pair_counts[pair] = (
                    pair_counts.get(pair, 0) + old_pre_token_frequency
                )
                pair_to_pre_token.setdefault(pair, set()).add(new_pre_token)

        pair_to_pre_token.pop(max_pair, None)
        pair_counts.pop(max_pair, None)

    return vocab, merges


class Tokenizer:
    """Byte-level BPE tokenizer with GPT-2-style pre-tokenization."""

    def __init__(
        self,
        vocab: dict[int, bytes | list[int]],
        bpe_merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        verbose: bool = False,
    ):
        if verbose:
            print("Initializing Tokenizer!")

        self.verbose = verbose
        self.vocab = {
            key: value if isinstance(value, bytes) else bytes(value)
            for key, value in vocab.items()
        }
        self.i_vocab = {value: key for key, value in self.vocab.items()}
        self.special_tokens = (
            []
            if special_tokens is None
            else sorted(set(special_tokens), key=len, reverse=True)
        )
        self.PAT = GPT2_PATTERN

        self.raw_bpe_merges = [
            (bytes(token1), bytes(token2)) for token1, token2 in bpe_merges
        ]
        self.bpe_merges = {
            (self.i_vocab[token1], self.i_vocab[token2]): self.i_vocab[
                token1 + token2
            ]
            for token1, token2 in self.raw_bpe_merges
        }
        self.bpe_merges_ranks = {
            (self.i_vocab[token1], self.i_vocab[token2]): rank
            for rank, (token1, token2) in enumerate(self.raw_bpe_merges)
        }

        self.b_int_to_token = {
            index: self.i_vocab[bytes([index])] for index in range(256)
        }
        self.pre_token_cache: dict[tuple, tuple] = {}
        self.pre_token_utf8_cache: dict[str, tuple] = {}

    def _to_utf8(self, pre_token: str) -> tuple[int, ...]:
        """Convert a pre-token's UTF-8 bytes into initial vocabulary IDs."""
        if pre_token in self.special_tokens:
            return (self.i_vocab[pre_token.encode("utf-8")],)

        if pre_token in self.pre_token_utf8_cache:
            return self.pre_token_utf8_cache[pre_token]

        token_ids = tuple(
            self.b_int_to_token[byte] for byte in pre_token.encode("utf-8")
        )
        self.pre_token_utf8_cache[pre_token] = token_ids
        return token_ids

    def encode(self, text: str) -> list[int]:
        """Encode text into BPE token IDs."""
        if self.special_tokens:
            special_pattern = "(" + "|".join(
                re.escape(token) for token in self.special_tokens
            ) + ")"
            splits = re.split(special_pattern, text)
        else:
            splits = [text]

        pre_tokens = []
        for split in splits:
            if split in self.special_tokens:
                pre_tokens.append(split)
            else:
                pre_tokens.extend(re.findall(self.PAT, split))

        output_tokens = []

        for pre_token in pre_tokens:
            tokens = self._to_utf8(pre_token)

            if tokens in self.pre_token_cache:
                output_tokens.extend(self.pre_token_cache[tokens])
                continue

            old_tokens = list(tokens)

            while len(old_tokens) > 1:
                # Find the adjacent pair with the earliest learned merge rank.
                max_rank = float("inf")
                max_pair = None

                for pair in get_all_pairs(old_tokens):
                    rank = self.bpe_merges_ranks.get(pair)
                    if rank is not None and rank < max_rank:
                        max_rank = rank
                        max_pair = pair

                if max_pair is None:
                    break

                merged_token = self.bpe_merges[max_pair]
                new_tokens = []
                index = 0

                while index < len(old_tokens):
                    if index == len(old_tokens) - 1:
                        new_tokens.append(old_tokens[index])
                        index += 1
                    elif (old_tokens[index], old_tokens[index + 1]) == max_pair:
                        new_tokens.append(merged_token)
                        index += 2
                    else:
                        new_tokens.append(old_tokens[index])
                        index += 1

                old_tokens = new_tokens

            self.pre_token_cache[tokens] = tuple(old_tokens)
            output_tokens.extend(old_tokens)

        return output_tokens

    def encode_iterable(self, iterable):
        """Yield token IDs for each text block in an iterable."""
        for block in iterable:
            yield from self.encode(block)

    def encode_large_file(self, large_file_path: str, output_path: str) -> None:
        """Encode a large text file and write token IDs as uint16 values."""
        with open(large_file_path, "r", encoding="utf-8") as input_file:
            with open(output_path, "wb") as output_file:
                token_generator = self.encode_iterable(input_file)
                buffer = []

                for token_id in tqdm(token_generator):
                    buffer.append(token_id)

                    if len(buffer) >= 10_000:
                        output_file.write(
                            np.asarray(buffer, dtype=np.uint16).tobytes()
                        )
                        buffer = []

                if buffer:
                    output_file.write(
                        np.asarray(buffer, dtype=np.uint16).tobytes()
                    )

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into text."""
        byte_string = b"".join(self.vocab[token_id] for token_id in ids)
        return byte_string.decode(encoding="utf-8", errors="replace")

    def export_to_files(
        self,
        vocab_path: str,
        merges_path: str,
        user_friendly: bool = False,
    ) -> None:
        """Export vocabulary and merges as JSON files."""
        vocab_export = {key: list(value) for key, value in self.vocab.items()}
        merges_export = [
            (list(token1), list(token2))
            for token1, token2 in self.raw_bpe_merges
        ]

        os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(merges_path) or ".", exist_ok=True)

        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(vocab_export, file, indent=2)

        with open(merges_path, "w", encoding="utf-8") as file:
            json.dump(merges_export, file, indent=2)

        print(f"{vocab_path} successfully exported!")
        print(f"{merges_path} successfully exported!")

        if user_friendly:
            vocab_friendly_path = vocab_path.removesuffix(".json") + "_friendly.json"
            merges_friendly_path = merges_path.removesuffix(".json") + "_friendly.json"

            vocab_export_friendly = {
                key: value.decode("utf-8", errors="replace")
                for key, value in self.vocab.items()
            }
            merges_export_friendly = [
                (
                    token1.decode("utf-8", errors="replace"),
                    token2.decode("utf-8", errors="replace"),
                )
                for token1, token2 in self.raw_bpe_merges
            ]

            with open(vocab_friendly_path, "w", encoding="utf-8") as file:
                json.dump(vocab_export_friendly, file, indent=2, ensure_ascii=False)

            with open(merges_friendly_path, "w", encoding="utf-8") as file:
                json.dump(merges_export_friendly, file, indent=2, ensure_ascii=False)

            print(f"{vocab_friendly_path} successfully exported!")
            print(f"{merges_friendly_path} successfully exported!")

    @classmethod
    def from_files(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None,
        verbose: bool = False,
    ) -> "Tokenizer":
        """Load a tokenizer previously exported with export_to_files()."""
        with open(vocab_path, "r", encoding="utf-8") as file:
            vocab_json = json.load(file)

        with open(merges_path, "r", encoding="utf-8") as file:
            merges_json = json.load(file)

        vocab = {int(key): bytes(value) for key, value in vocab_json.items()}
        merges = [(bytes(token1), bytes(token2)) for token1, token2 in merges_json]

        return cls(vocab, merges, special_tokens, verbose)

    @classmethod
    def train(
        cls,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = 1,
        verbose: bool = False,
    ) -> "Tokenizer":
        vocab, merges = train_bpe_tokenizer(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            num_processes=num_processes,
        )
        return cls(vocab, merges, special_tokens, verbose)


def _read_int(prompt: str, default: int | None = None) -> int:
    """Read an integer from the terminal, optionally using a default value."""
    while True:
        value = input(prompt).strip()
        if not value and default is not None:
            return default
        try:
            return int(value)
        except ValueError:
            print("Please enter a valid integer.")


def _read_special_tokens() -> list[str]:
    """Read comma-separated special tokens from the terminal."""
    value = input("Special tokens (comma-separated, default: <|endoftext|>): ").strip()
    if not value:
        return ["<|endoftext|>"]
    return [token.strip() for token in value.split(",") if token.strip()]


def run_menu() -> None:
    """Run the interactive tokenizer menu until the user chooses Exit."""
    tokenizer: Tokenizer | None = None

    while True:
        print("\n" + "=" * 42)
        print("          BYTE-LEVEL BPE TOKENIZER")
        print("=" * 42)
        print("1. Train a new tokenizer")
        print("2. Load a tokenizer from files")
        print("3. Encode text")
        print("4. Decode token IDs")
        print("5. Encode a large text file")
        print("6. Export current tokenizer")
        print("0. Exit")
        print("-" * 42)
        print("Status:", "Tokenizer loaded" if tokenizer else "No tokenizer loaded")

        choice = input("Choose an option: ").strip()

        try:
            if choice == "1":
                input_path = input("Training text file path: ").strip()
                if not os.path.isfile(input_path):
                    print("File not found:", input_path)
                    continue

                vocab_size = _read_int("Vocabulary size [10000]: ", 10_000)
                num_processes = _read_int("Number of processes [1]: ", 1)
                special_tokens = _read_special_tokens()

                print("\nTraining tokenizer...")
                tokenizer = Tokenizer.train(
                    input_path=input_path,
                    vocab_size=vocab_size,
                    special_tokens=special_tokens,
                    num_processes=num_processes,
                    verbose=True,
                )
                print("Tokenizer training completed.")

                export_now = input("Export tokenizer now? [Y/n]: ").strip().lower()
                if export_now in ("", "y", "yes"):
                    vocab_path = input(
                        "Vocabulary output path [./models/tokenizers/vocab.json]: "
                    ).strip() or "./models/tokenizers/vocab.json"
                    merges_path = input(
                        "Merges output path [./models/tokenizers/merges.json]: "
                    ).strip() or "./models/tokenizers/merges.json"
                    tokenizer.export_to_files(
                        vocab_path=vocab_path,
                        merges_path=merges_path,
                        user_friendly=True,
                    )

            elif choice == "2":
                vocab_path = input("Vocabulary JSON path: ").strip()
                merges_path = input("Merges JSON path: ").strip()
                if not os.path.isfile(vocab_path) or not os.path.isfile(merges_path):
                    print("Could not find the vocabulary or merges file.")
                    continue

                special_tokens = _read_special_tokens()
                tokenizer = Tokenizer.from_files(
                    vocab_path=vocab_path,
                    merges_path=merges_path,
                    special_tokens=special_tokens,
                    verbose=True,
                )
                print("Tokenizer loaded successfully.")

            elif choice == "3":
                if tokenizer is None:
                    print("Train or load a tokenizer first.")
                    continue

                text = input("Text to encode: ")
                token_ids = tokenizer.encode(text)
                print("\nToken IDs:")
                print(token_ids)
                print("Number of tokens:", len(token_ids))

            elif choice == "4":
                if tokenizer is None:
                    print("Train or load a tokenizer first.")
                    continue

                raw_ids = input("Token IDs (separated by spaces or commas): ").strip()
                try:
                    token_ids = [
                        int(value)
                        for value in raw_ids.replace(",", " ").split()
                    ]
                    print("\nDecoded text:")
                    print(tokenizer.decode(token_ids))
                except ValueError:
                    print("Invalid token IDs. Enter integers separated by spaces or commas.")
                except KeyError as error:
                    print(f"Unknown token ID: {error}")

            elif choice == "5":
                if tokenizer is None:
                    print("Train or load a tokenizer first.")
                    continue

                input_path = input("Input text file path: ").strip()
                if not os.path.isfile(input_path):
                    print("File not found:", input_path)
                    continue
                output_path = input("Output token file path: ").strip()
                if not output_path:
                    print("Output path cannot be empty.")
                    continue

                tokenizer.encode_large_file(input_path, output_path)
                print("Encoded token IDs written to:", output_path)
                print("Note: the output is a binary uint16 file.")

            elif choice == "6":
                if tokenizer is None:
                    print("Train or load a tokenizer first.")
                    continue

                vocab_path = input("Vocabulary output path: ").strip()
                merges_path = input("Merges output path: ").strip()
                if not vocab_path or not merges_path:
                    print("Both output paths are required.")
                    continue

                tokenizer.export_to_files(
                    vocab_path=vocab_path,
                    merges_path=merges_path,
                    user_friendly=True,
                )

            elif choice == "0":
                print("Exiting tokenizer. Goodbye!")
                break

            else:
                print("Invalid option. Choose one of the listed options.")

        except (ValueError, OSError, RuntimeError) as error:
            print(f"\nOperation failed: {error}")
        except KeyboardInterrupt:
            print("\nOperation cancelled. Returning to the menu...")


if __name__ == "__main__":
    run_menu()
