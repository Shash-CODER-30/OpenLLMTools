import json
import mmap
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.txt"
VOCAB_FILE = BASE_DIR / "vocab.json"

# Tune these for your corpus and available machine. The vocabulary limit also
# affects model size because the model has an embedding and output row per token.
MAX_VOCAB_SIZE = 32_000
MIN_FREQUENCY = 10
NUM_WORKERS = min(mp.cpu_count(), 8)
READ_BLOCK_BYTES = 1024 * 1024
BOUNDARY_LOOKAHEAD_BYTES = 4096


def scan_file_chunk(file_path: Path, start_pos: int, end_pos: int) -> dict[str, int]:
    """Count complete whitespace-delimited tokens in one mmap file range."""
    local_freqs = defaultdict(int)

    with file_path.open("rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:

            def next_whitespace(start_idx: int) -> int:
                """Return the closest ASCII whitespace offset in a small window."""
                window = mm[
                    start_idx : min(
                        start_idx + BOUNDARY_LOOKAHEAD_BYTES, mm.size()
                    )
                ]
                indices = [window.find(bytes((value,))) for value in b" \t\r\n\v\f"]
                valid = [index for index in indices if index != -1]
                return min(valid) if valid else -1

            # Move worker starts past the partial token shared with the
            # preceding worker. End offsets stop before the boundary token.
            if start_pos > 0:
                boundary_offset = next_whitespace(start_pos)
                if boundary_offset != -1:
                    start_pos += boundary_offset + 1
            if end_pos < mm.size():
                boundary_offset = next_whitespace(end_pos)
                if boundary_offset != -1:
                    end_pos += boundary_offset

            current_pos = start_pos
            while current_pos < end_pos and current_pos < mm.size():
                target_end = min(current_pos + READ_BLOCK_BYTES, end_pos)
                if target_end < end_pos:
                    boundary_offset = next_whitespace(target_end)
                    if boundary_offset != -1:
                        # Include the delimiter so a boundary at offset zero
                        # still advances the scan instead of repeating a block.
                        target_end += boundary_offset + 1

                text = mm[current_pos:target_end].decode(
                    "utf-8", errors="ignore"
                )
                for token in text.split():
                    local_freqs[token] += 1
                current_pos = target_end

    return local_freqs


def train_tokenizer(
    max_vocab_size: int = MAX_VOCAB_SIZE,
    min_frequency: int = MIN_FREQUENCY,
    num_workers: int = NUM_WORKERS,
) -> dict[int, str]:
    """Build the frequency-limited word vocabulary used by the training scripts."""
    if max_vocab_size < 1:
        raise ValueError("max_vocab_size must be at least 1 for the <unk> token.")
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1.")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1.")
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Training data not found: {DATA_FILE}. "
            'Save your UTF-8 corpus as "data.txt" beside the scripts.'
        )

    file_size = DATA_FILE.stat().st_size
    if file_size == 0:
        raise ValueError(f"Training data is empty: {DATA_FILE}")

    worker_count = min(num_workers, file_size)
    chunk_size = (file_size + worker_count - 1) // worker_count
    tasks = [
        (
            DATA_FILE,
            worker_index * chunk_size,
            min((worker_index + 1) * chunk_size, file_size),
        )
        for worker_index in range(worker_count)
    ]

    print(f"Counting words in {DATA_FILE.name} with {worker_count} worker(s)...")
    with mp.Pool(processes=worker_count) as pool:
        results = pool.starmap(scan_file_chunk, tasks)

    frequencies = defaultdict(int)
    for result in results:
        for token, count in result.items():
            frequencies[token] += count

    # Reserve token ID 0 for words not present in the vocabulary. Leave one
    # slot for it when applying the configured vocabulary-size limit.
    frequent_tokens = [
        (token, count)
        for token, count in frequencies.items()
        if count >= min_frequency and token != "<unk>"
    ]
    frequent_tokens.sort(key=lambda item: (-item[1], item[0]))
    frequent_tokens = frequent_tokens[: max_vocab_size - 1]

    vocab = {0: "<unk>"}
    vocab.update(
        (token_id, token)
        for token_id, (token, _) in enumerate(frequent_tokens, start=1)
    )
    print(f"Built a {len(vocab):,}-token vocabulary.")
    return vocab


if __name__ == "__main__":
    # `spawn` is required for safe multiprocessing startup on macOS and Windows.
    mp.set_start_method("spawn", force=True)
    vocab = train_tokenizer()

    print(f"Writing vocabulary to {VOCAB_FILE}...")
    with VOCAB_FILE.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("Tokenizer training complete.")
