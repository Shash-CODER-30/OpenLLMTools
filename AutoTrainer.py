import codecs
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from device_utils import select_device


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_FILE = BASE_DIR / "model_weights.pth"
CHECKPOINT_FILE = BASE_DIR / "training_checkpoint.pth"
VOCAB_FILE = BASE_DIR / "vocab.json"

# Save the training corpus as exactly data.txt beside these scripts. Add more
# filenames here only if you want to train on additional corpora in sequence.
CORPUS_FILES = ("data.txt",)
# Optional held-out text used to measure loss; do not copy training text here.
VALIDATION_FILE = BASE_DIR / "MiniDataset.txt"
CHECKPOINT_INTERVAL_SECONDS = 5 * 60
READ_BLOCK_BYTES = 64 * 1024

# Tune these to trade model capacity and training memory/speed. Keep them in
# sync with Trainer.py and LLM.py; embed_dim must be divisible by num_heads.
MAX_SEQ_LEN, LEARNING_RATE, EMBED_DIM, NUM_HEADS, NUM_LAYERS = (
    128,
    0.0003,
    512,
    8,
    6,
)
# LEARNING_RATE controls optimizer update size; adjust cautiously to avoid
# unstable training loss.
# Increase epochs for repeated passes over the corpus; each additional pass
# adds training time and can overfit small datasets.
EPOCHS = 1


def load_vocabulary(vocab_file):
  try:
    with vocab_file.open("r", encoding="utf-8") as f:
      raw_vocab = json.load(f)
    vocab, id_to_token = {}, {}
    is_id_to_token = any(
        isinstance(v, str) and not v.isdigit() and str(k).isdigit()
        for k, v in raw_vocab.items()
    )
    for k, v in raw_vocab.items():
      try:
        if is_id_to_token:
          vocab[str(v)], id_to_token[int(str(k).strip())] = (
              int(str(k).strip()),
              str(v),
          )
        else:
          vocab[str(k)], id_to_token[int(str(v).strip())] = (
              int(str(v).strip()),
              str(k),
          )
      except ValueError:
        continue
    return vocab, id_to_token
  except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"Vocabulary not found at {vocab_file}. Run Tokenizer.py first."
    ) from exc


VOCAB, ID_TO_TOKEN = load_vocabulary(VOCAB_FILE)
VOCAB_SIZE = max(ID_TO_TOKEN.keys()) + 1 if ID_TO_TOKEN else 4
UNK_ID = VOCAB.get("<unk>", 0)


class TransformerBlock(nn.Module):

  def __init__(self, embed_dim, num_heads):
    super().__init__()
    self.attention = nn.MultiheadAttention(
        embed_dim, num_heads, batch_first=True
    )
    self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
    self.feed_forward = nn.Sequential(
        nn.Linear(embed_dim, embed_dim * 4),
        nn.GELU(),
        nn.Linear(embed_dim * 4, embed_dim),
    )

  def forward(self, x, mask):
    # Disable attention weight computation since we don't use them.
    # This reduces memory/computation overhead when supported by PyTorch.
    try:
      attn_output, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
    except TypeError:
      # Fallback for older PyTorch versions that don't support need_weights.
      attn_output, _ = self.attention(x, x, x, attn_mask=mask)
    x = self.norm1(x + attn_output)
    return self.norm2(x + self.feed_forward(x))


class MiniLLM(nn.Module):

  def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
    super().__init__()
    self.token_embedding, self.position_embedding = nn.Embedding(
        vocab_size, embed_dim
    ), nn.Embedding(max_seq_len, embed_dim)
    self.blocks = nn.ModuleList(
        [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
    )
    self.final_norm, self.lm_head = nn.LayerNorm(embed_dim), nn.Linear(
        embed_dim, vocab_size
    )

  def forward(self, x):
    seq_len = x.size(1)
    mask = (
        torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        .to(x.device)
    )
    positions = (
        torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).to(x.device)
    )
    x = self.token_embedding(x) + self.position_embedding(positions)
    for block in self.blocks:
      x = block(x, mask)
    return self.lm_head(self.final_norm(x))


def resolve_corpora():
  corpora = []
  for corpus in CORPUS_FILES:
    path = Path(corpus)
    corpora.append(path if path.is_absolute() else BASE_DIR / path)

  missing = [str(path) for path in corpora if not path.is_file()]
  if missing:
    raise FileNotFoundError(
        "Configured corpus file(s) not found: " + ", ".join(missing)
    )
  if not corpora:
    raise ValueError("CORPUS_FILES must contain at least one corpus file.")
  return corpora


def corpus_signature(corpora):
  signature = []
  for path in corpora:
    metadata = path.stat()
    signature.append(
        {
            "path": str(path.resolve()),
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
        }
    )
  return signature


def fresh_training_state():
  return {
      "epoch": 0,
      "corpus_index": 0,
      "next_byte_offset": 0,
      "processed_bytes": 0,
      "pending_text": "",
      "decoder_buffer": b"",
      "token_buffer": [],
      "global_step": 0,
      "processed_tokens": 0,
      "loss_total": 0.0,
      "loss_count": 0,
      "last_loss": None,
      "last_reported_percent": 0,
      "training_start_time": time.time(),
      "tokens_processed": 0,  # Total target tokens used in loss computation
      "percent_interval_losses": {},  # {percent: [loss1, loss2, ...]} for current interval
      "percent_interval_steps": {},    # {percent: step_count} for current interval
      "last_validation_percent": 0,  # Track last validated percent to avoid duplicates
  }


def normalize_training_state(saved_state):
  state = fresh_training_state()
  if not isinstance(saved_state, dict):
    return state

  for key in (
      "epoch",
      "corpus_index",
      "next_byte_offset",
      "processed_bytes",
      "global_step",
      "processed_tokens",
      "loss_count",
      "last_reported_percent",
      "tokens_processed",
      "last_validation_percent",
  ):
    state[key] = int(saved_state.get(key, state[key]))
  state["loss_total"] = float(saved_state.get("loss_total", 0.0))
  last_loss = saved_state.get("last_loss")
  state["last_loss"] = None if last_loss is None else float(last_loss)
  state["pending_text"] = str(saved_state.get("pending_text", ""))
  state["decoder_buffer"] = bytes(saved_state.get("decoder_buffer", b""))
  state["token_buffer"] = [
      int(token_id) for token_id in saved_state.get("token_buffer", [])
  ]
  # Restore training start time if available, otherwise use current time.
  state["training_start_time"] = float(
      saved_state.get("training_start_time", time.time())
  )

  # Backward-compatible recovery for checkpoints created before processed_tokens
  # was introduced. Every completed normal training step processes exactly
  # MAX_SEQ_LEN target tokens, so this provides the correct cumulative count
  # for checkpoints taken during normal sequence training.
  if "processed_tokens" not in saved_state:
    state["processed_tokens"] = state["global_step"] * MAX_SEQ_LEN
  # Restore per-interval loss tracking.
  interval_losses = saved_state.get("percent_interval_losses", {})
  state["percent_interval_losses"] = {
      int(k): [float(v) for v in vs]
      for k, vs in interval_losses.items()
      if isinstance(vs, (list, tuple))
  }
  interval_steps = saved_state.get("percent_interval_steps", {})
  state["percent_interval_steps"] = {
      int(k): int(v) for k, v in interval_steps.items()
  }
  return state


def cpu_copy(value):
  """Copy tensors for serialization without moving the live model off-device."""
  if torch.is_tensor(value):
    return value.detach().cpu()
  if isinstance(value, dict):
    return {key: cpu_copy(item) for key, item in value.items()}
  if isinstance(value, list):
    return [cpu_copy(item) for item in value]
  if isinstance(value, tuple):
    return tuple(cpu_copy(item) for item in value)
  return value


def atomic_torch_save(value, destination):
  """Write a checkpoint atomically so an interrupted save keeps the old one."""
  temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
  try:
    with temporary.open("wb") as f:
      torch.save(value, f)
      f.flush()
      os.fsync(f.fileno())
    os.replace(temporary, destination)
  except BaseException:
    if temporary.exists():
      temporary.unlink()
    raise


def load_torch_file(path, map_location):
  """Support both current and older PyTorch versions."""
  try:
    return torch.load(path, map_location=map_location, weights_only=False)
  except TypeError:
    return torch.load(path, map_location=map_location)


def move_optimizer_state(optimizer, device):
  for state in optimizer.state.values():
    for key, value in state.items():
      if torch.is_tensor(value):
        state[key] = value.to(device)


def restore_rng_state(checkpoint, device):
  """Restore RNG state for the active device backend."""
  python_rng_state = checkpoint.get("python_rng_state")
  if python_rng_state is not None:
    random.setstate(python_rng_state)

  torch_rng_state = checkpoint.get("torch_rng_state")
  if torch_rng_state is not None:
    torch.set_rng_state(torch_rng_state)

  # CUDA RNG state
  cuda_rng_states = checkpoint.get("cuda_rng_states")
  if device.type == "cuda" and cuda_rng_states is not None:
    torch.cuda.set_rng_state_all(cuda_rng_states)

  # MPS RNG state (Apple Silicon)
  mps_rng_state = checkpoint.get("mps_rng_state")
  if device.type == "mps" and mps_rng_state is not None:
    try:
      if hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(mps_rng_state)
    except Exception as e:
      # MPS RNG restoration may not be fully supported on all PyTorch versions.
      # Log the issue but don't fail training.
      print(f"[Warning] Could not restore MPS RNG state: {e}")


def restore_training(model, optimizer, device, signature):
  if CHECKPOINT_FILE.exists():
    try:
      checkpoint = load_torch_file(CHECKPOINT_FILE, "cpu")
    except Exception as exc:
      print(f"[Checkpoint] Could not read {CHECKPOINT_FILE.name}: {exc}")
    else:
      if (
          isinstance(checkpoint, dict)
          and checkpoint.get("corpus_signature") == signature
          and "model_state_dict" in checkpoint
          and "optimizer_state_dict" in checkpoint
      ):
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state(optimizer, device)
        restore_rng_state(checkpoint, device)
        state = normalize_training_state(checkpoint.get("training_state"))
        print(
            "[Checkpoint] Resumed training at "
            f"step {state['global_step']:,}, epoch {state['epoch'] + 1}."
        )
        return state
      print(
          "[Checkpoint] Ignoring checkpoint because its corpus configuration "
          "does not match this run."
      )

  if WEIGHTS_FILE.exists():
    # Load on CPU first so restoring weights never creates a second full model
    # state_dict on the accelerator alongside the live model.
    weights = load_torch_file(WEIGHTS_FILE, "cpu")
    model.load_state_dict(weights)
    print(f"Loaded starting weights from {WEIGHTS_FILE.name}.")
  return fresh_training_state()


def save_checkpoint(model, optimizer, state, signature, reason):
  model_state = cpu_copy(model.state_dict())
  checkpoint = {
      "format_version": 1,
      "model_state_dict": model_state,
      "optimizer_state_dict": cpu_copy(optimizer.state_dict()),
      "training_state": state,
      "corpus_signature": signature,
      "python_rng_state": random.getstate(),
      "torch_rng_state": torch.get_rng_state(),
      "cuda_rng_states": (
          torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
      ),
      "saved_at": time.time(),
  }

  # Save MPS RNG state if available and on MPS device.
  try:
    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps_backend is not None
        and mps_backend.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
    ):
      checkpoint["mps_rng_state"] = torch.mps.get_rng_state()
  except Exception:
    # MPS RNG save may not be fully supported on all PyTorch versions.
    # This is optional; training continues without it.
    pass

  atomic_torch_save(checkpoint, CHECKPOINT_FILE)
  # Keep this file as a plain state_dict for Trainer.py and LLM.py.
  atomic_torch_save(model_state, WEIGHTS_FILE)
  print(
      f"\n[Checkpoint] {reason}: saved step {state['global_step']:,} to "
      f"{CHECKPOINT_FILE.name} and {WEIGHTS_FILE.name}."
  )


def latest_loss_text(state):
  latest_loss = state["last_loss"]
  if latest_loss is None:
    return "loss: waiting for the first trainable sequence"
  average_loss = state["loss_total"] / max(1, state["loss_count"])
  return f"loss: {latest_loss:.4f} (running avg: {average_loss:.4f})"


def compute_percent_interval_average(state, percent):
  """Compute average loss for a specific 1% interval."""
  losses = state["percent_interval_losses"].get(percent, [])
  if not losses:
    return None
  return sum(losses) / len(losses)


def format_duration(seconds):
  """Format duration as HhMmSs."""
  if seconds < 0:
    return "--:--:--"
  hours = int(seconds // 3600)
  minutes = int((seconds % 3600) // 60)
  secs = int(seconds % 60)
  return f"{hours:d}h {minutes:02d}m {secs:02d}s"


def estimate_throughput(state, total_bytes):
  """Calculate actual training throughput in tokens/sec."""
  elapsed = time.time() - state["training_start_time"]
  if elapsed <= 0 or state["processed_tokens"] <= 0:
    return None

  return state["processed_tokens"] / elapsed


class ProgressReporter:

  def __init__(self, state, total_bytes, corpus_count):
    self.state = state
    self.total_bytes = total_bytes
    self.corpus_count = corpus_count
    self.next_percent = max(1, state["last_reported_percent"] + 1)

  def report(self):
    completed_percent = int(
        min(100, self.state["processed_bytes"] * 100 / self.total_bytes)
    )
    while self.next_percent <= completed_percent:
      # Compute interval statistics for this percentage.
      current_avg = compute_percent_interval_average(
          self.state, self.next_percent
      )
      previous_avg = compute_percent_interval_average(
          self.state, self.next_percent - 1
      )

      # Compute elapsed time and ETA.
      elapsed = time.time() - self.state["training_start_time"]
      if completed_percent > 0:
        eta_total = elapsed * 100 / completed_percent
        eta_remaining = eta_total - elapsed
      else:
        eta_remaining = 0

      # Compute loss trend.
      loss_trend = 0.0
      if current_avg is not None and previous_avg is not None:
        loss_trend = current_avg - previous_avg

      # Format throughput using corrected calculation.
      throughput = estimate_throughput(self.state, self.total_bytes)
      throughput_str = (
          f"{throughput:,.0f} tok/s"
          if throughput is not None and throughput > 0
          else "computing"
      )

      # Format loss values
      current_loss_str = (
          f"{self.state['last_loss']:.4f}"
          if self.state['last_loss'] is not None
          else "N/A"
      )
      current_avg_str = (
          f"{current_avg:.4f}"
          if current_avg is not None
          else "N/A"
      )
      previous_avg_str = (
          f"{previous_avg:.4f}"
          if previous_avg is not None
          else "N/A"
      )
      loss_trend_str = (
          f"{loss_trend:+.4f}"
          if previous_avg is not None
          else "N/A"
      )

      # Build the detailed progress message.
      print(
          f"[{self.next_percent}%] overall\n"
          f"  step: {self.state['global_step']:,}\n"
          f"  current loss: {current_loss_str}\n"
          f"  1% avg loss: {current_avg_str}\n"
          f"  previous 1% avg loss: {previous_avg_str}\n"
          f"  loss trend: {loss_trend_str}\n"
          f"  elapsed: {format_duration(elapsed)}\n"
          f"  ETA: {format_duration(eta_remaining)}\n"
          f"  throughput: {throughput_str}",
          flush=True,
      )
      self.state["last_reported_percent"] = self.next_percent
      self.next_percent += 1


def split_completed_text(text):
  """Leave a trailing partial word for the next block."""
  if not text or text[-1].isspace():
    return text, ""
  for index in range(len(text) - 1, -1, -1):
    if text[index].isspace():
      return text[: index + 1], text[index + 1 :]
  return "", text


def tokenize(text):
  return [VOCAB.get(word, UNK_ID) for word in text.split()]


def train_one_sequence(model, optimizer, loss_fn, device, token_ids, state, total_bytes):
  train_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(
      device
  )
  logits = model(train_tensor[:, :-1])
  loss = loss_fn(
      logits.reshape(-1, VOCAB_SIZE), train_tensor[:, 1:].reshape(-1)
  )
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  loss_value = float(loss.detach().item())
  state["global_step"] += 1
  state["processed_tokens"] += len(token_ids) - 1
  state["loss_total"] += loss_value
  state["loss_count"] += 1
  state["last_loss"] = loss_value

  # Track tokens processed: exactly the number of target tokens used in loss.
  # For a sequence of length N, we compute loss on N-1 targets (positions 1 to N-1).
  num_target_tokens = len(token_ids) - 1
  state["tokens_processed"] += num_target_tokens

  # Track loss for the current 1% interval.
  # Determine which 1% bucket this step belongs to.
  if total_bytes > 0:
    current_percent = int(
        min(100, state["processed_bytes"] * 100 / total_bytes)
    )
  else:
    current_percent = 0
  if current_percent not in state["percent_interval_losses"]:
    state["percent_interval_losses"][current_percent] = []
  state["percent_interval_losses"][current_percent].append(loss_value)
  if current_percent not in state["percent_interval_steps"]:
    state["percent_interval_steps"][current_percent] = 0
  state["percent_interval_steps"][current_percent] += 1


def validate_model(model, loss_fn, device, validation_path, state):
  """
  Run validation on held-out data without updating model weights.
  Returns validation loss or None if validation is not available.
  
  Validates using exactly the same token stream logic as training:
  - Maintains pending_text between read blocks
  - Respects MAX_SEQ_LEN + 1 sequence length bounds
  - Preserves UTF-8 incremental decoding
  """
  if not validation_path.exists():
    return None

  model.eval()
  total_val_loss = 0.0
  val_count = 0
  decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
  token_buffer = []
  pending_text = ""

  try:
    with validation_path.open("rb") as val_file:
      while raw_block := val_file.read(READ_BLOCK_BYTES):
        # Decode incrementally, preserving UTF-8 state across blocks.
        text = pending_text + decoder.decode(raw_block, final=False)
        complete_text, pending_text = split_completed_text(text)
        token_buffer.extend(tokenize(complete_text))

        # Evaluate full sequences (exactly bounded by MAX_SEQ_LEN + 1).
        while len(token_buffer) >= MAX_SEQ_LEN + 1:
          with torch.no_grad():
            val_tensor = torch.tensor(
                token_buffer[: MAX_SEQ_LEN + 1], dtype=torch.long
            ).unsqueeze(0).to(device)
            logits = model(val_tensor[:, :-1])
            loss = loss_fn(
                logits.reshape(-1, VOCAB_SIZE), val_tensor[:, 1:].reshape(-1)
            )
            total_val_loss += float(loss.item())
            val_count += 1
          # Preserve one token for context (matching training behavior).
          del token_buffer[:MAX_SEQ_LEN]

    # Process final block: complete any pending UTF-8 sequences.
    final_text = pending_text + decoder.decode(b"", final=True)
    complete_text, _ = split_completed_text(final_text)
    token_buffer.extend(tokenize(complete_text))

    # Process remaining full sequences bounded by MAX_SEQ_LEN + 1.
    while len(token_buffer) >= MAX_SEQ_LEN + 1:
      with torch.no_grad():
        val_tensor = torch.tensor(
            token_buffer[: MAX_SEQ_LEN + 1], dtype=torch.long
        ).unsqueeze(0).to(device)
        logits = model(val_tensor[:, :-1])
        loss = loss_fn(
            logits.reshape(-1, VOCAB_SIZE), val_tensor[:, 1:].reshape(-1)
        )
        total_val_loss += float(loss.item())
        val_count += 1
      del token_buffer[:MAX_SEQ_LEN]

    # Final remainder: only evaluate if at least 2 tokens (input + target).
    if len(token_buffer) > 1:
      with torch.no_grad():
        val_tensor = torch.tensor(
            token_buffer, dtype=torch.long
        ).unsqueeze(0).to(device)
        logits = model(val_tensor[:, :-1])
        loss = loss_fn(
            logits.reshape(-1, VOCAB_SIZE), val_tensor[:, 1:].reshape(-1)
        )
        total_val_loss += float(loss.item())
        val_count += 1

    validation_loss = total_val_loss / max(1, val_count) if val_count > 0 else None
  finally:
    model.train()  # Resume training mode.

  return validation_loss


def checkpoint_if_due(
    checkpoint_deadline, model, optimizer, state, signature
):
  if time.monotonic() < checkpoint_deadline:
    return checkpoint_deadline
  save_checkpoint(model, optimizer, state, signature, "Periodic checkpoint")
  return time.monotonic() + CHECKPOINT_INTERVAL_SECONDS


def train_full_sequences(
    model,
    optimizer,
    loss_fn,
    device,
    state,
    signature,
    checkpoint_deadline,
    total_bytes,
):
  token_buffer = state["token_buffer"]
  while len(token_buffer) >= MAX_SEQ_LEN + 1:
    train_one_sequence(
        model,
        optimizer,
        loss_fn,
        device,
        token_buffer[: MAX_SEQ_LEN + 1],
        state,
        total_bytes,
    )
    # Preserve one token as the context for the next sequence.
    del token_buffer[:MAX_SEQ_LEN]
    checkpoint_deadline = checkpoint_if_due(
        checkpoint_deadline, model, optimizer, state, signature
    )
  return checkpoint_deadline


def train_corpus(
    corpus_path,
    model,
    optimizer,
    loss_fn,
    device,
    state,
    signature,
    progress,
    checkpoint_deadline,
    total_bytes,
):
  corpus_size = corpus_path.stat().st_size
  decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
  decoder.setstate((state["decoder_buffer"], 0))

  with corpus_path.open("rb") as corpus:
    corpus.seek(state["next_byte_offset"])
    while raw_block := corpus.read(READ_BLOCK_BYTES):
      state["next_byte_offset"] += len(raw_block)
      state["processed_bytes"] += len(raw_block)

      text = state["pending_text"] + decoder.decode(raw_block, final=False)
      state["decoder_buffer"] = decoder.getstate()[0]
      complete_text, state["pending_text"] = split_completed_text(text)
      state["token_buffer"].extend(tokenize(complete_text))
      checkpoint_deadline = train_full_sequences(
          model,
          optimizer,
          loss_fn,
          device,
          state,
          signature,
          checkpoint_deadline,
          total_bytes,
      )

      # Check if we've crossed a 5% validation boundary.
      if total_bytes > 0 and VALIDATION_FILE.exists():
        current_percent = int(
            min(100, state["processed_bytes"] * 100 / total_bytes)
        )
        # Validate at 5% boundaries (5, 10, 15, ... 100), but only once per boundary.
        if (
            current_percent % 5 == 0
            and current_percent > state["last_validation_percent"]
        ):
          val_loss = validate_model(model, loss_fn, device, VALIDATION_FILE, state)
          if val_loss is not None:
            print(
                f"[Validation] {current_percent}% progress: "
                f"validation loss = {val_loss:.4f}"
            )
          state["last_validation_percent"] = current_percent

      # Delay the last report until the corpus tail has produced its loss.
      if state["next_byte_offset"] < corpus_size:
        progress.report()

  final_text = state["pending_text"] + decoder.decode(b"", final=True)
  state["pending_text"] = ""
  state["decoder_buffer"] = b""
  state["token_buffer"].extend(tokenize(final_text))
  checkpoint_deadline = train_full_sequences(
      model,
      optimizer,
      loss_fn,
      device,
      state,
      signature,
      checkpoint_deadline,
      total_bytes,
  )

  # Train the final short sequence instead of dropping the tail of a corpus.
  if len(state["token_buffer"]) > 1:
    train_one_sequence(
        model,
        optimizer,
        loss_fn,
        device,
        state["token_buffer"],
        state,
        total_bytes,
    )
  state["token_buffer"].clear()
  checkpoint_deadline = checkpoint_if_due(
      checkpoint_deadline, model, optimizer, state, signature
  )
  progress.report()
  return checkpoint_deadline


def run_training(model, optimizer, loss_fn, device, corpora, state, signature):
  total_bytes = sum(path.stat().st_size for path in corpora) * EPOCHS
  if total_bytes <= 0:
    raise ValueError("Configured corpora must contain at least one byte of data.")
  progress = ProgressReporter(state, total_bytes, len(corpora))
  checkpoint_deadline = time.monotonic() + CHECKPOINT_INTERVAL_SECONDS

  if state["epoch"] >= EPOCHS:
    print(
        "[Checkpoint] The saved run is already complete; starting a new "
        "training pass from its saved weights."
    )
    state = fresh_training_state()
    state["training_start_time"] = time.time()
    progress = ProgressReporter(state, total_bytes, len(corpora))

  while state["epoch"] < EPOCHS:
    while state["corpus_index"] < len(corpora):
      corpus_path = corpora[state["corpus_index"]]
      print(
          f"\n[System] Epoch {state['epoch'] + 1}/{EPOCHS}: training "
          f"{corpus_path.name} with the single {device.type} model."
      )
      checkpoint_deadline = train_corpus(
          corpus_path,
          model,
          optimizer,
          loss_fn,
          device,
          state,
          signature,
          progress,
          checkpoint_deadline,
          total_bytes,
      )

      state["corpus_index"] += 1
      state["next_byte_offset"] = 0
      state["pending_text"] = ""
      state["decoder_buffer"] = b""
      state["token_buffer"].clear()

    state["epoch"] += 1
    state["corpus_index"] = 0
    state["next_byte_offset"] = 0

  return state


def main():
  corpora = resolve_corpora()
  signature = corpus_signature(corpora)
  device = select_device()

  # This is deliberately the only model moved to the GPU/MPS device.
  model = MiniLLM(
      VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN
  ).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  loss_fn = nn.CrossEntropyLoss()
  state = restore_training(model, optimizer, device, signature)
  model.train()

  # Validation file setup message.
  if VALIDATION_FILE.exists():
    print(
        f"[System] Validation corpus found at {VALIDATION_FILE.name}; "
        f"validation will run every 5% of training progress."
    )
  else:
    print(
        f"[System] No validation corpus at {VALIDATION_FILE.name}; "
        f"to enable validation, provide a validation dataset at that path."
    )

  total_size_gb = sum(path.stat().st_size for path in corpora) / (1024**3)
  print(
      "Starting single-model sequential corpus training "
      f"({EPOCHS} epoch(s), {total_size_gb:.2f} GiB total per epoch)."
  )

  try:
    state = run_training(
        model, optimizer, loss_fn, device, corpora, state, signature
    )
  except KeyboardInterrupt:
    save_checkpoint(model, optimizer, state, signature, "Interrupted training")
    print("Training interrupted; the latest completed optimizer step was saved.")
    return

  save_checkpoint(model, optimizer, state, signature, "Training complete")
  print("Autonomous training session complete.")


if __name__ == "__main__":
  main()
