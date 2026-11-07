import json
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from device_utils import select_device

# Keep the vocabulary and model settings identical to AutoTrainer.py and
# Trainer.py; otherwise saved model weights will have incompatible dimensions.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(BASE_DIR, "model_weights.pth")
VOCAB_FILE = os.path.join(BASE_DIR, "vocab.json")
# Sampling temperature controls randomness (lower is more conservative).
# EMBED_DIM must be divisible by NUM_HEADS; MAX_SEQ_LEN limits input context.
TEMPERATURE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN = (
    0.5,
    512,
    8,
    6,
    128,
)
# Shorten or lengthen generated replies here; HARD_MAX_TOKENS is the safety cap.
MAX_GEN_TOKENS, HARD_MAX_TOKENS = 50, 250

try:
  with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    raw_vocab = json.load(f)
  vocab, id_to_token = {}, {}
  is_id_to_token = any(
      isinstance(v, str) and not v.isdigit() and str(k).isdigit()
      for k, v in raw_vocab.items()
  )
  for k, v in raw_vocab.items():
    try:
      if is_id_to_token:
        vocab[str(v)] = int(str(k).strip())
        id_to_token[int(str(k).strip())] = str(v)
      else:
        vocab[str(k)] = int(str(v).strip())
        id_to_token[int(str(v).strip())] = str(k)
    except ValueError:
      continue
except FileNotFoundError as exc:
  raise FileNotFoundError(
      f"Vocabulary not found at {VOCAB_FILE}. Run Tokenizer.py first."
  ) from exc

VOCAB_SIZE = max(id_to_token.keys()) + 1 if id_to_token else 4
UNK_ID = vocab.get("<unk>", 0)
device = select_device()


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


context_history = []
model = MiniLLM(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(
    device
)
if not os.path.isfile(WEIGHTS_FILE):
  raise FileNotFoundError(
      f"Model weights not found at {WEIGHTS_FILE}. Run AutoTrainer.py first."
  )
model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
model.eval()


def generate_reply_stream(history, model, vocab_dict, temp):
  full_text = " ".join([msg["content"] for msg in history])
  input_ids = [vocab_dict.get(w, UNK_ID) for w in full_text.split()] or [UNK_ID]
  input_tensor = (
      torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
  )
  consecutive_blanks = 0
  for i in range(HARD_MAX_TOKENS):
    with torch.no_grad():
      logits = model(input_tensor[:, -MAX_SEQ_LEN:])
    next_token_logits = logits[0, -1, :] / temp
    if consecutive_blanks >= 2:
      for tok_id, tok_str in id_to_token.items():
        if not tok_str.strip():
          next_token_logits[tok_id] = float("-inf")
    probs = F.softmax(next_token_logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1).item()
    token_str = str(id_to_token.get(next_token_id, "<unk>"))
    clean_token = token_str.strip()
    consecutive_blanks = consecutive_blanks + 1 if not clean_token else 0
    input_tensor = torch.cat(
        [input_tensor, torch.tensor([[next_token_id]]).to(device)], dim=1
    )
    time.sleep(0.08)  # Increase/decrease this delay to adjust display speed.
    yield token_str + " "
    if clean_token == "<eos>" or (
        i >= MAX_GEN_TOKENS and clean_token[-1:] in [".", "!", "?"]
    ):
      break


print("LLM Read-Only Prototype\nType 'KILL' to exit.")
while True:
  try:
    print("Prompt:", flush=True)
    lines = []
    while True:
      start_time = time.time()
      line = input()
      if line.strip() == "KILL":
        sys.exit(0)
      if not line and (time.time() - start_time) > 0.05:
        break
      lines.append(line)
    prompt = "\n".join(lines).strip()
    if not prompt:
      continue
    context_history.append({"role": "user", "content": prompt})
    print("\nReply: ", end="", flush=True)
    complete_reply = ""
    for token_text in generate_reply_stream(
        context_history, model, vocab, TEMPERATURE
    ):
      print(token_text, end="", flush=True)
      complete_reply += token_text
    context_history.append({"role": "model", "content": complete_reply.strip()})
    print("\n\n" + "-" * 50 + "\n")
  except KeyboardInterrupt:
    sys.exit(0)