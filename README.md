# Mini LLM

A small, local, educational language-model prototype built with PyTorch. This
repository provides only the core tools for experimenting with making a basic
LLM: a simple word-level vocabulary builder, a streaming trainer, an
interactive prompt-based trainer, and an inference script.

These tools alone cannot create a production-ready LLM. They are intended for
learning and small experiments, not for building or deploying a production
system. A production LLM requires substantial additional data pipelines,
compute, model and tokenizer improvements, evaluation, safety work, and
reliable serving infrastructure that this repository does not provide.

## Project files

| File | Purpose |
| --- | --- |
| `Tokenizer.py` | Reads `data.txt` and creates `vocab.json`. |
| `AutoTrainer.py` | Trains on `data.txt`, periodically saves checkpoints, and writes model weights. |
| `LLM.py` | Loads trained weights and starts a read-only interactive chat. |
| `Trainer.py` | Runs an interactive session that also trains on each submitted prompt. |
| `device_utils.py` | Selects CUDA, Apple MPS, or CPU without assuming a specific operating system. |
| `requirements.txt` | Lists the Python package dependency. |

Generated files (`vocab.json`, model weights, and checkpoints) are local
artifacts; the scripts do not upload them anywhere.

## Requirements

- Python 3.10 or later, using a version supported by the selected PyTorch
  release.
- PyTorch 2.0 or later. No other third-party Python packages are imported by
  the project; `json`, `mmap`, `multiprocessing`, `pathlib`, and the other
  non-PyTorch imports are part of Python's standard library.
- The scripts are designed to run on Windows, macOS, and Linux. A CPU is
  sufficient, though training can be very slow. The shared device selector
  chooses CUDA when available, Apple Silicon MPS when supported by the
  installed PyTorch build, and CPU otherwise. Accelerator availability depends
  on your hardware, operating system, and PyTorch installation; CPU is the
  portable fallback.

Create and activate a virtual environment from this directory. Use the
activation command for your shell:

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux (bash or zsh)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, create and activate the environment with:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In Windows Command Prompt, activate it with
`.venv\Scripts\activate.bat` instead. If PowerShell blocks activation, follow
your organization's policy or use Command Prompt; you can also run the
environment's Python directly as `.venv\Scripts\python.exe`.

For NVIDIA CUDA or Apple Silicon acceleration, choose a PyTorch wheel that
matches your operating system, Python version, and accelerator using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
Then check the installation with `python -c "import torch; print(torch.__version__)"`.

## Prepare the training data

1. Put your plain-text training corpus in a UTF-8 encoded file named exactly
   **`data.txt`** (lowercase, including the `.txt` extension).
2. Save it in the same directory as `Tokenizer.py` and `AutoTrainer.py`.
3. Use ordinary text, not JSON/CSV markup. Paragraphs, lines, and documents
   should be separated with whitespace.

All three training components use Python's `text.split()` convention: each
whitespace-delimited word is one token, and punctuation stays attached to its
word. For example, `Hello, world!` contains the tokens `Hello,` and `world!`.
This is a simple word-level tokenizer, not byte-pair encoding (BPE) or a
subword tokenizer. Words outside the vocabulary map to `<unk>`.

Only use data you have the right to use. Check its license, provenance, privacy
requirements, and suitability before training. Avoid placing private or
confidential material in the corpus. `data.txt` is ignored by Git by default so
it is not accidentally included in a source release.

### Optional validation text

To print validation loss during corpus training, put a separate, held-out UTF-8
text file named `MiniDataset.txt` beside the scripts. It must use the same
plain-text format. It is optional and is never used for optimizer updates.
Do not use the training corpus as validation data.

## Build the vocabulary

Run the tokenizer from the project directory:

```bash
python Tokenizer.py
```
or
```bash
python3 Tokenizer.py
```

It counts whitespace-delimited words in `data.txt` and writes `vocab.json`.
Token ID `0` is reserved for `<unk>`. Vocabulary IDs are stable for a given
corpus and configuration, with more frequent words assigned first.

Settings near the top of `Tokenizer.py` can be adjusted before running:

- `MAX_VOCAB_SIZE` caps vocabulary size. A larger vocabulary increases model
  size and memory use.
- `MIN_FREQUENCY` excludes rare words, which will become `<unk>`. For a small
  corpus, lower this to `1` so the vocabulary is not mostly unknown tokens.
- `NUM_WORKERS` controls CPU parallelism. Lower it if your machine runs out of
  memory or becomes unresponsive.

Rebuild the vocabulary whenever `data.txt` or the vocabulary settings change.
The model's embedding and output layers depend on the vocabulary, so its
weights must be trained against the matching `vocab.json`.

## Train on the corpus

After vocabulary creation, run:

```bash
python AutoTrainer.py
```
or
```bash
python3 AutoTrainer.py
```

The trainer reads `data.txt` in blocks, trains one sequence at a time, and
reports loss and progress. It saves:

- `model_weights.pth` — model weights used by `LLM.py` and `Trainer.py`.
- `training_checkpoint.pth` — model, optimizer, progress, and random-state data
  for resuming an interrupted run.

Training checkpoints are saved periodically and when training completes. Press
Ctrl+C to save the current state and stop. A later run resumes an interrupted
run when the checkpoint's recorded corpus still matches; after a completed
run, a new pass starts from the saved weights. If you change the data,
vocabulary, or model configuration, back up or remove the old checkpoint and
weights before starting a fresh run.

The default model uses a 512-dimensional embedding, 8 attention heads, 6
transformer layers, and a 128-token context window. These settings require
`EMBED_DIM` to be divisible by `NUM_HEADS`. You can tune `MAX_SEQ_LEN`,
`LEARNING_RATE`, `EMBED_DIM`, `NUM_HEADS`, `NUM_LAYERS`, and `EPOCHS` near the
top of `AutoTrainer.py`. Keep the model architecture settings identical in
`Trainer.py` and `LLM.py` so their models can load the same weights. Increasing
model size, sequence length, or epochs increases training time and/or memory
use.

## Generate text

Once `AutoTrainer.py` has produced `model_weights.pth`, start inference:

```bash
python LLM.py
```
or
```bash
python3 LLM.py
```

Enter a prompt and submit it with a blank line. Enter `KILL` to exit. Inference
does not update the model weights. The `TEMPERATURE`, `MAX_GEN_TOKENS`, and
`HARD_MAX_TOKENS` settings in `LLM.py` control sampling and reply length.

## Optional interactive trainer

`Trainer.py` is an alternative interactive mode, not a replacement for corpus
training with `AutoTrainer.py`. It updates the model from each submitted prompt
and then generates a reply. This is a tiny online-training demonstration; it
does not read `data.txt`. It saves weights on `KILL` or Ctrl+C. Its
`TEMPERATURE`, `LEARNING_RATE`, sequence length, and model settings can be
adjusted near the top of the file.

## Limitations

- This repository is a basic starting point, not a complete LLM development
  stack. Running its scripts does not produce a production-ready model.
- It does not include production data preparation, distributed training,
  comprehensive evaluation, alignment or safety systems, monitoring, or
  scalable serving infrastructure.
- Word-level tokenization means spelling variations and rare words often map
  to `<unk>`. Punctuation is not separated from words.
- There is no pretrained model: weights start from random initialization unless
  a prior compatible checkpoint or weights file is present.
- The default one-epoch training configuration is only a starting point.
  Output quality depends heavily on corpus size and quality, training time,
  vocabulary coverage, and hardware.
- Generation is stochastic and may repeat or produce incoherent text. It
  should not be treated as factual or professional advice.
- This repository does not include automated tests or a benchmark suite.
