---
name: ml-training
description: Use when training or fine-tuning a model with PyTorch here (CPU only): sizing to the sandbox, tokenizers, data splits, metrics logging, checkpoints, long background runs, evaluation and samples.
---

# Training models on the sandbox CPU

## Size to the machine
- Read CPUs and RAM from Resources in <environment>. There is no GPU, so set `device = "cpu"` and use
  `torch.set_num_threads(<CPUs>)`.
- Start small and grow only if the throughput allows. For a language model, 4–8 layers, d_model 256–384,
  4–8 heads and context 128–256 is a few million parameters.
- Estimate memory before running: params × 4 B × 4 (weights, grads, Adam) plus activations
  (batch × context × d_model × layers × ~20 B). Exit code 137 with an out-of-memory note means: reduce
  the batch or context, or use a memory-mapped dataset.
- Measure tokens/s over 20 steps first, then choose the step count that fits the time budget.

## Data
- Hold out a validation split (e.g. 1% of documents) before training, and never train on it.
- Store token ids as numpy arrays (`np.uint16` when vocab < 65536), memory-mapped with `np.memmap` or
  `np.load(mmap_mode="r")`. Sample random windows of the context length for each batch.
- Tokenizers: use the `tokenizers` library. Train it on a subset (~50–100 MB of text) and save it to JSON.
  A byte-level BPE needs the matching decoder; without it `decode` joins tokens with spaces and leaves the
  `Ġ` space markers in (`'Ġhel lo Ġworld'`). Set it before training and never post-process decode output:
  ```python
  from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
  tok = Tokenizer(models.BPE())
  tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
  tok.decoder = decoders.ByteLevel()
  trainer = trainers.BpeTrainer(vocab_size=16000, special_tokens=["<|endoftext|>"],
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
  tok.train(["data/train.txt"], trainer)   # a list of file paths
  tok.save("data/tokenizer.json")
  ```
  Check `tok.decode(tok.encode(s).ids) == s` on a few sentences before tokenizing the whole corpus.

## Training loop
- Use AdamW (lr 3e-4 to 1e-3, weight decay 0.1), a short warmup, then cosine decay, and gradient
  clipping at 1.0.
- Every N steps (e.g. 50), evaluate on a fixed number of validation batches under `torch.no_grad()` and
  append one JSON line to `metrics.jsonl`, flushing after each write:
  `{"step", "train_loss", "val_loss", "perplexity", "lr", "tokens_per_sec", "elapsed_s"}`, with
  `perplexity = exp(val_loss)`.
- Save a checkpoint (model + optimizer + step) every few hundred steps and at the end. Write to a temp file,
  then rename it, so a checkpoint is never half-written.
- Stop on a time budget or when val_loss hasn't improved for K evaluations. Write a final record
  `{"done": true, "reason": ...}`.
- Write `{"pid", "started", "status"}` to a small status file, so a UI can show whether training is running.

## Running it
- Test first: one forward pass with the real shapes, then 20 training steps, and check the loss
  decreases. Only then start the full run.
- Start the full run with `uv run python -u train.py` as its own command. It continues in the
  background, and you watch it with `read_process <id> wait_seconds=600`. Don't start a second copy.
- A loss that is NaN or not falling: lower the learning rate and check the data pipeline (decode a
  batch and look at it).

## Evaluate honestly
- Generate a few samples from the latest checkpoint (temperature 0.8, top-k 50) and look at them.
  Repeated tokens or gibberish means the model didn't learn yet.
- Typical success criteria for the plan (copy the form exactly):
  ```
  - [ ] val_loss decreased — verify: `uv run python -c "import json; m=[json.loads(l) for l in open('metrics.jsonl') if 'val_loss' in l]; assert m[-1]['val_loss'] < m[0]['val_loss']"`
  - [ ] trained >= 1000 steps — verify: `uv run python -c "import json; assert max(json.loads(l).get('step', 0) for l in open('metrics.jsonl')) >= 1000"`
  ```
