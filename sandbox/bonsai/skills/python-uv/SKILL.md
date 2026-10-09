---
name: python-uv
description: Use for any Python work here (installing packages, PyTorch, scripts, training jobs, data processing, serving results). Explains the uv workflow; there is no pip.
---

# Python with uv

The sandbox has Python 3.11 and uv, but no pip. Every Python dependency goes through uv.

## The project
/workspace is already a uv project (pyproject.toml + .venv), so there's no `uv init` and no activation.
```bash
uv add numpy requests               # install: written to pyproject.toml, installed into /workspace/.venv
uv run python train.py              # run inside the project venv (always use uv run, not plain python3)
uv run python -c "import numpy; print(numpy.__version__)"
uv remove requests                  # uninstall
```
- Never delete `.venv` or create another venv. uv is on PATH, not inside `.venv`.
- `pip` doesn't exist, and `uv pip install` without a venv fails ("No virtual environment found"). Use `uv add`.

## PyTorch (CPU only, no GPU in this container)
```bash
uv add torch                        # about 20 s; the CPU index is preconfigured in pyproject.toml
uv run python -c "import torch; print(torch.__version__, torch.get_num_threads())"
```
Size models, batches and data to the CPUs and RAM listed under Resources in <environment> (`nproc` and
`cat /sys/fs/cgroup/memory.max` show them too). Exceeding the RAM limit kills the process (exit code 137).

## Check APIs instead of guessing: pyapi
Your memory of library APIs can be wrong or out of date for the installed version. Look them up:
```bash
pyapi tokenizers.trainers.BpeTrainer          # signature, docstring, members, installed version
pyapi torch.nn --search transformer           # find names containing "transformer"
pyapi xml.etree.ElementTree.iterparse         # the standard library works too
```
Wrong names get "did you mean" suggestions (e.g. `pyapi tokenizers.models.BPEE` suggests `BPE`). Use it
before coding against a library you're unsure about, and after ImportError, AttributeError or a TypeError
about arguments. Never retry the same guess twice.

## Use libraries for standard algorithms
Hand-written versions of standard algorithms cost many steps and are often buggy or slow. Use the library:
- tokenizers (BPE/WordPiece training in seconds): `uv add tokenizers`, then `from tokenizers import Tokenizer, models, trainers`
- data loading: `datasets`; numerics: `numpy`; models and training: `torch` (`torch.nn.TransformerEncoderLayer`,
  `torch.nn.functional.scaled_dot_product_attention`); web APIs: `fastapi`.
Test on a tiny input first, with `timeout 60`, before running on the full data.

## Build and verify one stage at a time
Run each script right after writing it, on a small sample, before writing the next one: download, then
preprocess (check the output files), then the model (one forward pass), then training (a few steps), then the UI.
Mark a plan step done only after it ran successfully. Writing everything first and running at the end piles
all the bugs into the last steps.
- Never name a file after a standard-library module (tokenize.py, random.py, types.py, queue.py, …): it
  shadows the real module for everything run from that folder. The python check flags this.

## Long-running jobs (downloads, training)
- A command still running after 30 s continues in the background as a process id. Check on it with
  read_process, and stop it with kill_process. Don't add `&`.
- Make long jobs observable: print progress regularly, and write metrics to a file (e.g. `metrics.jsonl`,
  one JSON object per line) that a UI can read. Flush output (`print(..., flush=True)` or `python -u`).
- Download big files with `curl -fL --retry 3 -o data/file URL` as their own command, then check the size
  with `ls -lh` and look at a sample before parsing: `bzcat f.bz2 | head -50`, `zcat f.gz | head`, `head -c 2000 f`,
  `jq '.[0]' f.json`, `file f`. Write the parser for the structure you saw, and test it on the first few records. Large compressed corpora: stream-decompress (`bz2.open`) and use a subset first.

## Showing results in the browser
- The simplest option is a Python server, with no Node needed: `uv add fastapi uvicorn`, then serve an HTML page plus a
  JSON endpoint, started with `uv run uvicorn app:app --host 0.0.0.0 --port 8000`. The preview attaches to
  the port it opens.
- Or use the React template (/opt/template) for a richer UI, with Vite proxying `/api` to the Python server.
