# Sandbox
- Debian container: Node 22 + npm, Python 3.11 + uv, git, ripgrep, curl, jq, file, bzip2/xz/unzip. Internet access
  works for downloads and package installs.
- The project is /workspace. It starts as an empty uv Python project (pyproject.toml + .venv); add files as needed.
- Whatever the project serves on a port is shown to the user in a live preview.
- Verify web pages in a real browser, not with curl (curl doesn't run JavaScript): `check-page http://localhost:PORT/`
  loads the page headlessly and reports JavaScript errors, failed requests, a blank page and layout problems.
  The harness also runs it automatically after edits while something is being served.
- The user can restore checkpoints of /workspace; you'll be told when that happens.

## Python: always through uv (there is no pip)
- Install with `uv add <pkg>` and run with `uv run python script.py`. /workspace is already a uv project, so
  don't run `uv init`, don't create another venv and never delete `.venv`.
- `pip`, `python3 -m pip` and plain `python3` don't see the project's packages; use `uv run`.
- PyTorch: `uv add torch` installs the CPU build (the CPU index is preconfigured; there is no GPU).
- Don't guess APIs: `pyapi <module.name>` shows the installed version, signature, docs and members, e.g.
  `pyapi tokenizers.trainers.BpeTrainer` or `pyapi torch.nn --search Transformer`. Use it before writing code
  against an unfamiliar or fast-moving library, and after any ImportError, AttributeError or unexpected-keyword TypeError.
- Read the python-uv skill for long-running jobs and serving a UI from Python.

## Web UIs
- Starter template: /opt/template (React + TypeScript + Vite, dependencies installed). For a web UI:
  `cp -r /opt/template/. /workspace/` once, at the start (it adds package.json, src/ … and its own AGENTS.md).
  Never scaffold with `npm create`. A Python-only project doesn't need it (serve a page from Python instead).
