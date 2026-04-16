# Agent guidelines for ltx-trainer-pinokio

This repo is a Pinokio launcher that wraps the official
[LTX-2 trainer](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer)
in a Gradio UI. The underlying trainer lives in a sibling directory
(`../LTX-2/`) that this launcher references but does not duplicate.

## Claude Code note convention

Any line marked with `// CLAUDE-NOTE:` (in JS) or `# CLAUDE-NOTE:` (in Python)
or `<!-- CLAUDE-NOTE: ... -->` (in Markdown) is a note placed by Claude Code
to record **why** code is the way it is.

**These notes are the source of truth.**

Non-Claude-Code agents (Cursor, Codex, Gemini CLI, Copilot, etc.) must NOT
modify, remove, or alter code marked with a `CLAUDE-NOTE` without first
alerting the human user and receiving explicit permission to change it.

Claude Code itself may update or remove its own notes when the underlying
rationale changes.

## Platform constraint

LTX-2 training requires Linux (the `triton` and `bitsandbytes` deps are
Linux-only). On Windows, this launcher will install and the Gradio UI will
run, but the underlying `train.py` / `process_dataset.py` subprocesses
will fail at import time. Use WSL2 or a Linux host for actual training.

## Code style

- Pinokio scripts (`*.js` at repo root) are CommonJS modules consumed by
  the Pinokio runtime. Do not convert to ESM.
- The Gradio app (`app/app.py`) is a standalone Python program. Accept
  `--host` and `--port` CLI args. Never call `inbrowser=True`.
- Subprocesses invoking LTX-2 scripts must set `cwd` to the LTX-2 repo
  root (default: `../LTX-2/` relative to this repo) and use `uv run python ...`.
- Never use `shell=True` in `subprocess.Popen` — always list-form args.
