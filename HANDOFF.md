# Handoff notes — ltx-trainer-pinokio

Notes for any agent (Claude Code or otherwise) picking up this repo fresh.
Companion repo: [`ltx-trainer-mcp`](../ltx-trainer-mcp) (MCP version of
the same trainer). Both wrap the upstream
[LTX-2 trainer](https://github.com/Lightricks/LTX-2) by Lightricks.

---

## What this is

Pinokio launcher with a Gradio UI that wraps the official
`packages/ltx-trainer` package. Gives a human-driven end-to-end
workflow: dataset prep → YAML config authoring → training → generation.
Every action shells out to `uv run python .../scripts/<name>.py` in the
LTX-2 monorepo — this repo never imports trainer internals.

## Status at handoff (2026-04-16)

- **All six tabs built and boot-verified** on Windows 11 + RTX 6000 Ada:
  - **Project** — paths persisted to `app/.user_settings.json` (gitignored)
  - **Dataset Prep** — `split_scenes`, `caption_videos`, `process_dataset`
    with cancellable streaming logs
  - **Config** — YAML template loader (4 templates: lora / lora_low_vram
    / ic_lora / synthesized full_finetune), structured form + raw YAML
    textarea source-of-truth at save time
  - **Train** — single-GPU or `accelerate launch` (ddp/ddp_compile/
    fsdp/fsdp_compile), checkpoint browser, Linux-only WSL2 banner
    shown on Windows
  - **Generate** — all 8 ltx-pipelines modules, gr.Video preview
  - **Settings** — HF token, `uv` path override, "Check Installation"
    report
- **Auto hardware detection** writes `detected_hardware.json` on install
  (gitignored); picks `lora` vs `lora_low_vram` from VRAM (split at 40 GB).
- **No GitHub remote yet.** User said "I'll create the remotes later."
  Commit history is local-only.
- **No `icon.png`** — referenced in pinokio.js but not shipped. Pinokio
  renders a default icon if missing; add one when available.

## How to resume

```bash
# On any host — UI runs everywhere, but training subprocesses need Linux.
cd f:/__PROJECTS/LTXTrainer/ltx-trainer-pinokio

# If the `env/` venv is missing (fresh clone), recreate it:
uv venv env --python 3.12
uv pip install --python env/Scripts/python.exe -r app/requirements.txt
env/Scripts/python.exe app/detect_hardware.py    # writes detected_hardware.json

# Boot the UI (Pinokio normally does this via start.js):
env/Scripts/python.exe app/app.py --host 127.0.0.1 --port 7870
# On Linux / WSL2: env/bin/python app/app.py ...

# Then browse http://127.0.0.1:7870
```

The launcher expects LTX-2 cloned at `../LTX-2/` (sibling). Configurable
via the Project tab's "LTX-2 repo path" field.

## Open items

- **Wire GitHub remote** when the user provides a URL.
  Convention: `git@github.com:hoodtronik/ltx-trainer-pinokio.git`.
- **Ship an `icon.png`** at the repo root.
- **No end-to-end training test yet** — the user's Windows machine can't
  run `train.py` natively (triton/bitsandbytes are Linux-only). First
  full integration test will need a WSL2 or Linux session.
- **`app/LTX-2/` nested directory** was removed from this tree but is
  gitignored — if a user re-creates it (e.g. a stale install), it won't
  pollute the repo. The canonical LTX-2 location is `../LTX-2/`.

## Gotchas / real bugs encountered

- **`uv venv` creates a pip-less venv by default.** Don't try
  `python -m pip install ...` inside `env/` — use
  `uv pip install --python env/Scripts/python.exe ...` instead. All three
  launcher scripts (`install.js`, `update.js`, `download_models.js`)
  follow this pattern.
- **Gradio 6 moved `theme` from `Blocks()` to `launch()`** and removed
  `show_copy_button` from `Textbox`. `app/app.py` is already adapted.
- **Gradio Dropdown `choices=[(label, value)]` tuple format works** in
  Gradio 6 — used in the Config tab's template picker.
- **Model naming:** the upstream repo has **two** model lines —
  `Lightricks/LTX-2.3` (22B, current) and `Lightricks/LTX-2` (19B,
  legacy). `download_models.js` defaults to 22B with a dropdown for 19B.
  Both are supported by the same trainer — don't hardcode either.

## File map

- `pinokio.js` — launcher manifest with dynamic menu
- `install.js`, `start.js`, `update.js`, `reset.js`, `download_models.js`
- `app/app.py` — Gradio UI, all six tabs, ~1000 lines
- `app/runner.py` — subprocess streaming (split_scenes, caption_videos,
  process_dataset, train, generate, list_checkpoints)
- `app/config_builder.py` — YAML template load + merge + serialize for
  the Config tab
- `app/settings.py` — JSON-backed user paths
- `app/hardware.py`, `app/detect_hardware.py` — GPU detection + template
  recommendation
- `app/requirements.txt` — Gradio 6, pyyaml, pillow
- `AGENTS.md` — non-Claude-Code agent rules (CLAUDE-NOTE convention)

## Commit log (local)

```
5faf049 add Generate tab with ltx-pipelines inference
e39ad1f add Train tab with multi-GPU, streaming logs, checkpoint browser
75e9251 add Config tab with YAML template loader + editor
fbe29a5 untrack stray app/LTX-2 clone; use sibling ../LTX-2 only
ae2c8e5 add Project, Dataset Prep, and Settings tabs
abf8109 scaffold Pinokio launcher for LTX-2 trainer
```

CLAUDE-NOTE comments throughout the code explain non-obvious decisions.
Trust them — they're the source of truth per the convention in
[`AGENTS.md`](./AGENTS.md).
