# LTX-Trainer Pinokio Launcher

One-click [Pinokio](https://pinokio.computer) launcher with a Gradio UI for
the official [LTX-2 trainer](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer)
by Lightricks — LoRA, audio-video LoRA, IC-LoRA, and full fine-tuning for
LTX-2 video generation.

## What this does

Wraps the official `ltx-trainer` package with:

- **Dataset prep** — scene splitting, auto-captioning, and preprocessing
  (latents + text embeddings) through a UI instead of CLI.
- **Config authoring** — YAML config editor with templates (standard LoRA,
  low-VRAM LoRA, IC-LoRA, full fine-tune) instead of hand-editing files.
- **Training** — streaming logs, cancel, checkpoint browser.
- **Generation** — inference with your trained LoRA using the
  `ltx-pipelines` package.

It does **not** reimplement any trainer logic — every action shells out to
`uv run python packages/ltx-trainer/scripts/<script>.py` in the LTX-2 repo.

## Platform support

| Platform | Launcher UI | Actual training |
|----------|-------------|-----------------|
| Linux (CUDA) | ✅ | ✅ |
| Windows (WSL2) | ✅ | ✅ |
| Windows (native) | ✅ | ❌ (triton + bitsandbytes are Linux-only) |
| macOS | ✅ | ❌ (no CUDA) |

## Install (via Pinokio)

1. Install [Pinokio](https://pinokio.computer/).
2. In Pinokio, click **Download from URL** and paste this repo's git URL.
3. Open the downloaded app and click **Install**. This:
   - Installs `uv` if it isn't already on PATH
   - Runs `uv sync` in `../LTX-2/` (the trainer monorepo — clone it to
     the sibling directory yourself, or point the launcher at it via
     Settings)
   - Creates a separate venv at `env/` for the Gradio UI
4. Optional — click **Download Models** to pull the LTX-2 checkpoint and
   Gemma text encoder from HuggingFace (~80 GB; skip if you already have
   them or want to use your own quantized copy).
5. Click **Start** — the Gradio URL will open automatically.

## Directory layout this launcher expects

```
<parent>/
├── LTX-2/                       ← official trainer monorepo (clone yourself)
│   └── packages/ltx-trainer/...
├── ltx-trainer-pinokio/         ← THIS REPO
│   ├── pinokio.js
│   ├── install.js, start.js, ...
│   ├── env/                     ← created by install
│   └── app/
│       ├── app.py               ← Gradio UI
│       ├── models/              ← optional, for Download Models
│       └── outputs/             ← training outputs land here by default
```

You can point the launcher at an LTX-2 repo in a different location via
the **Settings** tab.

## Companion tools

- [klippbok-mcp](https://github.com/hoodtronik/klippbok-mcp) — dataset curation MCP server
- [musubi-mcp](https://github.com/hoodtronik/musubi-mcp) — Wan/FLUX/Z-Image training MCP server
- [ltx-trainer-mcp](https://github.com/hoodtronik/ltx-trainer-mcp) — agent-driven LTX-2 training (MCP version of this launcher)

## License

Apache 2.0. Wraps the Apache 2.0-licensed
[LTX-2 trainer](https://github.com/Lightricks/LTX-2).
