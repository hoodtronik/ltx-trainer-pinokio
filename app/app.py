"""LTX-Trainer Pinokio Gradio UI.

TODO (incremental build plan):
  [x] Scaffolding + hello world (this commit)
  [ ] Project tab — LTX-2 dir + model paths, persisted in .user_settings.json
  [ ] Dataset Prep tab — split_scenes, caption_videos, process_dataset
      with streaming subprocess output
  [ ] Config tab — YAML template loader + structured editor +
      raw YAML textarea
  [ ] Train tab — launch train.py, stream logs, checkpoint browser
  [ ] Generate tab — ltx-pipelines inference with LoRA
  [ ] Settings tab — paths + Check Installation + HF token

Runs as a standalone Python program. Accepts --host and --port.
Never auto-opens a browser (Pinokio does that).
"""

from __future__ import annotations

import argparse

import gradio as gr

from hardware import get_hardware_info

APP_TITLE = "LTX-Trainer"
APP_SUBTITLE = "Gradio UI for the official LTX-2 video model trainer"


def _hardware_banner_md() -> str:
    info = get_hardware_info()
    if info.get("gpu_detected"):
        gpu_line = (
            f"**GPU:** {info['gpu_name']} &nbsp;·&nbsp; "
            f"**VRAM:** {info['vram_mib']} MiB &nbsp;·&nbsp; "
            f"**Recommended template:** `{info['recommended_template']}`"
        )
    else:
        gpu_line = f"**GPU:** _not detected_ &nbsp;·&nbsp; {info.get('reason', '')}"
    os_line = ""
    if info.get("training_supported_natively") is False:
        os_line = (
            "\n\n> ⚠️ **Training not supported on this OS natively.** "
            "LTX-2 depends on `triton` and `bitsandbytes`, which are "
            "Linux-only. The UI will run, but `train.py` / "
            "`process_dataset.py` subprocesses will fail. Use WSL2 or a "
            "Linux host for actual training."
        )
    return gpu_line + os_line


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as ui:
        gr.Markdown(f"# {APP_TITLE}\n\n{APP_SUBTITLE}")
        gr.Markdown(_hardware_banner_md())
        gr.Markdown(
            "⚠️ **Scaffolding only — tabs not wired up yet.** This is the "
            "initial commit; real functionality lands in the next commits. "
            "The launcher, install, and Pinokio plumbing are complete; the "
            "UI tabs themselves are the next milestone.",
        )
        with gr.Tabs():
            with gr.Tab("Project"):
                gr.Markdown("*Coming next commit.*")
            with gr.Tab("Dataset Prep"):
                gr.Markdown("*Coming next commit.*")
            with gr.Tab("Config"):
                gr.Markdown("*Coming after Dataset Prep.*")
            with gr.Tab("Train"):
                gr.Markdown("*Coming after Config.*")
            with gr.Tab("Generate"):
                gr.Markdown("*Coming last.*")
            with gr.Tab("Settings"):
                gr.Markdown("*Coming with Project tab.*")

    return ui


def main() -> None:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    args = parser.parse_args()

    ui = build_ui()
    # CLAUDE-NOTE: inbrowser=False — Pinokio captures the URL from stdout
    # and opens it through its own menu, so launching a browser here would
    # create a duplicate window.
    ui.launch(server_name=args.host, server_port=args.port, inbrowser=False)


if __name__ == "__main__":
    main()
