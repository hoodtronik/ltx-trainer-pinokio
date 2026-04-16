"""LTX-Trainer Pinokio Gradio UI.

TODO (incremental build plan):
  [x] Scaffolding + hello world
  [x] Project tab — LTX-2 dir + model paths, persisted in .user_settings.json
  [x] Dataset Prep tab — split_scenes, caption_videos, process_dataset
      with streaming subprocess output
  [ ] Config tab — YAML template loader + structured editor +
      raw YAML textarea
  [ ] Train tab — launch train.py, stream logs, checkpoint browser
  [ ] Generate tab — ltx-pipelines inference with LoRA
  [x] Settings tab — paths + Check Installation + HF token

Runs as a standalone Python program. Accepts --host and --port.
Never auto-opens a browser (Pinokio does that).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr

import runner
import settings as settings_mod
from hardware import get_hardware_info

APP_TITLE = "LTX-Trainer"
APP_SUBTITLE = "Gradio UI for the official LTX-2 video model trainer"

RESOLUTION_BUCKETS_HELP = (
    'Format: `"WxHxF"` where W,H are multiples of 32 and F % 8 == 1. '
    "Valid F: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121. "
    'Chain multiple with `;`, e.g. `"960x544x49;512x512x81"` '
    "(requires batch_size=1 in the training config)."
)


# =============================================================================
# Shared helpers
# =============================================================================


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
            "Linux-only. The UI runs, but `train.py` / `process_dataset.py` "
            "subprocesses will fail. Use WSL2 or a Linux host for training."
        )
    return gpu_line + os_line


def _load_settings_dict() -> dict[str, str]:
    """Return the current settings as a plain dict (for populating Textboxes)."""
    s = settings_mod.load()
    return {
        "ltx_repo_path": s.ltx_repo_path,
        "model_path": s.model_path,
        "text_encoder_path": s.text_encoder_path,
        "spatial_upscaler_path": s.spatial_upscaler_path,
        "distilled_lora_path": s.distilled_lora_path,
        "output_dir": s.output_dir,
        "hf_token": s.hf_token,
        "uv_path": s.uv_path,
        "python_path": s.python_path,
    }


def _save_settings_from_ui(
    ltx_repo_path: str,
    model_path: str,
    text_encoder_path: str,
    spatial_upscaler_path: str,
    distilled_lora_path: str,
    output_dir: str,
    hf_token: str,
    uv_path: str,
    python_path: str,
) -> str:
    s = settings_mod.Settings(
        ltx_repo_path=ltx_repo_path.strip(),
        model_path=model_path.strip(),
        text_encoder_path=text_encoder_path.strip(),
        spatial_upscaler_path=spatial_upscaler_path.strip(),
        distilled_lora_path=distilled_lora_path.strip(),
        output_dir=output_dir.strip(),
        hf_token=hf_token.strip(),
        uv_path=uv_path.strip() or "uv",
        python_path=python_path.strip(),
    )
    settings_mod.save(s)
    return f"✅ Saved to `{settings_mod.SETTINGS_PATH}`"


# =============================================================================
# Dataset Prep generators (wrap runner.py with session-state process handle)
# =============================================================================


def _stream_with_state(gen):
    """Adapt a runner generator `(log, process)` into Gradio's yield shape.

    Yields (log_text, process_handle) so the UI can attach the handle to
    a gr.State for cancellation.
    """
    proc = None
    try:
        for log, handle in gen:
            if handle is not None:
                proc = handle
            yield log, proc
    except runner.RunnerError as exc:
        yield f"[runner error] {exc}\n", proc


def split_scenes_fn(
    input_video: str,
    output_dir: str,
    detector: str,
    filter_shorter_than: str,
    min_scene_length: str,
    threshold: str,
    extra_args: str,
):
    if not input_video.strip() or not output_dir.strip():
        yield "Please provide both an input video and an output directory.", None
        return
    s = settings_mod.load()
    yield from _stream_with_state(
        runner.run_split_scenes(
            s,
            input_video=input_video,
            output_dir=output_dir,
            detector=detector,
            filter_shorter_than=filter_shorter_than,
            min_scene_length=min_scene_length,
            threshold=threshold,
            extra_args=extra_args,
        ),
    )


def caption_videos_fn(
    input_dir: str,
    output_json: str,
    captioner_type: str,
    api_key: str,
    num_workers: str,
    no_audio: bool,
    use_8bit: bool,
    clean_caption: bool,
    extra_args: str,
):
    if not input_dir.strip() or not output_json.strip():
        yield "Please provide both an input directory and an output JSON path.", None
        return
    s = settings_mod.load()
    yield from _stream_with_state(
        runner.run_caption_videos(
            s,
            input_dir=input_dir,
            output_json=output_json,
            captioner_type=captioner_type,
            api_key=api_key,
            num_workers=num_workers,
            no_audio=no_audio,
            use_8bit=use_8bit,
            clean_caption=clean_caption,
            extra_args=extra_args,
        ),
    )


def process_dataset_fn(
    dataset_json: str,
    resolution_buckets: str,
    with_audio: bool,
    reference_column: str,
    batch_size: str,
    vae_tiling: bool,
    decode: bool,
    output_dir: str,
    lora_trigger: str,
    extra_args: str,
):
    if not dataset_json.strip() or not resolution_buckets.strip():
        yield "Please provide both a dataset JSON path and resolution buckets.", None
        return
    s = settings_mod.load()
    yield from _stream_with_state(
        runner.run_process_dataset(
            s,
            dataset_json=dataset_json,
            resolution_buckets=resolution_buckets,
            with_audio=with_audio,
            reference_column=reference_column,
            batch_size=batch_size,
            vae_tiling=vae_tiling,
            decode=decode,
            output_dir=output_dir,
            lora_trigger=lora_trigger,
            extra_args=extra_args,
        ),
    )


def cancel_fn(proc):
    msg = runner.cancel_process(proc)
    return msg


# =============================================================================
# Check Installation (Settings tab)
# =============================================================================


def check_installation_fn() -> str:
    """Return a human-readable report of what's set up and what's missing."""
    s = settings_mod.load()
    lines: list[str] = []

    lines.append("## Installation check\n")

    # Hardware
    info = get_hardware_info()
    if info.get("gpu_detected"):
        lines.append(
            f"- ✅ GPU: **{info['gpu_name']}** ({info['vram_mib']} MiB)",
        )
    else:
        lines.append(f"- ❌ GPU: not detected — {info.get('reason', '')}")

    if info.get("training_supported_natively") is False:
        lines.append(
            f"- ⚠️  OS: **{info.get('os', '?')}** — training requires Linux "
            "(use WSL2 for actual training).",
        )
    elif info.get("training_supported_natively") is True:
        lines.append(f"- ✅ OS: **{info.get('os', '?')}** (training supported)")

    lines.append("")
    lines.append("### Paths\n")

    # LTX-2 repo
    ltx_root = Path(settings_mod.resolve_path(s.ltx_repo_path)) if s.ltx_repo_path else None
    if ltx_root and ltx_root.is_dir():
        scripts_dir = ltx_root / "packages" / "ltx-trainer" / "scripts"
        if scripts_dir.is_dir():
            lines.append(f"- ✅ LTX-2 repo: `{ltx_root}`")
        else:
            lines.append(
                f"- ⚠️  LTX-2 path exists but is not the monorepo: `{ltx_root}`",
            )
    else:
        lines.append(f"- ❌ LTX-2 repo not found: `{s.ltx_repo_path or '(unset)'}`")

    # Individual model files
    for label, path in [
        ("Model checkpoint", s.model_path),
        ("Text encoder dir", s.text_encoder_path),
        ("Spatial upscaler", s.spatial_upscaler_path),
        ("Distilled LoRA", s.distilled_lora_path),
    ]:
        if not path:
            lines.append(f"- ⚪ {label}: _(not set)_")
            continue
        resolved = Path(settings_mod.resolve_path(path))
        if resolved.exists():
            lines.append(f"- ✅ {label}: `{resolved}`")
        else:
            lines.append(f"- ❌ {label} not found: `{resolved}`")

    # Output dir
    out_dir = Path(settings_mod.resolve_path(s.output_dir)) if s.output_dir else None
    if out_dir:
        lines.append(f"- 📁 Output dir: `{out_dir}` (will be created if missing)")

    lines.append("")
    lines.append("### Raw detected_hardware.json\n")
    lines.append("```json\n" + json.dumps(info, indent=2) + "\n```")

    return "\n".join(lines)


# =============================================================================
# UI construction
# =============================================================================


def _project_tab(initial: dict) -> tuple[dict, gr.Markdown]:
    """Build the Project tab. Returns ({field_name: component}, save_status)."""
    gr.Markdown(
        "Configure the paths the rest of the UI uses. Relative paths are "
        "resolved against the launcher repo root. Changes save to "
        "`app/.user_settings.json` (gitignored).",
    )

    with gr.Row():
        ltx_repo_path = gr.Textbox(
            label="LTX-2 repo path",
            value=initial["ltx_repo_path"],
            placeholder="../LTX-2",
            info="The cloned LTX-2 monorepo. Default: `../LTX-2` (sibling dir).",
        )
        output_dir = gr.Textbox(
            label="Training output dir",
            value=initial["output_dir"],
            placeholder="app/outputs",
            info="Where checkpoints and validation samples land.",
        )

    gr.Markdown("### Model paths")
    with gr.Row():
        model_path = gr.Textbox(
            label="Model checkpoint (.safetensors)",
            value=initial["model_path"],
            placeholder="app/models/ltx-2/ltx-2-19b-dev.safetensors",
        )
        text_encoder_path = gr.Textbox(
            label="Gemma text encoder (directory)",
            value=initial["text_encoder_path"],
            placeholder="app/models/gemma-3-12b-it-qat-q4_0-unquantized",
            info="Must be a directory, not a .safetensors file.",
        )
    with gr.Row():
        spatial_upscaler_path = gr.Textbox(
            label="Spatial upscaler (optional, inference only)",
            value=initial["spatial_upscaler_path"],
            placeholder="app/models/ltx-2/ltx-2-spatial-upscaler-x2-1.0.safetensors",
        )
        distilled_lora_path = gr.Textbox(
            label="Distilled LoRA (optional, inference only)",
            value=initial["distilled_lora_path"],
            placeholder="app/models/ltx-2/ltx-2-22b-distilled-lora-384.safetensors",
        )

    save_btn = gr.Button("💾 Save paths", variant="primary")
    save_status = gr.Markdown("")

    fields = {
        "ltx_repo_path": ltx_repo_path,
        "model_path": model_path,
        "text_encoder_path": text_encoder_path,
        "spatial_upscaler_path": spatial_upscaler_path,
        "distilled_lora_path": distilled_lora_path,
        "output_dir": output_dir,
    }
    return fields, save_btn, save_status


def _dataset_prep_tab(default_output_dir: str) -> None:
    gr.Markdown(
        "Three-step pipeline. Scene split and captioning are optional — "
        "the only required step is **Preprocess** (computes latents + text "
        "embeddings that training actually reads). All three invoke "
        "`uv run python packages/ltx-trainer/scripts/<name>.py` inside the "
        "LTX-2 repo.",
    )

    # --- Step 1: split_scenes ---
    with gr.Accordion("① Scene split (optional)", open=False):
        gr.Markdown(
            "Splits a long video into per-scene clips using PySceneDetect. "
            "Useful if your source is one continuous file.",
        )
        ss_input = gr.Textbox(label="Input video path", placeholder="/path/to/long_source.mp4")
        ss_output = gr.Textbox(
            label="Output directory",
            placeholder=f"{default_output_dir}/scenes",
        )
        with gr.Row():
            ss_detector = gr.Dropdown(
                label="Detector",
                choices=["content", "adaptive", "threshold", "histogram"],
                value="content",
            )
            ss_filter = gr.Textbox(
                label="Filter shorter than",
                placeholder="5s",
                info="Drop scenes shorter than this (e.g. 5s, 2000ms).",
            )
            ss_min_len = gr.Textbox(
                label="Min scene length",
                placeholder="e.g. 48 (frames)",
            )
            ss_threshold = gr.Textbox(
                label="Detector threshold",
                placeholder="e.g. 27.0",
            )
        ss_extra = gr.Textbox(
            label="Extra args (space-separated, advanced)",
            placeholder="--save-images 1",
        )
        with gr.Row():
            ss_run = gr.Button("▶️ Run scene split", variant="primary")
            ss_cancel = gr.Button("■ Cancel")
        ss_log = gr.Textbox(label="Output", lines=16, max_lines=40, interactive=False)
        ss_proc = gr.State(None)

        ss_run.click(
            fn=split_scenes_fn,
            inputs=[ss_input, ss_output, ss_detector, ss_filter, ss_min_len, ss_threshold, ss_extra],
            outputs=[ss_log, ss_proc],
        )
        ss_cancel.click(fn=cancel_fn, inputs=[ss_proc], outputs=[ss_log])

    # --- Step 2: caption_videos ---
    with gr.Accordion("② Caption videos (optional)", open=False):
        gr.Markdown(
            "Auto-generate `[VISUAL] / [SPEECH] / [SOUNDS] / [TEXT]` "
            "captions for each clip. Two captioner backends: `qwen_omni` "
            "(local multimodal) and `gemini_flash` (Google API — "
            "requires API key).",
        )
        cv_input = gr.Textbox(label="Input directory (videos)", placeholder=f"{default_output_dir}/scenes")
        cv_output = gr.Textbox(
            label="Output dataset JSON",
            placeholder=f"{default_output_dir}/dataset.json",
        )
        with gr.Row():
            cv_type = gr.Dropdown(
                label="Captioner",
                choices=["qwen_omni", "gemini_flash"],
                value="qwen_omni",
            )
            cv_api_key = gr.Textbox(
                label="API key (gemini_flash only)",
                type="password",
                placeholder="Leave blank for qwen_omni",
            )
            cv_workers = gr.Textbox(label="Num workers", placeholder="e.g. 5")
        with gr.Row():
            cv_no_audio = gr.Checkbox(label="No audio (visual-only captions)", value=False)
            cv_8bit = gr.Checkbox(label="Use 8-bit model (qwen_omni, lower VRAM)", value=False)
            cv_clean = gr.Checkbox(label="Clean captions (strip bracketed prefixes)", value=False)
        cv_extra = gr.Textbox(
            label="Extra args (space-separated, advanced)",
            placeholder="--recursive --fps 1",
        )
        with gr.Row():
            cv_run = gr.Button("▶️ Run caption", variant="primary")
            cv_cancel = gr.Button("■ Cancel")
        cv_log = gr.Textbox(label="Output", lines=16, max_lines=40, interactive=False)
        cv_proc = gr.State(None)

        cv_run.click(
            fn=caption_videos_fn,
            inputs=[cv_input, cv_output, cv_type, cv_api_key, cv_workers, cv_no_audio, cv_8bit, cv_clean, cv_extra],
            outputs=[cv_log, cv_proc],
        )
        cv_cancel.click(fn=cancel_fn, inputs=[cv_proc], outputs=[cv_log])

    # --- Step 3: process_dataset ---
    with gr.Accordion("③ Preprocess (REQUIRED)", open=True):
        gr.Markdown(
            "Compute video latents + text embeddings. This is the heavy step "
            "(VAE encode + Gemma embed) and is what `train.py` actually reads. "
            "**Model paths come from the Project tab.**",
        )
        pd_json = gr.Textbox(
            label="Dataset JSON",
            placeholder=f"{default_output_dir}/dataset.json",
        )
        pd_buckets = gr.Textbox(
            label="Resolution buckets",
            value="960x544x49",
            info=RESOLUTION_BUCKETS_HELP,
        )
        with gr.Row():
            pd_with_audio = gr.Checkbox(label="With audio (joint A/V preprocessing)", value=False)
            pd_vae_tiling = gr.Checkbox(label="VAE tiling (lower peak VRAM)", value=False)
            pd_decode = gr.Checkbox(label="Decode sanity check (writes preview mp4s)", value=False)
        with gr.Row():
            pd_batch = gr.Textbox(label="Batch size", placeholder="default")
            pd_ref_col = gr.Textbox(
                label="Reference column (IC-LoRA)",
                placeholder="e.g. reference_path",
            )
            pd_trigger = gr.Textbox(label="LoRA trigger word (optional)")
        pd_output = gr.Textbox(
            label="Output dir (optional, defaults to <dataset_dir>/.precomputed)",
            placeholder="",
        )
        pd_extra = gr.Textbox(
            label="Extra args (space-separated, advanced)",
            placeholder="--reference-downscale-factor 2",
        )
        with gr.Row():
            pd_run = gr.Button("▶️ Run preprocess", variant="primary")
            pd_cancel = gr.Button("■ Cancel")
        pd_log = gr.Textbox(label="Output", lines=20, max_lines=60, interactive=False)
        pd_proc = gr.State(None)

        pd_run.click(
            fn=process_dataset_fn,
            inputs=[
                pd_json, pd_buckets, pd_with_audio, pd_ref_col, pd_batch,
                pd_vae_tiling, pd_decode, pd_output, pd_trigger, pd_extra,
            ],
            outputs=[pd_log, pd_proc],
        )
        pd_cancel.click(fn=cancel_fn, inputs=[pd_proc], outputs=[pd_log])


def _settings_tab(initial: dict) -> None:
    gr.Markdown(
        "Advanced knobs rarely need tweaking. The HuggingFace token is used "
        "only for model downloads and never written to config files or logs.",
    )
    with gr.Row():
        hf_token = gr.Textbox(
            label="HuggingFace token",
            value=initial["hf_token"],
            type="password",
            info="For downloading gated models (Gemma). Optional otherwise.",
        )
    with gr.Row():
        uv_path = gr.Textbox(
            label="uv executable",
            value=initial["uv_path"],
            info="Override if `uv` isn't on PATH.",
        )
        python_path = gr.Textbox(
            label="Python path (rarely used)",
            value=initial["python_path"],
            info="Leave blank — `uv run python` handles this. Power users only.",
        )

    save_btn = gr.Button("💾 Save advanced settings", variant="primary")
    save_status = gr.Markdown("")

    gr.Markdown("---")
    gr.Markdown("### Check installation")
    check_btn = gr.Button("🔍 Check installation")
    check_output = gr.Markdown("")
    check_btn.click(fn=check_installation_fn, inputs=[], outputs=[check_output])

    return hf_token, uv_path, python_path, save_btn, save_status


def build_ui() -> gr.Blocks:
    initial = _load_settings_dict()

    # CLAUDE-NOTE: Gradio 6+ moved `theme` from Blocks() to launch(). The
    # theme is applied in main() below via ui.launch(theme=...).
    with gr.Blocks(title=APP_TITLE) as ui:
        gr.Markdown(f"# {APP_TITLE}\n\n{APP_SUBTITLE}")
        gr.Markdown(_hardware_banner_md())

        with gr.Tabs():
            with gr.Tab("Project"):
                project_fields, project_save_btn, project_save_status = _project_tab(initial)

            with gr.Tab("Dataset Prep"):
                _dataset_prep_tab(default_output_dir=initial["output_dir"])

            with gr.Tab("Config"):
                gr.Markdown("*Coming next — YAML template loader + editor.*")

            with gr.Tab("Train"):
                gr.Markdown("*Coming after Config.*")

            with gr.Tab("Generate"):
                gr.Markdown("*Coming last.*")

            with gr.Tab("Settings"):
                hf_token, uv_path, python_path, settings_save_btn, settings_save_status = _settings_tab(initial)

        # CLAUDE-NOTE: Wire Save buttons across tabs. Both Project and Settings
        # save the same Settings object — we pass all fields from both tabs
        # into `_save_settings_from_ui`. This means clicking Save on either
        # tab persists the full settings snapshot, which avoids having to
        # merge partial writes from two places.
        all_inputs = [
            project_fields["ltx_repo_path"],
            project_fields["model_path"],
            project_fields["text_encoder_path"],
            project_fields["spatial_upscaler_path"],
            project_fields["distilled_lora_path"],
            project_fields["output_dir"],
            hf_token,
            uv_path,
            python_path,
        ]
        project_save_btn.click(fn=_save_settings_from_ui, inputs=all_inputs, outputs=[project_save_status])
        settings_save_btn.click(fn=_save_settings_from_ui, inputs=all_inputs, outputs=[settings_save_status])

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
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
