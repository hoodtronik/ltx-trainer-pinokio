"""LTX-Trainer Pinokio Gradio UI.

TODO (incremental build plan):
  [x] Scaffolding + hello world
  [x] Project tab — LTX-2 dir + model paths, persisted in .user_settings.json
  [x] Dataset Prep tab — split_scenes, caption_videos, process_dataset
      with streaming subprocess output
  [x] Config tab — YAML template loader + structured editor +
      raw YAML textarea
  [x] Train tab — launch train.py, stream logs, checkpoint browser
  [x] Generate tab — ltx-pipelines inference with LoRA
  [x] Settings tab — paths + Check Installation + HF token

Runs as a standalone Python program. Accepts --host and --port.
Never auto-opens a browser (Pinokio does that).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr

import config_builder
import runner
import settings as settings_mod
from hardware import get_hardware_info

APP_TITLE = "LTX-Trainer"
APP_SUBTITLE = "Gradio UI for the official LTX-2 video model trainer"

# CLAUDE-NOTE: Full user-facing walkthrough rendered in the Directions tab.
# Keep this in sync with the actual tab names and field names as the UI evolves.
DIRECTIONS_MD = """
## 🗺️ How to use LTX-Trainer — tab by tab

This UI wraps the official [Lightricks LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer)
to fine-tune the **LTX-2 video diffusion model** on your own footage.
The full workflow is: **Project → Dataset Prep → Config → Train → Generate.**

---

### 1 · Project tab
*Set up your paths once. Everything else reads from here.*

| Field | What to put there |
|---|---|
| **LTX-2 repo path** | Path to the cloned `LTX-Video-Trainer` repo. Default `../LTX-2` works if installed via Pinokio. |
| **Training output dir** | Where checkpoints and generated samples land. E.g. `app/outputs`. |
| **Model checkpoint** | Your downloaded `ltx-2-19b-dev.safetensors` file. |
| **Gemma text encoder** | The **directory** containing the Gemma-3 12B text encoder (not a single file). |
| **Spatial upscaler** | Optional. Used only at inference for 2× resolution boost. |
| **Distilled LoRA** | Optional. A pre-trained distilled LoRA for faster 8-step inference. |

**Click 💾 Save paths** when done. Paths can be relative (resolved against the launcher root) or absolute.

---

### 2 · Dataset Prep tab
*Prepare your video clips for training. Three steps — only Preprocess is required.*

#### ① Scene split *(optional)*
Splits a long source video into short per-scene clips. Skip this if you already have individual clips.
- **Input video path** — your long source `.mp4`.
- **Output directory** — where clips land (e.g. `app/data/scenes`).
- **Detector** — `content` works for most footage; `adaptive` handles fast-cuts; `threshold` for static/timelapse.
- **Filter shorter than** — drop clips below this duration (e.g. `3s`). Clips under ~2 s are usually unusable for training.
- **Min scene length** — minimum frames (e.g. `24`). Ignored if empty.
- **Detector threshold** — sensitivity; lower = more splits. Default (`27.0`) is usually fine.

#### ② Caption videos *(optional but strongly recommended)*
Auto-generates text captions for each clip. Captions become the training prompts.
- **Input directory** — your clips folder (output of step ①, or your own folder).
- **Output dataset JSON** — where the caption file lands (e.g. `app/data/dataset.json`).
- **Captioner** — `qwen_omni` runs locally (needs ~20 GB VRAM); `gemini_flash` calls Google's API (fast, cheap, requires API key).
- **API key** — required only for `gemini_flash`. Leave blank for local `qwen_omni`.
- **Num workers** — parallel threads. `1`–`4` is safe; more workers need more VRAM.
- **No audio** — tick if your clips have no meaningful audio, or to speed things up.
- **Use 8-bit** — halves VRAM for `qwen_omni` at minor quality cost.
- **Clean captions** — strips the `[VISUAL]`/`[SPEECH]` brackets from captions.

> 💡 If you skip captioning, manually create `dataset.json` with `{"video_path": "...", "caption": "..."}` entries.

#### ③ Preprocess *(REQUIRED)*
Comp utes video **latents** (VAE-encoded) and **text embeddings** (Gemma-encoded). This is what `train.py` actually reads — training never touches raw video files.
- **Dataset JSON** — the `.json` from step ②, or your hand-crafted one.
- **Resolution buckets** — format `WxHxF` where W and H are multiples of 32, and F follows the sequence `1, 9, 17, 25, … 121`. Example: `960x544x49` (≈ 2 s at 24 fps). Chain multiple with `;`.
- **With audio** — tick for joint audio-video training.
- **VAE tiling** — lowers peak VRAM during encoding. Use if you get OOM here.
- **Decode sanity check** — writes small preview `.mp4` files so you can verify the latents look correct before committing to a long training run.
- **LoRA trigger** — an optional single word baked into every caption (e.g. `sks`). Use this to create a LoRA triggered by that token.

---

### 3 · Config tab
*Author the training YAML. Change settings in the form, then Save.*

1. **Pick a template** from the dropdown — `LoRA (standard)` is the right starting point for most users. Load it with **📥 Load template**.
2. Tweak the form fields (see below). Click **🔄 Regenerate YAML** to update the raw YAML preview.
3. Edit the raw **YAML box** directly for anything the form doesn't expose (wandb, STG, custom validation prompts).
4. Click **🔍 Validate** to catch missing paths.
5. Set the **Save config to** path and click **💾 Save config**.

**Key form fields:**

| Field | Guidance |
|---|---|
| **Training mode** | `lora` for LoRA (recommended, needs 24–48 GB). `full` for full fine-tune (needs 80 GB+). |
| **Resume from checkpoint** | Path to a prior `.safetensors` checkpoint to continue training. Leave blank for a fresh run. |
| **Rank / Alpha** | LoRA rank (complexity). `32` is a good default. Higher = more capacity but more VRAM. `alpha = rank` is standard. |
| **Strategy** | `text_to_video` = standard LoRA. `video_to_video` = IC-LoRA (needs reference videos in dataset). |
| **First-frame conditioning prob** | `0.5` = 50/50 text-to-video vs image-to-video conditioning. Push toward `1.0` for image-anchored results. |
| **Learning rate** | `1e-4` is the standard starting point. Lower (`5e-5`) if the model collapses or overtrains. |
| **Steps** | `500`–`2000` is typical for LoRA. More steps = more capacity, but also more overfit risk. |
| **Batch size** | Keep at `1` unless you have 80 GB+ VRAM. |
| **Gradient accumulation** | Simulates larger batch size without extra VRAM. `4`–`8` is common. |
| **Optimizer** | `adamw8bit` saves ~75% optimizer VRAM — use it on ≤48 GB cards. |
| **Scheduler** | `linear` or `cosine` are both fine. |
| **Gradient checkpointing** | Leave ON unless you have unlimited VRAM. |
| **Mixed precision** | `bf16` is the recommended default for Ampere/Ada cards. |
| **Quantization** | `none` unless VRAM is very tight. `int8-quanto` on 32 GB cards. |
| **Checkpoint every N steps** | `250` is a safe default. Lower if you want to recover quickly from a bad run. |

---

### 4 · Train tab
*Launch the trainer and watch the live log.*

1. Paste (or type) the path to your saved config YAML in **Config YAML path**.
2. For single-GPU: leave **Multi-GPU** unticked and click **▶️ Run training**.
3. For multi-GPU: tick **Multi-GPU**, choose an **Accelerate config** (`ddp` = standard, `fsdp` = VRAM sharding across cards), then run.
4. Watch the log stream. Training writes checkpoints to the `output_dir` from your config every N steps.
5. Click **🔄 Refresh checkpoint list** at any time to see what's been saved.
6. Click **■ Cancel** to stop the training subprocess cleanly.

> ⚠️ **Windows note:** LTX-2 training requires `triton` and `bitsandbytes`, which are Linux-only. On Windows, use **WSL2** for actual training — the Config tab is fully functional for authoring YAML on Windows before pushing to a Linux host.

---

### 5 · Generate tab
*Run inference with your trained LoRA (or the base model).*

1. **Pipeline** — `ti2vid_two_stages` is the best-quality option (text-to-video + upscaling, needs all four model paths). `distilled` is fastest (8 steps, good for iteration). `ic_lora` = video-to-video with an IC-LoRA.
2. **Prompt** — describe the target video in detail. Include your LoRA trigger word if you set one.
3. **Negative prompt** — leave as-is unless you have a reason to change it.
4. **Your trained LoRA** — paste the path to your `.safetensors` checkpoint and set the **LoRA strength** (1.0 = full effect; 0.5 = blend with base; try 0.7–1.2).
5. **Resolution / timing** — leave at `0` to use pipeline defaults (`960×544×49` frames ≈ 2 s at 24 fps). Override as needed.
6. **Seed** — fix a seed for reproducible results; `-1` for random.
7. **Runtime quantization** — `none` is recommended. `fp8-cast` reduces VRAM at minor quality cost.
8. Click **🎬 Generate** and watch the log. The result video appears below the log when the subprocess exits.

---

### 6 · Settings tab
*Power-user knobs — rarely need changing.*

- **HuggingFace token** — required to download gated models (Gemma-3). Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Never appears in config files or logs.
- **uv executable** — override the `uv` path if it's not on your system PATH.
- **Python path** — leave blank; `uv run python` manages this automatically.
- **🔍 Check installation** — runs a health check: GPU detected, all model paths found, LTX-2 repo structure intact. Always run this first if something seems broken.

---

### Typical end-to-end workflow

```
1. Project tab      → set LTX-2 repo + model paths → Save
2. Dataset Prep     → Scene split (optional) → Caption → Preprocess
3. Config tab       → Load LoRA template → tweak LR/steps/rank → Save YAML
4. Train tab        → paste config path → Run → wait (hours)
5. Generate tab     → paste checkpoint path → Generate → review output
6. Iterate          → adjust steps/LR/LoRA strength and repeat from step 3 or 5
```
"""

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
# Config tab — template load, form → YAML, validate, save
# =============================================================================


def _quantization_for_ui(yaml_val) -> str:
    """YAML null → 'none' string so Gradio Dropdown has a clean value."""
    if yaml_val is None:
        return "none"
    return str(yaml_val)


def _config_template_to_form(cfg: dict) -> tuple:
    """Extract the form-field values from a parsed config dict.

    Returns a tuple in the exact order the Config tab's Load button
    outputs expect. Keep this order in sync with the component list in
    _config_tab().
    """
    model = cfg.get("model", {})
    lora = cfg.get("lora") or {}
    ts = cfg.get("training_strategy", {})
    opt = cfg.get("optimization", {})
    accel = cfg.get("acceleration", {})
    data = cfg.get("data", {})
    ck = cfg.get("checkpoints", {})

    return (
        # model
        model.get("training_mode", "lora"),
        model.get("load_checkpoint") or "",
        # lora
        int(lora.get("rank", 32)),
        int(lora.get("alpha", 32)),
        float(lora.get("dropout", 0.0)),
        # training_strategy
        ts.get("name", "text_to_video"),
        float(ts.get("first_frame_conditioning_p", 0.5)),
        bool(ts.get("with_audio", False)),
        # optimization
        float(opt.get("learning_rate", 1e-4)),
        int(opt.get("steps", 2000)),
        int(opt.get("batch_size", 1)),
        int(opt.get("gradient_accumulation_steps", 1)),
        opt.get("optimizer_type", "adamw"),
        opt.get("scheduler_type", "linear"),
        bool(opt.get("enable_gradient_checkpointing", True)),
        # acceleration
        accel.get("mixed_precision_mode", "bf16"),
        _quantization_for_ui(accel.get("quantization")),
        bool(accel.get("load_text_encoder_in_8bit", False)),
        # data
        int(data.get("num_dataloader_workers", 2)),
        # checkpoints
        int(ck.get("interval") or 250),
        int(ck.get("keep_last_n") if ck.get("keep_last_n") is not None else -1),
        # top-level
        int(cfg.get("seed", 42)),
    )


def config_load_template_fn(template_name: str):
    """Load a template and return (form_values..., yaml_text, status)."""
    try:
        s = settings_mod.load()
        cfg = config_builder.load_template(template_name, s)
        cfg = config_builder.apply_project_paths(cfg, s)
    except config_builder.ConfigBuilderError as exc:
        # On failure, leave the form alone (gr.update()) and show an error.
        skip = [gr.update() for _ in range(22)]
        return (*skip, gr.update(), f"❌ {exc}")

    form_values = _config_template_to_form(cfg)
    yaml_text = config_builder.dict_to_yaml(cfg)
    status = (
        f"✅ Loaded **{config_builder.TEMPLATE_LABELS[template_name]}** "
        "from the trainer's config directory (paths injected from Project tab)."
    )
    return (*form_values, yaml_text, status)


def config_regenerate_yaml_fn(
    template_name: str,
    training_mode: str,
    load_checkpoint: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    strategy_name: str,
    first_frame_conditioning_p: float,
    with_audio: bool,
    learning_rate: float,
    steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    optimizer_type: str,
    scheduler_type: str,
    enable_gradient_checkpointing: bool,
    mixed_precision_mode: str,
    quantization: str,
    load_text_encoder_in_8bit: bool,
    num_dataloader_workers: int,
    checkpoint_interval: int,
    keep_last_n: int,
    seed: int,
) -> tuple[str, str]:
    """Build YAML from the template + form overrides. Returns (yaml, status)."""
    try:
        s = settings_mod.load()
        cfg = config_builder.load_template(template_name, s)
        cfg = config_builder.apply_project_paths(cfg, s)
    except config_builder.ConfigBuilderError as exc:
        return "", f"❌ {exc}"

    form = config_builder.FormOverrides(
        training_mode=training_mode,
        load_checkpoint=load_checkpoint.strip() or None,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        strategy_name=strategy_name,
        first_frame_conditioning_p=first_frame_conditioning_p,
        with_audio=with_audio,
        learning_rate=learning_rate,
        steps=steps,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        mixed_precision_mode=mixed_precision_mode,
        quantization=quantization,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        num_dataloader_workers=num_dataloader_workers,
        checkpoint_interval=checkpoint_interval,
        keep_last_n=keep_last_n,
        seed=seed,
    )
    cfg = config_builder.apply_form_overrides(cfg, form)
    return (
        config_builder.dict_to_yaml(cfg),
        "🔄 Regenerated YAML from form. The raw YAML below is now the "
        "source of truth for Save — edit it directly to override any "
        "advanced keys the form doesn't expose (validation prompts, "
        "STG settings, wandb, etc.).",
    )


def config_validate_fn(yaml_text: str) -> str:
    try:
        cfg = config_builder.yaml_to_dict(yaml_text)
    except config_builder.ConfigBuilderError as exc:
        return f"❌ {exc}"
    warnings = config_builder.validate_for_save(cfg)
    if not warnings:
        return "✅ Config parses and all paths look valid."
    return "\n\n".join(warnings)


def config_save_fn(yaml_text: str, out_path: str) -> str:
    if not out_path.strip():
        return "❌ Please specify a save path."
    ok, msg = config_builder.save_config(yaml_text, Path(out_path.strip()))
    return msg


# =============================================================================
# Train tab
# =============================================================================


def train_fn(config_path: str, multi_gpu: bool, accelerate_config: str, extra_args: str):
    if not config_path.strip():
        yield "Please provide a config YAML path.", None
        return
    s = settings_mod.load()
    yield from _stream_with_state(
        runner.run_train(
            s,
            config_path=config_path,
            multi_gpu=multi_gpu,
            accelerate_config=accelerate_config,
            extra_args=extra_args,
        ),
    )


def generate_fn(
    pipeline: str,
    prompt: str,
    negative_prompt: str,
    output_path: str,
    lora_path: str,
    lora_multiplier: float,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    seed: int,
    quantization: str,
    extra_args: str,
):
    """Stream log + emit the output video path once it exists on disk."""
    s = settings_mod.load()
    proc = None
    final_log = ""
    try:
        for log, handle in runner.run_generate(
            s,
            pipeline=pipeline,
            prompt=prompt,
            output_path=output_path,
            negative_prompt=negative_prompt,
            lora_path=lora_path,
            lora_multiplier=lora_multiplier,
            width=int(width) if width else 0,
            height=int(height) if height else 0,
            num_frames=int(num_frames) if num_frames else 0,
            frame_rate=float(frame_rate) if frame_rate else 0.0,
            num_inference_steps=int(num_inference_steps) if num_inference_steps else 0,
            seed=int(seed) if seed is not None else -1,
            quantization=quantization,
            extra_args=extra_args,
        ):
            if handle is not None:
                proc = handle
            final_log = log
            yield log, proc, None
    except runner.RunnerError as exc:
        yield f"[runner error] {exc}\n", proc, None
        return

    # CLAUDE-NOTE: After the subprocess exits, check if the output mp4
    # landed. Only return a path if it exists — avoids Gradio trying to
    # load a missing file.
    out_path = settings_mod.resolve_path(output_path) if output_path.strip() else ""
    video_result = out_path if out_path and Path(out_path).exists() else None
    yield final_log, proc, video_result


def list_checkpoints_fn() -> str:
    """Return a markdown table of checkpoints under the current output_dir."""
    s = settings_mod.load()
    entries = runner.list_checkpoints(s.output_dir)
    if not entries:
        return (
            f"_No `.safetensors` files under `{s.output_dir}` yet._\n\n"
            "Checkpoints appear here once training writes them "
            "(every `checkpoints.interval` steps)."
        )
    import datetime as _dt  # local import; used nowhere else
    header = "| File | Size (MB) | Modified |\n|---|---:|---|\n"
    rows = []
    for e in entries[:50]:  # CLAUDE-NOTE: cap at 50 to keep the markdown render fast
        ts = _dt.datetime.fromtimestamp(e["mtime"]).strftime("%Y-%m-%d %H:%M")  # noqa: DTZ006
        rows.append(f"| `{e['path']}` | {e['size_mb']:.1f} | {ts} |")
    suffix = f"\n\n_Showing newest 50 of {len(entries)}._" if len(entries) > 50 else ""
    return header + "\n".join(rows) + suffix


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


def _generate_tab(initial: dict) -> None:
    gr.Markdown(
        "Generate video from a trained checkpoint using the "
        "`ltx-pipelines` package (the same production-grade inference "
        "code LTX-2 ships). Paths for the base model, text encoder, "
        "spatial upsampler, and distilled LoRA come from the Project "
        "tab — two-stage pipelines need all four."
    )

    with gr.Row():
        pipeline = gr.Dropdown(
            label="Pipeline",
            choices=runner.ALL_PIPELINES,
            value="ti2vid_two_stages",
            info="ti2vid_two_stages is the recommended starting point. "
                 "distilled is fastest (8 steps). ic_lora = video-to-video "
                 "with an IC-LoRA you trained.",
        )

    prompt = gr.Textbox(
        label="Prompt",
        placeholder="A slow pan across a neon-lit street at night, steam rising from a manhole, …",
        lines=3,
    )
    negative_prompt = gr.Textbox(
        label="Negative prompt (optional)",
        value="worst quality, inconsistent motion, blurry, jittery, distorted",
        lines=2,
    )

    default_output = str(Path(initial["output_dir"]) / "generated.mp4")
    output_path = gr.Textbox(
        label="Output video path (.mp4)",
        value=default_output,
    )

    with gr.Accordion("Your trained LoRA (optional)", open=True):
        with gr.Row():
            lora_path = gr.Textbox(
                label="LoRA path (.safetensors)",
                placeholder="app/outputs/my_run/lora_weights.safetensors",
            )
            lora_multiplier = gr.Slider(
                label="LoRA strength",
                minimum=0.0, maximum=2.0, step=0.05, value=1.0,
            )

    with gr.Accordion("Resolution / timing / sampling", open=True):
        with gr.Row():
            width = gr.Number(label="Width (0 = pipeline default)", value=0, precision=0)
            height = gr.Number(label="Height (0 = default)", value=0, precision=0)
            num_frames = gr.Number(label="Num frames (0 = default)", value=0, precision=0)
            frame_rate = gr.Number(label="Frame rate (0 = default)", value=0)
        with gr.Row():
            num_inference_steps = gr.Number(
                label="Inference steps (0 = default, ignored for distilled)",
                value=0, precision=0,
            )
            seed = gr.Number(label="Seed (-1 = random)", value=42, precision=0)
            quantization = gr.Dropdown(
                label="Runtime quantization",
                choices=["none", "fp8-cast", "int8-cast"],
                value="none",
                info="Lower VRAM at small quality cost. Must match what the model supports.",
            )

    extra_args = gr.Textbox(
        label="Extra args (space-separated, advanced)",
        placeholder="--enhance-prompt --images path/to/first_frame.png",
    )

    with gr.Row():
        gen_btn = gr.Button("🎬 Generate", variant="primary")
        gen_cancel = gr.Button("■ Cancel")

    log_box = gr.Textbox(
        label="Inference log",
        lines=18,
        max_lines=50,
        interactive=False,
    )
    proc_state = gr.State(None)
    # CLAUDE-NOTE: gr.Video accepts a filesystem path; Gradio serves it back
    # to the browser. It stays blank until the subprocess exits and the
    # generator emits a real path.
    video_out = gr.Video(label="Result", interactive=False)

    gen_btn.click(
        fn=generate_fn,
        inputs=[
            pipeline, prompt, negative_prompt, output_path,
            lora_path, lora_multiplier,
            width, height, num_frames, frame_rate,
            num_inference_steps, seed, quantization,
            extra_args,
        ],
        outputs=[log_box, proc_state, video_out],
    )
    gen_cancel.click(fn=cancel_fn, inputs=[proc_state], outputs=[log_box])


def _train_tab() -> dict:
    """Build the Train tab."""
    hw = get_hardware_info()
    if hw.get("training_supported_natively") is False:
        gr.Markdown(
            "### ⚠️ This OS cannot run LTX-2 training natively\n\n"
            f"You're on **{hw.get('os', '?')}**. The trainer depends on "
            "`triton` and `bitsandbytes`, which are Linux-only. If you "
            "click **Run** below on this machine, `train.py` will fail at "
            "import time.\n\n"
            "**To actually train**, switch to:\n"
            "- **WSL2** on the same box (recommended — GPU passthrough works for the RTX 6000 Ada), or\n"
            "- A Linux host (local or cloud).\n\n"
            "The Config tab is still fully useful here — author your YAML "
            "on Windows, push it to the Linux box, and run training there."
        )
    else:
        gr.Markdown(
            "Launch training. Streams the live log from "
            "`uv run python packages/ltx-trainer/scripts/train.py <config>` "
            "(or `accelerate launch` for multi-GPU). Checkpoints land in "
            "whatever `output_dir` the config specifies.",
        )

    config_path = gr.Textbox(
        label="Config YAML path",
        placeholder="e.g. outputs/../configs/my_training.yaml (from the Config tab)",
    )

    with gr.Row():
        multi_gpu = gr.Checkbox(label="Multi-GPU (accelerate launch)", value=False)
        accelerate_config = gr.Dropdown(
            label="Accelerate config",
            choices=["ddp", "ddp_compile", "fsdp", "fsdp_compile"],
            value="ddp",
            info="ddp=standard, fsdp=parameter sharding, *_compile adds torch.compile.",
            interactive=False,
        )
    extra_args = gr.Textbox(
        label="Extra args (space-separated, advanced)",
        placeholder="e.g. --some-override value",
    )

    # CLAUDE-NOTE: Enable the accelerate config dropdown only when Multi-GPU
    # is checked. Keeps the UI from implying it matters in single-GPU mode.
    multi_gpu.change(
        fn=lambda mg: gr.update(interactive=bool(mg)),
        inputs=[multi_gpu],
        outputs=[accelerate_config],
    )

    with gr.Row():
        run_btn = gr.Button("▶️ Run training", variant="primary")
        cancel_btn = gr.Button("■ Cancel")

    log_box = gr.Textbox(
        label="Training log",
        lines=24,
        max_lines=80,
        interactive=False,
    )
    proc_state = gr.State(None)

    gr.Markdown("---\n### Checkpoints")
    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh checkpoint list")
    ckpt_table = gr.Markdown(
        "_Click refresh after training writes a checkpoint._",
    )

    run_btn.click(
        fn=train_fn,
        inputs=[config_path, multi_gpu, accelerate_config, extra_args],
        outputs=[log_box, proc_state],
    )
    cancel_btn.click(fn=cancel_fn, inputs=[proc_state], outputs=[log_box])
    refresh_btn.click(fn=list_checkpoints_fn, inputs=[], outputs=[ckpt_table])

    return {
        "config_path": config_path,
        "log_box": log_box,
    }


def _config_tab(initial: dict) -> dict:
    """Build the Config tab. Returns a dict of handles the builder wires up."""
    gr.Markdown(
        "Author a training YAML config. Pick a template, tweak common "
        "fields in the form, then optionally edit the raw YAML for "
        "advanced keys (validation prompts, STG settings, wandb, etc.). "
        "Model paths auto-populate from the Project tab."
    )

    with gr.Row():
        template = gr.Dropdown(
            label="Template",
            choices=[
                (label, key) for key, label in config_builder.TEMPLATE_LABELS.items()
            ],
            value="lora",
        )
        load_btn = gr.Button("📥 Load template", variant="primary")

    default_save = str(Path(initial["output_dir"]).parent / "configs" / "my_training.yaml")
    save_path = gr.Textbox(
        label="Save config to",
        value=default_save,
        info="Where the YAML will be written when you click Save.",
    )

    load_status = gr.Markdown("")

    gr.Markdown("### Common fields")

    with gr.Accordion("Model", open=True):
        with gr.Row():
            training_mode = gr.Dropdown(
                label="Training mode",
                choices=["lora", "full"],
                value="lora",
                info="Full fine-tune needs 80GB+ VRAM.",
            )
            load_checkpoint = gr.Textbox(
                label="Resume from checkpoint (optional)",
                placeholder="/path/to/checkpoint-1000.safetensors",
            )

    with gr.Accordion("LoRA (ignored if training_mode=full)", open=True):
        with gr.Row():
            lora_rank = gr.Number(label="Rank", value=32, precision=0)
            lora_alpha = gr.Number(label="Alpha", value=32, precision=0)
            lora_dropout = gr.Slider(label="Dropout", minimum=0.0, maximum=0.5, step=0.05, value=0.0)

    with gr.Accordion("Training strategy", open=True):
        with gr.Row():
            strategy_name = gr.Dropdown(
                label="Strategy",
                choices=["text_to_video", "video_to_video"],
                value="text_to_video",
                info="video_to_video = IC-LoRA mode (needs reference videos).",
            )
            first_frame_p = gr.Slider(
                label="First-frame conditioning probability",
                minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                info="Higher = closer to image-to-video.",
            )
            with_audio = gr.Checkbox(label="Joint audio-video training", value=True)

    with gr.Accordion("Optimization", open=True):
        with gr.Row():
            learning_rate = gr.Number(label="Learning rate", value=1e-4)
            steps = gr.Number(label="Training steps", value=2000, precision=0)
            batch_size = gr.Number(label="Batch size", value=1, precision=0)
            grad_accum = gr.Number(label="Gradient accumulation", value=1, precision=0)
        with gr.Row():
            optimizer_type = gr.Dropdown(
                label="Optimizer",
                choices=["adamw", "adamw8bit"],
                value="adamw",
                info="adamw8bit saves ~75% optimizer VRAM.",
            )
            scheduler_type = gr.Dropdown(
                label="Scheduler",
                choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"],
                value="linear",
            )
            grad_ckpt = gr.Checkbox(label="Gradient checkpointing", value=True,
                                    info="Slower, but required on <48GB GPUs.")

    with gr.Accordion("Acceleration / precision", open=False):
        with gr.Row():
            mixed_precision = gr.Dropdown(
                label="Mixed precision",
                choices=["no", "fp16", "bf16"],
                value="bf16",
            )
            quantization = gr.Dropdown(
                label="Model quantization",
                choices=["none", "int8-quanto", "int4-quanto", "fp8-quanto"],
                value="none",
                info="Use int8-quanto on 32GB cards.",
            )
            te_8bit = gr.Checkbox(label="Text encoder in 8-bit", value=False)

    with gr.Accordion("Data / Checkpoints / Misc", open=False):
        with gr.Row():
            dl_workers = gr.Number(label="Dataloader workers", value=2, precision=0)
            ckpt_interval = gr.Number(label="Checkpoint every N steps", value=250, precision=0)
            keep_last_n = gr.Number(label="Keep last N checkpoints (-1 = all)", value=-1, precision=0)
            seed = gr.Number(label="Seed", value=42, precision=0)

    gr.Markdown("---\n### Raw YAML (source of truth for Save)")

    yaml_box = gr.Textbox(
        label="YAML",
        lines=30,
        max_lines=80,
        interactive=True,
        value="# Click 'Load template' above to populate this box.\n",
    )

    with gr.Row():
        regen_btn = gr.Button("🔄 Regenerate YAML from form")
        validate_btn = gr.Button("🔍 Validate")
        save_btn = gr.Button("💾 Save config", variant="primary")

    save_status = gr.Markdown("")

    return {
        "template": template,
        "load_btn": load_btn,
        "save_path": save_path,
        "load_status": load_status,
        "training_mode": training_mode,
        "load_checkpoint": load_checkpoint,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "strategy_name": strategy_name,
        "first_frame_p": first_frame_p,
        "with_audio": with_audio,
        "learning_rate": learning_rate,
        "steps": steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "optimizer_type": optimizer_type,
        "scheduler_type": scheduler_type,
        "grad_ckpt": grad_ckpt,
        "mixed_precision": mixed_precision,
        "quantization": quantization,
        "te_8bit": te_8bit,
        "dl_workers": dl_workers,
        "ckpt_interval": ckpt_interval,
        "keep_last_n": keep_last_n,
        "seed": seed,
        "yaml_box": yaml_box,
        "regen_btn": regen_btn,
        "validate_btn": validate_btn,
        "save_btn": save_btn,
        "save_status": save_status,
    }


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


def _directions_tab() -> None:
    """Render the Directions tab — a full user-facing walkthrough of the UI."""
    gr.Markdown(DIRECTIONS_MD)


def build_ui() -> gr.Blocks:
    initial = _load_settings_dict()

    # CLAUDE-NOTE: Gradio 6+ moved `theme` from Blocks() to launch(). The
    # theme is applied in main() below via ui.launch(theme=...).
    with gr.Blocks(title=APP_TITLE) as ui:
        gr.Markdown(f"# {APP_TITLE}\n\n{APP_SUBTITLE}")
        gr.Markdown(_hardware_banner_md())

        with gr.Tabs():
            with gr.Tab("📖 Directions"):
                _directions_tab()

            with gr.Tab("Project"):
                project_fields, project_save_btn, project_save_status = _project_tab(initial)

            with gr.Tab("Dataset Prep"):
                _dataset_prep_tab(default_output_dir=initial["output_dir"])

            with gr.Tab("Config"):
                cfg = _config_tab(initial)

            with gr.Tab("Train"):
                _train_tab()

            with gr.Tab("Generate"):
                _generate_tab(initial)

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

        # ---------- Config tab wiring ----------
        # Form components in the exact order _config_template_to_form emits.
        form_components = [
            cfg["training_mode"], cfg["load_checkpoint"],
            cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"],
            cfg["strategy_name"], cfg["first_frame_p"], cfg["with_audio"],
            cfg["learning_rate"], cfg["steps"], cfg["batch_size"], cfg["grad_accum"],
            cfg["optimizer_type"], cfg["scheduler_type"], cfg["grad_ckpt"],
            cfg["mixed_precision"], cfg["quantization"], cfg["te_8bit"],
            cfg["dl_workers"],
            cfg["ckpt_interval"], cfg["keep_last_n"],
            cfg["seed"],
        ]

        cfg["load_btn"].click(
            fn=config_load_template_fn,
            inputs=[cfg["template"]],
            outputs=[*form_components, cfg["yaml_box"], cfg["load_status"]],
        )

        cfg["regen_btn"].click(
            fn=config_regenerate_yaml_fn,
            inputs=[cfg["template"], *form_components],
            outputs=[cfg["yaml_box"], cfg["save_status"]],
        )

        cfg["validate_btn"].click(
            fn=config_validate_fn,
            inputs=[cfg["yaml_box"]],
            outputs=[cfg["save_status"]],
        )

        cfg["save_btn"].click(
            fn=config_save_fn,
            inputs=[cfg["yaml_box"], cfg["save_path"]],
            outputs=[cfg["save_status"]],
        )

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
