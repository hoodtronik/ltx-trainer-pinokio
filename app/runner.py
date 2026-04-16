"""Subprocess runner for LTX-2 trainer scripts.

Streams stdout+stderr line-by-line from `uv run python scripts/<script>.py`
invocations. The Gradio tabs use these generators as live-updating Textbox
sources.

All commands run with cwd set to the LTX-2 repo root, never `shell=True`,
and arguments always in list form to avoid quoting issues on Windows.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

from settings import Settings, resolve_path

# CLAUDE-NOTE: The trainer's scripts live at `packages/ltx-trainer/scripts/`
# inside the LTX-2 monorepo. We pass the path relative to the LTX-2 root so
# `uv run` resolves the right workspace member automatically.
SCRIPTS_SUBDIR = "packages/ltx-trainer/scripts"


class RunnerError(RuntimeError):
    """Raised when a command cannot be assembled (missing paths, etc.)."""


def _ltx_root(settings: Settings) -> Path:
    root = Path(resolve_path(settings.ltx_repo_path))
    if not root.is_dir():
        raise RunnerError(
            f"LTX-2 repo not found at {root}. Set the path in the Project tab.",
        )
    if not (root / "packages" / "ltx-trainer" / "scripts").is_dir():
        raise RunnerError(
            f"{root} does not look like the LTX-2 monorepo "
            "(missing packages/ltx-trainer/scripts).",
        )
    return root


def build_script_cmd(
    settings: Settings,
    script_name: str,
    positional: Sequence[str],
    options: dict[str, str | bool | None],
) -> list[str]:
    """Construct the `uv run python scripts/<script>.py ...` argv list.

    `options` values:
      - str: emitted as `--key value`
      - True: emitted as `--key` (flag)
      - False / None / "": omitted
    """
    uv = settings.uv_path or "uv"
    script_path = f"{SCRIPTS_SUBDIR}/{script_name}"
    cmd: list[str] = [uv, "run", "python", script_path, *positional]
    for key, value in options.items():
        if value in (None, False, ""):
            continue
        flag = f"--{key}"
        if value is True:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def stream_command(
    cmd: Sequence[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> Iterator[tuple[str, subprocess.Popen | None]]:
    """Run `cmd` and yield (accumulated_log, process_handle) after each line.

    The process handle is yielded alongside the log so the UI's Cancel button
    can terminate it. The final yield has an exit-code marker appended and
    the handle's returncode set.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    start = time.time()
    header = f"$ {' '.join(cmd)}\n  (cwd: {cwd})\n\n"
    log = header
    yield log, None

    # CLAUDE-NOTE: stderr merged into stdout so UI users see everything in
    # one stream (rich/tqdm often write to stderr). bufsize=1 = line-buffered.
    process = subprocess.Popen(  # noqa: S603 - argv is list-form, no shell
        list(cmd),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        env=full_env,
    )
    assert process.stdout is not None  # noqa: S101

    try:
        for line in process.stdout:
            log += line
            yield log, process
    finally:
        process.wait()

    duration = time.time() - start
    log += f"\n[exit code: {process.returncode}]  [elapsed: {duration:.1f}s]\n"
    yield log, process


def cancel_process(process: subprocess.Popen | None) -> str:
    """Terminate a running subprocess. Returns a short status string."""
    if process is None:
        return "Nothing to cancel."
    if process.poll() is not None:
        return f"Process already exited (code {process.returncode})."
    # CLAUDE-NOTE: terminate() = SIGTERM on POSIX, TerminateProcess on
    # Windows. If the child ignores it (tqdm loops, stuck CUDA init), we
    # escalate to kill() = SIGKILL / Windows hard-kill after a 3s grace
    # period so the UI doesn't hang waiting for cleanup.
    try:
        process.terminate()
    except OSError:
        return "Failed to send terminate signal."

    def _kill_after_grace() -> None:
        time.sleep(3)
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    threading.Thread(target=_kill_after_grace, daemon=True).start()
    return "Cancel signal sent."


# -----------------------------
# High-level wrappers per script
# -----------------------------


def run_split_scenes(
    settings: Settings,
    input_video: str,
    output_dir: str,
    *,
    detector: str = "content",
    filter_shorter_than: str = "",
    min_scene_length: str = "",
    threshold: str = "",
    extra_args: str = "",
) -> Iterator[tuple[str, subprocess.Popen | None]]:
    root = _ltx_root(settings)
    positional = [resolve_path(input_video), resolve_path(output_dir)]
    options: dict[str, str | bool | None] = {
        "detector": detector,
        "filter-shorter-than": filter_shorter_than or None,
        "min-scene-length": min_scene_length or None,
        "threshold": threshold or None,
    }
    cmd = build_script_cmd(settings, "split_scenes.py", positional, options)
    if extra_args.strip():
        cmd.extend(extra_args.split())
    yield from stream_command(cmd, root)


def run_caption_videos(
    settings: Settings,
    input_dir: str,
    output_json: str,
    *,
    captioner_type: str = "qwen_omni",
    api_key: str = "",
    num_workers: str = "",
    no_audio: bool = False,
    use_8bit: bool = False,
    clean_caption: bool = False,
    extra_args: str = "",
) -> Iterator[tuple[str, subprocess.Popen | None]]:
    root = _ltx_root(settings)
    positional = [resolve_path(input_dir)]
    options: dict[str, str | bool | None] = {
        "output": resolve_path(output_json),
        "captioner-type": captioner_type,
        "api-key": api_key or None,
        "num-workers": num_workers or None,
        "no-audio": no_audio,
        "use-8bit": use_8bit,
        "clean-caption": clean_caption,
    }
    cmd = build_script_cmd(settings, "caption_videos.py", positional, options)
    if extra_args.strip():
        cmd.extend(extra_args.split())
    yield from stream_command(cmd, root)


def run_train(
    settings: Settings,
    config_path: str,
    *,
    multi_gpu: bool = False,
    accelerate_config: str = "ddp",
    extra_args: str = "",
) -> Iterator[tuple[str, subprocess.Popen | None]]:
    """Launch training. Streams logs until train.py exits (or user cancels).

    Single-GPU: `uv run python packages/ltx-trainer/scripts/train.py <config>`
    Multi-GPU:  `uv run accelerate launch --config_file configs/accelerate/<name>.yaml \\
                       packages/ltx-trainer/scripts/train.py <config>`
    """
    root = _ltx_root(settings)
    config_abs = resolve_path(config_path)
    if not Path(config_abs).is_file():
        raise RunnerError(f"Config file not found: {config_abs}")

    uv = settings.uv_path or "uv"
    script_rel = f"{SCRIPTS_SUBDIR}/train.py"

    if multi_gpu:
        # CLAUDE-NOTE: Accelerate configs live at
        # packages/ltx-trainer/configs/accelerate/<name>.yaml. Four options:
        # ddp, ddp_compile, fsdp, fsdp_compile. We pass the path relative to
        # the LTX-2 repo root since `uv run` sets that as cwd.
        accel_config = (
            f"packages/ltx-trainer/configs/accelerate/{accelerate_config}.yaml"
        )
        cmd = [
            uv, "run", "accelerate", "launch",
            "--config_file", accel_config,
            script_rel, config_abs,
        ]
    else:
        cmd = [uv, "run", "python", script_rel, config_abs]

    if extra_args.strip():
        cmd.extend(extra_args.split())

    yield from stream_command(cmd, root)


PIPELINES_TWO_STAGE = {"ti2vid_two_stages", "ti2vid_two_stages_hq", "ic_lora"}
PIPELINES_SINGLE_STAGE = {"ti2vid_one_stage", "distilled", "a2vid_two_stage",
                         "keyframe_interpolation", "retake"}
ALL_PIPELINES = sorted(PIPELINES_TWO_STAGE | PIPELINES_SINGLE_STAGE)


def run_generate(
    settings: Settings,
    pipeline: str,
    prompt: str,
    output_path: str,
    *,
    negative_prompt: str = "",
    lora_path: str = "",
    lora_multiplier: float = 1.0,
    width: int = 0,
    height: int = 0,
    num_frames: int = 0,
    frame_rate: float = 0.0,
    num_inference_steps: int = 0,
    seed: int = -1,
    quantization: str = "",
    extra_args: str = "",
) -> Iterator[tuple[str, subprocess.Popen | None]]:
    """Run one of the ltx-pipelines via `uv run python -m ltx_pipelines.<name>`.

    Paths for the checkpoint, text encoder, spatial upsampler, and distilled
    LoRA are pulled from Settings (Project tab) automatically.
    """
    root = _ltx_root(settings)
    if pipeline not in ALL_PIPELINES:
        raise RunnerError(f"Unknown pipeline: {pipeline!r}. Choose from {ALL_PIPELINES}")
    if not prompt.strip():
        raise RunnerError("Prompt is required.")
    if not output_path.strip():
        raise RunnerError("Output path is required.")

    for label, path in [
        ("model_path", settings.model_path),
        ("text_encoder_path", settings.text_encoder_path),
    ]:
        if not path:
            raise RunnerError(f"{label} not set — configure it in the Project tab.")

    if pipeline in PIPELINES_TWO_STAGE:
        for label, path in [
            ("spatial_upscaler_path", settings.spatial_upscaler_path),
            ("distilled_lora_path", settings.distilled_lora_path),
        ]:
            if not path:
                raise RunnerError(
                    f"{label} not set — the two-stage pipeline {pipeline!r} needs it. "
                    "Configure it in the Project tab.",
                )

    uv = settings.uv_path or "uv"
    cmd: list[str] = [
        uv, "run", "python", "-m", f"ltx_pipelines.{pipeline}",
        "--prompt", prompt,
        "--output-path", resolve_path(output_path),
        "--gemma-root", resolve_path(settings.text_encoder_path),
    ]

    # CLAUDE-NOTE: `distilled` pipeline takes --distilled-checkpoint-path
    # (the distilled model itself is the only checkpoint). All other
    # pipelines take --checkpoint-path (the regular LTX-2 model) and the
    # two-stage family additionally takes --distilled-lora for stage 2.
    if pipeline == "distilled":
        cmd += ["--distilled-checkpoint-path", resolve_path(settings.model_path)]
    else:
        cmd += ["--checkpoint-path", resolve_path(settings.model_path)]

    if pipeline in PIPELINES_TWO_STAGE:
        cmd += [
            "--spatial-upsampler-path", resolve_path(settings.spatial_upscaler_path),
            "--distilled-lora", resolve_path(settings.distilled_lora_path), "0.8",
        ]

    if negative_prompt.strip():
        cmd += ["--negative-prompt", negative_prompt]
    if lora_path.strip():
        cmd += ["--lora", resolve_path(lora_path), str(lora_multiplier)]
    if width > 0:
        cmd += ["--width", str(width)]
    if height > 0:
        cmd += ["--height", str(height)]
    if num_frames > 0:
        cmd += ["--num-frames", str(num_frames)]
    if frame_rate > 0:
        cmd += ["--frame-rate", str(frame_rate)]
    if num_inference_steps > 0 and pipeline != "distilled":
        # distilled pipeline has a fixed step schedule, no --num-inference-steps.
        cmd += ["--num-inference-steps", str(num_inference_steps)]
    if seed >= 0:
        cmd += ["--seed", str(seed)]
    if quantization and quantization != "none":
        cmd += ["--quantization", quantization]
    if extra_args.strip():
        cmd.extend(extra_args.split())

    yield from stream_command(cmd, root)


def list_checkpoints(output_dir: str) -> list[dict[str, object]]:
    """List .safetensors files under `output_dir`, newest first.

    Returns entries like `{"path": str, "size_mb": float, "mtime": float}`.
    Used by the Train tab's checkpoint browser.
    """
    out = Path(resolve_path(output_dir)) if output_dir else None
    if not out or not out.is_dir():
        return []
    results: list[dict[str, object]] = []
    for p in out.rglob("*.safetensors"):
        try:
            st = p.stat()
        except OSError:
            continue
        results.append(
            {
                "path": str(p),
                "size_mb": round(st.st_size / (1024 * 1024), 1),
                "mtime": st.st_mtime,
            },
        )
    results.sort(key=lambda r: r["mtime"], reverse=True)
    return results


def run_process_dataset(
    settings: Settings,
    dataset_json: str,
    resolution_buckets: str,
    *,
    with_audio: bool = False,
    reference_column: str = "",
    batch_size: str = "",
    vae_tiling: bool = False,
    decode: bool = False,
    output_dir: str = "",
    lora_trigger: str = "",
    extra_args: str = "",
) -> Iterator[tuple[str, subprocess.Popen | None]]:
    root = _ltx_root(settings)

    if not settings.model_path:
        raise RunnerError("model_path not set — configure it in the Project tab.")
    if not settings.text_encoder_path:
        raise RunnerError("text_encoder_path not set — configure it in the Project tab.")

    positional = [resolve_path(dataset_json)]
    options: dict[str, str | bool | None] = {
        "resolution-buckets": resolution_buckets,
        "model-path": resolve_path(settings.model_path),
        "text-encoder-path": resolve_path(settings.text_encoder_path),
        "output-dir": resolve_path(output_dir) if output_dir else None,
        "with-audio": with_audio,
        "reference-column": reference_column or None,
        "batch-size": batch_size or None,
        "vae-tiling": vae_tiling,
        "decode": decode,
        "lora-trigger": lora_trigger or None,
    }
    cmd = build_script_cmd(settings, "process_dataset.py", positional, options)
    if extra_args.strip():
        cmd.extend(extra_args.split())
    yield from stream_command(cmd, root)
