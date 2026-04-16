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
