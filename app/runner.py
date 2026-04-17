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
    # CLAUDE-NOTE: Force UTF-8 for all child processes on Windows so tools
    # like `hf` (which prints ✓ checkmarks) don't crash with a charmap
    # codec error. PYTHONUTF8=1 covers Python-based tools; PYTHONIOENCODING
    # covers older tools that read the env var instead of the mode flag.
    full_env["PYTHONUTF8"] = "1"
    full_env["PYTHONIOENCODING"] = "utf-8"
    if env:
        full_env.update(env)

    start = time.time()
    header = f"$ {' '.join(cmd)}\n  (cwd: {cwd})\n\n"
    log = header
    yield log, None

    # CLAUDE-NOTE: stderr merged into stdout so UI users see everything in
    # one stream (rich/tqdm often write to stderr). bufsize=1 = line-buffered.
    # encoding='utf-8' + errors='replace': hf/tqdm print Unicode checkmarks
    # (✓ U+2713) that Windows cp1252 can't handle — 'replace' turns any
    # un-decodable byte into '?' instead of raising UnicodeDecodeError.
    process = subprocess.Popen(  # noqa: S603 - argv is list-form, no shell
        list(cmd),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
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


def run_hf_download(
    settings: Settings,
    repo_id: str,
    filename: str,
    local_dir: str,
    *,
    hf_token: str = "",
) -> Iterator[tuple[str, subprocess.Popen | None]]:
    """Download a single file (or whole repo) from HuggingFace Hub.

    CLAUDE-NOTE: We use the Python huggingface_hub API via a subprocess so we
    can import it with the correct venv Python without pulling it into the
    Gradio process. We write a small inline Python script that downloads and
    prints progress lines — one line every 512 MB — so stream_command can
    relay them to the Gradio log box.

    Why not `hf download` CLI: the `hf` binary uses Rich with \\r carriage-
    return progress updates that never write newlines, so our line-reader never
    sees any output until the subprocess exits (useless for a 40 GB file).
    """
    token_arg = f'token="{hf_token.strip()}"' if hf_token.strip() else "token=None"
    cwd = Path(__file__).parent

    if filename.strip():
        # Single-file download with manual progress reporting every 512 MB.
        # CLAUDE-NOTE: hf_hub_download() and snapshot_download() are silent
        # until completion; they use tqdm internally which writes \r updates.
        # To get live lines in the Gradio log we use the HF Hub's internal
        # URL resolution then download via requests with a manual read loop
        # that prints a progress line every CHUNK bytes.
        script = f"""
import sys, os, time, requests
from pathlib import Path
from huggingface_hub import hf_hub_url, HfApi

repo_id   = {repo_id!r}
filename  = {filename!r}
local_dir = Path({local_dir!r})
token     = ({hf_token.strip()!r}) or None
chunk     = 512 * 1024 * 1024  # report every 512 MB

local_dir.mkdir(parents=True, exist_ok=True)
dest = local_dir / Path(filename).name

url = hf_hub_url(repo_id=repo_id, filename=filename)
headers = {{"Authorization": f"Bearer {{token}}"}} if token else {{}}
print(f"Connecting to HuggingFace for {{filename}} ...", flush=True)

with requests.get(url, headers=headers, stream=True, timeout=60) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    total_gb = total / (1024**3) if total else 0
    print(f"File size: {{total_gb:.2f}} GB  |  dest: {{dest}}", flush=True)
    downloaded = 0
    next_report = chunk
    start = time.time()
    with open(dest, "wb") as f:
        for data in r.iter_content(chunk_size=8*1024*1024):
            f.write(data)
            downloaded += len(data)
            if downloaded >= next_report:
                elapsed = time.time() - start
                pct = (downloaded / total * 100) if total else 0
                speed = downloaded / elapsed / (1024**2)
                print(f"  {{downloaded/(1024**3):.2f}} / {{total_gb:.2f}} GB  ({{pct:.1f}}%)  {{speed:.1f}} MB/s", flush=True)
                next_report += chunk

elapsed = time.time() - start
print(f"Done. {{downloaded/(1024**3):.2f}} GB downloaded in {{elapsed:.0f}}s  ->  {{dest}}", flush=True)
"""
    else:
        # Whole-repo download — use snapshot_download (no good way to stream
        # individual file progress, but at least print start/done).
        script = f"""
import sys, os, time
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id   = {repo_id!r}
local_dir = {local_dir!r}
token     = ({hf_token.strip()!r}) or None

Path(local_dir).mkdir(parents=True, exist_ok=True)
print(f"Starting snapshot download of {{repo_id}} ...", flush=True)
print("(This may appear frozen \u2014 large repos take time to enumerate.)", flush=True)
start = time.time()

dest = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token=token,
)

elapsed = time.time() - start
print(f"Done. Repo snapshot saved in {{elapsed:.0f}}s  ->  {{dest}}", flush=True)
"""

    # CLAUDE-NOTE: Run via the venv Python so huggingface_hub resolves from
    # the installed packages. We inline the script via -c to avoid creating
    # a temp file on disk.
    # stderr=STDOUT so any huggingface_hub warnings appear in the log too.
    py = Path(__file__).parent.parent / "env" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(__file__).parent.parent / "env" / "bin" / "python"

    env: dict[str, str] = {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    if hf_token.strip():
        env["HF_TOKEN"] = hf_token.strip()

    cmd = [str(py), "-c", script.strip()]
    yield from stream_command(cmd, cwd, env=env)


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
