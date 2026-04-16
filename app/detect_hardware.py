"""Detect the user's GPU and pick a default training config template.

Runs at the end of install.js. Writes `detected_hardware.json` in the repo
root. The UI (hardware.py + Config tab) reads that file to decide which
template to pre-select and which warnings to show.

Why a file instead of runtime-only detection: the user might run install on
one session and start the UI in another; caching the result avoids re-running
nvidia-smi on every app launch, and gives us a stable artifact we can surface
in the Settings tab's "Check Installation" output.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

# CLAUDE-NOTE: Thresholds chosen from the LTX-2 trainer docs:
#   • 80 GB = recommended for standard config
#   • 32 GB = minimum for low-VRAM config (INT8 quant + adamw8bit + rank 16)
#   • 40 GB = our split point — standard config fits on 48 GB cards like the
#     RTX 6000 Ada (the user's hardware); below 40 GB we default to low-VRAM.
VRAM_STANDARD_THRESHOLD_MB = 40_000


def _query_nvidia_smi() -> tuple[str, int] | None:
    """Return (gpu_name, vram_mib) for the first GPU, or None on any failure."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if proc.returncode != 0:
        return None
    first_line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    if not first_line or "," not in first_line:
        return None
    name, vram_str = (part.strip() for part in first_line.split(",", 1))
    try:
        vram_mib = int(vram_str)
    except ValueError:
        return None
    return name, vram_mib


def detect() -> dict:
    nvidia = _query_nvidia_smi()
    os_name = platform.system()

    # CLAUDE-NOTE: Triton + bitsandbytes are Linux-only per LTX-2 deps. On
    # Windows native, training will fail at import regardless of GPU. Flag it
    # so the UI can show a prominent warning.
    training_supported = os_name == "Linux"

    result: dict[str, object] = {
        "os": os_name,
        "os_release": platform.release(),
        "training_supported_natively": training_supported,
    }

    if nvidia is None:
        result.update(
            {
                "gpu_detected": False,
                "gpu_name": None,
                "vram_mib": None,
                "recommended_template": "lora_low_vram",
                "reason": "No NVIDIA GPU detected (nvidia-smi failed or missing). "
                "Defaulting to low-VRAM template. Override via the Config tab.",
            },
        )
        return result

    gpu_name, vram_mib = nvidia
    if vram_mib >= VRAM_STANDARD_THRESHOLD_MB:
        template = "lora"
        reason = (
            f"{gpu_name} has {vram_mib} MiB VRAM (>= {VRAM_STANDARD_THRESHOLD_MB}). "
            "Using the standard audio-video LoRA template."
        )
    else:
        template = "lora_low_vram"
        reason = (
            f"{gpu_name} has {vram_mib} MiB VRAM (< {VRAM_STANDARD_THRESHOLD_MB}). "
            "Using the low-VRAM template (INT8 quantization + 8-bit optimizer + rank 16)."
        )

    result.update(
        {
            "gpu_detected": True,
            "gpu_name": gpu_name,
            "vram_mib": vram_mib,
            "recommended_template": template,
            "reason": reason,
        },
    )
    return result


def main() -> int:
    out_path = Path(__file__).resolve().parent.parent / "detected_hardware.json"
    data = detect()
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"[detect_hardware] Wrote {out_path}")
    print(json.dumps(data, indent=2))
    if not data["training_supported_natively"]:
        print(
            "[detect_hardware] WARNING: Native training not supported on "
            f"{data['os']}. Use WSL2 or a Linux host. The UI will install "
            "and launch, but train.py / process_dataset.py subprocesses will fail.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
