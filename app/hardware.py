"""Read-side helper for the hardware detection result.

The Config tab calls `get_hardware_info()` to decide which YAML template to
pre-select. The Settings tab displays the full info dict in "Check Installation".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DETECTED_FILE = _REPO_ROOT / "detected_hardware.json"


def get_hardware_info() -> dict[str, Any]:
    """Return the detected hardware dict, or a safe default if detection
    hasn't run (e.g. user is running app.py standalone without installing
    via Pinokio)."""
    if _DETECTED_FILE.exists():
        try:
            return json.loads(_DETECTED_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "os": None,
        "training_supported_natively": None,
        "gpu_detected": False,
        "gpu_name": None,
        "vram_mib": None,
        "recommended_template": "lora",  # CLAUDE-NOTE: optimistic default; the
        # Config tab will warn if detection hasn't run.
        "reason": "Hardware detection has not run yet. Re-install via Pinokio, "
        "or run `python app/detect_hardware.py` manually.",
    }


def recommended_template() -> str:
    return get_hardware_info().get("recommended_template", "lora")
