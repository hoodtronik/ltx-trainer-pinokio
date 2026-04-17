"""User-configurable paths, persisted across sessions.

Stored as JSON at `app/.user_settings.json` (gitignored). Loaded once at UI
startup and mutated via the Project and Settings tabs.

All path fields are stored as strings. They may be relative to the launcher
root or absolute — callers that shell out to LTX-2 should resolve via
`resolve_path()` to get an absolute, OS-normalized path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# CLAUDE-NOTE: Repo root = parent of the `app/` dir containing this file.
# Resolve once at import time so the whole module agrees on it.
REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = REPO_ROOT / "app" / ".user_settings.json"


def _default_ltx_repo_path() -> str:
    """Default to the sibling `../LTX-2` directory (where install.js clones it)."""
    return str((REPO_ROOT.parent / "LTX-2").resolve())


def _default_output_dir() -> str:
    return str((REPO_ROOT / "app" / "outputs").resolve())


@dataclass
class Settings:
    # LTX-2 repo location. install.js defaults this to ../LTX-2.
    ltx_repo_path: str = field(default_factory=_default_ltx_repo_path)

    # Model paths. Empty strings mean "not set" — UI warns before any action
    # that needs them.
    model_path: str = ""
    text_encoder_path: str = ""
    spatial_upscaler_path: str = ""
    distilled_lora_path: str = ""

    # Where training runs write checkpoints + validation samples.
    output_dir: str = field(default_factory=_default_output_dir)

    # HuggingFace token — used only for model downloads, never written to
    # config YAML or logs.
    hf_token: str = ""

    # uv executable — usually just "uv" (on PATH), but overridable for
    # pinned installs.
    uv_path: str = "uv"

    # Python executable for the trainer env. Empty = "use uv run python".
    # Rarely set manually; mostly here for power users.
    python_path: str = ""


def load() -> Settings:
    if not SETTINGS_PATH.exists():
        return Settings()
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Settings()

    # CLAUDE-NOTE: Tolerate unknown keys (forward compat) and missing keys
    # (back compat) by only applying fields the dataclass declares.
    known = {f.name for f in fields(Settings)}
    filtered = {k: v for k, v in data.items() if k in known}
    return Settings(**filtered)


def save(settings: Settings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(asdict(settings), indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_path(p: str) -> str:
    """Normalize a user-supplied path to an absolute, OS-correct string.

    Relative paths are resolved against REPO_ROOT. Absolute paths are left
    alone but still normalized (e.g. forward/backslash consistency).
    """
    if not p:
        return ""
    candidate = Path(p)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate.resolve())


# ---------------------------------------------------------------------------
# Model auto-detection
# ---------------------------------------------------------------------------

# CLAUDE-NOTE: Ordered by priority — first match wins for each field.
# Relative to REPO_ROOT so paths stored in settings work cross-machine.
_MODEL_CANDIDATES: dict[str, list[str]] = {
    "model_path": [
        "app/models/ltx-2.3/ltx-2.3-22b-dev.safetensors",
        "app/models/ltx-2/ltx-2-19b-dev.safetensors",
    ],
    "text_encoder_path": [
        "app/models/gemma-3-12b-it-qat-q4_0-unquantized",
    ],
    "spatial_upscaler_path": [
        "app/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        "app/models/ltx-2/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    ],
    "distilled_lora_path": [
        "app/models/ltx-2.3/ltx-2.3-22b-distilled-lora-384.safetensors",
        "app/models/ltx-2/ltx-2.3-22b-distilled-lora-384.safetensors",
    ],
}


def scan_model_paths() -> dict[str, str]:
    """Scan app/models/ for known model files and return relative paths.

    CLAUDE-NOTE: Returns only fields for which a file/dir was actually found.
    The caller merges this with existing settings — existing non-empty values
    are always preferred over detected ones so the user's manual overrides are
    never clobbered.
    """
    found: dict[str, str] = {}
    for field_name, candidates in _MODEL_CANDIDATES.items():
        for rel in candidates:
            candidate = REPO_ROOT / rel
            # text_encoder_path must be a directory; the rest must be files.
            if field_name == "text_encoder_path":
                if candidate.is_dir():
                    found[field_name] = rel
                    break
            else:
                if candidate.is_file():
                    found[field_name] = rel
                    break
    return found


def autodetect_and_save() -> dict[str, str]:
    """Scan for models, merge into settings (non-destructively), save, return found dict."""
    s = load()
    detected = scan_model_paths()
    changed = False
    for field_name, rel_path in detected.items():
        if not getattr(s, field_name, ""):
            setattr(s, field_name, rel_path)
            changed = True
    if changed:
        save(s)
    return detected
