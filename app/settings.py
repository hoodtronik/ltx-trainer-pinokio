"""User-configurable paths, persisted across sessions.

Stored as JSON at `app/.user_settings.json` (gitignored). Loaded once at UI
startup and mutated via the Project and Settings tabs.

All path fields are stored as strings. They may be relative to the launcher
root or absolute — callers that shell out to LTX-2 should resolve via
`resolve_path()` to get an absolute, OS-normalized path.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# CLAUDE-NOTE: Repo root = parent of the `app/` dir containing this file.
# Resolve once at import time so the whole module agrees on it.
REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = REPO_ROOT / "app" / ".user_settings.json"


def _default_ltx_repo_path() -> str:
    """Default to app/LTX-2/ (where install.js clones the monorepo)."""
    return str((REPO_ROOT / "app" / "LTX-2").resolve())


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


def _stored_path_exists(s: "Settings", field_name: str) -> bool:
    """Return True if the stored path for field_name actually exists on disk."""
    raw = getattr(s, field_name, "")
    if not raw:
        return False
    resolved = Path(resolve_path(raw))
    if field_name == "text_encoder_path":
        return resolved.is_dir()
    return resolved.is_file()


def autodetect_and_save() -> dict[str, str]:
    """Scan app/models/ for known files, update settings, save, return found dict.

    CLAUDE-NOTE: Overrides a stored path when it is empty OR when it no longer
    exists on disk (stale path from a previous install or location change).
    Existing paths that resolve to real files/dirs are left untouched so
    manual overrides survive app restarts.
    """
    s = load()
    detected = scan_model_paths()
    changed = False
    for field_name, rel_path in detected.items():
        if not _stored_path_exists(s, field_name):
            setattr(s, field_name, rel_path)
            changed = True
    if changed:
        save(s)
    return detected


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

# CLAUDE-NOTE: Size ranges are ±30% of the real HF file sizes, wide enough to
# tolerate minor version bumps but narrow enough to catch wrong-model-class
# errors (e.g. a 3 GB FP8 inference variant instead of the 40 GB BF16 checkpoint).
_MODEL_SIZE_RANGES: dict[str, tuple[int, int]] = {
    "ltx-2.3-22b-dev.safetensors":                 (28_000_000_000, 56_000_000_000),  # ~40 GB
    "ltx-2-19b-dev.safetensors":                   ( 7_000_000_000, 18_000_000_000),  # ~12 GB
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors": (  600_000_000,  3_500_000_000),  # ~1.8 GB
    "ltx-2.3-22b-distilled-lora-384.safetensors":  (  100_000_000,  1_500_000_000),  # ~500 MB
}

# Filenames to search for when recursively scanning an external directory.
# Each tuple is (filename, field_name).
_EXTERNAL_FILE_PATTERNS: list[tuple[str, str]] = [
    ("ltx-2.3-22b-dev.safetensors",                 "model_path"),
    ("ltx-2-19b-dev.safetensors",                   "model_path"),
    ("ltx-2.3-spatial-upscaler-x2-1.0.safetensors", "spatial_upscaler_path"),
    ("ltx-2.3-22b-distilled-lora-384.safetensors",  "distilled_lora_path"),
]

_GEMMA_DIR_NAME = "gemma-3-12b-it-qat-q4_0-unquantized"
# Files that must exist inside the Gemma directory for it to be considered valid.
_GEMMA_MARKER_FILES = ["tokenizer.json", "config.json"]


def _read_safetensors_header(path: Path) -> dict | None:
    """Read the JSON metadata section of a .safetensors file without loading tensors."""
    try:
        with open(path, "rb") as f:
            size_bytes = f.read(8)
            if len(size_bytes) < 8:
                return None
            header_size = struct.unpack("<Q", size_bytes)[0]
            if header_size > 50_000_000:  # sanity: header > 50 MB is malformed
                return None
            return json.loads(f.read(header_size))
    except Exception:
        return None


def validate_model_file(path: str, field_name: str) -> tuple[bool, str]:
    """Check whether a path is likely the correct model variant for LTX training.

    Returns (is_valid, human-readable message).
    is_valid=False means the file is definitively wrong, missing, or corrupt.
    """
    p = Path(resolve_path(path))

    if field_name == "text_encoder_path":
        if not p.is_dir():
            return False, "Not a directory — text encoder must be a folder"
        missing = [f for f in _GEMMA_MARKER_FILES if not (p / f).exists()]
        if missing:
            return False, f"Missing {', '.join(missing)} — may not be the Gemma model folder"
        try:
            n = sum(1 for _ in p.iterdir())
        except OSError:
            n = 0
        return True, f"✅ Directory OK ({n} files)"

    if not p.is_file():
        return False, "File not found"

    size = p.stat().st_size
    if size < 1000:
        return False, "File is empty or nearly empty — download may be incomplete"

    fname = p.name
    gb = size / 1_000_000_000

    if fname in _MODEL_SIZE_RANGES:
        lo, hi = _MODEL_SIZE_RANGES[fname]
        if size < lo:
            return False, (
                f"File too small ({gb:.1f} GB, expected ≥{lo/1e9:.0f} GB). "
                "This is likely a quantized inference variant — LTX training "
                "requires the full-precision BF16 base checkpoint."
            )
        if size > hi:
            return False, (
                f"File larger than expected ({gb:.1f} GB, expected ≤{hi/1e9:.0f} GB). "
                "Verify this is the correct model."
            )

    if fname.endswith(".safetensors"):
        header = _read_safetensors_header(p)
        if header is None:
            return False, "Cannot read safetensors header — file may be corrupt or incomplete"

        # CLAUDE-NOTE: Training requires BF16 weights. If >10% of tensors are
        # quantized (FP8/INT8), this is an inference-only model that will fail
        # during training. Many community reposts have the same filename as the
        # official checkpoint but are FP8-quantized for VRAM savings.
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if tensors:
            quant_dtypes = {"F8_E4M3", "F8_E5M2", "I8", "UI8", "I4"}
            quant_count = sum(
                1 for v in tensors.values()
                if isinstance(v, dict) and v.get("dtype") in quant_dtypes
            )
            if quant_count > len(tensors) * 0.10:
                pct = quant_count / len(tensors) * 100
                return False, (
                    f"⚠️ {pct:.0f}% of tensors are quantized (FP8/INT8). "
                    "Training needs the full-precision BF16 checkpoint — "
                    "this appears to be an inference-optimized model."
                )

        return True, f"✅ Valid safetensors, BF16 ({gb:.1f} GB)"

    return True, f"✅ File found ({gb:.1f} GB)"


# ---------------------------------------------------------------------------
# External directory scan
# ---------------------------------------------------------------------------

def scan_external_directory(directory: str) -> dict[str, list[dict]]:
    """Recursively scan any directory for LTX-compatible model files.

    Returns {field_name: [{"path": str, "valid": bool, "message": str}, ...]}.
    All matches are returned (not just the first) so the UI can show every
    option and let the user decide — useful when someone has both FP8 and BF16
    copies in the same ComfyUI models folder.
    """
    results: dict[str, list[dict]] = {k: [] for k in _MODEL_CANDIDATES}
    root = Path(directory)
    if not root.is_dir():
        return results

    try:
        for fname, field_name in _EXTERNAL_FILE_PATTERNS:
            for match in root.rglob(fname):
                if match.is_file():
                    valid, msg = validate_model_file(str(match), field_name)
                    results[field_name].append({"path": str(match), "valid": valid, "message": msg})

        for match in root.rglob(_GEMMA_DIR_NAME):
            if match.is_dir():
                valid, msg = validate_model_file(str(match), "text_encoder_path")
                results["text_encoder_path"].append({"path": str(match), "valid": valid, "message": msg})
    except PermissionError:
        pass

    return results
