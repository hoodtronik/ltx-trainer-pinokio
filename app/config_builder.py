"""Training YAML config authoring helpers.

Load templates from the LTX-2 monorepo's `packages/ltx-trainer/configs/`,
inject project paths (model, text encoder, data, output), apply user
overrides from the Config tab's form fields, and serialize back to YAML.

Design choices:
  • PyYAML (not ruamel) — the trainer parses YAML and doesn't care about
    comments or key order, so we don't need round-trip fidelity.
  • Full-finetune template is synthesized from the LoRA template at
    runtime (no example YAML ships with that mode). We strip the `lora`
    section and flip `training_mode: "full"`.
  • Form → YAML is one-way by design. The UI regenerates the YAML
    preview from the form on demand; the user can then edit the raw
    YAML directly and that edited text is what gets saved.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from settings import Settings, resolve_path

# CLAUDE-NOTE: Template files live in the trainer monorepo. We use the
# LTX-2 repo path from Settings so a user who moved the repo still finds
# them. The synthetic "full_finetune" template has no file path — it's
# built in-memory from the lora template.
TEMPLATE_FILES = {
    "lora": "packages/ltx-trainer/configs/ltx2_av_lora.yaml",
    "lora_low_vram": "packages/ltx-trainer/configs/ltx2_av_lora_low_vram.yaml",
    "ic_lora": "packages/ltx-trainer/configs/ltx2_v2v_ic_lora.yaml",
    "full_finetune": None,  # synthesized
}

TEMPLATE_LABELS = {
    "lora": "Audio-Video LoRA (standard, 48GB+ VRAM)",
    "lora_low_vram": "Audio-Video LoRA (low-VRAM, 32GB, INT8 quant)",
    "ic_lora": "IC-LoRA video-to-video",
    "full_finetune": "Full fine-tune (80GB+ VRAM)",
}


class ConfigBuilderError(RuntimeError):
    pass


@dataclass
class FormOverrides:
    """Flat field set the Config tab lets users edit directly. Keys that
    stay `None` are not written (template default wins). All values are
    already-typed — the UI-side handlers do the coercion."""

    # model
    training_mode: str | None = None  # "lora" | "full"
    load_checkpoint: str | None = None

    # lora (only if training_mode == "lora")
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None

    # training_strategy
    strategy_name: str | None = None  # "text_to_video" | "video_to_video"
    first_frame_conditioning_p: float | None = None
    with_audio: bool | None = None

    # optimization
    learning_rate: float | None = None
    steps: int | None = None
    batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    optimizer_type: str | None = None  # "adamw" | "adamw8bit"
    scheduler_type: str | None = None
    enable_gradient_checkpointing: bool | None = None

    # acceleration
    mixed_precision_mode: str | None = None  # "no" | "fp16" | "bf16"
    quantization: str | None = None  # "none" | "int8-quanto" | "fp8-quanto"
    load_text_encoder_in_8bit: bool | None = None

    # data
    num_dataloader_workers: int | None = None

    # checkpoints
    checkpoint_interval: int | None = None
    keep_last_n: int | None = None

    # top-level
    seed: int | None = None


def list_template_names() -> list[str]:
    return list(TEMPLATE_FILES.keys())


def load_template(name: str, settings: Settings) -> dict[str, Any]:
    """Load a template as a nested dict. Raises on unknown name / missing file."""
    if name not in TEMPLATE_FILES:
        raise ConfigBuilderError(f"Unknown template: {name}")

    if name == "full_finetune":
        # Synthesize from the standard LoRA template.
        base = load_template("lora", settings)
        base.setdefault("model", {})["training_mode"] = "full"
        # CLAUDE-NOTE: Full fine-tune ignores the `lora` block — the trainer
        # branches on `training_mode`. Dropping it keeps the YAML clean.
        base.pop("lora", None)
        return base

    ltx_root = Path(resolve_path(settings.ltx_repo_path))
    tpl_path = ltx_root / TEMPLATE_FILES[name]
    if not tpl_path.is_file():
        raise ConfigBuilderError(
            f"Template file not found: {tpl_path}\n"
            "Check the LTX-2 repo path in the Project tab.",
        )

    try:
        data = yaml.safe_load(tpl_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigBuilderError(f"Failed to read/parse {tpl_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigBuilderError(f"{tpl_path} did not parse as a mapping.")
    return data


def apply_project_paths(cfg: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Inject model/text-encoder/output paths from the Project tab.

    We only overwrite if the user has actually set the corresponding
    path — blank settings leave the template's placeholder in place so
    the user sees exactly what they need to fill in.
    """
    cfg = copy.deepcopy(cfg)

    model = cfg.setdefault("model", {})
    if settings.model_path:
        model["model_path"] = resolve_path(settings.model_path)
    if settings.text_encoder_path:
        model["text_encoder_path"] = resolve_path(settings.text_encoder_path)

    if settings.output_dir:
        cfg["output_dir"] = resolve_path(settings.output_dir)

    return cfg


def apply_form_overrides(cfg: dict[str, Any], form: FormOverrides) -> dict[str, Any]:
    """Apply overrides onto a template dict. Mutates a deep copy and returns it."""
    cfg = copy.deepcopy(cfg)

    model = cfg.setdefault("model", {})
    if form.training_mode is not None:
        model["training_mode"] = form.training_mode
    if form.load_checkpoint is not None:
        # Empty string means "no resume" — preserve the null in YAML.
        model["load_checkpoint"] = form.load_checkpoint or None

    # If the mode just switched to "full", drop the lora section; if it
    # switched to "lora" and there's no section yet, seed a minimal one.
    if form.training_mode == "full":
        cfg.pop("lora", None)
    elif form.training_mode == "lora" and "lora" not in cfg:
        cfg["lora"] = {"rank": 32, "alpha": 32, "dropout": 0.0}

    if (form.lora_rank is not None or form.lora_alpha is not None or
            form.lora_dropout is not None) and "lora" in cfg:
        lora = cfg["lora"]
        if form.lora_rank is not None:
            lora["rank"] = form.lora_rank
        if form.lora_alpha is not None:
            lora["alpha"] = form.lora_alpha
        if form.lora_dropout is not None:
            lora["dropout"] = form.lora_dropout

    ts = cfg.setdefault("training_strategy", {})
    if form.strategy_name is not None:
        ts["name"] = form.strategy_name
    if form.first_frame_conditioning_p is not None:
        ts["first_frame_conditioning_p"] = form.first_frame_conditioning_p
    if form.with_audio is not None:
        ts["with_audio"] = form.with_audio
        # CLAUDE-NOTE: text_to_video + with_audio needs audio_latents_dir.
        # If user flips with_audio on and the key is missing, inject the
        # default. Leave it alone when off (trainer ignores the key).
        if form.with_audio and ts.get("name") == "text_to_video":
            ts.setdefault("audio_latents_dir", "audio_latents")

    opt = cfg.setdefault("optimization", {})
    for (attr, key) in [
        ("learning_rate", "learning_rate"),
        ("steps", "steps"),
        ("batch_size", "batch_size"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
        ("optimizer_type", "optimizer_type"),
        ("scheduler_type", "scheduler_type"),
        ("enable_gradient_checkpointing", "enable_gradient_checkpointing"),
    ]:
        val = getattr(form, attr)
        if val is not None:
            opt[key] = val

    accel = cfg.setdefault("acceleration", {})
    if form.mixed_precision_mode is not None:
        accel["mixed_precision_mode"] = form.mixed_precision_mode
    if form.quantization is not None:
        # UI sends "none" for the no-quant option; normalize to YAML null.
        accel["quantization"] = None if form.quantization == "none" else form.quantization
    if form.load_text_encoder_in_8bit is not None:
        accel["load_text_encoder_in_8bit"] = form.load_text_encoder_in_8bit

    if form.num_dataloader_workers is not None:
        cfg.setdefault("data", {})["num_dataloader_workers"] = form.num_dataloader_workers

    ck = cfg.setdefault("checkpoints", {})
    if form.checkpoint_interval is not None:
        ck["interval"] = form.checkpoint_interval
    if form.keep_last_n is not None:
        ck["keep_last_n"] = form.keep_last_n

    if form.seed is not None:
        cfg["seed"] = form.seed

    return cfg


def dict_to_yaml(cfg: dict[str, Any]) -> str:
    # CLAUDE-NOTE: sort_keys=False preserves logical section order
    # (model, lora, training_strategy, ...) which matches the example
    # configs the user is likely comparing against.
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, allow_unicode=True)


def yaml_to_dict(text: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigBuilderError(f"Invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigBuilderError("YAML must be a top-level mapping.")
    return data


def validate_for_save(cfg: dict[str, Any]) -> list[str]:
    """Return a list of warning strings. Empty = config looks OK."""
    warnings: list[str] = []

    model = cfg.get("model", {})
    for key, label in [("model_path", "model.model_path"),
                       ("text_encoder_path", "model.text_encoder_path")]:
        val = model.get(key)
        if not val or "path/to/" in str(val):
            warnings.append(f"⚠️ {label} looks like a placeholder — set it in the Project tab.")
        elif not Path(val).exists():
            warnings.append(f"⚠️ {label} does not exist on disk: `{val}`")

    data_root = cfg.get("data", {}).get("preprocessed_data_root")
    if not data_root or "path/to/" in str(data_root):
        warnings.append(
            "⚠️ data.preprocessed_data_root is a placeholder. Fill it in with "
            "the directory produced by the Preprocess step (usually ends in `/.precomputed`).",
        )
    elif not Path(data_root).exists():
        warnings.append(f"⚠️ data.preprocessed_data_root does not exist on disk: `{data_root}`")

    if model.get("training_mode") not in ("lora", "full"):
        warnings.append(f"⚠️ model.training_mode should be 'lora' or 'full', got {model.get('training_mode')!r}")

    if model.get("training_mode") == "lora" and "lora" not in cfg:
        warnings.append("⚠️ training_mode is 'lora' but there's no `lora:` section.")

    # Sanity: multi-bucket preprocessing requires batch_size=1.
    # Can't tell bucket count from the config alone (buckets are decided at
    # preprocess time), but we can flag the combination of batch_size>1 +
    # no_explicit_note as a gentle reminder.
    bs = cfg.get("optimization", {}).get("batch_size", 1)
    if isinstance(bs, int) and bs > 1:
        warnings.append(
            f"ℹ️ optimization.batch_size={bs}. If your dataset was preprocessed with "
            "multiple resolution buckets, this will fail — keep batch_size=1 in that case.",
        )

    return warnings


def save_config(yaml_text: str, out_path: Path) -> tuple[bool, str]:
    """Write YAML text to disk after confirming it parses. Returns (ok, message)."""
    try:
        yaml_to_dict(yaml_text)
    except ConfigBuilderError as exc:
        return False, f"❌ Refusing to save — {exc}"

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_text, encoding="utf-8")
    except OSError as exc:
        return False, f"❌ Write failed: {exc}"

    return True, f"✅ Saved to `{out_path}`"
