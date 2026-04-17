// CLAUDE-NOTE: Optional model download. Pulls the main artifacts
// from HuggingFace into app/models/ using huggingface-cli (installed in
// the UI venv so we don't need a second dep tree).
//
// FIX (2026-04-17): removed `type: "select"` form field — that type is not
// supported in this Pinokio version and silently prevented the Done button
// from firing at all. Replaced with two checkboxes (one per model line).
// User checks whichever variant they want; if both are checked, both download.
//
// Downloads are HUGE (~80 GB total for all items). Intentionally a separate
// menu entry so the user opts in explicitly.

module.exports = {
  run: [
    {
      method: "input",
      params: {
        title: "Download LTX-2.3 models from HuggingFace",
        description: [
          "LTX-2 has two current model lines:",
          "  • LTX-2.3 (22B) — current release, recommended",
          "  • LTX-2   (19B) — older, still supported by the trainer",
          "",
          "Check the items you want to download (~80 GB total if all checked).",
          "A HuggingFace token is required for Gemma (it is a gated model).",
          "",
          "Tip: Skip everything here if you already have the models on disk",
          "— just update the paths in the Project tab instead.",
        ].join("\n"),
        form: [
          { key: "hf_token", type: "password", title: "HuggingFace token (required for Gemma)", default: "" },
          { key: "dl_main_22b",   type: "checkbox", title: "Main checkpoint LTX-2.3 22B ★ recommended (~40 GB)", default: true },
          { key: "dl_main_19b",   type: "checkbox", title: "Main checkpoint LTX-2 19B — older/legacy (~12 GB)", default: false },
          { key: "dl_gemma",      type: "checkbox", title: "Gemma 3 12B text encoder — gated, needs HF token (~24 GB)", default: true },
          { key: "dl_upscaler",   type: "checkbox", title: "Spatial upscaler x2 (~2 GB) — inference only", default: true },
          { key: "dl_distilled",  type: "checkbox", title: "Distilled LoRA (~1 GB) — inference only (fast 8-step generation)", default: true },
        ],
      },
    },

    // Ensure huggingface-cli is in the venv.
    // CLAUDE-NOTE: Gradio venv is pip-less — use `uv pip install --python`
    // to install against the venv's interpreter without pip bootstrap.
    {
      method: "shell.run",
      params: {
        message: [
          "{{platform === 'win32' ? 'uv pip install --python env\\\\Scripts\\\\python.exe -U huggingface_hub[cli,hf_transfer]' : 'uv pip install --python env/bin/python -U huggingface_hub[cli,hf_transfer]'}}",
        ],
      },
    },

    // LTX-2.3 22B checkpoint (recommended).
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          HF_TOKEN: "{{input.hf_token}}",
          HF_HUB_ENABLE_HF_TRANSFER: "1",
        },
        message: [
          "huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-dev.safetensors --local-dir app/models/ltx-2.3",
        ],
      },
      when: "{{input.dl_main_22b}}",
    },

    // LTX-2 19B checkpoint (legacy).
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          HF_TOKEN: "{{input.hf_token}}",
          HF_HUB_ENABLE_HF_TRANSFER: "1",
        },
        message: [
          "huggingface-cli download Lightricks/LTX-2 ltx-2-19b-dev.safetensors --local-dir app/models/ltx-2",
        ],
      },
      when: "{{input.dl_main_19b}}",
    },

    // Gemma 3 12B text encoder (gated — needs HF token).
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          HF_TOKEN: "{{input.hf_token}}",
          HF_HUB_ENABLE_HF_TRANSFER: "1",
        },
        message: [
          "huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir app/models/gemma-3-12b-it-qat-q4_0-unquantized",
        ],
      },
      when: "{{input.dl_gemma}}",
    },

    // Spatial upscaler (shared across both model lines).
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          HF_TOKEN: "{{input.hf_token}}",
          HF_HUB_ENABLE_HF_TRANSFER: "1",
        },
        message: [
          "huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.0.safetensors --local-dir app/models/ltx-2.3",
        ],
      },
      when: "{{input.dl_upscaler}}",
    },

    // Distilled LoRA (for fast two-stage inference pipelines).
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          HF_TOKEN: "{{input.hf_token}}",
          HF_HUB_ENABLE_HF_TRANSFER: "1",
        },
        message: [
          "huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-lora-384.safetensors --local-dir app/models/ltx-2.3",
        ],
      },
      when: "{{input.dl_distilled}}",
    },

    {
      method: "log",
      params: {
        text: [
          "",
          "✅ Model downloads complete (for all checked items).",
          "",
          "Models landed in app/models/. Update the paths in the",
          "Project tab to point at them before training or generating.",
          "",
        ].join("\n"),
      },
    },
  ],
};
