// CLAUDE-NOTE: Optional model download. Pulls the three main artifacts
// from HuggingFace into app/models/ using huggingface-cli (installed in
// the UI venv so we don't need a second dep tree).
//
// Downloads are HUGE (~80 GB total). This is intentionally a separate menu
// entry so the user opts in explicitly. Skippable — the Settings tab lets
// the user point at any on-disk location (e.g. their existing quantized
// wan2gp LTX-2.3 copy).

module.exports = {
  run: [
    {
      method: "input",
      params: {
        title: "Download LTX-2 models from HuggingFace",
        description: [
          "This will download ~80 GB total to app/models/:",
          "  • Lightricks/LTX-2 → ltx-2-19b-dev.safetensors",
          "  • google/gemma-3-12b-it-qat-q4_0-unquantized (full dir)",
          "  • Lightricks/LTX-2 → ltx-2-spatial-upscaler-x2-1.0.safetensors (for inference)",
          "  • Lightricks/LTX-2 → ltx-2-22b-distilled-lora-384.safetensors (for inference)",
          "",
          "You can skip any of these by un-checking them. If you already",
          "have models elsewhere, skip everything and set paths in the",
          "Settings tab of the UI instead.",
          "",
          "A HuggingFace token is required for Gemma (gated model).",
        ].join("\n"),
        form: [
          { key: "hf_token", type: "password", title: "HuggingFace token (required for Gemma)", default: "" },
          { key: "dl_ltx", type: "checkbox", title: "LTX-2 checkpoint (ltx-2-19b-dev.safetensors, ~38 GB)", default: true },
          { key: "dl_gemma", type: "checkbox", title: "Gemma 3 12B text encoder (~24 GB)", default: true },
          { key: "dl_upscaler", type: "checkbox", title: "Spatial upscaler (~2 GB) — inference only", default: true },
          { key: "dl_distilled", type: "checkbox", title: "Distilled LoRA (~1 GB) — inference only", default: true },
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

    // LTX-2 main checkpoint.
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
      when: "{{input.dl_ltx}}",
    },

    // Gemma text encoder.
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

    // Spatial upscaler (inference).
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          HF_TOKEN: "{{input.hf_token}}",
          HF_HUB_ENABLE_HF_TRANSFER: "1",
        },
        message: [
          "huggingface-cli download Lightricks/LTX-2 ltx-2-spatial-upscaler-x2-1.0.safetensors --local-dir app/models/ltx-2",
        ],
      },
      when: "{{input.dl_upscaler}}",
    },

    // Distilled LoRA (inference).
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          HF_TOKEN: "{{input.hf_token}}",
          HF_HUB_ENABLE_HF_TRANSFER: "1",
        },
        message: [
          "huggingface-cli download Lightricks/LTX-2 ltx-2-22b-distilled-lora-384.safetensors --local-dir app/models/ltx-2",
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
          "Models are in app/models/. They will appear in the Settings tab's",
          "path pickers automatically.",
          "",
        ].join("\n"),
      },
    },
  ],
};
