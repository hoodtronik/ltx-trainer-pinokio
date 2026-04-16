// CLAUDE-NOTE: Install flow — four distinct phases:
//   1. Prereq check (uv, git). Install uv via Pinokio's shell if missing.
//   2. Resolve LTX-2 repo location: default ../LTX-2 relative to this repo,
//      clone if absent. User can override later in Settings tab.
//   3. `uv sync` inside the LTX-2 repo to install the trainer's deps.
//   4. Create a separate Gradio UI venv at env/ and install the UI deps from
//      app/requirements.txt. Kept separate so Gradio's dep tree never
//      conflicts with the trainer's pinned torch/triton/etc.

module.exports = {
  run: [
    {
      method: "log",
      params: {
        text: [
          "========================================",
          "  LTX-Trainer Pinokio — Install",
          "========================================",
          "",
          "This will:",
          "  1. Verify uv + git are available (install uv if missing)",
          "  2. Locate or clone the LTX-2 trainer repo to ../LTX-2",
          "  3. Run `uv sync` inside LTX-2/",
          "  4. Create a separate Gradio UI venv at env/",
          "",
          "NOTE: LTX-2 training requires Linux (triton + bitsandbytes).",
          "On Windows, the UI will install and launch, but training",
          "subprocesses will fail. Use WSL2 or a Linux host for training.",
          "",
        ].join("\n"),
      },
    },

    // Phase 1: uv + git check.
    {
      method: "shell.run",
      params: {
        message: [
          "git --version",
          "uv --version || (echo 'Installing uv...' && {{which('powershell') ? \"powershell -Command \\\"irm https://astral.sh/uv/install.ps1 | iex\\\"\" : \"curl -LsSf https://astral.sh/uv/install.sh | sh\"}})",
        ],
      },
    },

    // Phase 2: resolve or clone LTX-2.
    // CLAUDE-NOTE: Clone into ../LTX-2 only if missing. The path is
    // configurable in the UI's Settings tab; install defaults to sibling.
    {
      method: "shell.run",
      params: {
        path: "..",
        message: [
          "{{fs.existsSync(path.join(__dirname, '..', 'LTX-2')) ? \"echo 'Found existing LTX-2 at ../LTX-2 — skipping clone.'\" : \"git clone https://github.com/Lightricks/LTX-2.git LTX-2\"}}",
        ],
      },
    },

    // Phase 3: uv sync in LTX-2.
    {
      method: "shell.run",
      params: {
        path: "../LTX-2",
        message: [
          "uv sync",
        ],
      },
    },

    // Phase 4: Gradio UI venv.
    // CLAUDE-NOTE: Use uv to create the venv because (a) uv is already
    // installed from phase 1, (b) it resolves a compatible Python (3.10-3.12,
    // since LTX-2 pins >=3.10 and Gradio works cleanly in that range) even
    // if system python is 3.13+.
    //
    // `uv venv` creates a pip-less venv, so we use `uv pip install --python`
    // to install directly against the venv's interpreter — no need to
    // bootstrap pip into the venv first.
    {
      method: "shell.run",
      params: {
        message: [
          "uv venv env --python 3.12",
          "{{platform === 'win32' ? 'uv pip install --python env\\\\Scripts\\\\python.exe -r app/requirements.txt' : 'uv pip install --python env/bin/python -r app/requirements.txt'}}",
        ],
      },
    },

    // Phase 5: GPU detection → writes detected_hardware.json which the
    // Config tab reads to pre-select the training template (standard vs.
    // low-VRAM). Runs inside the UI venv so it shares the same Python the
    // UI will use.
    // CLAUDE-NOTE: Non-fatal on failure — the UI has a safe fallback.
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "python app/detect_hardware.py",
        ],
      },
    },

    {
      method: "log",
      params: {
        text: [
          "",
          "✅ Install complete.",
          "",
          "Detected hardware has been written to detected_hardware.json —",
          "the Config tab will use it to pick the right training template.",
          "",
          "Next steps:",
          "  • Click 'Download Models' to pull LTX-2 + Gemma from HuggingFace",
          "    (OR skip and use your own model paths via Settings tab)",
          "  • Click 'Start' to launch the Gradio UI",
          "",
        ].join("\n"),
      },
    },
  ],
};
