// CLAUDE-NOTE: Install flow — five phases:
//   1. Prereq check (uv, git).
//   2. Clone the LTX-2 monorepo into app/LTX-2/ (training code, ~50 MB).
//      - If app/LTX-2/ exists but is missing packages/ (wrong repo or empty),
//        rename it to app/LTX-2-backup/ to preserve any user data.
//      - If packages/ltx-trainer/ already exists, skip clone (idempotent).
//   3. `uv sync` inside app/LTX-2 (monorepo's own venv, separate from UI venv).
//   4. Create a separate Gradio UI venv at env/ and install the UI deps.
//   5. GPU detection.
//
// Repo: https://github.com/Lightricks/LTX-2 (monorepo — trainer + pipelines + core)
// Models live separately in app/models/ and are never touched by this script.

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
          "  2. Clone LTX-2 monorepo into app/LTX-2/ from github.com/Lightricks/LTX-2",
          "  3. Run `uv sync` inside app/LTX-2/ (monorepo's own venv)",
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

    // Phase 2: clone the LTX-2 monorepo into app/LTX-2/.
    // CLAUDE-NOTE: Split into separate steps with `when` guards because Pinokio
    //   template expressions only expose `exists()` (relative-path check), not
    //   Node's `fs.existsSync` or `path.join` or `__dirname`. Using `when` on
    //   each step lets Pinokio evaluate the condition at runtime and skip
    //   inapplicable steps cleanly.
    //   Use --depth 1 for fast initial checkout — full history isn't needed for training.

    // Step 2a: backup broken LTX-2 dir on Windows (exists but missing packages/)
    {
      when: "{{platform === 'win32' && exists('app/LTX-2') && !exists('app/LTX-2/packages')}}",
      method: "shell.run",
      params: {
        message: "move app\\LTX-2 app\\LTX-2-backup && echo Backed up app/LTX-2 to app/LTX-2-backup",
      },
    },
    // Step 2b: backup broken LTX-2 dir on Unix
    {
      when: "{{platform !== 'win32' && exists('app/LTX-2') && !exists('app/LTX-2/packages')}}",
      method: "shell.run",
      params: {
        message: "mv app/LTX-2 app/LTX-2-backup && echo Backed up app/LTX-2 to app/LTX-2-backup",
      },
    },
    // Step 2c: clone only if packages/ not already present
    {
      when: "{{!exists('app/LTX-2/packages')}}",
      method: "shell.run",
      params: {
        message: "git clone --depth 1 https://github.com/Lightricks/LTX-2.git app/LTX-2",
      },
    },
    // Step 2d: skip message if already installed
    {
      when: "{{exists('app/LTX-2/packages')}}",
      method: "log",
      params: {
        text: "LTX-2 monorepo already installed at app/LTX-2 — skipping clone.",
      },
    },

    // Phase 3: uv sync inside the monorepo (its own venv, separate from the UI venv).
    // CLAUDE-NOTE: Pin to --python 3.12. sentencepiece==0.2.0 (a transitive dep)
    //   has no pre-built Windows wheel for Python 3.13 and fails to compile from
    //   source because its CMakeLists.txt is incompatible with modern CMake (>3.5
    //   policy removed). Python 3.12 has binary wheels for all deps. (2026-04-17)
    {
      method: "shell.run",
      params: {
        path: "app/LTX-2",
        message: [
          "uv sync --python 3.12",
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
