// CLAUDE-NOTE: Reinstalls the LTX-2 monorepo (training code only).
// Models in app/models/ are NEVER touched — they live in a separate
// directory and this script only operates on app/LTX-2/.
// Safe to run when the app is stopped; do NOT run while training.

module.exports = {
  run: [
    {
      method: "log",
      params: {
        text: [
          "========================================",
          "  Reinstall LTX-2 Training Code",
          "========================================",
          "",
          "This deletes app/LTX-2/ and re-clones from GitHub.",
          "Your models in app/models/ are NOT affected.",
          "",
        ].join("\n"),
      },
    },

    // Delete existing app/LTX-2/ (may be wrong repo or corrupt).
    {
      method: "shell.run",
      params: {
        message: [
          "{{platform === 'win32' ? 'if exist app\\\\LTX-2 rmdir /s /q app\\\\LTX-2 && echo Removed old app/LTX-2' : 'rm -rf app/LTX-2 && echo Removed old app/LTX-2'}}",
        ],
      },
    },

    // Fresh clone.
    {
      method: "shell.run",
      params: {
        message: [
          "git clone --depth 1 https://github.com/Lightricks/LTX-2.git app/LTX-2",
        ],
      },
    },

    // Install dependencies inside the monorepo's own venv.
    // CLAUDE-NOTE: Pin --python 3.12 — sentencepiece==0.2.0 (transitive dep)
    // has no 3.13 Windows wheel and fails to build from source.
    {
      method: "shell.run",
      params: {
        path: "app/LTX-2",
        message: [
          "uv sync --python 3.12",
        ],
      },
    },

    {
      method: "log",
      params: {
        text: [
          "",
          "✅ LTX-2 training code reinstalled.",
          "   Restart the app to pick up the new code.",
          "",
        ].join("\n"),
      },
    },
  ],
};
