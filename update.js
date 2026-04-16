// CLAUDE-NOTE: Update flow — pulls latest LTX-2 trainer code, re-syncs its
// deps, and refreshes the Gradio UI venv. Does NOT touch models or user
// data in app/models/, app/outputs/, etc.

module.exports = {
  run: [
    {
      method: "log",
      params: {
        text: "Pulling latest LTX-2 trainer code and refreshing deps…",
      },
    },

    // Update this launcher itself.
    {
      method: "shell.run",
      params: {
        message: [
          "git pull",
        ],
      },
    },

    // Update LTX-2 trainer repo.
    {
      method: "shell.run",
      params: {
        path: "../LTX-2",
        message: [
          "git pull",
          "uv sync",
        ],
      },
    },

    // Refresh Gradio UI deps (in case app/requirements.txt changed).
    // CLAUDE-NOTE: Uses `uv pip install --python <venv-python>` — the
    // Gradio venv is pip-less (uv venv default) so we can't call
    // `python -m pip` inside it.
    {
      method: "shell.run",
      params: {
        message: [
          "{{platform === 'win32' ? 'uv pip install --python env\\\\Scripts\\\\python.exe -r app/requirements.txt' : 'uv pip install --python env/bin/python -r app/requirements.txt'}}",
        ],
      },
    },

    {
      method: "log",
      params: { text: "✅ Update complete." },
    },
  ],
};
