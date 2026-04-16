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
    {
      method: "shell.run",
      params: {
        message: [
          "{{platform === 'win32' ? 'env\\\\Scripts\\\\python.exe -m pip install -r app/requirements.txt' : 'env/bin/python -m pip install -r app/requirements.txt'}}",
        ],
      },
    },

    {
      method: "log",
      params: { text: "✅ Update complete." },
    },
  ],
};
