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
    // CLAUDE-NOTE: Plain `git pull` aborts if the Pinokio runtime dir has any
    // local modifications or untracked files that conflict with incoming changes.
    // Use fetch + clean + hard reset instead so updates always succeed.
    // git clean only removes non-gitignored files; user data (app/models/,
    // app/outputs/, .user_settings.json, env/) is gitignored and is untouched.
    {
      method: "shell.run",
      params: {
        message: [
          "git fetch origin",
          "git clean -fd",
          "git reset --hard origin/main",
        ],
      },
    },

    // Update LTX-2 monorepo (app/LTX-2) and re-sync its venv.
    // CLAUDE-NOTE: Use reset --hard like the launcher update above so a dirty
    // working tree (e.g. from a failed previous sync) never blocks the pull.
    {
      method: "shell.run",
      params: {
        path: "app/LTX-2",
        message: [
          "git fetch origin",
          "git reset --hard origin/main",
          "uv sync --frozen",
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
