// CLAUDE-NOTE: Start the Gradio UI. The `on` hook captures Gradio's
// "Running on local URL: http://..." line and stores the URL via local.set
// so pinokio.js's menu can render a clickable "Open Web UI" item.
//
// We pass --host 127.0.0.1 and a specific port so the URL is predictable
// even if Gradio chooses a different port on retry.

module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "python app/app.py --host 127.0.0.1 --port 7870",
        ],
        on: [
          {
            // Match Gradio's standard "Running on local URL:  http://..." line.
            // The 2nd capture group is the URL.
            event: "/(Running on local URL:\\s+)(https?:\\/\\/\\S+)/",
            done: true,
          },
        ],
      },
      next: null,
    },

    {
      method: "local.set",
      params: {
        // CLAUDE-NOTE: input.event[2] is the URL from the regex above.
        url: "{{input.event[2]}}",
      },
    },

    {
      method: "proceed",
    },

    // Keep the process alive — this step never completes.
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "python app/app.py --host 127.0.0.1 --port 7870",
        ],
      },
    },
  ],
};
