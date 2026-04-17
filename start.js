// CLAUDE-NOTE: Start the Gradio UI. The `on` hook captures Gradio's
// "Running on local URL: http://..." line and stores the URL via local.set
// so pinokio.js's menu can render a clickable "Open Web UI" item.
//
// FIX (2026-04-17): removed `next: null` from the shell.run step.
// `next: null` was preventing the pipeline from advancing to `local.set`
// after `done: true` fired — the URL was captured but never stored,
// so the menu stayed stuck on "Starting...".
//
// Regex has 2 capture groups:
//   event[0] = full match
//   event[1] = "Running on local URL:  " (prefix)
//   event[2] = "http://127.0.0.1:7870"  (the URL we want)

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
            // Match Gradio's "Running on local URL:  http://..." line.
            // event[2] = the URL (2nd capture group).
            event: "/(Running on local URL:\\s+)(https?:\\/\\/\\S+)/",
            done: true,
          },
        ],
      },
      // CLAUDE-NOTE: next:null deliberately removed — it was blocking the
      // pipeline from reaching local.set after done:true fired.
    },

    {
      method: "local.set",
      params: {
        url: "{{input.event[2]}}",
      },
    },

    {
      method: "proceed",
    },
  ],
};
