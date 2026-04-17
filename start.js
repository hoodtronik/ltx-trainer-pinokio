// CLAUDE-NOTE: Start the Gradio UI. The `on` hook captures Gradio's
// "Running on local URL: http://..." line and stores the URL via local.set
// so pinokio.js's menu can render a clickable "Open Web UI" item.
//
// History of fixes (2026-04-17):
//   - Removed next:null which blocked pipeline from reaching local.set.
//   - Removed `proceed` step — not a valid RPC method in this Pinokio version.
//   - daemon:true keeps the shell process alive; no keepalive shell needed.
//
// Regex capture groups:
//   event[0] = full match string
//   event[1] = "Running on local URL:  " (prefix — not used)
//   event[2] = "http://127.0.0.1:7870"  (the URL stored via local.set)

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
            event: "/(Running on local URL:\\s+)(https?:\\/\\/\\S+)/",
            done: true,
          },
        ],
      },
    },

    {
      method: "local.set",
      params: {
        url: "{{input.event[2]}}",
      },
    },
  ],
};
