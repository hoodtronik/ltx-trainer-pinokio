// CLAUDE-NOTE: Start the Gradio UI. The `on` hook captures Gradio's
// "Running on local URL: http://..." line and stores the URL via local.set
// so pinokio.js's menu can render a clickable "Open Web UI" item.
//
// FIX (2026-04-17):
//   - Regex simplified to ONE capture group so event[1] = the URL.
//     Old version used event[2] (second group) which was unreliable.
//   - Port changed 7870 -> 7878 because 7870 was occupied; Gradio
//     auto-incremented and the hardcoded event[2] URL never matched.
//   - Removed duplicate shell.run at bottom — daemon:true keeps alive.

module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "python app/app.py --host 127.0.0.1 --port 7878",
        ],
        on: [
          {
            // CLAUDE-NOTE: Single capture group — event[1] = full URL string.
            // Gradio prints: "Running on local URL:  http://127.0.0.1:PORT"
            // \\s+ handles the double-space Gradio emits after the colon.
            event: "/Running on local URL:\\s+(https?:\\/\\/\\S+)/",
            done: true,
          },
        ],
      },
      next: null,
    },

    {
      method: "local.set",
      params: {
        // CLAUDE-NOTE: event[1] = capture group 1 = the http://... URL.
        url: "{{input.event[1]}}",
      },
    },

    {
      method: "proceed",
    },
  ],
};
