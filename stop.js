// CLAUDE-NOTE: Stops the running Gradio UI daemon (start.js).
// Pinokio's shell.stop terminates the shell session that start.js owns.
// The "id" must match the script filename (without path) that launched
// the daemon — in this case "start.js".

module.exports = {
  run: [
    {
      method: "shell.stop",
      params: {
        id: "start.js",
      },
    },
  ],
};
