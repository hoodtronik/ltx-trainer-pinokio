// CLAUDE-NOTE: Opens an interactive shell pre-activated with the Gradio venv
// (env/). Pinokio renders this as a full terminal the user can type into.
// Works just like other apps' "Terminal" tabs — e.g. you can run
// `hf download ...`, `python app/app.py`, `uv pip list`, etc.
//
// The `message` field is intentionally empty so Pinokio drops the user
// straight into an interactive prompt rather than running any command.

module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "",
      },
    },
  ],
};
