// CLAUDE-NOTE: Reset — removes the Gradio UI venv at env/ and clears ruff
// / pycache artifacts so the next Install starts from clean. Intentionally
// does NOT touch:
//   • app/models/ (huge model downloads)
//   • app/outputs/ (training runs)
//   • ../LTX-2/ (the trainer repo itself — use `git clean` there if needed)
// User must explicitly confirm before anything is deleted.

module.exports = {
  run: [
    {
      method: "input",
      params: {
        title: "Reset launcher?",
        description: [
          "This will delete the Gradio UI venv (env/) and Python caches",
          "in this launcher's folder.",
          "",
          "It will NOT delete:",
          "  • Downloaded models in app/models/",
          "  • Training outputs in app/outputs/",
          "  • The LTX-2 trainer at ../LTX-2/",
        ].join("\n"),
        form: [
          {
            key: "confirm",
            type: "checkbox",
            title: "Yes, remove the env/ venv",
            default: false,
          },
        ],
      },
    },

    {
      method: "fs.rm",
      params: {
        path: "env",
      },
      when: "{{input.confirm}}",
    },

    {
      method: "fs.rm",
      params: { path: "app/__pycache__" },
      when: "{{input.confirm}}",
    },

    {
      method: "log",
      params: {
        text: [
          "{{input.confirm ? '✅ Reset complete. Click Install to rebuild the venv.' : 'Reset cancelled.'}}",
        ].join("\n"),
      },
    },
  ],
};
