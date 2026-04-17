// CLAUDE-NOTE: Pinokio launcher manifest. Dynamic menu — items switch based
// on (a) whether install.js has populated env/, and (b) whether start.js
// is currently running. The menu entries each invoke one of the sibling
// .js scripts.

const path = require("path");

module.exports = {
  version: "1.0",
  title: "LTX-Trainer",
  description: "Gradio UI for the official LTX-2 video model trainer — LoRA, audio-video, IC-LoRA, and full fine-tuning.",
  icon: "icon.png",

  menu: async (kernel, info) => {
    // CLAUDE-NOTE: `installed` test — env/ directory exists after install.js
    // creates the Gradio venv. Use this rather than a sentinel file so a
    // partial install is still recognized and can be completed with Update.
    const installed = await kernel.exists(__dirname, "env");

    const startRunning = info.running("start.js");
    const installRunning = info.running("install.js");
    const updateRunning = info.running("update.js");
    const downloadRunning = info.running("download_models.js");
    const resetRunning = info.running("reset.js");
    const reinstallCodeRunning = info.running("reinstall_ltx2.js");

    if (installRunning) {
      return [{ default: true, icon: "fa-solid fa-plug", text: "Installing…", href: "install.js" }];
    }
    if (updateRunning) {
      return [{ default: true, icon: "fa-solid fa-rotate", text: "Updating…", href: "update.js" }];
    }
    if (downloadRunning) {
      return [{ default: true, icon: "fa-solid fa-download", text: "Downloading models…", href: "download_models.js" }];
    }
    if (resetRunning) {
      return [{ default: true, icon: "fa-solid fa-eraser", text: "Resetting…", href: "reset.js" }];
    }
    if (reinstallCodeRunning) {
      return [{ default: true, icon: "fa-solid fa-code", text: "Reinstalling LTX-2 code…", href: "reinstall_ltx2.js" }];
    }

    if (!installed) {
      return [
        { default: true, icon: "fa-solid fa-plug", text: "Install", href: "install.js" },
      ];
    }

    if (startRunning) {
      // CLAUDE-NOTE: url is stored via local.set(..) in start.js once Gradio
      // prints its address. Until then, show a "Starting…" item with no link.
      const local = info.local("start.js");
      if (local && local.url) {
        return [
          { default: true, icon: "fa-solid fa-rocket", text: "Open Web UI", href: local.url },
          { icon: "fa-solid fa-terminal", text: "Terminal", href: "terminal.js" },
          { icon: "fa-solid fa-scroll", text: "View Logs", href: "start.js" },
          { icon: "fa-solid fa-stop", text: "Stop", href: "stop.js" },
        ];
      }
      return [
        { default: true, icon: "fa-solid fa-rocket", text: "Starting…", href: "start.js" },
      ];
    }

    return [
      { default: true, icon: "fa-solid fa-power-off", text: "Start", href: "start.js" },
      { icon: "fa-solid fa-terminal", text: "Terminal", href: "terminal.js" },
      { icon: "fa-solid fa-download", text: "Download Models", href: "download_models.js" },
      { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
      { icon: "fa-solid fa-code", text: "Reinstall LTX-2 Code", href: "reinstall_ltx2.js" },
      { icon: "fa-solid fa-plug", text: "Re-install (full)", href: "install.js" },
      { icon: "fa-solid fa-eraser", text: "Reset (clean venvs)", href: "reset.js" },
    ];
  },
};
