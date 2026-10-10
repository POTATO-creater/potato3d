import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { setHost } from "./lib/api";
import { tauriHost } from "./lib/tauri";

async function boot() {
  if (import.meta.env.VITE_MOCK === "1") {
    const { installMock } = await import("./dev/mock");
    installMock();
  }
  setHost(tauriHost);
  ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
}

boot();
