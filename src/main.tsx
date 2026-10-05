import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

async function boot() {
  if (import.meta.env.VITE_MOCK === "1") {
    const { installMock } = await import("./dev/mock");
    installMock();
  }
  ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
}

boot();
