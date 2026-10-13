// Web version: the same App as the desktop (src/), with a host that runs the
// pipeline in the browser instead of Tauri commands.
import React from "react";
import ReactDOM from "react-dom/client";
import App from "../../src/App";
import { setHost } from "../../src/lib/api";
import { webHost } from "./host";
import { registerServiceWorker } from "./host/update";

setHost(webHost);
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
registerServiceWorker();
