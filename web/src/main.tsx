// Entry point: the App (src/) with the host that runs the pipeline in the
// browser.
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
