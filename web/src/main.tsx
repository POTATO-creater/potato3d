// Versão web: o mesmo App do desktop (src/), com o host que roda o pipeline
// no navegador no lugar dos comandos do Tauri.
import React from "react";
import ReactDOM from "react-dom/client";
import App from "../../src/App";
import { setHost } from "../../src/lib/api";
import { webHost } from "./host";

setHost(webHost);
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
