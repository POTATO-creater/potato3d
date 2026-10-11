/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Servidor dos links de compartilhamento (padrão: ver src/lib/tauri.ts e web/src/host). */
  readonly VITE_SHARE_ORIGIN?: string;
}
