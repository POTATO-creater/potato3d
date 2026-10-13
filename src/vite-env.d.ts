/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Share-link server (default: see src/lib/tauri.ts and web/src/host). */
  readonly VITE_SHARE_ORIGIN?: string;
}
