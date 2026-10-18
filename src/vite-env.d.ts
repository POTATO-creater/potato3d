/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Share-link server (default: the page's own origin, see web/src/host). */
  readonly VITE_SHARE_ORIGIN?: string;
}
