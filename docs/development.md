# Development

## Repository layout

| Path | What |
|---|---|
| `src/` | React UI shared by both apps (`App.tsx`, `components/`, `i18n/`, `lib/`) |
| `src/lib/api.ts` | The `Host` interface: everything the UI asks of the platform |
| `src/lib/tauri.ts` | Desktop `Host`, backed by Tauri commands |
| `src-tauri/` | Rust backend: frames, masks, COLMAP, Brush, export, project packages |
| `web/` | Browser `Host` (`web/src/host/`), pipeline, SfM, export, share viewer |
| `web/brush/` | Patches applied to the pinned Brush commit (desktop and web) |
| `worker/` | Cloudflare Worker for share links |
| `scripts/` | Tool setup, Brush build, benchmarks and a ComfyUI mock |

## UI without the backend

`pnpm dev:mock` serves only the UI in the browser and replays events recorded
from the CLI. Configure it in `.env.local`:

```
VITE_MOCK_VIDEO=/path/video.mp4
VITE_MOCK_EVENTS=/path/events.jsonl   # CLI stdout
VITE_MOCK_SCENE=/path/scene.json      # output of `cli scene`
```

In the URL, `?os=windows` simulates the Windows UI and `?events=…&scene=…`
swaps files without editing `.env.local`. `?project=<folder>` opens a real
project; export is simulated and returns the `lastExport` that `cli export`
saved in it. Use a folder outside `~/Downloads`/`~/Documents`: macOS may block
Vite's access to them and the server hangs.

## Tests

```bash
pnpm test:rust
```
```bash
pnpm test:web
```

Benchmark scripts for the web pipeline live in `scripts/web-bench/`; see
[web.md](web.md#benchmarks).
