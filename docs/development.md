# Development

## Repository layout

| Path | What |
|---|---|
| `src/` | React UI (`App.tsx`, `components/`, `i18n/`, `lib/`) |
| `src/lib/api.ts` | The `Host` interface: everything the UI asks of the platform |
| `web/` | Entry point, `Host` implementation (`web/src/host/`), pipeline, SfM, export, mesh, share viewer |
| `web/brush/` | Patches applied to the pinned Brush commit |
| `worker/` | Cloudflare Worker for share links |
| `scripts/` | Brush build, benchmarks and a ComfyUI mock |

## Running

```bash
pnpm install
pnpm web:wasm      # Brush → WASM (web/pkg), once per Brush pin or patch change
pnpm web:dev       # http://localhost:1440
```

## Tests

```bash
pnpm test:web
```

Benchmark scripts for the pipeline live in `scripts/web-bench/`; see
[web.md](web.md#benchmarks).
