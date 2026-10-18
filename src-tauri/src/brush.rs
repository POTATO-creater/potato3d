//! Stage 3: Gaussian Splatting training with Brush (Rust + wgpu/Metal).
//!
//! Brush prints no progress outside a TTY, so we enable `info` logging for
//! its process modules and parse the refine lines ("Refine iter N, X splats.")
//! and eval lines ("Eval iter N: PSNR …").

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::proc::run_streaming;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainOptions {
    pub total_steps: u32,
    /// 0–3. AI-generated video has inconsistent lighting: low degrees avoid memorizing noise.
    pub sh_degree: u32,
    pub max_splats: u32,
    pub max_resolution: u32,
    /// Exports an intermediate .ply every N steps (live preview).
    pub export_every: u32,
    /// Holds out 1 of every N images to measure PSNR (0 = off).
    pub eval_split_every: u32,
    /// SSIM loss weight (0 = off). With the pinned Brush it costs ~6–11% more
    /// training time on Metal (3× with Brush 0.3); see src/lib/presets.ts.
    #[serde(default = "default_ssim")]
    pub ssim_weight: f32,
    /// Negative-contribution pruning and DeG regularization (web/brush/deg.patch);
    /// see `TrainOptions` in src/lib/api.ts. Omitted or 0 = off.
    #[serde(default)]
    pub contrib_prune_min_views: Option<u32>,
    #[serde(default)]
    pub contrib_prune_ratio: Option<f32>,
    #[serde(default)]
    pub contrib_prune_start: Option<f32>,
    #[serde(default)]
    pub opac_solid_weight: Option<f32>,
    #[serde(default)]
    pub opac_binary_weight: Option<f32>,
    #[serde(default)]
    pub volume_reg_weight: Option<f32>,
}

fn default_ssim() -> f32 {
    0.2
}

/// Background for masked training: a uniform color drawn at random every step
/// (0.5 ± 0.5). With Brush's black ± 0.1, dark semi-transparent layers outside
/// the silhouette cost almost nothing and turn into a ghost silhouette around
/// the subject; with a varying background, any stray transparency shows up in
/// the loss. Same values as `MASKED_BACKGROUND` in web/src/brush.ts.
const MASKED_BACKGROUND: ([f32; 3], f32) = ([0.5, 0.5, 0.5], 0.5);

/// Training arguments for `brush-cli` (without the dataset and output).
fn train_args(opts: &TrainOptions, masked: bool, export_every: u32) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    let mut push = |k: &str, v: &[String]| {
        a.push(k.into());
        a.extend_from_slice(v);
    };
    let s = |v: &dyn ToString| [v.to_string()];
    push("--total-train-iters", &s(&opts.total_steps));
    push("--sh-degree", &s(&opts.sh_degree.min(3)));
    push("--max-splats", &s(&opts.max_splats));
    push("--max-resolution", &s(&opts.max_resolution));
    push("--ssim-weight", &s(&opts.ssim_weight.max(0.0)));
    push("--export-every", &s(&export_every));
    if opts.eval_split_every > 1 {
        push("--eval-split-every", &s(&opts.eval_split_every));
        push("--eval-every", &s(&export_every));
    }
    let positive = |v: Option<f32>| v.filter(|w| *w > 0.0);
    if let Some(n) = opts.contrib_prune_min_views.filter(|n| *n > 0) {
        push("--contrib-prune-min-views", &s(&n));
        if let Some(r) = opts.contrib_prune_ratio {
            push("--contrib-prune-ratio", &s(&r));
        }
        if let Some(st) = opts.contrib_prune_start {
            push("--contrib-prune-start", &s(&st));
        }
    }
    if let Some(w) = positive(opts.opac_solid_weight) {
        push("--opac-solid-weight", &s(&w));
    }
    if let Some(w) = positive(opts.opac_binary_weight) {
        push("--opac-binary-weight", &s(&w));
    }
    if let Some(w) = positive(opts.volume_reg_weight) {
        push("--volume-reg-weight", &s(&w));
    }
    if masked {
        let (color, noise) = MASKED_BACKGROUND;
        push("--background-color", &color.map(|c| c.to_string()));
        push("--background-noise-strength", &s(&noise));
    }
    a
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum TrainEvent {
    Step { iter: u32, splats: u64 },
    Eval { iter: u32, psnr: f64, ssim: f64 },
    Snapshot { iter: u32, path: String },
}

fn parse_refine(line: &str) -> Option<(u32, u64)> {
    let rest = line.split("Refine iter ").nth(1)?;
    let (iter, rest) = rest.split_once(',')?;
    let splats = rest.trim().split_whitespace().next()?;
    Some((iter.trim().parse().ok()?, splats.parse().ok()?))
}

fn parse_eval(line: &str) -> Option<(u32, f64, f64)> {
    let rest = line.split("Eval iter ").nth(1)?;
    let (iter, rest) = rest.split_once(':')?;
    let psnr = rest.split("PSNR").nth(1)?.split(',').next()?.trim();
    let ssim = rest.split("ssim").nth(1)?.trim();
    Some((iter.trim().parse().ok()?, psnr.parse().ok()?, ssim.parse().ok()?))
}

/// Strips the `[2026-10-04T19:51:15Z INFO  module] ` prefix.
fn strip_env_logger(line: &str) -> &str {
    if line.starts_with('[') {
        if let Some(i) = line.find("] ") {
            return &line[i + 2..];
        }
    }
    line
}

/// `masked`: the dataset images carry the mask alpha (dataset_alpha).
pub async fn run(
    brush: &Path,
    dataset: &Path,
    out_dir: &Path,
    opts: &TrainOptions,
    masked: bool,
    cancel: &CancellationToken,
    mut on_event: impl FnMut(TrainEvent),
    mut on_log: impl FnMut(&str),
) -> Result<PathBuf> {
    if out_dir.exists() {
        std::fs::remove_dir_all(out_dir)?;
    }
    std::fs::create_dir_all(out_dir)?;

    let export_every = opts.export_every.clamp(500, opts.total_steps.max(500));
    let mut cmd = Command::new(brush);
    cmd.env("RUST_LOG", "brush_process=info,brush_cli=info,brush_train=info")
        .arg(dataset)
        .args(train_args(opts, masked, export_every))
        .arg("--export-path")
        .arg(out_dir)
        .args(["--export-name", "splat_{iter}.ply"]);

    // Brush zero-pads `{iter}` to the digit count of the total
    // (splat_02500.ply with 20000 steps), so we scan the folder instead of
    // building the name.
    let mut last_reported = 0u32;
    let mut check_snapshots = |on_event: &mut dyn FnMut(TrainEvent)| {
        let snaps = list_snapshots(out_dir);
        let Some((iter, path)) = snaps.last().cloned() else { return };
        if iter > last_reported {
            last_reported = iter;
            on_event(TrainEvent::Snapshot { iter, path: path.to_string_lossy().into_owned() });
        }
        // Each .ply can be hundreds of MB: keep only the last two
        // (the second-to-last may still be being read by the viewer).
        for (_, old) in snaps.iter().rev().skip(2) {
            let _ = std::fs::remove_file(old);
        }
    };

    run_streaming(cmd, cancel, |raw| {
        let line = strip_env_logger(raw);
        if let Some((iter, splats)) = parse_refine(line) {
            on_event(TrainEvent::Step { iter, splats });
            check_snapshots(&mut on_event);
        } else if let Some((iter, psnr, ssim)) = parse_eval(line) {
            on_event(TrainEvent::Eval { iter, psnr, ssim });
        } else if !line.contains("load_scene_img") && !line.contains("Loaded COLMAP image") {
            on_log(line);
        }
    })
    .await
    .context("Brush training failed")?;

    check_snapshots(&mut on_event);

    // The last export is the result.
    list_snapshots(out_dir)
        .pop()
        .map(|(_, p)| p)
        .context("Brush finished without exporting any .ply")
}

/// `splat_<iter>.ply` snapshots in ascending iteration order.
fn list_snapshots(dir: &Path) -> Vec<(u32, PathBuf)> {
    let mut v: Vec<(u32, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter_map(|p| {
            let iter: u32 = p.file_stem()?.to_str()?.strip_prefix("splat_")?.parse().ok()?;
            (p.extension()? == "ply").then_some((iter, p))
        })
        .collect();
    v.sort_by_key(|(i, _)| *i);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_brush_logs() {
        let r = "[2026-10-04T19:51:20Z INFO  brush_process::train_stream] Refine iter 801, 13799 splats.";
        assert_eq!(parse_refine(strip_env_logger(r)), Some((801, 13799)));
        let e = "Eval iter 1200: PSNR 32.510826, ssim 0.9530626";
        assert_eq!(parse_eval(e), Some((1200, 32.510826, 0.9530626)));
    }

    #[test]
    fn masked_training_uses_random_background() {
        // `scaleLossWeight` (the removed "smooth" option) still comes in older
        // settings and packages: it must deserialize and do nothing.
        let opts: TrainOptions = serde_json::from_value(serde_json::json!({
            "totalSteps": 15000, "shDegree": 1, "maxSplats": 2000000, "maxResolution": 1920,
            "exportEvery": 2500, "evalSplitEvery": 0, "scaleLossWeight": 1e-6, "opacBinaryWeight": 0.01
        }))
        .unwrap();
        let plain = train_args(&opts, false, 2500).join(" ");
        assert!(plain.starts_with("--total-train-iters 15000 --sh-degree 1"));
        assert!(plain.contains("--opac-binary-weight 0.01"));
        assert!(!plain.contains("background") && !plain.contains("scale-loss") && !plain.contains("eval"));
        let masked = train_args(&opts, true, 2500).join(" ");
        assert!(masked.ends_with("--background-color 0.5 0.5 0.5 --background-noise-strength 0.5"));
    }

    #[test]
    fn finds_zero_padded_snapshots() {
        let dir = std::env::temp_dir().join("splatforge_snap_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["splat_02500.ply", "splat_10000.ply", "splat_05000.ply", "other.ply"] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
        let iters: Vec<u32> = list_snapshots(&dir).into_iter().map(|(i, _)| i).collect();
        assert_eq!(iters, vec![2500, 5000, 10000]);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
