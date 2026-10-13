//! Optimized export of the trained splat: opacity pruning, SH band cutting,
//! merge-based decimation, and output as .ply or compressed .sog.
//!
//! Runs on the final `splat.ply` without retraining, so several settings can
//! be tried and compared in the viewer.

pub mod decimate;
pub mod ply;
pub mod sog;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::msg;
use crate::msg::Msg;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Ply,
    Sog,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportOptions {
    pub format: Format,
    /// Fraction of gaussians to remove by merging (0–0.9).
    pub decimate: f32,
    /// Removes gaussians with opacity below this (0–1; 0 = off).
    pub min_opacity: f32,
    /// Maximum SH degree in the output (`None` = the training one).
    pub sh_degree: Option<u32>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self { format: Format::Sog, decimate: 0.0, min_opacity: 0.0, sh_degree: None }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportStats {
    pub path: String,
    pub input_count: usize,
    /// After opacity pruning.
    pub pruned_count: usize,
    pub output_count: usize,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub sh_degree: u32,
    pub seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SplatInfo {
    pub count: usize,
    pub sh_degree: u32,
    pub bytes: u64,
}

pub fn info(path: &Path) -> Result<SplatInfo> {
    let (count, sh_degree) = ply::info(path)?;
    Ok(SplatInfo { count, sh_degree, bytes: std::fs::metadata(path)?.len() })
}

/// Output file name; it changes with the settings so exports with other
/// parameters are not overwritten.
pub fn output_name(opts: &ExportOptions) -> String {
    let mut name = String::from("splat");
    if opts.decimate > 0.0 {
        name += &format!("_dec{}", (opts.decimate * 100.0).round() as u32);
    }
    if opts.min_opacity > 0.0 {
        name += &format!("_op{}", (opts.min_opacity * 1000.0).round() as u32);
    }
    if let Some(d) = opts.sh_degree {
        name += &format!("_sh{d}");
    }
    if opts.format == Format::Ply && name == "splat" {
        // Don't overwrite the training splat.ply.
        name += "_export";
    }
    name + match opts.format {
        Format::Ply => ".ply",
        Format::Sog => ".sog",
    }
}

pub fn run(input: &Path, out_dir: &Path, opts: &ExportOptions, mut progress: impl FnMut(f32, &Msg)) -> Result<ExportStats> {
    let t0 = Instant::now();
    progress(0.0, &msg!("export.readingPly"));
    let input_bytes = std::fs::metadata(input)?.len();
    let mut s = ply::read(input)?;
    let input_count = s.count;

    // Pruning: drop low opacities and any row with NaN/∞.
    let min_logit = if opts.min_opacity > 0.0 {
        let p = opts.min_opacity.clamp(1e-6, 0.999) as f64;
        (p / (1.0 - p)).ln() as f32
    } else {
        f32::NEG_INFINITY
    };
    let rl = s.rest_len();
    let keep: Vec<u32> = (0..s.count)
        .filter(|&i| {
            s.opacity[i] >= min_logit
                && s.pos[i * 3..i * 3 + 3].iter().chain(&s.scale[i * 3..i * 3 + 3]).all(|v| v.is_finite())
                && s.rot[i * 4..i * 4 + 4].iter().any(|v| *v != 0.0)
                && s.rest[i * rl..(i + 1) * rl].iter().all(|v| v.is_finite())
        })
        .map(|i| i as u32)
        .collect();
    if keep.len() < s.count {
        s = s.select(&keep);
    }
    let pruned_count = s.count;
    if pruned_count == 0 {
        anyhow::bail!("opacity pruning removed every gaussian");
    }

    if let Some(d) = opts.sh_degree {
        s.truncate_sh(d);
    }

    // Weight of each phase in the progress bar.
    let (dec_span, write_start) = if opts.decimate > 0.0 { (0.55, 0.6) } else { (0.0, 0.1) };
    if opts.decimate > 0.0 {
        progress(0.05, &msg!("export.merging"));
        if let Some(d) = decimate::decimate(&s, opts.decimate, |f| progress(0.05 + f * dec_span, &msg!("export.merging"))) {
            s = d;
        }
    }

    std::fs::create_dir_all(out_dir)?;
    let path: PathBuf = out_dir.join(output_name(opts));
    let span = 1.0 - write_start;
    let output_bytes = match opts.format {
        Format::Ply => {
            progress(write_start, &msg!("export.writingPly"));
            ply::write(&path, &s)?
        }
        Format::Sog => sog::write(&path, &s, |f, m| progress(write_start + f * span, m))?,
    };
    progress(1.0, &msg!("export.done"));

    Ok(ExportStats {
        path: path.to_string_lossy().into_owned(),
        input_count,
        pruned_count,
        output_count: s.count,
        input_bytes,
        output_bytes,
        sh_degree: s.sh_degree,
        seconds: t0.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_reflect_options() {
        let o = ExportOptions::default();
        assert_eq!(output_name(&o), "splat.sog");
        assert_eq!(output_name(&ExportOptions { format: Format::Ply, ..o.clone() }), "splat_export.ply");
        let o = ExportOptions { decimate: 0.5, min_opacity: 0.02, sh_degree: Some(0), ..o };
        assert_eq!(output_name(&o), "splat_dec50_op20_sh0.sog");
    }

    /// Benchmarks with a real splat:
    /// `SPLATFORGE_TEST_PLY=<splat.ply> cargo test --release export::tests::real -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real() {
        let input = PathBuf::from(std::env::var("SPLATFORGE_TEST_PLY").expect("SPLATFORGE_TEST_PLY"));
        let out = std::env::temp_dir().join("splatforge-export-test");
        for opts in [
            ExportOptions { format: Format::Ply, decimate: 0.0, min_opacity: 0.0, sh_degree: None },
            ExportOptions { format: Format::Sog, decimate: 0.0, min_opacity: 0.0, sh_degree: None },
            ExportOptions { format: Format::Ply, decimate: 0.5, min_opacity: 0.0, sh_degree: None },
            ExportOptions { format: Format::Sog, decimate: 0.5, min_opacity: 0.02, sh_degree: Some(0) },
        ] {
            let st = run(&input, &out, &opts, |_, _| {}).unwrap();
            println!(
                "{:?} → {} · {} → {} → {} splats · {:.1} MB → {:.2} MB ({:.1}×) · {:.2} s",
                opts,
                st.path,
                st.input_count,
                st.pruned_count,
                st.output_count,
                st.input_bytes as f64 / 1e6,
                st.output_bytes as f64 / 1e6,
                st.input_bytes as f64 / st.output_bytes as f64,
                st.seconds
            );
        }
    }
}
