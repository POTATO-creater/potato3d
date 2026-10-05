//! Etapa 1: extração de frames candidatos com ffmpeg (decodificação por
//! hardware) e seleção dos mais nítidos, distribuídos uniformemente ao
//! longo da órbita.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::proc::{run_capture, run_streaming};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration: f64,
    pub codec: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameOptions {
    /// Quantos frames entram na reconstrução.
    pub target_frames: u32,
    /// Lado maior máximo dos frames (px).
    pub max_side: u32,
    /// Candidatos extraídos por frame final; o mais nítido de cada janela vence.
    pub oversample: u32,
    pub trim_start: f64,
    /// `None` = até o fim do vídeo.
    pub trim_end: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameInfo {
    pub path: String,
    pub time: f64,
    pub sharpness: f32,
}

pub async fn probe(ffprobe: &Path, video: &Path) -> Result<VideoInfo> {
    let mut cmd = Command::new(ffprobe);
    cmd.args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=width,height,avg_frame_rate,codec_name:format=duration"])
        .args(["-of", "json"])
        .arg(video);
    let out = run_capture(cmd).await?;
    let json: serde_json::Value = serde_json::from_str(&out).context("saída inesperada do ffprobe")?;
    let stream = json["streams"]
        .get(0)
        .context("o arquivo não tem trilha de vídeo")?;
    let fps = stream["avg_frame_rate"]
        .as_str()
        .and_then(|r| {
            let (n, d) = r.split_once('/')?;
            let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
            (d > 0.0).then_some(n / d)
        })
        .unwrap_or(30.0);
    Ok(VideoInfo {
        width: stream["width"].as_u64().unwrap_or(0) as u32,
        height: stream["height"].as_u64().unwrap_or(0) as u32,
        fps,
        duration: json["format"]["duration"]
            .as_str()
            .and_then(|d| d.parse().ok())
            .unwrap_or(0.0),
        codec: stream["codec_name"].as_str().unwrap_or("?").to_string(),
    })
}

/// Extrai candidatos para `raw_dir`. Retorna o fps de amostragem usado.
pub async fn extract(
    ffmpeg: &Path,
    video: &Path,
    info: &VideoInfo,
    opts: &FrameOptions,
    raw_dir: &Path,
    cancel: &CancellationToken,
    mut on_progress: impl FnMut(f32),
    mut on_log: impl FnMut(&str),
) -> Result<f64> {
    let start = opts.trim_start.max(0.0);
    let end = opts
        .trim_end
        .filter(|e| *e > start)
        .unwrap_or(info.duration)
        .min(info.duration);
    let span = end - start;
    if span <= 0.1 {
        bail!("trecho selecionado do vídeo é vazio");
    }

    let wanted = (opts.target_frames * opts.oversample.max(1)) as f64;
    let sample_fps = (wanted / span).min(info.fps).max(0.1);

    std::fs::create_dir_all(raw_dir)?;
    let max = opts.max_side;
    let filter = format!(
        "fps={sample_fps:.6},scale=w='min(iw,{max})':h='min(ih,{max})':force_original_aspect_ratio=decrease:flags=lanczos"
    );

    let mut cmd = Command::new(ffmpeg);
    cmd.args(["-hide_banner", "-nostats", "-loglevel", "error", "-y"])
        .args(["-progress", "pipe:1"])
        // VideoToolbox no macOS; no Windows/Linux o ffmpeg escolhe
        // (D3D11VA, CUDA, VAAPI…) e cai para software se nada servir.
        .args(["-hwaccel", if cfg!(target_os = "macos") { "videotoolbox" } else { "auto" }])
        .args(["-ss", &format!("{start:.3}"), "-t", &format!("{span:.3}")])
        .arg("-i")
        .arg(video)
        .args(["-vf", &filter])
        .args(["-q:v", "2", "-start_number", "0"])
        .arg(raw_dir.join("%05d.jpg"));

    let total_us = span * 1_000_000.0;
    run_streaming(cmd, cancel, |line| {
        if let Some(v) = line.strip_prefix("out_time_us=") {
            if let Ok(us) = v.parse::<f64>() {
                on_progress((us / total_us).clamp(0.0, 1.0) as f32);
            }
        } else if !line.contains('=') {
            on_log(line);
        }
    })
    .await?;
    Ok(sample_fps)
}

/// Variância do Laplaciano numa versão reduzida em tons de cinza: métrica
/// clássica e barata de foco/nitidez.
fn sharpness(path: &Path) -> Result<f32> {
    let img = image::open(path)?.thumbnail(640, 640).to_luma8();
    let (w, h) = img.dimensions();
    if w < 3 || h < 3 {
        return Ok(0.0);
    }
    let px = img.as_raw();
    let at = |x: u32, y: u32| px[(y * w + x) as usize] as f32;
    let (mut sum, mut sum_sq, mut n) = (0f64, 0f64, 0f64);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let lap = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y);
            sum += lap as f64;
            sum_sq += (lap * lap) as f64;
            n += 1.0;
        }
    }
    let mean = sum / n;
    Ok((sum_sq / n - mean * mean) as f32)
}

/// Escolhe o frame mais nítido de cada janela e grava em `images_dir` com
/// numeração sequencial (a ordem importa para o matching circular).
pub fn select(
    raw_dir: &Path,
    images_dir: &Path,
    opts: &FrameOptions,
    sample_fps: f64,
    on_progress: impl Fn(f32) + Sync,
) -> Result<Vec<FrameInfo>> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(raw_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "jpg"))
        .collect();
    candidates.sort();
    if candidates.is_empty() {
        bail!("o ffmpeg não gerou nenhum frame");
    }

    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = candidates.len();
    let scores: Vec<f32> = candidates
        .par_iter()
        .map(|p| {
            let s = sharpness(p).unwrap_or(0.0);
            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if d % 8 == 0 || d == total {
                on_progress(d as f32 / total as f32);
            }
            s
        })
        .collect();

    let target = (opts.target_frames as usize).clamp(1, total);
    if images_dir.exists() {
        std::fs::remove_dir_all(images_dir)?;
    }
    std::fs::create_dir_all(images_dir)?;

    let mut out = Vec::with_capacity(target);
    for w in 0..target {
        let lo = w * total / target;
        let hi = ((w + 1) * total / target).max(lo + 1);
        let best = (lo..hi)
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap();
        let dest = images_dir.join(format!("{:05}.jpg", out.len()));
        std::fs::rename(&candidates[best], &dest)?;
        out.push(FrameInfo {
            path: dest.to_string_lossy().into_owned(),
            time: opts.trim_start + best as f64 / sample_fps,
            sharpness: scores[best],
        });
    }
    std::fs::remove_dir_all(raw_dir)?;
    Ok(out)
}
