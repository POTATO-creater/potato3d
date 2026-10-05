//! Etapa 3: treino do Gaussian Splatting com o Brush (Rust + wgpu/Metal).
//!
//! O Brush não imprime progresso fora de um TTY, então ligamos o log
//! `info` dos módulos de processo e lemos as linhas de refinamento
//! ("Refine iter N, X splats.") e de avaliação ("Eval iter N: PSNR …").

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
    /// 0–3. Vídeo de IA tem iluminação incoerente: graus baixos evitam decorar ruído.
    pub sh_degree: u32,
    pub max_splats: u32,
    pub max_resolution: u32,
    /// Exporta um .ply intermediário a cada N passos (visualização ao vivo).
    pub export_every: u32,
    /// Separa 1 a cada N imagens para medir PSNR (0 = desliga).
    pub eval_split_every: u32,
    /// Peso da perda SSIM (0 = desliga). No Metal ela triplica o custo do
    /// passo; no mesmo tempo, sem SSIM e com mais passos deu PSNR maior
    /// (32,6 dB em 15k passos vs 31,8 dB em 7k, ambos ~9 min).
    #[serde(default = "default_ssim")]
    pub ssim_weight: f32,
    /// Penaliza splats grandes/alongados. 1e-6 tirou o "azul" e as agulhas
    /// do topo não filmado de uma cabeça, mas custou 1,3 dB de PSNR.
    /// `None` = padrão do Brush (1e-8).
    #[serde(default)]
    pub scale_loss_weight: Option<f32>,
}

fn default_ssim() -> f32 {
    0.2
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

/// Remove o prefixo `[2026-10-04T19:51:15Z INFO  modulo] `.
fn strip_env_logger(line: &str) -> &str {
    if line.starts_with('[') {
        if let Some(i) = line.find("] ") {
            return &line[i + 2..];
        }
    }
    line
}

pub async fn run(
    brush: &Path,
    dataset: &Path,
    out_dir: &Path,
    opts: &TrainOptions,
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
        .args(["--total-steps", &opts.total_steps.to_string()])
        .args(["--sh-degree", &opts.sh_degree.min(3).to_string()])
        .args(["--max-splats", &opts.max_splats.to_string()])
        .args(["--max-resolution", &opts.max_resolution.to_string()])
        .args(["--ssim-weight", &opts.ssim_weight.max(0.0).to_string()])
        .args(["--export-every", &export_every.to_string()])
        .arg("--export-path")
        .arg(out_dir)
        .args(["--export-name", "splat_{iter}.ply"]);
    if let Some(w) = opts.scale_loss_weight {
        cmd.args(["--scale-loss-weight", &format!("{w:e}")]);
    }
    if opts.eval_split_every > 1 {
        cmd.args(["--eval-split-every", &opts.eval_split_every.to_string()])
            .args(["--eval-every", &export_every.to_string()]);
    }

    // O Brush preenche `{iter}` com zeros conforme o nº de dígitos do total
    // (splat_02500.ply com 20000 passos), então varremos a pasta em vez de
    // montar o nome.
    let mut last_reported = 0u32;
    let mut check_snapshots = |on_event: &mut dyn FnMut(TrainEvent)| {
        let snaps = list_snapshots(out_dir);
        let Some((iter, path)) = snaps.last().cloned() else { return };
        if iter > last_reported {
            last_reported = iter;
            on_event(TrainEvent::Snapshot { iter, path: path.to_string_lossy().into_owned() });
        }
        // Cada .ply pode ter centenas de MB: guarda só os dois últimos
        // (o penúltimo ainda pode estar sendo lido pelo visualizador).
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
    .context("falha no treino do Brush")?;

    check_snapshots(&mut on_event);

    // O último export é o resultado.
    list_snapshots(out_dir)
        .pop()
        .map(|(_, p)| p)
        .context("o Brush terminou sem exportar nenhum .ply")
}

/// Snapshots `splat_<iter>.ply` em ordem crescente de iteração.
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
    fn finds_zero_padded_snapshots() {
        let dir = std::env::temp_dir().join("splatforge_snap_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["splat_02500.ply", "splat_10000.ply", "splat_05000.ply", "outro.ply"] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
        let iters: Vec<u32> = list_snapshots(&dir).into_iter().map(|(i, _)| i).collect();
        assert_eq!(iters, vec![2500, 5000, 10000]);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
