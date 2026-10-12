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
    /// Penalizava splats grandes/alongados no Brush 0.3. O Brush fixado
    /// (scripts/build-brush-web.mjs) não tem mais essa perda: é ignorada,
    /// como no web.
    #[serde(default)]
    pub scale_loss_weight: Option<f32>,
    /// Poda por contribuição negativa e regularização do DeG (web/brush/deg.patch);
    /// ver `TrainOptions` em src/lib/api.ts. Omitido ou 0 = desligado.
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

/// Fundo do treino com máscara: cor uniforme sorteada a cada passo
/// (0,5 ± 0,5). No preto ± 0,1 do Brush, camadas escuras e semitransparentes
/// fora da silhueta quase não custam nada e viram uma silhueta fantasma em
/// volta do sujeito; com o fundo variando, qualquer transparência indevida
/// pesa na perda. Mesmos valores de `MASKED_BACKGROUND` em web/src/brush.ts.
const MASKED_BACKGROUND: ([f32; 3], f32) = ([0.5, 0.5, 0.5], 0.5);

/// Argumentos de treino do `brush-cli` (sem o dataset e a saída).
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

/// Remove o prefixo `[2026-10-04T19:51:15Z INFO  modulo] `.
fn strip_env_logger(line: &str) -> &str {
    if line.starts_with('[') {
        if let Some(i) = line.find("] ") {
            return &line[i + 2..];
        }
    }
    line
}

/// `masked`: as imagens do dataset têm o alpha das máscaras (dataset_alpha).
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
    if opts.scale_loss_weight.is_some() {
        on_log("Opção \"suavizar\" (scale-loss-weight) não existe nesta versão do Brush; ignorada.");
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
    fn masked_training_uses_random_background() {
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
        for name in ["splat_02500.ply", "splat_10000.ply", "splat_05000.ply", "outro.ply"] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
        let iters: Vec<u32> = list_snapshots(&dir).into_iter().map(|(i, _)| i).collect();
        assert_eq!(iters, vec![2500, 5000, 10000]);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
