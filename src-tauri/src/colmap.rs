//! Etapa 2: poses de câmera com COLMAP 4.x.
//!
//! Otimizações para vídeo orbital:
//! - câmera única pinhole (vídeo gerado não tem distorção de lente);
//! - pares de matching circulares (cada frame casa com os vizinhos, com
//!   "wrap-around" para fechar a órbita) em vez de exaustivo/vocab tree;
//! - calibração do grafo de vistas + mapeador global (ex-GLOMAP);
//! - se o SIFT comum registrar menos de 95% das câmeras, nova tentativa com
//!   SIFT "reforçado" (texturas pobres: cabelo, pele, fundo desfocado);
//! - fallback para o mapeador incremental se ainda assim registrar pouco;
//! - opcional: órbita forçada num plano, para vídeos que giram em nível.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::proc::{run_capture, run_streaming};
use crate::msg;
use crate::msg::Msg;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Features {
    /// SIFT padrão: o mais rápido na CPU.
    Sift,
    /// SIFT com limiar de detecção baixo, agrupamento de escalas e forma
    /// afim. ~2x o custo do padrão; recupera regiões de pouca textura.
    /// Num vídeo real de IA (volta atrás de uma cabeça): 83 → 107/107 câmeras.
    SiftPlus,
    /// ALIKED + LightGlue: registra bem, mas ~10x mais lento que o
    /// SiftPlus na CPU e com erro de reprojeção maior.
    Aliked,
}

impl Features {
    fn label(self) -> &'static str {
        match self {
            Features::Sift => "SIFT",
            Features::SiftPlus => "SIFT reforçado",
            Features::Aliked => "ALIKED + LightGlue",
        }
    }

    fn extractor_args(self) -> &'static [&'static str] {
        match self {
            Features::Sift => &[],
            Features::SiftPlus => &[
                "--SiftExtraction.peak_threshold", "0.002",
                "--SiftExtraction.domain_size_pooling", "1",
                "--SiftExtraction.estimate_affine_shape", "1",
            ],
            Features::Aliked => &["--FeatureExtraction.type", "ALIKED_N16ROT"],
        }
    }

    fn matcher_args(self) -> &'static [&'static str] {
        match self {
            Features::Aliked => &["--FeatureMatching.type", "ALIKED_LIGHTGLUE"],
            _ => &[],
        }
    }
}

/// Abaixo disso o SIFT padrão ganha uma segunda tentativa reforçada.
const RETRY_BELOW: f32 = 0.95;
/// Vizinhos casados na tentativa reforçada. Com 10 o resultado variou entre
/// 97 e 106/107 câmeras de uma execução para outra (pontes fracas na região
/// pobre em textura); com 20 ficou estável em 106–107, por ~10% a mais de tempo.
const RETRY_WINDOW: u32 = 20;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PoseOptions {
    pub features: Features,
    /// "SIMPLE_PINHOLE" ou "PINHOLE".
    pub camera_model: String,
    /// Quantos vizinhos de cada lado são casados.
    pub match_window: u32,
    /// A órbita dá a volta completa (casa o fim com o começo).
    pub closed_loop: bool,
    /// Força todas as câmeras no mesmo plano (vídeo que gira em nível).
    #[serde(default)]
    pub flat_orbit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStats {
    pub registered: u32,
    pub total: u32,
    pub points: u64,
    pub reprojection_error: Option<f64>,
    pub mapper: String,
}

pub struct Paths<'a> {
    pub colmap: &'a Path,
    pub images: &'a Path,
    pub work: &'a Path,
    /// Destino final: `<dataset>/sparse/0`.
    pub model_out: &'a Path,
    /// COLMAP com CUDA. O SIFT reforçado (forma afim/DSP) continua na CPU:
    /// o próprio COLMAP faz esse fallback; o matching vai para a GPU.
    pub use_gpu: bool,
    /// Máscaras `<imagem>.png` (preto = ignorar features do fundo).
    pub masks: Option<PathBuf>,
}

fn gpu_flag(on: bool) -> &'static str {
    if on {
        "1"
    } else {
        "0"
    }
}

fn base(colmap: &Path, sub: &str) -> Command {
    let mut cmd = Command::new(colmap);
    cmd.arg(sub).args(["--log_target", "stderr", "--log_color", "0"]);
    cmd
}

/// Remove o prefixo do glog (`I20261004 16:45:25.417 0x… file.cc:294] `).
fn strip_glog(line: &str) -> &str {
    match line.find("] ") {
        Some(i) if line.starts_with(['I', 'W', 'E', 'F']) => &line[i + 2..],
        _ => line,
    }
}

/// Extrai `[i/N]` de uma linha.
fn bracket_progress(line: &str) -> Option<(u32, u32)> {
    let s = line.find('[')?;
    let e = line[s..].find(']')? + s;
    let (a, b) = line[s + 1..e].split_once('/')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

fn write_pairs(images: &Path, path: &Path, window: u32, closed: bool) -> Result<usize> {
    let mut names: Vec<String> = std::fs::read_dir(images)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".jpg") || n.ends_with(".png"))
        .collect();
    names.sort();
    let n = names.len();
    if n < 3 {
        bail!("poucos frames para reconstruir ({n})");
    }

    // Vizinhos próximos + alguns saltos longos (16, 32, 64) que ajudam a
    // amarrar a geometria global sem o custo do matching exaustivo.
    let mut offsets: Vec<usize> = (1..=window as usize).collect();
    offsets.extend([16, 32, 64].into_iter().filter(|&o| o > window as usize));
    // Na órbita fechada, o salto `o` equivale a `n - o`; não passe da metade.
    let limit = if closed { n / 2 } else { n - 1 };
    offsets.retain(|&o| o <= limit);

    let mut pairs = std::collections::BTreeSet::new();
    for i in 0..n {
        for &o in &offsets {
            let j = if closed {
                (i + o) % n
            } else if i + o < n {
                i + o
            } else {
                continue;
            };
            if i != j {
                pairs.insert((i.min(j), i.max(j)));
            }
        }
    }
    let text: String = pairs
        .iter()
        .map(|&(a, b)| format!("{} {}\n", names[a], names[b]))
        .collect();
    std::fs::write(path, text)?;
    Ok(pairs.len())
}

async fn analyze(colmap: &Path, model: &Path) -> Result<(u32, u64, Option<f64>)> {
    let mut cmd = base(colmap, "model_analyzer");
    cmd.arg("--path").arg(model);
    let out = run_capture(cmd).await?;
    let (mut reg, mut pts, mut err) = (0u32, 0u64, None);
    for line in out.lines().map(strip_glog) {
        if let Some(v) = line.strip_prefix("Registered images:") {
            reg = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("Points:") {
            pts = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("Mean reprojection error:") {
            err = v.trim().trim_end_matches("px").parse().ok();
        }
    }
    Ok((reg, pts, err))
}

/// Entre os modelos gerados (`0/`, `1/`, …) escolhe o que registrou mais imagens.
async fn best_model(colmap: &Path, dir: &Path) -> Result<Option<(PathBuf, u32, u64, Option<f64>)>> {
    let mut best: Option<(PathBuf, u32, u64, Option<f64>)> = None;
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(None);
    };
    for e in entries.flatten() {
        let p = e.path();
        if !p.join("images.bin").exists() {
            continue;
        }
        let (reg, pts, err) = analyze(colmap, &p).await?;
        if best.as_ref().map_or(true, |b| reg > b.1) {
            best = Some((p, reg, pts, err));
        }
    }
    Ok(best)
}

type Best = (PathBuf, u32, u64, Option<f64>);

/// Uma tentativa completa (features → matching → calibração → mapeador
/// global) numa subpasta própria. Progresso local de 0 a 1.
async fn attempt(
    colmap: &Path,
    images: &Path,
    masks: Option<&Path>,
    use_gpu: bool,
    dir: &Path,
    opts: &PoseOptions,
    features: Features,
    cancel: &CancellationToken,
    on_progress: &mut impl FnMut(f32, Msg),
    on_log: &mut impl FnMut(&str),
) -> Result<Option<Best>> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)?;
    let db = dir.join("database.db");

    // 1. Features
    on_progress(0.0, msg!("poses.extracting", features = features));
    let mut cmd = base(colmap, "feature_extractor");
    cmd.arg("--database_path").arg(&db)
        .arg("--image_path").arg(images)
        .args(["--ImageReader.single_camera", "1"])
        .args(["--ImageReader.camera_model", &opts.camera_model])
        .args(["--FeatureExtraction.use_gpu", gpu_flag(use_gpu)])
        .args(features.extractor_args());
    if let Some(m) = masks {
        cmd.arg("--ImageReader.mask_path").arg(m);
    }
    run_streaming(cmd, cancel, |l| {
        let l = strip_glog(l);
        if l.starts_with("Processed file") {
            if let Some((i, n)) = bracket_progress(l) {
                on_progress(0.4 * i as f32 / n as f32, msg!("poses.extractingN", features = features, i = i, n = n));
            }
        } else if !l.starts_with(' ') {
            on_log(l);
        }
    })
    .await
    .context("falha na extração de features")?;

    // 2. Matching circular
    let pairs_path = dir.join("pairs.txt");
    let n_pairs = write_pairs(images, &pairs_path, opts.match_window, opts.closed_loop)?;
    on_progress(0.4, msg!("poses.matching", pairs = n_pairs));
    let mut cmd = base(colmap, "matches_importer");
    cmd.arg("--database_path").arg(&db)
        .arg("--match_list_path").arg(&pairs_path)
        .args(["--match_type", "pairs"])
        .args(["--FeatureMatching.use_gpu", gpu_flag(use_gpu)])
        .args(features.matcher_args());
    run_streaming(cmd, cancel, |l| {
        let l = strip_glog(l);
        if l.starts_with("Processing block") {
            if let Some((i, n)) = bracket_progress(l) {
                let f = (i - 1) as f32 / n as f32;
                on_progress(0.4 + 0.45 * f, msg!("poses.matchingBlock", pairs = n_pairs, i = i, n = n));
            }
        }
        on_log(l);
    })
    .await
    .context("falha no matching")?;

    // 3. Calibração do grafo de vistas (melhora o foco inicial do mapeador global)
    on_progress(0.85, msg!("poses.calibrating"));
    let mut cmd = base(colmap, "view_graph_calibrator");
    cmd.arg("--database_path").arg(&db);
    if let Err(e) = run_streaming(cmd, cancel, |l| on_log(strip_glog(l))).await {
        if crate::proc::is_cancelled(&e) {
            return Err(e);
        }
        on_log(&format!("calibração falhou, seguindo sem ela: {e:#}"));
    }

    // 4. Mapeador global
    on_progress(0.9, msg!("poses.globalMapper"));
    let global_dir = dir.join("sparse_global");
    std::fs::create_dir_all(&global_dir)?;
    let mut cmd = base(colmap, "global_mapper");
    cmd.arg("--database_path").arg(&db)
        .arg("--image_path").arg(images)
        .arg("--output_path").arg(&global_dir)
        .args(["--GlobalMapper.gp_use_gpu", gpu_flag(use_gpu)])
        .args(["--GlobalMapper.ba_ceres_use_gpu", gpu_flag(use_gpu)]);
    if let Err(e) = run_streaming(cmd, cancel, |l| on_log(strip_glog(l))).await {
        if crate::proc::is_cancelled(&e) {
            return Err(e);
        }
        on_log(&format!("mapeador global falhou: {e:#}"));
    }
    best_model(colmap, &global_dir).await
}

/// Achata a órbita (ver `scene::flatten_orbit`) e reajusta só os pontos 3D
/// com as poses e a intrínseca fixas, para que continuem coerentes com as
/// novas câmeras. Na falha, `model` fica intocado. Retorna o desvio corrigido
/// (fração do raio) e o novo erro de reprojeção.
async fn flatten(
    colmap: &Path,
    work: &Path,
    model: &Path,
    cancel: &CancellationToken,
    on_log: &mut impl FnMut(&str),
) -> Result<(f64, Option<f64>)> {
    let flat = work.join("flat");
    let adjusted = work.join("flat_ba");
    for d in [&flat, &adjusted] {
        if d.exists() {
            std::fs::remove_dir_all(d)?;
        }
    }
    let dev = crate::scene::flatten_orbit(model, &flat)?;
    std::fs::create_dir_all(&adjusted)?;
    let mut cmd = base(colmap, "bundle_adjuster");
    cmd.arg("--input_path").arg(&flat)
        .arg("--output_path").arg(&adjusted)
        .args(["--BundleAdjustment.refine_rig_from_world", "0"])
        .args(["--BundleAdjustment.refine_focal_length", "0"])
        .args(["--BundleAdjustment.refine_extra_params", "0"]);
    run_streaming(cmd, cancel, |l| on_log(strip_glog(l))).await.context("falha no bundle adjustment")?;
    let (_, _, err) = analyze(colmap, &adjusted).await?;
    std::fs::remove_dir_all(model)?;
    std::fs::rename(&adjusted, model)?;
    Ok((dev, err))
}

pub async fn run(
    paths: Paths<'_>,
    opts: &PoseOptions,
    cancel: &CancellationToken,
    mut on_progress: impl FnMut(f32, Msg),
    mut on_log: impl FnMut(&str),
) -> Result<ModelStats> {
    let Paths { colmap, images, work, model_out, use_gpu, masks } = paths;
    if work.exists() {
        std::fs::remove_dir_all(work)?;
    }
    std::fs::create_dir_all(work)?;
    let total = std::fs::read_dir(images)?.count() as u32;
    let ratio = |b: &Option<Best>| b.as_ref().map_or(0.0, |b| b.1 as f32 / total.max(1) as f32);

    let mut features = opts.features;
    let masks = masks.as_deref();
    let mut best = attempt(colmap, images, masks, use_gpu, &work.join("attempt1"), opts, features, cancel, &mut on_progress, &mut on_log).await?;
    let mut mapper = "global";
    let mut db_dir = work.join("attempt1");

    // Segunda tentativa: SIFT reforçado quando o padrão deixou buracos na órbita.
    if features == Features::Sift && ratio(&best) < RETRY_BELOW {
        let got = best.as_ref().map_or(0, |b| b.1);
        let msg = format!("{got}/{total} câmeras; repetindo com SIFT reforçado");
        on_log(&msg);
        let dir = work.join("attempt2");
        let retry_opts = PoseOptions { match_window: opts.match_window.max(RETRY_WINDOW), ..opts.clone() };
        let retry = attempt(
            colmap,
            images,
            masks,
            use_gpu,
            &dir,
            &retry_opts,
            Features::SiftPlus,
            cancel,
            &mut |f, m| on_progress(f, msg!("poses.retry", step = m)),
            &mut on_log,
        )
        .await?;
        if ratio(&retry) > ratio(&best) {
            best = retry;
            features = Features::SiftPlus;
            db_dir = dir;
        }
    }

    // Fallback: incremental, mais lento porém mais tolerante.
    if ratio(&best) < 0.6 {
        on_progress(0.95, msg!("poses.incremental"));
        let inc_dir = work.join("sparse_incremental");
        std::fs::create_dir_all(&inc_dir)?;
        let mut cmd = base(colmap, "mapper");
        cmd.arg("--database_path").arg(db_dir.join("database.db"))
            .arg("--image_path").arg(images)
            .arg("--output_path").arg(&inc_dir)
            .args(["--Mapper.ba_use_gpu", gpu_flag(use_gpu)]);
        run_streaming(cmd, cancel, |l| {
            let l = strip_glog(l);
            if let Some(rest) = l.strip_prefix("Registering image #") {
                on_progress(0.95, msg!("poses.registering", image = rest));
            }
            on_log(l);
        })
        .await
        .context("falha no mapeador incremental")?;
        if let Some(inc) = best_model(colmap, &inc_dir).await? {
            if best.as_ref().map_or(true, |b| inc.1 > b.1) {
                best = Some(inc);
                mapper = "incremental";
            }
        }
    }

    let Some((model, registered, points, err)) = best else {
        bail!(
            "o COLMAP não conseguiu reconstruir a cena. Tente SIFT reforçado ou ALIKED, \
             mais frames, ou um trecho mais estável do vídeo."
        );
    };

    if model_out.exists() {
        std::fs::remove_dir_all(model_out)?;
    }
    std::fs::create_dir_all(model_out.parent().unwrap())?;
    std::fs::rename(&model, model_out)?;

    let mut label = format!("{mapper} · {}", features.label());
    let mut err = err;
    if opts.flat_orbit {
        on_progress(0.98, msg!("poses.flattening"));
        match flatten(colmap, work, model_out, cancel, &mut on_log).await {
            Ok((dev, e)) => {
                on_log(&format!("órbita plana: maior desvio de altura corrigido {:.2}% do raio", dev * 100.0));
                label.push_str(" · órbita plana");
                err = e.or(err);
            }
            Err(e) if crate::proc::is_cancelled(&e) => return Err(e),
            Err(e) => on_log(&format!("órbita plana não aplicada: {e:#}")),
        }
    }

    on_progress(1.0, msg!("poses.ready", features = features));
    Ok(ModelStats {
        registered,
        total,
        points,
        reprojection_error: err,
        mapper: label,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_glog_progress() {
        let l = "I20261004 16:45:25.843585 0x170243000 feature_extraction.cc:270] Processed file [12/120]";
        assert_eq!(bracket_progress(strip_glog(l)), Some((12, 120)));
        assert_eq!(strip_glog(l), "Processed file [12/120]");
    }

    #[test]
    fn circular_pairs_wrap_around() {
        let dir = std::env::temp_dir().join("splatforge_pairs_test");
        let imgs = dir.join("images");
        std::fs::create_dir_all(&imgs).unwrap();
        for i in 0..10 {
            std::fs::write(imgs.join(format!("{i:05}.jpg")), b"").unwrap();
        }
        let out = dir.join("pairs.txt");
        let closed = write_pairs(&imgs, &out, 2, true).unwrap();
        let open = write_pairs(&imgs, &out, 2, false).unwrap();
        // fechado: 10 frames × 2 vizinhos = 20 pares; aberto perde os 3 que dão a volta.
        assert_eq!(closed, 20);
        assert_eq!(open, 17);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
