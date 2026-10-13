//! Step 2: camera poses with COLMAP 4.x.
//!
//! Optimizations for orbit video:
//! - single pinhole camera (generated video has no lens distortion);
//! - circular matching pairs (each frame matches its neighbours, with
//!   wrap-around to close the orbit) instead of exhaustive/vocab tree;
//! - view graph calibration + global mapper (formerly GLOMAP);
//! - if plain SIFT registers under 95% of the cameras, retry with
//!   "enhanced" SIFT (poor textures: hair, skin, blurry background);
//! - fall back to the incremental mapper if that still registers too few;
//! - optional: orbit forced onto a plane, for videos that rotate level.

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
    /// Default SIFT: the fastest on CPU.
    Sift,
    /// SIFT with a low detection threshold, domain-size pooling and affine
    /// shape. ~2x the cost of the default; recovers low-texture regions.
    /// On a real AI video (orbit behind a head): 83 → 107/107 cameras.
    SiftPlus,
    /// ALIKED + LightGlue: registers well, but ~10x slower than SiftPlus on
    /// CPU and with higher reprojection error.
    Aliked,
}

impl Features {
    fn label(self) -> &'static str {
        match self {
            Features::Sift => "SIFT",
            Features::SiftPlus => "enhanced SIFT",
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

/// Below this ratio, default SIFT gets a second, enhanced attempt.
const RETRY_BELOW: f32 = 0.95;
/// Neighbours matched in the enhanced attempt. With 10 the result varied
/// between 97 and 106/107 cameras from run to run (weak bridges in the
/// low-texture region); with 20 it held steady at 106–107, for ~10% more time.
const RETRY_WINDOW: u32 = 20;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PoseOptions {
    pub features: Features,
    /// "SIMPLE_PINHOLE" or "PINHOLE".
    pub camera_model: String,
    /// How many neighbours on each side are matched.
    pub match_window: u32,
    /// The orbit is a full loop (matches the end with the start).
    pub closed_loop: bool,
    /// Forces all cameras onto the same plane (video that rotates level).
    #[serde(default)]
    pub flat_orbit: bool,
    /// Frames without a registered pose get one inferred from their
    /// neighbours (`scene::fill_missing`, applied by the pipeline).
    #[serde(default)]
    pub fill_missing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStats {
    pub registered: u32,
    pub total: u32,
    pub points: u64,
    pub reprojection_error: Option<f64>,
    pub mapper: String,
    /// Poses inferred from neighbours (frames COLMAP did not register).
    #[serde(default)]
    pub inferred: u32,
    /// Of those, the ones refined against the 3D points.
    #[serde(default)]
    pub refined: u32,
}

pub struct Paths<'a> {
    pub colmap: &'a Path,
    pub images: &'a Path,
    pub work: &'a Path,
    /// Final destination: `<dataset>/sparse/0`.
    pub model_out: &'a Path,
    /// COLMAP with CUDA. Enhanced SIFT (affine shape/DSP) stays on the CPU:
    /// COLMAP does that fallback itself; matching goes to the GPU.
    pub use_gpu: bool,
    /// `<image>.png` masks (black = ignore background features).
    pub masks: Option<PathBuf>,
    /// Timestamp of each frame by name, for `PoseOptions::fill_missing`.
    pub times: &'a std::collections::HashMap<String, f64>,
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

/// Strips the glog prefix (`I20261004 16:45:25.417 0x… file.cc:294] `).
fn strip_glog(line: &str) -> &str {
    match line.find("] ") {
        Some(i) if line.starts_with(['I', 'W', 'E', 'F']) => &line[i + 2..],
        _ => line,
    }
}

/// Extracts `[i/N]` from a line.
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
        bail!("too few frames to reconstruct ({n})");
    }

    // Near neighbours + a few long jumps (16, 32, 64) that help tie down the
    // global geometry without the cost of exhaustive matching.
    let mut offsets: Vec<usize> = (1..=window as usize).collect();
    offsets.extend([16, 32, 64].into_iter().filter(|&o| o > window as usize));
    // On a closed orbit, jump `o` is equivalent to `n - o`; don't go past half.
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

/// Among the generated models (`0/`, `1/`, …), picks the one that registered the most images.
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

/// One full attempt (features → matching → calibration → global mapper) in
/// its own subfolder. Local progress from 0 to 1.
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
    .context("feature extraction failed")?;

    // 2. Circular matching
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
    .context("matching failed")?;

    // 3. View graph calibration (improves the global mapper's initial focal length)
    on_progress(0.85, msg!("poses.calibrating"));
    let mut cmd = base(colmap, "view_graph_calibrator");
    cmd.arg("--database_path").arg(&db);
    if let Err(e) = run_streaming(cmd, cancel, |l| on_log(strip_glog(l))).await {
        if crate::proc::is_cancelled(&e) {
            return Err(e);
        }
        on_log(&format!("calibration failed, continuing without it: {e:#}"));
    }

    // 4. Global mapper
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
        on_log(&format!("global mapper failed: {e:#}"));
    }
    best_model(colmap, &global_dir).await
}

/// Flattens the orbit (see `scene::flatten_orbit`) and re-adjusts only the 3D
/// points with poses and intrinsics fixed, so they stay consistent with the
/// new cameras. On failure, `model` is left untouched. Returns the corrected
/// deviation (fraction of the radius) and the new reprojection error.
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
    run_streaming(cmd, cancel, |l| on_log(strip_glog(l))).await.context("bundle adjustment failed")?;
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
    let Paths { colmap, images, work, model_out, use_gpu, masks, times } = paths;
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

    // Second attempt: enhanced SIFT when the default left gaps in the orbit.
    if features == Features::Sift && ratio(&best) < RETRY_BELOW {
        let got = best.as_ref().map_or(0, |b| b.1);
        let msg = format!("{got}/{total} cameras; retrying with enhanced SIFT");
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

    // Fallback: incremental, slower but more tolerant.
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
        .context("incremental mapper failed")?;
        if let Some(inc) = best_model(colmap, &inc_dir).await? {
            if best.as_ref().map_or(true, |b| inc.1 > b.1) {
                best = Some(inc);
                mapper = "incremental";
            }
        }
    }

    let Some((model, registered, points, err)) = best else {
        bail!(
            "COLMAP could not reconstruct the scene. Try enhanced SIFT or ALIKED, \
             more frames, or a steadier part of the video."
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
                on_log(&format!("flat orbit: largest height deviation corrected {:.2}% of the radius", dev * 100.0));
                label.push_str(" · flat orbit");
                err = e.or(err);
            }
            Err(e) if crate::proc::is_cancelled(&e) => return Err(e),
            Err(e) => on_log(&format!("flat orbit not applied: {e:#}")),
        }
    }

    let (mut inferred, mut refined) = (0, 0);
    if opts.fill_missing {
        // Correspondences for the frames left out, from the database of the
        // attempt that produced the model; without them, poses are only inferred.
        let corr = correspondences(&db_dir.join("database.db"), model_out).unwrap_or_else(|e| {
            on_log(&format!("no correspondences to refine the inferred poses: {e:#}"));
            Default::default()
        });
        let fill = crate::scene::FillOptions { flat: opts.flat_orbit, correspondences: &corr };
        match crate::scene::fill_missing(model_out, images, times, &fill) {
            Ok(f) => {
                if !f.inferred.is_empty() {
                    on_log(&format!("poses inferred from neighbours: {}", f.inferred.join(", ")));
                }
                for (name, inliers) in &f.refined {
                    on_log(&format!("pose of {name} refined against {inliers} 3D points"));
                }
                (inferred, refined) = (f.inferred.len() as u32, f.refined.len() as u32);
            }
            Err(e) => on_log(&format!("poses not inferred: {e:#}")),
        }
    }

    on_progress(1.0, msg!("poses.ready", features = features));
    Ok(ModelStats {
        registered,
        total,
        points,
        reprojection_error: err,
        mapper: label,
        inferred,
        refined,
    })
}

/// 2D → 3D correspondences for each database image the model did not
/// register: its geometric matches (two_view_geometries) with registered
/// images, carried over to the 3D points those images see. A 2D point matched
/// to several 3D points keeps the most voted one.
pub fn correspondences(db: &Path, model: &Path) -> Result<std::collections::HashMap<String, crate::scene::Correspondences>> {
    use rusqlite::{Connection, OpenFlags};
    use std::collections::HashMap;
    // pair_id = id1 · 2³¹−1 + id2 (id1 < id2), as in COLMAP.
    const MAX_IMAGES: i64 = 2_147_483_647;

    let (tracks, points) = crate::scene::read_tracks(model)?;
    let conn = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut names = HashMap::new();
    {
        let mut st = conn.prepare("SELECT image_id, name FROM images")?;
        for row in st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
            let (id, name) = row?;
            names.insert(id, name);
        }
    }
    let keypoints = |id: i64| -> Result<Vec<[f64; 2]>> {
        let (rows, cols, data): (i64, i64, Vec<u8>) =
            conn.query_row("SELECT rows, cols, data FROM keypoints WHERE image_id = ?1", [id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        let f = |i: usize| f32::from_le_bytes(data[4 * i..4 * i + 4].try_into().unwrap()) as f64;
        let cols = cols as usize;
        anyhow::ensure!(data.len() >= rows as usize * cols * 4 && cols >= 2, "malformed keypoints for image {id}");
        Ok((0..rows as usize).map(|k| [f(k * cols), f(k * cols + 1)]).collect())
    };

    // 2D point k of a registered image is keypoint k in the database: only
    // valid when the counts match.
    let mut aligned = HashMap::new();
    let mut votes: HashMap<i64, HashMap<u32, HashMap<i64, u32>>> = HashMap::new();
    let mut st = conn.prepare("SELECT pair_id, rows, data FROM two_view_geometries WHERE rows > 0")?;
    for row in st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, Vec<u8>>(2)?)))? {
        let (pair, rows, data) = row?;
        let (id1, id2) = (pair / MAX_IMAGES, pair % MAX_IMAGES);
        let (Some(n1), Some(n2)) = (names.get(&id1), names.get(&id2)) else { continue };
        // Exactly one of the two is outside the model.
        let (missing, reg, missing_first) = match (tracks.contains_key(n1), tracks.contains_key(n2)) {
            (false, true) => (id1, n2, true),
            (true, false) => (id2, n1, false),
            _ => continue,
        };
        let reg_id = if missing_first { id2 } else { id1 };
        let ids = &tracks[reg];
        if !*aligned.entry(reg_id).or_insert_with(|| keypoints(reg_id).map_or(false, |k| k.len() == ids.len())) {
            continue;
        }
        let v = votes.entry(missing).or_default();
        let u = |i: usize| u32::from_le_bytes(data[4 * i..4 * i + 4].try_into().unwrap());
        for m in 0..(rows as usize).min(data.len() / 8) {
            let (a, b) = (u(2 * m), u(2 * m + 1));
            let (km, kr) = if missing_first { (a, b) } else { (b, a) };
            if let Some(&pid) = ids.get(kr as usize).filter(|&&p| p >= 0) {
                *v.entry(km).or_default().entry(pid).or_default() += 1;
            }
        }
    }

    let mut out = HashMap::new();
    for (id, v) in votes {
        let kps = keypoints(id)?;
        let corr: crate::scene::Correspondences = v
            .into_iter()
            .filter_map(|(k, pids)| {
                let pid = pids.into_iter().max_by_key(|&(_, n)| n)?.0;
                Some((*kps.get(k as usize)?, *points.get(&(pid as u64))?))
            })
            .collect();
        out.insert(names[&id].clone(), corr);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a real project (poses computed by the app), drops cameras from the
    /// model and compares the inferred and refined poses with COLMAP's:
    ///   SPLATFORGE_TEST_PROJECT=<project> cargo test refine_real -- --ignored --nocapture
    #[test]
    #[ignore]
    fn refine_real_project() {
        let project = PathBuf::from(std::env::var("SPLATFORGE_TEST_PROJECT").expect("SPLATFORGE_TEST_PROJECT"));
        let (model, images) = (project.join("dataset/sparse/0"), project.join("dataset/images"));
        let db = project.join("colmap/attempt1/database.db");
        let truth = crate::scene::read_poses(&model).unwrap();
        let mut names: Vec<&String> = truth.keys().collect();
        names.sort();
        // A burst in the middle, isolated losses and both ends.
        let lost: Vec<String> = [0, 20, 21, 22, 23, 24, 50, 70, 71, names.len() - 1].iter().map(|&i| names[i].clone()).collect();

        let mut results = Vec::new();
        for refine in [false, true] {
            let dir = std::env::temp_dir().join(format!("splatforge_refine_real_{refine}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            for f in ["cameras.bin", "points3D.bin", "frames.bin", "rigs.bin"] {
                let _ = std::fs::copy(model.join(f), dir.join(f));
            }
            crate::scene::drop_images(&model.join("images.bin"), &dir.join("images.bin"), &lost).unwrap();
            let corr = if refine { correspondences(&db, &dir).unwrap() } else { Default::default() };
            let times = Default::default();
            let opts = crate::scene::FillOptions { flat: false, correspondences: &corr };
            let filled = crate::scene::fill_missing(&dir, &images, &times, &opts).unwrap();
            assert_eq!(filled.inferred.len(), lost.len());
            let radius = crate::scene::read(&model).unwrap().radius;
            let got = crate::scene::read_poses(&dir).unwrap();
            let errs: Vec<(f64, f64)> = lost
                .iter()
                .map(|n| crate::scene::pose_error(&got[n], &truth[n], radius))
                .collect();
            for (n, (dc, da)) in lost.iter().zip(&errs) {
                let k = corr.get(n).map_or(0, |c| c.len());
                println!("{} {n}: center {:.3}% of radius, rotation {:.3}°, {k} correspondences", if refine { "refined" } else { "inferred" }, dc * 100.0, da);
            }
            println!("refined: {:?}", filled.refined);
            results.push(errs);
            let _ = std::fs::remove_dir_all(&dir);
        }
        let mean = |e: &[(f64, f64)]| e.iter().map(|x| x.0).sum::<f64>() / e.len() as f64;
        println!("mean: inferred {:.3}%, refined {:.3}%", mean(&results[0]) * 100.0, mean(&results[1]) * 100.0);
        assert!(mean(&results[1]) <= mean(&results[0]));
    }

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
        // closed: 10 frames × 2 neighbours = 20 pairs; open loses the 3 that wrap around.
        assert_eq!(closed, 20);
        assert_eq!(open, 17);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
