//! Orchestrates the stages (frames → masks → poses → training), persists the
//! project state on disk and sends progress events to the UI over a Tauri
//! `Channel`.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::brush::{self, TrainEvent, TrainOptions};
use crate::colmap::{self, ModelStats, PoseOptions};
use crate::export::{self, ExportOptions, ExportStats};
use crate::frames::{self, FrameInfo, FrameOptions, VideoInfo};
use crate::masks::{self, MaskMethod, MaskOptions, MaskStats};
use crate::msg;
use crate::msg::Msg;
use crate::proc::is_cancelled;
use crate::tools::{install_hint, Tools};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    Frames,
    Masks,
    Poses,
    Train,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobConfig {
    /// The video, or the images folder when `images` is set.
    pub video_path: String,
    /// Already-extracted frames (in orbit order) instead of the video.
    #[serde(default)]
    pub images: Option<Vec<String>>,
    pub project_dir: String,
    pub from_stage: Stage,
    /// Last stage to run (inclusive); absent = through training.
    #[serde(default)]
    pub to_stage: Option<Stage>,
    pub frames: FrameOptions,
    #[serde(default)]
    pub masks: MaskOptions,
    pub poses: PoseOptions,
    pub train: TrainOptions,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum JobEvent {
    StageStarted { stage: Stage },
    Progress { stage: Stage, fraction: f32, message: Msg },
    Log { stage: Stage, line: String },
    FramesReady { frames: Vec<FrameInfo> },
    MasksReady { stats: MaskStats, dir: String },
    PosesReady { stats: ModelStats },
    /// Skipped stage (disabled, or a previous result was reused).
    StageSkipped { stage: Stage, reason: Msg },
    Train { event: TrainEvent },
    StageDone { stage: Stage, seconds: f64 },
    /// `ply` is absent when the job stopped before training.
    Finished { ply: Option<String>, seconds: f64 },
    Failed { stage: Option<Stage>, error: String },
    Cancelled,
}

/// Event sink (Tauri Channel in the UI, stdout in the CLI).
pub type Emit = Arc<dyn Fn(JobEvent) + Send + Sync>;

/// What already exists on disk for a project (allows resuming from any stage).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectState {
    pub video_path: Option<String>,
    pub frames: Vec<FrameInfo>,
    #[serde(default)]
    pub masks: Option<MaskStats>,
    #[serde(default)]
    pub masks_dir: Option<String>,
    pub poses: Option<ModelStats>,
    pub final_ply: Option<String>,
    pub last_psnr: Option<f64>,
    /// Last optimized export (cleared when the splat is retrained).
    #[serde(default)]
    pub last_export: Option<ExportStats>,
}

pub struct Layout {
    pub root: PathBuf,
}

impl Layout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    pub(crate) fn state_file(&self) -> PathBuf {
        self.root.join("project.json")
    }
    fn raw_frames(&self) -> PathBuf {
        self.root.join("frames_raw")
    }
    /// Folder passed to Brush: only `images/` and `sparse/0/` (it scans
    /// everything inside, .ply files included, so the output lives outside).
    pub fn dataset(&self) -> PathBuf {
        self.root.join("dataset")
    }
    pub fn images(&self) -> PathBuf {
        self.dataset().join("images")
    }
    /// Manually rejected frames: outside `dataset/` so COLMAP, masking and
    /// Brush don't see them.
    pub(crate) fn rejected(&self) -> PathBuf {
        self.root.join("rejected")
    }
    pub fn model(&self) -> PathBuf {
        self.dataset().join("sparse").join("0")
    }
    fn colmap_work(&self) -> PathBuf {
        self.root.join("colmap")
    }
    /// Outside `dataset/`: Brush would automatically treat `dataset/masks` as
    /// "ignore" masks.
    pub fn masks(&self) -> PathBuf {
        self.root.join("masks")
    }
    /// Outside `colmap/`, which the poses stage recreates from scratch.
    fn colmap_masks(&self) -> PathBuf {
        self.root.join("colmap_masks")
    }
    /// Training dataset with a transparent background (RGBA PNG).
    fn alpha_dataset(&self) -> PathBuf {
        self.root.join("dataset_alpha")
    }
    fn train_out(&self) -> PathBuf {
        self.root.join("training")
    }
    pub(crate) fn final_ply(&self) -> PathBuf {
        self.root.join("splat.ply")
    }
    /// Optimized exports (.sog/.ply), outside `dataset/`.
    pub(crate) fn exports(&self) -> PathBuf {
        self.root.join("export")
    }

    pub fn load(&self) -> ProjectState {
        std::fs::read(self.state_file())
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    pub(crate) fn save(&self, state: &ProjectState) -> Result<()> {
        std::fs::write(self.state_file(), serde_json::to_vec_pretty(state)?)?;
        Ok(())
    }
}

/// Sends progress events at a limited rate (the UI needs no more than ~15
/// updates per second).
struct Reporter<'a> {
    emit: &'a Emit,
    stage: Stage,
    last: std::sync::Mutex<Instant>,
}

impl<'a> Reporter<'a> {
    fn new(emit: &'a Emit, stage: Stage) -> Self {
        emit(JobEvent::StageStarted { stage });
        Self { emit, stage, last: (Instant::now() - Duration::from_secs(1)).into() }
    }
    fn progress(&self, fraction: f32, message: Msg) {
        {
            let mut last = self.last.lock().unwrap();
            if fraction < 1.0 && last.elapsed() < Duration::from_millis(66) {
                return;
            }
            *last = Instant::now();
        }
        (self.emit)(JobEvent::Progress {
            stage: self.stage,
            fraction,
            message,
        });
    }
    fn log(&self, line: &str) {
        (self.emit)(JobEvent::Log { stage: self.stage, line: line.to_string() });
    }
}

pub async fn run(cfg: JobConfig, tools: Tools, emit: Emit, cancel: CancellationToken) {
    let started = Instant::now();
    let mut current = None;
    let result = run_inner(&cfg, &tools, &emit, &cancel, &mut current).await;
    let event = match result {
        Ok(ply) => JobEvent::Finished { ply, seconds: started.elapsed().as_secs_f64() },
        Err(e) if is_cancelled(&e) => JobEvent::Cancelled,
        Err(e) => JobEvent::Failed { stage: current, error: format!("{e:#}") },
    };
    emit(event);
}

async fn run_inner(
    cfg: &JobConfig,
    tools: &Tools,
    emit: &Emit,
    cancel: &CancellationToken,
    current: &mut Option<Stage>,
) -> Result<Option<String>> {
    let layout = Layout::new(&cfg.project_dir);
    let to = cfg.to_stage.unwrap_or(Stage::Train).max(cfg.from_stage);
    std::fs::create_dir_all(&layout.root).context("could not create the project folder")?;
    let mut state = layout.load();
    let video = PathBuf::from(&cfg.video_path);

    // ---------- 1. Frames ----------
    if cfg.from_stage <= Stage::Frames {
        *current = Some(Stage::Frames);
        let t0 = Instant::now();
        let rep = Reporter::new(emit, Stage::Frames);
        let picked = match cfg.images.as_deref() {
            Some(sources) if !sources.is_empty() => {
                rep.log(&format!("{} images from {}", sources.len(), cfg.video_path));
                let sources: Vec<PathBuf> = sources.iter().map(PathBuf::from).collect();
                let (images, opts, progress_emit) = (layout.images(), cfg.frames.clone(), emit.clone());
                tokio::task::spawn_blocking(move || {
                    let last = std::sync::Mutex::new(Instant::now() - Duration::from_secs(1));
                    frames::from_images(&sources, &images, &opts, |f| {
                        let mut last = last.lock().unwrap();
                        if f >= 1.0 || last.elapsed() > Duration::from_millis(66) {
                            *last = Instant::now();
                            progress_emit(JobEvent::Progress {
                                stage: Stage::Frames,
                                fraction: f,
                                message: if f < 0.5 { msg!("frames.sharpness") } else { msg!("frames.converting") },
                            });
                        }
                    })
                })
                .await??
            }
            _ => extract_video(cfg, tools, emit, &rep, cancel, &layout, &video).await?,
        };
        rep.progress(1.0, msg!("frames.selected", n = picked.len()));

        // Later stages become invalid.
        let _ = std::fs::remove_dir_all(layout.dataset().join("sparse"));
        let _ = std::fs::remove_dir_all(layout.masks());
        let _ = std::fs::remove_dir_all(layout.rejected());
        state = ProjectState {
            video_path: Some(cfg.video_path.clone()),
            frames: picked.clone(),
            ..Default::default()
        };
        layout.save(&state)?;
        emit(JobEvent::FramesReady { frames: picked });
        emit(JobEvent::StageDone { stage: Stage::Frames, seconds: t0.elapsed().as_secs_f64() });
    } else if !layout.images().exists() {
        anyhow::bail!("this project has no extracted frames; run from the frames stage");
    }
    if to == Stage::Frames {
        return Ok(None);
    }

    // ---------- 2. Masks (optional) ----------
    let masks_ready = |state: &ProjectState| state.masks.is_some() && layout.masks().is_dir();
    if cfg.from_stage <= Stage::Masks {
        if cfg.masks.enabled {
            *current = Some(Stage::Masks);
            let t0 = Instant::now();
            let rep = Reporter::new(emit, Stage::Masks);
            rep.progress(0.0, msg!("masks.detecting"));
            let stats = masks::run(tools, &cfg.masks, &layout.images(), &layout.masks(), cancel, |f, m| rep.progress(f, m), |l| rep.log(l)).await?;
            rep.progress(
                1.0,
                if cfg.masks.method == MaskMethod::Chroma {
                    msg!("masks.doneChroma", n = stats.count)
                } else {
                    msg!("masks.done", n = stats.count, backend = stats.backend)
                },
            );
            state.masks = Some(stats.clone());
            state.masks_dir = Some(layout.masks().to_string_lossy().into_owned());
            if cfg.masks.for_poses {
                // The poses used the previous masks.
                state.poses = None;
                state.final_ply = None;
                state.last_psnr = None;
                state.last_export = None;
            }
            layout.save(&state)?;
            emit(JobEvent::MasksReady { stats, dir: layout.masks().to_string_lossy().into_owned() });
            emit(JobEvent::StageDone { stage: Stage::Masks, seconds: t0.elapsed().as_secs_f64() });
        } else {
            emit(JobEvent::StageSkipped { stage: Stage::Masks, reason: msg!("masks.disabled") });
        }
    }
    if to == Stage::Masks {
        return Ok(None);
    }
    if cfg.masks.enabled && !masks_ready(&state) {
        anyhow::bail!("this project has no masks; run from the masks stage");
    }

    // ---------- 3. Poses ----------
    // Re-masking without feeding the masks to COLMAP doesn't change the poses.
    let reuse_poses = cfg.from_stage == Stage::Masks
        && !(cfg.masks.enabled && cfg.masks.for_poses)
        && layout.model().join("images.bin").exists()
        && state.poses.is_some();
    if reuse_poses {
        emit(JobEvent::StageSkipped { stage: Stage::Poses, reason: msg!("poses.reused") });
    } else if cfg.from_stage <= Stage::Poses {
        *current = Some(Stage::Poses);
        let t0 = Instant::now();
        let colmap_bin = Tools::require(&tools.colmap, "COLMAP", install_hint("colmap"))?;
        let rep = Reporter::new(emit, Stage::Poses);
        // Timestamp of each frame, so pose inference follows the camera speed.
        let times = state
            .frames
            .iter()
            .filter(|f| f.source.is_none() && !f.rejected)
            .filter_map(|f| Some((Path::new(&f.path).file_name()?.to_string_lossy().into_owned(), f.time)))
            .collect();
        let stats = colmap::run(
            colmap::Paths {
                colmap: &colmap_bin,
                images: &layout.images(),
                work: &layout.colmap_work(),
                model_out: &layout.model(),
                use_gpu: tools.colmap_cuda,
                masks: if cfg.masks.enabled && cfg.masks.for_poses {
                    masks::write_colmap_masks(&layout.images(), &layout.masks(), &layout.colmap_masks())?;
                    Some(layout.colmap_masks())
                } else {
                    None
                },
                times: &times,
            },
            &cfg.poses,
            cancel,
            |f, m| rep.progress(f, m),
            |l| rep.log(l),
        )
        .await?;

        state.poses = Some(stats.clone());
        state.final_ply = None;
        state.last_psnr = None;
        state.last_export = None;
        layout.save(&state)?;
        emit(JobEvent::PosesReady { stats });
        emit(JobEvent::StageDone { stage: Stage::Poses, seconds: t0.elapsed().as_secs_f64() });
    } else if state.poses.is_none() || !layout.model().join("images.bin").exists() {
        anyhow::bail!("this project has no computed poses; run from the poses stage");
    }
    if to == Stage::Poses {
        return Ok(None);
    }

    // ---------- 4. Training ----------
    *current = Some(Stage::Train);
    let t0 = Instant::now();
    let brush_bin = Tools::require(&tools.brush, "Brush", install_hint("brush"))?;
    let rep = Reporter::new(emit, Stage::Train);
    let total = cfg.train.total_steps.max(1);
    let train_dataset = if cfg.masks.enabled {
        rep.progress(0.0, msg!("train.alphaDataset"));
        let (dataset, masks_dir, out) = (layout.dataset(), layout.masks(), layout.alpha_dataset());
        let despill = cfg.masks.despill.then_some(cfg.masks.despill_strength);
        let key = tokio::task::spawn_blocking(move || masks::build_alpha_dataset(&dataset, &masks_dir, &out, despill)).await??;
        match key {
            Some(k) => rep.log(&format!("despill: #{:02x}{:02x}{:02x} spill removed from the subject", k[0], k[1], k[2])),
            None if despill.is_some() => rep.log("despill: background is neither green nor blue; colors kept"),
            None => {}
        }
        layout.alpha_dataset()
    } else {
        let _ = std::fs::remove_dir_all(layout.alpha_dataset());
        layout.dataset()
    };
    rep.progress(0.0, msg!("train.loading"));
    let mut last_psnr = None;
    let last_ply = brush::run(
        &brush_bin,
        &train_dataset,
        &layout.train_out(),
        &cfg.train,
        cfg.masks.enabled,
        cancel,
        |ev| {
            match &ev {
                TrainEvent::Step { iter, splats } => {
                    rep.progress(*iter as f32 / total as f32, msg!("train.step", iter = iter, total = total, splats = splats))
                }
                TrainEvent::Eval { psnr, .. } => last_psnr = Some(*psnr),
                TrainEvent::Snapshot { .. } => {}
            }
            emit(JobEvent::Train { event: ev });
        },
        |l| rep.log(l),
    )
    .await?;

    let final_ply = layout.final_ply();
    std::fs::copy(&last_ply, &final_ply)?;
    rep.progress(1.0, msg!("train.done"));
    let ply = final_ply.to_string_lossy().into_owned();
    state.final_ply = Some(ply.clone());
    state.last_psnr = last_psnr;
    state.last_export = None;
    layout.save(&state)?;
    emit(JobEvent::StageDone { stage: Stage::Train, seconds: t0.elapsed().as_secs_f64() });
    Ok(Some(ply))
}

/// Video frames: candidates from ffmpeg, keeping the sharpest of each window.
async fn extract_video(
    cfg: &JobConfig,
    tools: &Tools,
    emit: &Emit,
    rep: &Reporter<'_>,
    cancel: &CancellationToken,
    layout: &Layout,
    video: &Path,
) -> Result<Vec<FrameInfo>> {
    let ffmpeg = Tools::require(&tools.ffmpeg, "ffmpeg", install_hint("ffmpeg"))?;
    let ffprobe = Tools::require(&tools.ffprobe, "ffprobe", install_hint("ffmpeg"))?;

    rep.progress(0.0, msg!("frames.reading"));
    let info: VideoInfo = frames::probe(&ffprobe, video).await?;
    rep.log(&format!(
        "{}x{} @ {:.2} fps, {:.1}s, {}",
        info.width, info.height, info.fps, info.duration, info.codec
    ));

    let raw = layout.raw_frames();
    let sample_fps = frames::extract(
        &ffmpeg,
        video,
        &info,
        &cfg.frames,
        &raw,
        cancel,
        |f| rep.progress(0.6 * f, msg!("frames.decoding")),
        |l| rep.log(l),
    )
    .await?;

    let images = layout.images();
    let opts = cfg.frames.clone();
    let progress_emit = emit.clone();
    tokio::task::spawn_blocking(move || {
        let last = std::sync::Mutex::new(Instant::now());
        frames::select(&raw, &images, &opts, sample_fps, |f| {
            let mut last = last.lock().unwrap();
            if f >= 1.0 || last.elapsed() > Duration::from_millis(66) {
                *last = Instant::now();
                progress_emit(JobEvent::Progress {
                    stage: Stage::Frames,
                    fraction: 0.6 + 0.4 * f,
                    message: msg!("frames.sharpness"),
                });
            }
        })
    })
    .await?
}

/// Rejects (moves to `rejected/`) or re-accepts frames by file name. The
/// poses become invalid; re-accepting a frame that has no mask also
/// invalidates the masks.
pub fn set_frames_rejected(project_dir: &str, names: &[String], rejected: bool) -> Result<ProjectState> {
    let layout = Layout::new(project_dir);
    let mut state = layout.load();
    let dest_dir = if rejected { layout.rejected() } else { layout.images() };
    std::fs::create_dir_all(&dest_dir)?;
    let (mut changed, mut missing_mask) = (false, false);
    let mut moved = Ok(());
    for f in state.frames.iter_mut().filter(|f| f.rejected != rejected) {
        let src = PathBuf::from(&f.path);
        let Some(name) = src.file_name() else { continue };
        if !names.iter().any(|n| n.as_str() == name) {
            continue;
        }
        let dest = dest_dir.join(name);
        if let Err(e) = std::fs::rename(&src, &dest) {
            moved = Err(anyhow::Error::new(e).context(format!("could not move {}", src.display())));
            break;
        }
        missing_mask |= !rejected && !layout.masks().join(name).with_extension("png").exists();
        f.path = dest.to_string_lossy().into_owned();
        f.rejected = rejected;
        changed = true;
    }
    // Save what was already moved even if a file failed midway.
    if changed {
        if missing_mask {
            state.masks = None;
            state.masks_dir = None;
        }
        state.poses = None;
        layout.save(&state)?;
    }
    moved.map(|_| state)
}
/// Exports the project's final splat with the optimization settings and
/// records the result in `project.json`.
pub fn export_project(project_dir: &str, opts: &ExportOptions, progress: impl FnMut(f32, &Msg)) -> Result<ExportStats> {
    let layout = Layout::new(project_dir);
    let input = layout.final_ply();
    if !input.exists() {
        anyhow::bail!("this project has no trained splat yet");
    }
    let stats = export::run(&input, &layout.exports(), opts, progress)?;
    let mut state = layout.load();
    state.last_export = Some(stats.clone());
    layout.save(&state)?;
    Ok(stats)
}

pub fn default_project_dir(video: &Path) -> PathBuf {
    let stem = video.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "project".into());
    video.with_file_name(format!("{stem}_splat"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_and_accept_frames() {
        let root = std::env::temp_dir().join("splatforge_reject_test");
        let _ = std::fs::remove_dir_all(&root);
        let layout = Layout::new(&root);
        std::fs::create_dir_all(layout.images()).unwrap();
        std::fs::create_dir_all(layout.masks()).unwrap();
        let frames = (0..3)
            .map(|i| {
                let path = layout.images().join(format!("{i:05}.jpg"));
                std::fs::write(&path, b"x").unwrap();
                FrameInfo { path: path.to_string_lossy().into_owned(), time: i as f64, sharpness: 1.0, source: None, rejected: false }
            })
            .collect();
        // Only frames 0 and 1 have masks.
        for i in 0..2 {
            std::fs::write(layout.masks().join(format!("{i:05}.png")), b"m").unwrap();
        }
        let stats = ModelStats { registered: 3, total: 3, points: 10, reprojection_error: None, mapper: "t".into(), inferred: 0, refined: 0 };
        let masks = MaskStats { count: 2, no_subject: 0, mean_coverage: 0.5, backend: "t".into(), key_color: None };
        layout
            .save(&ProjectState { frames, masks: Some(masks), poses: Some(stats), ..Default::default() })
            .unwrap();
        let dir = root.to_string_lossy().into_owned();

        let s = set_frames_rejected(&dir, &["00001.jpg".into(), "00002.jpg".into()], true).unwrap();
        assert!(s.poses.is_none() && s.masks.is_some());
        assert!(s.frames[1].rejected && s.frames[2].rejected && !s.frames[0].rejected);
        assert!(layout.rejected().join("00001.jpg").exists() && !layout.images().join("00001.jpg").exists());
        assert_eq!(layout.load().frames[2].path, layout.rejected().join("00002.jpg").to_string_lossy());

        // Re-accepting a frame that has a mask keeps the masks...
        let s = set_frames_rejected(&dir, &["00001.jpg".into()], false).unwrap();
        assert!(!s.frames[1].rejected && s.masks.is_some());
        assert!(layout.images().join("00001.jpg").exists());
        // ...and one without a mask invalidates them.
        let s = set_frames_rejected(&dir, &["00002.jpg".into()], false).unwrap();
        assert!(s.masks.is_none() && !s.frames[2].rejected);
        let _ = std::fs::remove_dir_all(&root);
    }
}
