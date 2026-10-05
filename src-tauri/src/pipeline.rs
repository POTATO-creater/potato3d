//! Orquestra as etapas (frames → recorte → poses → treino), persiste o estado do projeto em disco e envia
//! eventos de progresso para a interface por um `Channel` do Tauri.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::brush::{self, TrainEvent, TrainOptions};
use crate::colmap::{self, ModelStats, PoseOptions};
use crate::frames::{self, FrameInfo, FrameOptions, VideoInfo};
use crate::masks::{self, MaskOptions, MaskStats};
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
    pub video_path: String,
    pub project_dir: String,
    pub from_stage: Stage,
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
    Progress { stage: Stage, fraction: f32, message: String },
    Log { stage: Stage, line: String },
    FramesReady { frames: Vec<FrameInfo> },
    MasksReady { stats: MaskStats, dir: String },
    PosesReady { stats: ModelStats },
    /// Etapa pulada (desativada ou resultado anterior reaproveitado).
    StageSkipped { stage: Stage, reason: String },
    Train { event: TrainEvent },
    StageDone { stage: Stage, seconds: f64 },
    Finished { ply: String, seconds: f64 },
    Failed { stage: Option<Stage>, error: String },
    Cancelled,
}

/// Destino dos eventos (Channel do Tauri na UI, stdout na CLI).
pub type Emit = Arc<dyn Fn(JobEvent) + Send + Sync>;

/// O que já existe em disco para um projeto (permite retomar de qualquer etapa).
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
}

pub struct Layout {
    pub root: PathBuf,
}

impl Layout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    fn state_file(&self) -> PathBuf {
        self.root.join("project.json")
    }
    fn raw_frames(&self) -> PathBuf {
        self.root.join("frames_raw")
    }
    /// Pasta passada ao Brush: só `images/` e `sparse/0/` (ele varre tudo
    /// dentro dela, inclusive .ply, então a saída fica fora).
    pub fn dataset(&self) -> PathBuf {
        self.root.join("dataset")
    }
    pub fn images(&self) -> PathBuf {
        self.dataset().join("images")
    }
    pub fn model(&self) -> PathBuf {
        self.dataset().join("sparse").join("0")
    }
    fn colmap_work(&self) -> PathBuf {
        self.root.join("colmap")
    }
    /// Fora de `dataset/`: o Brush trataria `dataset/masks` como máscara de
    /// "ignorar" automaticamente.
    pub fn masks(&self) -> PathBuf {
        self.root.join("masks")
    }
    /// Fora de `colmap/`, que a etapa de poses recria do zero.
    fn colmap_masks(&self) -> PathBuf {
        self.root.join("colmap_masks")
    }
    /// Dataset de treino com fundo transparente (PNG RGBA).
    fn alpha_dataset(&self) -> PathBuf {
        self.root.join("dataset_alpha")
    }
    fn train_out(&self) -> PathBuf {
        self.root.join("training")
    }
    fn final_ply(&self) -> PathBuf {
        self.root.join("splat.ply")
    }

    pub fn load(&self) -> ProjectState {
        std::fs::read(self.state_file())
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    fn save(&self, state: &ProjectState) -> Result<()> {
        std::fs::write(self.state_file(), serde_json::to_vec_pretty(state)?)?;
        Ok(())
    }
}

/// Envia eventos de progresso limitando a frequência (a UI não precisa de
/// mais de ~15 atualizações por segundo).
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
    fn progress(&self, fraction: f32, message: &str) {
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
            message: message.to_string(),
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
) -> Result<String> {
    let layout = Layout::new(&cfg.project_dir);
    std::fs::create_dir_all(&layout.root).context("não foi possível criar a pasta do projeto")?;
    let mut state = layout.load();
    let video = PathBuf::from(&cfg.video_path);

    // ---------- 1. Frames ----------
    if cfg.from_stage <= Stage::Frames {
        *current = Some(Stage::Frames);
        let t0 = Instant::now();
        let ffmpeg = Tools::require(&tools.ffmpeg, "ffmpeg", install_hint("ffmpeg"))?;
        let ffprobe = Tools::require(&tools.ffprobe, "ffprobe", install_hint("ffmpeg"))?;
        let rep = Reporter::new(emit, Stage::Frames);

        rep.progress(0.0, "Lendo vídeo");
        let info: VideoInfo = frames::probe(&ffprobe, &video).await?;
        rep.log(&format!(
            "{}x{} @ {:.2} fps, {:.1}s, {}",
            info.width, info.height, info.fps, info.duration, info.codec
        ));

        let raw = layout.raw_frames();
        let sample_fps = frames::extract(
            &ffmpeg,
            &video,
            &info,
            &cfg.frames,
            &raw,
            cancel,
            |f| rep.progress(0.6 * f, "Decodificando vídeo"),
            |l| rep.log(l),
        )
        .await?;

        let images = layout.images();
        let opts = cfg.frames.clone();
        let progress_emit = emit.clone();
        let picked = tokio::task::spawn_blocking(move || {
            let last = std::sync::Mutex::new(Instant::now());
            frames::select(&raw, &images, &opts, sample_fps, |f| {
                let mut last = last.lock().unwrap();
                if f >= 1.0 || last.elapsed() > Duration::from_millis(66) {
                    *last = Instant::now();
                    progress_emit(JobEvent::Progress {
                        stage: Stage::Frames,
                        fraction: 0.6 + 0.4 * f,
                        message: "Medindo nitidez".into(),
                    });
                }
            })
        })
        .await??;
        rep.progress(1.0, &format!("{} frames selecionados", picked.len()));

        // Etapas seguintes ficam inválidas.
        let _ = std::fs::remove_dir_all(layout.dataset().join("sparse"));
        let _ = std::fs::remove_dir_all(layout.masks());
        state = ProjectState {
            video_path: Some(cfg.video_path.clone()),
            frames: picked.clone(),
            ..Default::default()
        };
        layout.save(&state)?;
        emit(JobEvent::FramesReady { frames: picked });
        emit(JobEvent::StageDone { stage: Stage::Frames, seconds: t0.elapsed().as_secs_f64() });
    } else if !layout.images().exists() {
        anyhow::bail!("não há frames extraídos neste projeto; rode a partir da etapa de frames");
    }

    // ---------- 2. Recorte (opcional) ----------
    let masks_ready = |state: &ProjectState| state.masks.is_some() && layout.masks().is_dir();
    if cfg.from_stage <= Stage::Masks {
        if cfg.masks.enabled {
            *current = Some(Stage::Masks);
            let t0 = Instant::now();
            let rep = Reporter::new(emit, Stage::Masks);
            rep.progress(0.0, "Detectando o sujeito");
            let stats = masks::run(tools, &layout.images(), &layout.masks(), cancel, |f, m| rep.progress(f, m), |l| rep.log(l)).await?;
            rep.progress(1.0, &format!("{} recortes · {}", stats.count, stats.backend));
            state.masks = Some(stats.clone());
            state.masks_dir = Some(layout.masks().to_string_lossy().into_owned());
            layout.save(&state)?;
            emit(JobEvent::MasksReady { stats, dir: layout.masks().to_string_lossy().into_owned() });
            emit(JobEvent::StageDone { stage: Stage::Masks, seconds: t0.elapsed().as_secs_f64() });
        } else {
            emit(JobEvent::StageSkipped { stage: Stage::Masks, reason: "Recorte desativado".into() });
        }
    }
    if cfg.masks.enabled && !masks_ready(&state) {
        anyhow::bail!("não há recortes neste projeto; rode a partir da etapa de recorte");
    }

    // ---------- 3. Poses ----------
    // Recortar de novo sem usar as máscaras no COLMAP não muda as poses.
    let reuse_poses = cfg.from_stage == Stage::Masks
        && !(cfg.masks.enabled && cfg.masks.for_poses)
        && layout.model().join("images.bin").exists()
        && state.poses.is_some();
    if reuse_poses {
        emit(JobEvent::StageSkipped { stage: Stage::Poses, reason: "Poses anteriores reaproveitadas".into() });
    } else if cfg.from_stage <= Stage::Poses {
        *current = Some(Stage::Poses);
        let t0 = Instant::now();
        let colmap_bin = Tools::require(&tools.colmap, "COLMAP", install_hint("colmap"))?;
        let rep = Reporter::new(emit, Stage::Poses);
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
        layout.save(&state)?;
        emit(JobEvent::PosesReady { stats });
        emit(JobEvent::StageDone { stage: Stage::Poses, seconds: t0.elapsed().as_secs_f64() });
    } else if !layout.model().join("images.bin").exists() {
        anyhow::bail!("não há poses calculadas neste projeto; rode a partir da etapa de poses");
    }

    // ---------- 4. Treino ----------
    *current = Some(Stage::Train);
    let t0 = Instant::now();
    let brush_bin = Tools::require(&tools.brush, "Brush", install_hint("brush"))?;
    let rep = Reporter::new(emit, Stage::Train);
    let total = cfg.train.total_steps.max(1);
    let train_dataset = if cfg.masks.enabled {
        rep.progress(0.0, "Montando frames com fundo transparente");
        let (dataset, masks_dir, out) = (layout.dataset(), layout.masks(), layout.alpha_dataset());
        tokio::task::spawn_blocking(move || masks::build_alpha_dataset(&dataset, &masks_dir, &out)).await??;
        layout.alpha_dataset()
    } else {
        let _ = std::fs::remove_dir_all(layout.alpha_dataset());
        layout.dataset()
    };
    rep.progress(0.0, "Carregando dataset na GPU");
    let mut last_psnr = None;
    let last_ply = brush::run(
        &brush_bin,
        &train_dataset,
        &layout.train_out(),
        &cfg.train,
        cancel,
        |ev| {
            match &ev {
                TrainEvent::Step { iter, splats } => rep.progress(
                    *iter as f32 / total as f32,
                    &format!("Passo {iter}/{total} · {} splats", fmt_count(*splats)),
                ),
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
    rep.progress(1.0, "Treino concluído");
    let ply = final_ply.to_string_lossy().into_owned();
    state.final_ply = Some(ply.clone());
    state.last_psnr = last_psnr;
    layout.save(&state)?;
    emit(JobEvent::StageDone { stage: Stage::Train, seconds: t0.elapsed().as_secs_f64() });
    Ok(ply)
}

fn fmt_count(n: u64) -> String {
    match n {
        n if n >= 1_000_000 => format!("{:.2}M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.0}k", n as f64 / 1e3),
        n => n.to_string(),
    }
}

pub fn default_project_dir(video: &Path) -> PathBuf {
    let stem = video.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "projeto".into());
    video.with_file_name(format!("{stem}_splat"))
}
