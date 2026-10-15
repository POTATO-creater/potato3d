//! Optional "Cutout" step: isolates the subject in each frame.
//!
//! Methods:
//! - macOS, automatic: Vision framework (Swift sidecar `splatforge_mask`),
//!   ~40 ms/frame on an M4 GPU, any subject, no model to download.
//! - Windows, people: MODNet (Apache 2.0), a light convolutional net — 144 ms
//!   on an M4 CPU vs 7.9 s for BiRefNet, with equivalent quality on people
//!   (including from behind).
//! - Windows, any object: BiRefNet-lite (MIT). More accurate beyond people,
//!   but its deformable convolutions become ~16k ONNX ops and stay slow even
//!   on GPU (~4 s/frame on an RTX 4070 Ti Super).
//! - RMBG-1.4 (BRIA, non-commercial), Windows and Apple Silicon macOS: an
//!   IS-Net of plain convolutions, better edges than Vision on objects.
//!   ~0.4 s/frame with CoreML on an M4 GPU; downloaded only if chosen.
//! - Solid color (chroma key): RGB distance to the background color, with
//!   tolerance, softness and flood fill from the borders. Instant.
//!
//! The masks (grayscale, 255 = subject) become the alpha of the training PNGs:
//! Brush learns the background as transparent and the splat holds only the
//! subject. With a green/blue background, despill removes the light the
//! background bounces onto the subject.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{bail, Context, Result};
use image::{GrayImage, RgbImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::msg;

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum MaskMethod {
    /// macOS: Vision. Windows: MODNet.
    #[default]
    Auto,
    /// Windows: MODNet (fast, people only).
    People,
    /// Windows: BiRefNet (any object, slow).
    General,
    /// RMBG-1.4 (any object; Windows and Apple Silicon macOS).
    Rmbg,
    /// Solid-color chroma key.
    Chroma,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ChromaOptions {
    /// Detects the background color from the median of each frame's borders.
    pub auto_color: bool,
    pub color: [u8; 3],
    /// 0–100: RGB distance treated as background.
    pub tolerance: f32,
    /// 0–100: soft transition band beyond the tolerance.
    pub softness: f32,
    /// Only removes regions connected to the border (doesn't punch holes in the subject).
    pub contiguous: bool,
}

impl Default for ChromaOptions {
    fn default() -> Self {
        // More conservative than typical editors: a semi-transparent hole in the
        // subject becomes a hole in the splat (18/8 already ate light blue tones).
        Self { auto_color: true, color: [0, 255, 0], tolerance: 12.0, softness: 6.0, contiguous: true }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MaskOptions {
    pub enabled: bool,
    /// Also passes the masks to COLMAP. Helps when the background "melts";
    /// hurts on low-texture subjects (e.g. the back of a head), hence optional.
    pub for_poses: bool,
    pub method: MaskMethod,
    pub chroma: ChromaOptions,
    /// Removes green or blue background spill from the subject (any method).
    pub despill: bool,
    /// 0–100.
    pub despill_strength: f32,
}

impl Default for MaskOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            for_poses: false,
            method: MaskMethod::Auto,
            chroma: ChromaOptions::default(),
            despill: true,
            despill_strength: 100.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskStats {
    pub count: u32,
    /// Frames where no subject was detected (full mask).
    pub no_subject: u32,
    /// Mean fraction of the image covered by the subject.
    pub mean_coverage: f32,
    pub backend: String,
    /// Chroma key: background color used (median across frames).
    #[serde(default)]
    pub key_color: Option<[u8; 3]>,
}

fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("png")))
        .collect();
    v.sort();
    Ok(v)
}

/// Messages from a blocking job: progress `(i, n)` or a log line.
type Tx = tokio::sync::mpsc::UnboundedSender<Msg>;
enum Msg {
    Progress(u32, u32),
    Log(String),
}

/// Runs `work` on a blocking thread, forwarding progress and log.
async fn drive<T: Send + 'static>(
    work: impl FnOnce(Tx) -> Result<T> + Send + 'static,
    on_progress: &mut impl FnMut(f32, crate::msg::Msg),
    on_log: &mut impl FnMut(&str),
) -> Result<T> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = tokio::task::spawn_blocking(move || work(tx));
    while let Some(msg) = rx.recv().await {
        match msg {
            Msg::Progress(i, n) => on_progress(i as f32 / n.max(1) as f32, msg!("masks.cutting", i = i, n = n)),
            Msg::Log(l) => on_log(&l),
        }
    }
    worker.await?
}

pub async fn run(
    tools: &crate::tools::Tools,
    opts: &MaskOptions,
    images: &Path,
    masks: &Path,
    cancel: &CancellationToken,
    mut on_progress: impl FnMut(f32, crate::msg::Msg),
    mut on_log: impl FnMut(&str),
) -> Result<MaskStats> {
    if masks.exists() {
        std::fs::remove_dir_all(masks)?;
    }
    std::fs::create_dir_all(masks)?;
    let total = list_images(images)?.len();
    if total == 0 {
        bail!("no frames to mask");
    }

    let mut no_subject = 0u32;
    let mut key_color = None;
    let backend = if opts.method == MaskMethod::Chroma {
        let (images, masks, chroma, cancel) = (images.to_path_buf(), masks.to_path_buf(), opts.chroma.clone(), cancel.clone());
        let key = drive(move |tx| chroma::run(&images, &masks, &chroma, &cancel, &tx), &mut on_progress, &mut on_log).await?;
        key_color = Some(key);
        "Solid color".to_string()
    } else {
        run_ai(tools, opts.method, images, masks, cancel, &mut on_progress, &mut on_log, &mut no_subject).await?
    };

    // Stats (in parallel): mean subject coverage.
    let files = list_images(masks)?;
    if files.len() != total {
        bail!("masking produced {} masks for {total} frames", files.len());
    }
    let coverage: Vec<f32> = files
        .par_iter()
        .map(|p| {
            image::open(p)
                .map(|m| {
                    let g = m.into_luma8();
                    g.as_raw().iter().map(|&v| v as f32).sum::<f32>() / (255.0 * g.as_raw().len().max(1) as f32)
                })
                .unwrap_or(1.0)
        })
        .collect();
    Ok(MaskStats {
        count: total as u32,
        no_subject,
        mean_coverage: coverage.iter().sum::<f32>() / coverage.len().max(1) as f32,
        backend,
        key_color,
    })
}

// Each OS uses only some of the arguments.
#[allow(clippy::too_many_arguments, unused_variables)]
async fn run_ai(
    tools: &crate::tools::Tools,
    method: MaskMethod,
    images: &Path,
    masks: &Path,
    cancel: &CancellationToken,
    on_progress: &mut impl FnMut(f32, crate::msg::Msg),
    on_log: &mut impl FnMut(&str),
    no_subject: &mut u32,
) -> Result<String> {
    if method == MaskMethod::Rmbg && !crate::tools::ONNX_AVAILABLE {
        bail!("RMBG-1.4 requires Windows or an Apple Silicon Mac");
    }

    // ONNX models: all of them on Windows; only RMBG on macOS (the rest is Vision).
    #[cfg(any(windows, all(target_os = "macos", target_arch = "aarch64")))]
    {
        let kind = match method {
            MaskMethod::Rmbg => Some(ModelKind::Rmbg),
            MaskMethod::General if cfg!(windows) => Some(ModelKind::BiRefNet),
            _ if cfg!(windows) => Some(ModelKind::Modnet),
            _ => None,
        };
        if let Some(kind) = kind {
            let model = match kind {
                ModelKind::BiRefNet => &tools.birefnet_model,
                ModelKind::Modnet => &tools.modnet_model,
                ModelKind::Rmbg => &tools.rmbg_model,
            };
            let model = crate::tools::Tools::require(model, kind.label(), crate::tools::install_hint("model"))?;
            let (images, masks, cancel) = (images.to_path_buf(), masks.to_path_buf(), cancel.clone());
            return drive(move |tx| onnx::run(&model, kind, &images, &masks, &cancel, &tx), on_progress, on_log).await;
        }
    }

    #[cfg(target_os = "macos")]
    {
        let bin = crate::tools::Tools::require(&tools.masker, "Masker (Vision)", "Rebuild the app on macOS 14+.")?;
        let mut cmd = tokio::process::Command::new(bin);
        cmd.arg(images).arg(masks);
        crate::proc::run_streaming(cmd, cancel, |line| {
            if let Some((i, n)) = line
                .strip_prefix("progress ")
                .and_then(|r| r.split_once('/'))
                .and_then(|(i, n)| Some((i.parse::<u32>().ok()?, n.parse::<u32>().ok()?)))
            {
                on_progress(i as f32 / n.max(1) as f32, msg!("masks.cutting", i = i, n = n));
            } else if let Some(name) = line.strip_prefix("nosubject ") {
                *no_subject += 1;
                on_log(&format!("no subject detected in {name}; keeping the whole frame"));
            } else {
                on_log(line);
            }
        })
        .await
        .context("masking failed (Vision)")?;
        Ok("Vision · GPU".to_string())
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (tools, images, masks, cancel, on_progress, on_log, no_subject);
        bail!("AI masking is not available on this system yet; use solid color");
    }
}

/// Background color spill suppression: the light a green screen bounces onto
/// the subject. The key color's dominant channel is clamped to the larger of
/// the other two across the whole subject (not just the edge: bounced green
/// covers entire surfaces, like the base of a gray ball) and the removed
/// luminance is added back as gray so the subject doesn't darken. Clamping to
/// the max (not the mean) of the other channels preserves yellows and cyans.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Despill {
    /// Key color channel: 1 = green, 2 = blue.
    channel: usize,
    /// 0–1.
    strength: f32,
}

/// Per-channel luminance weights (Rec. 709).
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

impl Despill {
    /// Only for well-saturated green or blue backgrounds: on a beige or gray
    /// background the "spill" would be the subject's own color (skin, wood…).
    pub fn new(key: [u8; 3], strength: f32) -> Option<Self> {
        let channel = (0..3).max_by_key(|&i| key[i]).unwrap();
        let others = (0..3).filter(|&i| i != channel).map(|i| key[i]).max().unwrap();
        (channel != 0 && key[channel] as i32 - others as i32 > 50 && strength > 0.0)
            .then(|| Self { channel, strength: (strength / 100.0).clamp(0.0, 1.0) })
    }

    fn apply(&self, px: &mut [u8]) {
        let c = self.channel;
        let limit = (0..3).filter(|&i| i != c).map(|i| px[i]).max().unwrap();
        let spill = (px[c] as f32 - limit as f32) * self.strength;
        if spill <= 0.0 {
            return;
        }
        let gray = spill * LUMA[c];
        for (i, v) in px.iter_mut().take(3).enumerate() {
            let x = *v as f32 - if i == c { spill } else { 0.0 } + gray;
            *v = x.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Erosion radius for the training alpha. The mask edge mixes subject and
/// background (and the cutout leaks a bit past the outline); without erosion,
/// Brush learns that fringe as a colored shell around the object. Same value
/// as TripoSplat (3×3 min filter).
const ALPHA_ERODE: u32 = 1;

/// Grayscale erosion: minimum over a (2r+1)² window, separable, with the
/// image border replicated (a subject touching the border doesn't shrink there).
pub fn erode(mask: &GrayImage, r: u32) -> GrayImage {
    if r == 0 {
        return mask.clone();
    }
    let (w, h) = mask.dimensions();
    let r = r as i64;
    let min_along = |src: &GrayImage, dx: i64, dy: i64| {
        GrayImage::from_fn(w, h, |x, y| {
            let m = (-r..=r)
                .map(|k| {
                    let sx = (x as i64 + k * dx).clamp(0, w as i64 - 1) as u32;
                    let sy = (y as i64 + k * dy).clamp(0, h as i64 - 1) as u32;
                    src.get_pixel(sx, sy)[0]
                })
                .min()
                .unwrap();
            image::Luma([m])
        })
    };
    min_along(&min_along(mask, 1, 0), 0, 1)
}

/// Reads a frame's mask, resized to the frame.
fn load_mask(masks: &Path, img_path: &Path, (w, h): (u32, u32)) -> Result<GrayImage> {
    let stem = img_path.file_stem().context("invalid frame name")?;
    let mask_path = masks.join(stem).with_extension("png");
    let mask = image::open(&mask_path)
        .with_context(|| format!("missing mask: {}", mask_path.display()))?
        .into_luma8();
    Ok(if mask.dimensions() == (w, h) {
        mask
    } else {
        image::imageops::resize(&mask, w, h, image::imageops::FilterType::Triangle)
    })
}

/// Background color: per-channel median of the pixels the mask marks as
/// background, over a sample of up to 24 frames. Works with any masking method
/// and is a single color for the whole video (consistent across views).
pub fn background_color(images: &[PathBuf], masks: &Path) -> Result<Option<[u8; 3]>> {
    let step = images.len().div_ceil(24).max(1);
    let hists = images
        .par_iter()
        .step_by(step)
        .map(|p| -> Result<[[u64; 256]; 3]> {
            let img = image::open(p)?.into_rgb8();
            let mask = load_mask(masks, p, img.dimensions())?;
            let mut h = [[0u64; 256]; 3];
            for (px, m) in img.pixels().zip(mask.pixels()).step_by(3) {
                if m[0] < 26 {
                    for c in 0..3 {
                        h[c][px[c] as usize] += 1;
                    }
                }
            }
            Ok(h)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut total = [[0u64; 256]; 3];
    for h in &hists {
        for c in 0..3 {
            for v in 0..256 {
                total[c][v] += h[c][v];
            }
        }
    }
    let n: u64 = total[0].iter().sum();
    if n < 1000 {
        return Ok(None);
    }
    Ok(Some(std::array::from_fn(|c| {
        let mut acc = 0;
        total[c].iter().position(|&k| {
            acc += k;
            acc * 2 >= n
        }).unwrap_or(0) as u8
    })))
}

/// Builds the training dataset with a transparent background: RGBA PNGs
/// (alpha = mask eroded by `ALPHA_ERODE` px, edge colors unmixed from the
/// background by `decontam`) and a copy of the COLMAP model
/// with names switched to `.png`.
/// Brush looks up each image by the name registered in `images.bin`.
/// `despill`: strength 0–100; returns the background color when applied.
pub fn build_alpha_dataset(dataset: &Path, masks: &Path, out: &Path, despill: Option<f32>) -> Result<Option<[u8; 3]>> {
    if out.exists() {
        std::fs::remove_dir_all(out)?;
    }
    let out_images = out.join("images");
    let out_model = out.join("sparse").join("0");
    std::fs::create_dir_all(&out_images)?;
    std::fs::create_dir_all(&out_model)?;

    let images = list_images(&dataset.join("images"))?;
    let key = match despill {
        Some(strength) => background_color(&images, masks)?.filter(|&k| Despill::new(k, strength).is_some()),
        None => None,
    };
    let despill = key.zip(despill).and_then(|(k, s)| Despill::new(k, s));

    images.par_iter().try_for_each(|img_path| -> Result<()> {
        let stem = img_path.file_stem().context("invalid frame name")?;
        let mut rgba = image::open(img_path)?.into_rgba8();
        let raw = load_mask(masks, img_path, rgba.dimensions())?;
        let mask = erode(&raw, ALPHA_ERODE);
        // Before despill: unmixing needs the photo's own background color.
        crate::decontam::decontaminate(&mut rgba, &raw, &mask);
        for (px, m) in rgba.pixels_mut().zip(mask.pixels()) {
            px[3] = m[0];
            if let Some(d) = despill.filter(|_| m[0] > 0) {
                d.apply(&mut px.0[..3]);
            }
        }
        rgba.save(out_images.join(stem).with_extension("png"))?;
        Ok(())
    })?;

    let model = dataset.join("sparse").join("0");
    for f in ["cameras.bin", "points3D.bin"] {
        std::fs::copy(model.join(f), out_model.join(f))?;
    }
    crate::scene::rewrite_image_names(&model.join("images.bin"), &out_model.join("images.bin"), |name| {
        Path::new(name).with_extension("png").to_string_lossy().into_owned()
    })?;
    Ok(key)
}

/// COLMAP expects `<image_name>.png` (e.g. `00000.jpg.png`), black = ignore.
pub fn write_colmap_masks(images: &Path, masks: &Path, out: &Path) -> Result<()> {
    if out.exists() {
        std::fs::remove_dir_all(out)?;
    }
    std::fs::create_dir_all(out)?;
    for img in list_images(images)? {
        let name = img.file_name().context("invalid name")?.to_string_lossy().into_owned();
        let stem = img.file_stem().context("invalid name")?;
        let src = masks.join(stem).with_extension("png");
        let dst = out.join(format!("{name}.png"));
        if std::fs::hard_link(&src, &dst).is_err() {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

// ---------- Chroma key ----------

pub mod chroma {
    use super::*;

    /// RGB distance to the key color, accepting a brighter key (up to 1.6x):
    /// screens tend to be lit more in the middle than at the borders, where the
    /// detected color comes from. Not darker: the shadowed base of an object,
    /// tinted by green spill, would be mistaken for background.
    pub fn dist(px: &[u8], key: [u8; 3]) -> f32 {
        let k: [f32; 3] = key.map(|v| v as f32);
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let p = [px[0] as f32, px[1] as f32, px[2] as f32];
        let s = (dot(p, k) / dot(k, k).max(1.0)).clamp(1.0, 1.6);
        let d: [f32; 3] = std::array::from_fn(|i| p[i] - s * k[i]);
        dot(d, d).sqrt()
    }

    /// Per-channel median of the border pixels (robust to a "dirty" corner).
    pub fn border_key(img: &RgbImage) -> [u8; 3] {
        let (w, h) = img.dimensions();
        let step = ((w + h) / 400).max(1) as usize;
        let mut ch: [Vec<u8>; 3] = Default::default();
        let mut push = |x: u32, y: u32| {
            let p = img.get_pixel(x, y);
            for c in 0..3 {
                ch[c].push(p[c]);
            }
        };
        for x in (0..w).step_by(step) {
            push(x, 0);
            push(x, h - 1);
        }
        for y in (0..h).step_by(step) {
            push(0, y);
            push(w - 1, y);
        }
        std::array::from_fn(|c| {
            ch[c].sort_unstable();
            ch[c][ch[c].len() / 2]
        })
    }

    /// Mask (255 = subject) from distance to the key color.
    pub fn mask(img: &RgbImage, key: [u8; 3], o: &ChromaOptions) -> GrayImage {
        let (w, h) = img.dimensions();
        let n = (w * h) as usize;
        let tol = o.tolerance.clamp(0.0, 100.0) * 4.42; // 0–100 → 0–442 (max RGB distance)
        let soft = (o.softness.clamp(0.0, 100.0) * 4.42).max(1.0);
        let d: Vec<f32> = img.as_raw().par_chunks_exact(3).map(|p| dist(p, key)).collect();

        // With "contiguous", only what connects to the border counts as background.
        let reach = o.contiguous.then(|| {
            let limit = tol + soft;
            let mut seen = vec![false; n];
            let mut stack = Vec::with_capacity(n / 8);
            let push = |i: usize, seen: &mut Vec<bool>, stack: &mut Vec<usize>| {
                if !seen[i] && d[i] <= limit {
                    seen[i] = true;
                    stack.push(i);
                }
            };
            let (wu, hu) = (w as usize, h as usize);
            for x in 0..wu {
                push(x, &mut seen, &mut stack);
                push((hu - 1) * wu + x, &mut seen, &mut stack);
            }
            for y in 0..hu {
                push(y * wu, &mut seen, &mut stack);
                push(y * wu + wu - 1, &mut seen, &mut stack);
            }
            while let Some(i) = stack.pop() {
                let x = i % wu;
                if x > 0 {
                    push(i - 1, &mut seen, &mut stack);
                }
                if x + 1 < wu {
                    push(i + 1, &mut seen, &mut stack);
                }
                if i >= wu {
                    push(i - wu, &mut seen, &mut stack);
                }
                if i + wu < n {
                    push(i + wu, &mut seen, &mut stack);
                }
            }
            seen
        });

        let out: Vec<u8> = d
            .par_iter()
            .enumerate()
            .map(|(i, &di)| {
                if reach.as_ref().is_some_and(|r| !r[i]) || di >= tol + soft {
                    255
                } else if di <= tol {
                    0
                } else {
                    (255.0 * (di - tol) / soft).round() as u8
                }
            })
            .collect();
        GrayImage::from_raw(w, h, out).unwrap()
    }

    /// Masks all frames; returns the median key color used.
    pub(super) fn run(images: &Path, masks: &Path, o: &ChromaOptions, cancel: &CancellationToken, tx: &Tx) -> Result<[u8; 3]> {
        let files = list_images(images)?;
        let n = files.len() as u32;
        let done = AtomicU32::new(0);
        let tx = std::sync::Mutex::new(tx.clone());
        let keys: Vec<[u8; 3]> = files
            .par_iter()
            .map(|p| -> Result<[u8; 3]> {
                if cancel.is_cancelled() {
                    return Err(crate::proc::Cancelled.into());
                }
                let img = image::open(p)?.into_rgb8();
                let key = if o.auto_color { border_key(&img) } else { o.color };
                mask(&img, key, o).save(masks.join(p.file_stem().context("invalid name")?).with_extension("png"))?;
                let i = done.fetch_add(1, Ordering::Relaxed) + 1;
                let _ = tx.lock().unwrap().send(Msg::Progress(i, n));
                Ok(key)
            })
            .collect::<Result<_>>()?;
        let key: [u8; 3] = std::array::from_fn(|c| {
            let mut v: Vec<u8> = keys.iter().map(|k| k[c]).collect();
            v.sort_unstable();
            v[v.len() / 2]
        });
        if o.auto_color {
            let _ = tx.lock().unwrap().send(Msg::Log(format!(
                "detected background color (median): #{:02x}{:02x}{:02x}",
                key[0], key[1], key[2]
            )));
        }
        Ok(key)
    }
}

// ---------- ONNX models: pre/post-processing (shared and testable) ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// BiRefNet-lite: fixed 1024² input, ImageNet normalization, logits output.
    BiRefNet,
    /// MODNet: short side 512 (multiples of 32), [-1, 1] normalization, 0–1 output.
    Modnet,
    /// RMBG-1.4: fixed 1024² input, `x - 0.5`, 0–1 output stretched between
    /// the frame's min and max (as in BRIA's `utilities.py`).
    Rmbg,
}

impl ModelKind {
    pub fn label(self) -> &'static str {
        match self {
            ModelKind::BiRefNet => "BiRefNet",
            ModelKind::Modnet => "MODNet",
            ModelKind::Rmbg => "RMBG-1.4",
        }
    }

    /// Model input size (width, height) for a frame.
    pub fn input_size(self, w: u32, h: u32) -> (u32, u32) {
        match self {
            ModelKind::BiRefNet | ModelKind::Rmbg => (1024, 1024),
            ModelKind::Modnet => {
                let s = 512.0 / w.min(h) as f32;
                let r = |v: u32| (((v as f32 * s) / 32.0).round() as u32 * 32).max(32);
                (r(w), r(h))
            }
        }
    }

    /// Image → NCHW tensor and its shape.
    pub fn preprocess(self, img: &RgbImage) -> (Vec<f32>, [usize; 4]) {
        let (iw, ih) = self.input_size(img.width(), img.height());
        let small = image::imageops::resize(img, iw, ih, image::imageops::FilterType::Triangle);
        let (mean, std) = match self {
            ModelKind::BiRefNet => ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ModelKind::Modnet => ([0.5; 3], [0.5; 3]),
            ModelKind::Rmbg => ([0.5; 3], [1.0; 3]),
        };
        let plane = (iw * ih) as usize;
        let mut out = vec![0f32; 3 * plane];
        for (i, px) in small.pixels().enumerate() {
            for c in 0..3 {
                out[c * plane + i] = (px[c] as f32 / 255.0 - mean[c]) / std[c];
            }
        }
        (out, [1, 3, ih as usize, iw as usize])
    }

    /// Model output → mask at the original size.
    pub fn postprocess(self, out: &[f32], (iw, ih): (u32, u32), width: u32, height: u32) -> GrayImage {
        // RMBG: stretch the result between the frame's min and max, except
        // when there's hardly any subject (stretching would only amplify noise).
        let (lo, scale) = match self {
            ModelKind::Rmbg => {
                let (lo, hi) = out.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                if hi - lo > 0.5 { (lo, 1.0 / (hi - lo)) } else { (0.0, 1.0) }
            }
            _ => (0.0, 1.0),
        };
        let to_u8 = |v: f32| match self {
            ModelKind::BiRefNet => (255.0 / (1.0 + (-v).exp())).round() as u8,
            ModelKind::Modnet => (v.clamp(0.0, 1.0) * 255.0).round() as u8,
            ModelKind::Rmbg => (((v - lo) * scale).clamp(0.0, 1.0) * 255.0).round() as u8,
        };
        let bytes: Vec<u8> = out.iter().map(|&v| to_u8(v)).collect();
        let m = GrayImage::from_raw(iw, ih, bytes).expect("model output has unexpected size");
        image::imageops::resize(&m, width, height, image::imageops::FilterType::Triangle)
    }
}

#[cfg(any(windows, all(target_os = "macos", target_arch = "aarch64")))]
mod onnx {
    use super::*;
    use ort::{ep, session::Session, value::Tensor};

    /// Builder errors carry the builder itself (not `Send`).
    fn ort_err<R>(e: ort::Error<R>) -> anyhow::Error {
        anyhow::anyhow!("{e}")
    }

    #[cfg(windows)]
    const GPU: &str = "DirectML";
    #[cfg(target_os = "macos")]
    const GPU: &str = "CoreML · GPU";

    #[cfg(windows)]
    fn gpu_provider() -> ort::ep::ExecutionProviderDispatch {
        ep::DirectML::default().build().error_on_failure()
    }

    /// GPU without the Neural Engine: compiling large nets for the ANE has hung
    /// the system's `ANECompilerService` for over an hour (BiRefNet).
    #[cfg(target_os = "macos")]
    fn gpu_provider() -> ort::ep::ExecutionProviderDispatch {
        use ep::coreml::{ComputeUnits, ModelFormat};
        ep::CoreML::default()
            .with_compute_units(ComputeUnits::CPUAndGPU)
            .with_model_format(ModelFormat::MLProgram)
            .with_static_input_shapes(true)
            .build()
            .error_on_failure()
    }

    fn session(model: &Path, gpu: bool) -> Result<Session> {
        // DirectML requires sequential execution and no memory pattern. Batch
        // fixed at 1: with a dynamic batch CoreML only takes part of the graph
        // and RMBG gets 3x slower.
        let mut b = Session::builder()
            .map_err(ort_err)?
            .with_parallel_execution(false)
            .map_err(ort_err)?
            .with_memory_pattern(false)
            .map_err(ort_err)?
            .with_dimension_override("batch_size", 1)
            .map_err(ort_err)?;
        if gpu {
            b = b.with_execution_providers([gpu_provider()]).map_err(ort_err)?;
        }
        b.commit_from_file(model).map_err(ort_err)
    }

    pub fn run(model: &Path, kind: ModelKind, images: &Path, masks: &Path, cancel: &CancellationToken, tx: &Tx) -> Result<String> {
        let (mut sess, device) = match session(model, true) {
            Ok(s) => (s, GPU),
            Err(e) => {
                let _ = tx.send(Msg::Log(format!("{GPU} unavailable ({e:#}); using CPU")));
                (session(model, false)?, "CPU")
            }
        };
        let input = sess.inputs()[0].name().to_string();
        let files = list_images(images)?;
        let n = files.len() as u32;
        // Preprocess the next frame in parallel while the GPU works.
        let load = |p: &PathBuf| -> Result<(u32, u32, Vec<f32>, [usize; 4])> {
            let img = image::open(p)?.into_rgb8();
            let (data, shape) = kind.preprocess(&img);
            Ok((img.width(), img.height(), data, shape))
        };
        let mut next = files.first().map(load).transpose()?;
        for (i, path) in files.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(crate::proc::Cancelled.into());
            }
            let (w, h, data, shape) = next.take().context("frame not loaded")?;
            let following = files.get(i + 1).cloned();
            let (prefetch, out) = rayon::join(
                || following.as_ref().map(load).transpose(),
                || -> Result<Vec<f32>> {
                    let t = Tensor::from_array((shape, data))?;
                    let out = sess.run(ort::inputs![input.as_str() => t])?;
                    // BiRefNet: the last output is the final one; MODNet and RMBG have only one.
                    let (_, v) = out[out.len() - 1].try_extract_tensor::<f32>()?;
                    Ok(v.to_vec())
                },
            );
            next = prefetch?;
            let mask = kind.postprocess(&out?, (shape[3] as u32, shape[2] as u32), w, h);
            mask.save(masks.join(path.file_stem().context("invalid name")?).with_extension("png"))?;
            let _ = tx.send(Msg::Progress(i as u32 + 1, n));
        }
        Ok(format!("{} · {device}", kind.label()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_pre_and_post_processing() {
        let img = RgbImage::from_pixel(64, 48, image::Rgb([124, 116, 104]));
        let (t, shape) = ModelKind::BiRefNet.preprocess(&img);
        assert_eq!(shape, [1, 3, 1024, 1024]);
        assert_eq!(t.len(), 3 * 1024 * 1024);
        // Color ≈ ImageNet mean → values near zero.
        assert!(t.iter().all(|v| v.abs() < 0.05));
        let m = ModelKind::BiRefNet.postprocess(&vec![10.0; 1024 * 1024], (1024, 1024), 64, 48);
        assert_eq!(m.dimensions(), (64, 48));
        assert!(m.pixels().all(|p| p[0] == 255));

        // MODNet: short side 512, multiples of 32 (960×1280 → 512×672).
        assert_eq!(ModelKind::Modnet.input_size(960, 1280), (512, 672));
        let (_, shape) = ModelKind::Modnet.preprocess(&RgbImage::new(960, 1280));
        assert_eq!(shape, [1, 3, 672, 512]);
        let m = ModelKind::Modnet.postprocess(&vec![0.5; 512 * 672], (512, 672), 960, 1280);
        assert!(m.pixels().all(|p| p[0] == 128));
    }

    /// A 4×4 square shrinks 1 px on each side; what touches the image border
    /// doesn't shrink there; the soft fringe moves 1 px inward.
    #[test]
    fn erode_shrinks_one_pixel() {
        let mut m = GrayImage::new(10, 8);
        for y in 2..6 {
            for x in 2..6 {
                m.put_pixel(x, y, image::Luma([255]));
            }
        }
        for y in 0..8 {
            m.put_pixel(9, y, image::Luma([255]));
            m.put_pixel(8, y, image::Luma([128]));
        }
        let e = erode(&m, 1);
        let on = |x, y| e.get_pixel(x, y)[0];
        assert_eq!((on(2, 2), on(3, 3), on(4, 4), on(5, 5)), (0, 255, 255, 0));
        assert_eq!((on(9, 0), on(9, 7), on(8, 3)), (128, 128, 0));
        assert_eq!(erode(&m, 0), m);
    }

    /// Green background with a red square in the middle and a green "hole"
    /// inside it: with "contiguous", the hole is not removed.
    #[test]
    fn chroma_key_contiguous_and_soft_edges() {
        let mut img = RgbImage::from_pixel(40, 40, image::Rgb([20, 200, 30]));
        for y in 10..30 {
            for x in 10..30 {
                img.put_pixel(x, y, image::Rgb([200, 40, 40]));
            }
        }
        img.put_pixel(20, 20, image::Rgb([20, 200, 30]));
        let key = chroma::border_key(&img);
        assert_eq!(key, [20, 200, 30]);

        let o = ChromaOptions::default();
        let m = chroma::mask(&img, key, &o);
        assert_eq!(m.get_pixel(0, 0)[0], 0, "background removed");
        assert_eq!(m.get_pixel(15, 15)[0], 255, "subject kept");
        assert_eq!(m.get_pixel(20, 20)[0], 255, "inner hole preserved (contiguous)");
        let m = chroma::mask(&img, key, &ChromaOptions { contiguous: false, ..o.clone() });
        assert_eq!(m.get_pixel(20, 20)[0], 0, "without contiguous, the hole goes");

        // A screen brighter than the key is still background; a dark object
        // tinted green (shadowed base of a ball) is not.
        assert!(chroma::dist(&[30, 250, 45], [20, 200, 30]) < 15.0);
        assert!(chroma::dist(&[30, 80, 40], [30, 200, 40]) > 100.0);

    }

    /// Despill: green light bounced onto a gray surface disappears across the
    /// whole subject without darkening it; yellow and skin stay intact.
    #[test]
    fn despill_removes_green_spill_everywhere() {
        let d = Despill::new([30, 200, 40], 100.0).expect("green screen");
        // Base of a gray ball lit by the green floor.
        let mut ball = [30u8, 80, 40];
        d.apply(&mut ball);
        assert!(ball[1] <= ball[0].max(ball[2]), "no dominant green: {ball:?}");
        let luma = |p: [u8; 3]| (0..3).map(|i| p[i] as f32 * LUMA[i]).sum::<f32>();
        assert!(luma(ball) > 50.0, "luminance preserved: {ball:?}");
        for keep in [[200u8, 150, 120], [220, 220, 30], [60, 60, 60]] {
            let mut p = keep;
            d.apply(&mut p);
            assert_eq!(p, keep);
        }
        // Half strength removes half the excess.
        let mut half = [40u8, 140, 40];
        Despill::new([30, 200, 40], 50.0).unwrap().apply(&mut half);
        assert!(half[1] > half[0] && half[1] < 140, "{half:?}");

        // Blue too; low-saturation or reddish backgrounds don't.
        assert_eq!(Despill::new([20, 60, 210], 100.0).map(|d| d.channel), Some(2));
        assert!(Despill::new([150, 190, 150], 100.0).is_none(), "light green wall");
        assert!(Despill::new([210, 120, 60], 100.0).is_none(), "orange background");
        assert!(Despill::new([30, 200, 40], 0.0).is_none());
    }

    #[test]
    fn rmbg_pre_and_post_processing() {
        let (t, shape) = ModelKind::Rmbg.preprocess(&RgbImage::from_pixel(64, 48, image::Rgb([255, 0, 128])));
        assert_eq!(shape, [1, 3, 1024, 1024]);
        let plane = 1024 * 1024;
        assert!((t[0] - 0.5).abs() < 1e-3 && (t[plane] + 0.5).abs() < 1e-3 && t[2 * plane].abs() < 0.01);
        // Output stretched between the frame's min and max.
        let mut out = vec![0.1f32; plane];
        out[..plane / 2].fill(0.9);
        let m = ModelKind::Rmbg.postprocess(&out, (1024, 1024), 1024, 1024);
        assert_eq!((m.get_pixel(0, 0)[0], m.get_pixel(0, 1023)[0]), (255, 0));
        // No subject: nothing is stretched.
        let m = ModelKind::Rmbg.postprocess(&vec![0.02; plane], (1024, 1024), 64, 48);
        assert!(m.pixels().all(|p| p[0] == 5));
    }

    /// Real masking run with an ONNX model, timed:
    /// `SPLATFORGE_TEST_MODEL=rmbg14.onnx SPLATFORGE_TEST_FRAMES=<folder>
    ///  cargo test --release onnx_model_on_frames -- --ignored --nocapture`
    #[cfg(any(windows, all(target_os = "macos", target_arch = "aarch64")))]
    #[test]
    #[ignore]
    fn onnx_model_on_frames() {
        let model = PathBuf::from(std::env::var("SPLATFORGE_TEST_MODEL").expect("SPLATFORGE_TEST_MODEL"));
        let frames = PathBuf::from(std::env::var("SPLATFORGE_TEST_FRAMES").expect("SPLATFORGE_TEST_FRAMES"));
        let kind = match model.file_name().unwrap().to_string_lossy() {
            n if n.contains("rmbg") => ModelKind::Rmbg,
            n if n.contains("modnet") => ModelKind::Modnet,
            _ => ModelKind::BiRefNet,
        };
        let out = frames.with_file_name("masks_test");
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let t0 = std::time::Instant::now();
        let backend = onnx::run(&model, kind, &frames, &out, &CancellationToken::new(), &tx).unwrap();
        let n = list_images(&out).unwrap().len();
        while let Ok(Msg::Log(l)) = rx.try_recv() {
            println!("{l}");
        }
        println!("{backend}: {n} frames in {:.1} s ({:.0} ms/frame) → {}", t0.elapsed().as_secs_f32(),
            t0.elapsed().as_secs_f32() * 1000.0 / n as f32, out.display());
    }

    /// Background color from the mask, and dataset with despill applied.
    #[test]
    fn background_color_from_masks() {
        let dir = std::env::temp_dir().join(format!("splatforge_bgcolor_{}", std::process::id()));
        let (images, masks) = (dir.join("images"), dir.join("masks"));
        std::fs::create_dir_all(&images).unwrap();
        std::fs::create_dir_all(&masks).unwrap();
        let mut img = RgbImage::from_pixel(64, 64, image::Rgb([20, 190, 40]));
        let mut mask = GrayImage::new(64, 64);
        for y in 16..48 {
            for x in 16..48 {
                img.put_pixel(x, y, image::Rgb([90, 140, 90]));
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }
        img.save(images.join("00000.png")).unwrap();
        mask.save(masks.join("00000.png")).unwrap();
        let key = background_color(&[images.join("00000.png")], &masks).unwrap();
        assert_eq!(key, Some([20, 190, 40]));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
