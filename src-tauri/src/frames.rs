//! Step 1: extract candidate frames with ffmpeg (hardware decoding) and pick
//! the sharpest ones, evenly spread along the orbit. Also accepts
//! pre-extracted frames (loose images or a folder), with the same selection.

use std::cmp::Ordering;
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
    /// How many frames go into the reconstruction.
    pub target_frames: u32,
    /// Maximum long side of the frames (px).
    pub max_side: u32,
    /// Candidates extracted per final frame; the sharpest in each window wins.
    pub oversample: u32,
    pub trim_start: f64,
    /// `None` = until the end of the video.
    pub trim_end: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameInfo {
    pub path: String,
    pub time: f64,
    pub sharpness: f32,
    /// Source image name, when the frames came from images (then `time`
    /// doesn't apply).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Manually rejected: the image lives in `rejected/`, outside the dataset.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rejected: bool,
}

/// Pre-extracted frames, in orbit order.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageSetInfo {
    pub paths: Vec<String>,
    /// Image folder: names the project.
    pub dir: String,
    /// Of the first image, already rotated per EXIF.
    pub width: u32,
    pub height: u32,
}

pub async fn probe(ffprobe: &Path, video: &Path) -> Result<VideoInfo> {
    let mut cmd = Command::new(ffprobe);
    cmd.args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=width,height,avg_frame_rate,codec_name:format=duration"])
        .args(["-of", "json"])
        .arg(video);
    let out = run_capture(cmd).await?;
    let json: serde_json::Value = serde_json::from_str(&out).context("unexpected ffprobe output")?;
    let stream = json["streams"]
        .get(0)
        .context("the file has no video track")?;
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

/// Extracts candidates into `raw_dir`. Returns the sampling fps used.
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
        bail!("selected video range is empty");
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
        // VideoToolbox on macOS; on Windows/Linux ffmpeg picks one
        // (D3D11VA, CUDA, VAAPI…) and falls back to software if none works.
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

/// Variance of the Laplacian on a downscaled grayscale copy: a classic,
/// cheap focus/sharpness metric.
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

/// Picks the sharpest frame in each window and writes it to `images_dir` with
/// sequential numbering (order matters for circular matching).
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
        bail!("ffmpeg produced no frames");
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

    if images_dir.exists() {
        std::fs::remove_dir_all(images_dir)?;
    }
    std::fs::create_dir_all(images_dir)?;

    let mut out = Vec::new();
    for best in sharpest_per_window(&scores, opts.target_frames) {
        let dest = images_dir.join(format!("{:05}.jpg", out.len()));
        std::fs::rename(&candidates[best], &dest)?;
        out.push(FrameInfo {
            path: dest.to_string_lossy().into_owned(),
            time: opts.trim_start + best as f64 / sample_fps,
            sharpness: scores[best],
            source: None,
            rejected: false,
        });
    }
    std::fs::remove_dir_all(raw_dir)?;
    Ok(out)
}

/// Splits the candidates into `target` consecutive windows and returns the
/// index of the sharpest in each (last one on ties).
fn sharpest_per_window(scores: &[f32], target: u32) -> Vec<usize> {
    let total = scores.len();
    let target = (target as usize).clamp(1, total.max(1));
    (0..target.min(total))
        .map(|w| {
            let lo = w * total / target;
            let hi = ((w + 1) * total / target).max(lo + 1);
            (lo..hi).max_by(|&a, &b| scores[a].total_cmp(&scores[b])).unwrap()
        })
        .collect()
}

/// Extensions accepted as pre-extracted frames.
pub const IMAGE_EXT: &[&str] = &["jpg", "jpeg", "png", "webp"];

fn is_image(p: &Path) -> bool {
    let name = p.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
    // `._foo.jpg`: macOS metadata on exFAT/FAT drives, not images.
    !name.starts_with('.')
        && p.extension().is_some_and(|e| IMAGE_EXT.contains(&e.to_string_lossy().to_lowercase().as_str()))
}

/// Natural order ("frame2" before "frame10"), case-insensitive.
fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (mut a, mut b) = (a.chars().peekable(), b.chars().peekable());
    loop {
        match (a.peek().copied(), b.peek().copied()) {
            (None, None) => return Ordering::Equal,
            (None, _) => return Ordering::Less,
            (_, None) => return Ordering::Greater,
            (Some(x), Some(y)) if x.is_ascii_digit() && y.is_ascii_digit() => {
                let take = |it: &mut std::iter::Peekable<std::str::Chars>| {
                    let mut d = String::new();
                    while let Some(c) = it.next_if(|c| c.is_ascii_digit()) {
                        d.push(c);
                    }
                    d
                };
                let (da, db) = (take(&mut a), take(&mut b));
                let (ta, tb) = (da.trim_start_matches('0'), db.trim_start_matches('0'));
                let ord = ta.len().cmp(&tb.len()).then_with(|| ta.cmp(tb)).then_with(|| da.len().cmp(&db.len()));
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            (Some(x), Some(y)) => {
                let ord = x.to_lowercase().cmp(y.to_lowercase());
                if ord != Ordering::Equal {
                    return ord;
                }
                a.next();
                b.next();
            }
        }
    }
}

/// Image decoder plus the rotation requested by EXIF (phone photos).
fn open_oriented(p: &Path) -> Result<(impl image::ImageDecoder, image::metadata::Orientation)> {
    use image::ImageDecoder;
    let mut dec = image::ImageReader::open(p)?.with_guessed_format()?.into_decoder()?;
    let orientation = dec.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    Ok((dec, orientation))
}

fn oriented_size(p: &Path) -> Result<(u32, u32)> {
    use image::metadata::Orientation::*;
    use image::ImageDecoder;
    let (dec, o) = open_oriented(p)?;
    let (w, h) = dec.dimensions();
    Ok(if matches!(o, Rotate90 | Rotate270 | Rotate90FlipH | Rotate270FlipH) { (h, w) } else { (w, h) })
}

/// Loose images and folders (top level only) → list in name order.
pub fn probe_images(paths: &[PathBuf]) -> Result<ImageSetInfo> {
    let mut files = Vec::new();
    for p in paths {
        if p.is_dir() {
            for e in std::fs::read_dir(p).with_context(|| format!("could not read folder {}", p.display()))? {
                let f = e?.path();
                if f.is_file() && is_image(&f) {
                    files.push(f);
                }
            }
        } else if p.is_file() && is_image(p) {
            files.push(p.clone());
        }
    }
    files.sort_by(|a, b| natural_cmp(&a.to_string_lossy(), &b.to_string_lossy()));
    files.dedup();
    let Some(first) = files.first() else {
        bail!("no images ({}) found", IMAGE_EXT.join(", "));
    };
    let (width, height) = oriented_size(first).with_context(|| format!("could not read {}", first.display()))?;
    let dir = match paths {
        [d] if d.is_dir() => d.clone(),
        _ => first.parent().map(Path::to_path_buf).unwrap_or_default(),
    };
    Ok(ImageSetInfo {
        paths: files.iter().map(|f| f.to_string_lossy().into_owned()).collect(),
        dir: dir.to_string_lossy().into_owned(),
        width,
        height,
    })
}

/// Frames from images: the sharpest in each window (as with video), rotated
/// per EXIF and downscaled to fit `max_side`, becomes 00000.jpg, …
pub fn from_images(
    sources: &[PathBuf],
    images_dir: &Path,
    opts: &FrameOptions,
    on_progress: impl Fn(f32) + Sync,
) -> Result<Vec<FrameInfo>> {
    if sources.is_empty() {
        bail!("no images to use as frames");
    }
    // The output folder is wiped before writing.
    if let Ok(out) = images_dir.canonicalize() {
        if sources.iter().any(|s| s.canonicalize().is_ok_and(|s| s.starts_with(&out))) {
            bail!("the images are inside the project folder ({}); choose another project folder", out.display());
        }
    }

    // Half the progress scores sharpness of all images, half writes the picked ones.
    let done = std::sync::atomic::AtomicUsize::new(0);
    let tick = |base: f32, total: usize| {
        let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if d % 4 == 0 || d == total {
            on_progress(base + 0.5 * d as f32 / total as f32);
        }
    };
    let scores: Vec<f32> = sources
        .par_iter()
        .map(|p| {
            let s = sharpness(p).with_context(|| format!("could not read {}", p.display()));
            tick(0.0, sources.len());
            s
        })
        .collect::<Result<_>>()?;
    done.store(0, std::sync::atomic::Ordering::Relaxed);
    let picked = sharpest_per_window(&scores, opts.target_frames);

    if images_dir.exists() {
        std::fs::remove_dir_all(images_dir)?;
    }
    std::fs::create_dir_all(images_dir)?;
    let max = opts.max_side.max(1);
    let sizes: Vec<(u32, u32)> = picked
        .par_iter()
        .enumerate()
        .map(|(i, &k)| {
            let src = &sources[k];
            let size = write_frame(src, &images_dir.join(format!("{i:05}.jpg")), max)
                .with_context(|| format!("could not convert {}", src.display()));
            tick(0.5, picked.len());
            size
        })
        .collect::<Result<_>>()?;

    // COLMAP and Brush use a single camera for all frames.
    if let Some(i) = sizes.iter().position(|s| *s != sizes[0]) {
        bail!(
            "the images have different sizes ({}: {}×{}, {}: {}×{}); use images from a single camera",
            name_of(&sources[picked[0]]),
            sizes[0].0,
            sizes[0].1,
            name_of(&sources[picked[i]]),
            sizes[i].0,
            sizes[i].1
        );
    }

    Ok(picked
        .iter()
        .enumerate()
        .map(|(i, &k)| FrameInfo {
            path: images_dir.join(format!("{i:05}.jpg")).to_string_lossy().into_owned(),
            time: 0.0,
            sharpness: scores[k],
            source: Some(name_of(&sources[k])),
            rejected: false,
        })
        .collect())
}

fn name_of(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Writes a JPEG frame; returns the final size. A JPEG that already fits and
/// needs no rotation is copied without re-encoding.
fn write_frame(src: &Path, dest: &Path, max_side: u32) -> Result<(u32, u32)> {
    use image::metadata::Orientation::NoTransforms;
    use image::ImageDecoder;
    let (dec, orientation) = open_oriented(src)?;
    let (w, h) = dec.dimensions();
    let jpeg = src.extension().is_some_and(|e| matches!(e.to_string_lossy().to_lowercase().as_str(), "jpg" | "jpeg"));
    if jpeg && orientation == NoTransforms && w.max(h) <= max_side {
        std::fs::copy(src, dest)?;
        return Ok((w, h));
    }
    let mut img = image::DynamicImage::from_decoder(dec)?;
    img.apply_orientation(orientation);
    let (w, h) = (img.width(), img.height());
    let scale = (max_side as f64 / w.max(h) as f64).min(1.0);
    if scale < 1.0 {
        let (nw, nh) = (((w as f64 * scale).round() as u32).max(1), ((h as f64 * scale).round() as u32).max(1));
        img = img.resize_exact(nw, nh, image::imageops::FilterType::Lanczos3);
    }
    let rgb = img.to_rgb8();
    let file = std::io::BufWriter::new(std::fs::File::create(dest)?);
    image::codecs::jpeg::JpegEncoder::new_with_quality(file, 95).encode_image(&rgb)?;
    Ok(rgb.dimensions())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_order() {
        let mut v = vec!["f10.png", "F2.png", "f1.png", "f02.png", "g.png", "f001.png"];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(v, ["f1.png", "f001.png", "F2.png", "f02.png", "f10.png", "g.png"]);
    }

    #[test]
    fn windows_pick_sharpest() {
        let s = [1.0, 3.0, 2.0, 5.0, 4.0, 0.0];
        assert_eq!(sharpest_per_window(&s, 3), [1, 3, 4]);
        // Fewer images than the target: use them all.
        assert_eq!(sharpest_per_window(&s, 100), [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn images_become_frames() {
        let dir = std::env::temp_dir().join(format!("splatforge-frames-{}", std::process::id()));
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        for i in [1, 2, 10] {
            let img = image::RgbImage::from_fn(400, 200, |x, y| image::Rgb([((x * i) % 256) as u8, (y % 256) as u8, 0]));
            img.save(src.join(format!("frame{i}.png"))).unwrap();
        }
        std::fs::write(src.join("._frame3.png"), b"macOS junk").unwrap();
        let set = probe_images(&[src.clone()]).unwrap();
        assert_eq!((set.width, set.height, set.paths.len()), (400, 200, 3));
        assert!(set.paths[2].ends_with("frame10.png"));

        let opts = FrameOptions { target_frames: 10, max_side: 100, oversample: 1, trim_start: 0.0, trim_end: None };
        let paths: Vec<PathBuf> = set.paths.iter().map(PathBuf::from).collect();
        let out = from_images(&paths, &dir.join("images"), &opts, |_| {}).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[2].source.as_deref(), Some("frame10.png"));
        assert_eq!(image::image_dimensions(&out[0].path).unwrap(), (100, 50));

        // Different sizes can't form a single camera.
        image::RgbImage::new(300, 300).save(src.join("frame11.png")).unwrap();
        let set = probe_images(&[src.clone()]).unwrap();
        let paths: Vec<PathBuf> = set.paths.iter().map(PathBuf::from).collect();
        assert!(from_images(&paths, &dir.join("images"), &opts, |_| {}).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
