//! Project package (`.gorbit`): a ZIP with everything processed so far —
//! frames, masks, poses, splat, the source (video or images) and the session
//! settings — to move the project to another machine or save it mid-run.
//! Opened again, it behaves like a cached project: finished stages show up as
//! saved and work resumes where it stopped.
//!
//! The layout inside the ZIP mirrors the project folder (`Layout`), with the
//! `project.json` paths relative to the root; the same package opens on
//! desktop and web (web/src/host/package.ts).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::frames::FrameInfo;
use crate::pipeline::{Layout, ProjectState};

pub const EXT: &str = "gorbit";
const FORMAT: &str = "gausify-orbit-project";
const VERSION: u32 = 1;
const MANIFEST: &str = "manifest.json";
const STATE: &str = "project.json";
const MODEL_FILES: [&str; 3] = ["cameras.bin", "images.bin", "points3D.bin"];
/// Folders and files a package may contain; anything else is ignored (future
/// versions may add more).
const ALLOWED: [&str; 7] = ["dataset/images/", "dataset/sparse/0/", "rejected/", "masks/", "export/", "source/", "splat.ply"];

/// Stages to include besides the frames: only those the UI knows are finished
/// (a running job may be rewriting the others).
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct Include {
    pub masks: bool,
    pub poses: bool,
    pub splat: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Source {
    Video { path: String },
    Images { paths: Vec<String> },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackRequest {
    pub project_dir: String,
    pub dest: String,
    pub include: Include,
    /// Source video or images, when still available.
    #[serde(default)]
    pub source: Option<Source>,
    /// UI settings (quality, masking, trim): opaque here.
    #[serde(default)]
    pub session: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PackStats {
    pub path: String,
    pub bytes: u64,
    pub files: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Unpacked {
    pub project_dir: String,
    pub video: Option<String>,
    pub images: Option<Vec<String>>,
    pub session: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
enum PackedSource {
    Video { file: String },
    Images { files: Vec<String> },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Manifest {
    format: String,
    version: u32,
    app: String,
    /// Milliseconds since 1970.
    created_at: u64,
    #[serde(default)]
    source: Option<PackedSource>,
    #[serde(default)]
    session: Option<serde_json::Value>,
}

fn file_name(path: &Path) -> Result<String> {
    Ok(path
        .file_name()
        .with_context(|| format!("path has no file name: {}", path.display()))?
        .to_string_lossy()
        .into_owned())
}

/// Top-level files of a folder, sorted by name.
fn files_in(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
        .filter(|(n, _)| !n.starts_with('.'))
        .collect();
    out.sort();
    out
}

/// Only compress what shrinks: images and video are already compressed.
fn method_for(name: &str) -> CompressionMethod {
    match name.rsplit('.').next().map(str::to_ascii_lowercase).as_deref() {
        Some("json" | "bin" | "ply") => CompressionMethod::Deflated,
        _ => CompressionMethod::Stored,
    }
}

/// Writes the package to `dest` (through `.part`, so a half-written package
/// never ends up under the final name).
pub fn pack(req: &PackRequest, mut progress: impl FnMut(f32)) -> Result<PackStats> {
    let layout = Layout::new(&req.project_dir);
    let mut state = layout.load();
    if state.frames.is_empty() {
        bail!("the project has no frames to save yet");
    }
    let mut entries: Vec<(String, PathBuf)> = Vec::new();

    let mut frames = Vec::with_capacity(state.frames.len());
    for f in &state.frames {
        let src = PathBuf::from(&f.path);
        let name = file_name(&src)?;
        let rel = format!("{}/{name}", if f.rejected { "rejected" } else { "dataset/images" });
        entries.push((rel.clone(), src));
        frames.push(FrameInfo { path: rel, ..f.clone() });
    }
    state.frames = frames;

    if req.include.masks && state.masks.is_some() && layout.masks().is_dir() {
        entries.extend(files_in(&layout.masks()).into_iter().filter(|(n, _)| n.ends_with(".png")).map(|(n, p)| (format!("masks/{n}"), p)));
        state.masks_dir = Some("masks".into());
    } else {
        state.masks = None;
        state.masks_dir = None;
    }

    let poses = req.include.poses && state.poses.is_some() && layout.model().join("images.bin").is_file();
    if poses {
        for f in MODEL_FILES {
            let p = layout.model().join(f);
            if p.is_file() {
                entries.push((format!("dataset/sparse/0/{f}"), p));
            }
        }
    } else {
        state.poses = None;
    }

    // The splat depends on the poses it was trained with.
    if poses && req.include.splat && state.final_ply.is_some() && layout.final_ply().is_file() {
        entries.push(("splat.ply".into(), layout.final_ply()));
        state.final_ply = Some("splat.ply".into());
        state.last_export = state.last_export.take().and_then(|mut e| {
            let p = PathBuf::from(&e.path);
            let rel = format!("export/{}", file_name(&p).ok()?);
            p.is_file().then(|| entries.push((rel.clone(), p)))?;
            e.path = rel;
            Some(e)
        });
    } else {
        state.final_ply = None;
        state.last_psnr = None;
        state.last_export = None;
    }

    let source = match &req.source {
        Some(Source::Video { path }) if Path::new(path).is_file() => {
            let rel = format!("source/{}", file_name(Path::new(path))?);
            entries.push((rel.clone(), PathBuf::from(path)));
            state.video_path = Some(rel.clone());
            Some(PackedSource::Video { file: rel })
        }
        Some(Source::Images { paths }) => {
            let mut files = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for (i, p) in paths.iter().map(Path::new).filter(|p| p.is_file()).enumerate() {
                let mut name = file_name(p)?;
                // Loose images from different folders may share a name.
                if !seen.insert(name.clone()) {
                    name = format!("{i:05}_{name}");
                    seen.insert(name.clone());
                }
                let rel = format!("source/images/{name}");
                entries.push((rel.clone(), p.to_path_buf()));
                files.push(rel);
            }
            if files.is_empty() {
                None
            } else {
                state.video_path = Some("source/images".into());
                Some(PackedSource::Images { files })
            }
        }
        _ => None,
    };

    let manifest = Manifest {
        format: FORMAT.into(),
        version: VERSION,
        app: env!("CARGO_PKG_VERSION").into(),
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
        source,
        session: req.session.clone(),
    };

    let mut dest = PathBuf::from(&req.dest);
    if !dest.extension().is_some_and(|e| e.eq_ignore_ascii_case(EXT)) {
        dest.as_mut_os_string().push(format!(".{EXT}"));
    }
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let result = write_zip(&part, &manifest, &state, &entries, &mut progress);
    if let Err(e) = result {
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }
    std::fs::rename(&part, &dest).with_context(|| format!("could not write {}", dest.display()))?;
    Ok(PackStats {
        bytes: std::fs::metadata(&dest)?.len(),
        path: dest.to_string_lossy().into_owned(),
        files: entries.len() + 2,
    })
}

fn write_zip(
    out: &Path,
    manifest: &Manifest,
    state: &ProjectState,
    entries: &[(String, PathBuf)],
    progress: &mut impl FnMut(f32),
) -> Result<()> {
    let mut sizes = Vec::with_capacity(entries.len());
    for (_, p) in entries {
        sizes.push(std::fs::metadata(p).with_context(|| format!("project file is missing: {}", p.display()))?.len());
    }
    let total = sizes.iter().sum::<u64>().max(1);
    let mut zip = ZipWriter::new(BufWriter::new(File::create(out).with_context(|| format!("could not create {}", out.display()))?));
    let opts = |name: &str, size: u64| {
        SimpleFileOptions::default().compression_method(method_for(name)).large_file(size >= u32::MAX as u64)
    };
    zip.start_file(MANIFEST, opts(MANIFEST, 0))?;
    zip.write_all(&serde_json::to_vec_pretty(manifest)?)?;
    zip.start_file(STATE, opts(STATE, 0))?;
    zip.write_all(&serde_json::to_vec_pretty(state)?)?;

    let mut done = 0u64;
    let mut buf = vec![0u8; 1 << 20];
    progress(0.0);
    for ((name, path), size) in entries.iter().zip(sizes) {
        zip.start_file(name.as_str(), opts(name, size))?;
        let mut f = File::open(path).with_context(|| format!("could not read {}", path.display()))?;
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            zip.write_all(&buf[..n])?;
            done += n as u64;
            progress(done as f32 / total as f32 * 0.99);
        }
    }
    zip.finish()?.flush()?;
    progress(1.0);
    Ok(())
}

/// Relative path inside the package → safe components (no `..`, root or
/// drive).
fn safe_rel(rel: &str) -> Result<PathBuf> {
    let mut out = PathBuf::new();
    for part in rel.split('/') {
        if part.is_empty() || part == "." || part == ".." || part.contains(['\\', ':']) {
            bail!("invalid path in package: {rel}");
        }
        out.push(part);
    }
    Ok(out)
}

/// `<folder>/<name>`, or `<name> (2)`, `(3)`… if it already exists.
fn free_dir(parent: &Path, name: &str) -> PathBuf {
    let first = parent.join(name);
    if !first.exists() {
        return first;
    }
    (2..).map(|i| parent.join(format!("{name} ({i})"))).find(|p| !p.exists()).unwrap()
}

fn read_json<T: serde::de::DeserializeOwned>(zip: &mut ZipArchive<BufReader<File>>, name: &str) -> Result<T> {
    let mut f = zip.by_name(name)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(serde_json::from_slice(&buf)?)
}

/// Extracts the package into a new folder next to it (named after the file)
/// and returns the project ready to open.
pub fn unpack(package: &Path, mut progress: impl FnMut(f32)) -> Result<Unpacked> {
    let file = File::open(package).with_context(|| format!("could not open {}", package.display()))?;
    let mut zip = ZipArchive::new(BufReader::new(file)).context("the file is not a Gausify Orbit package")?;
    let manifest: Manifest = read_json(&mut zip, MANIFEST).context("the file is not a Gausify Orbit package")?;
    if manifest.format != FORMAT {
        bail!("the file is not a Gausify Orbit package");
    }
    if manifest.version > VERSION {
        bail!("the package was saved by a newer version of Gausify Orbit ({}); update the app", manifest.app);
    }
    let state: ProjectState = read_json(&mut zip, STATE).context("package has no valid project.json")?;

    let parent = package.parent().unwrap_or(Path::new("."));
    let stem = package.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "project".into());
    let root = free_dir(parent, &stem);
    std::fs::create_dir_all(&root).with_context(|| format!("could not create {}", root.display()))?;
    match extract(&mut zip, &root, &manifest, state, &mut progress) {
        Ok(r) => Ok(r),
        Err(e) => {
            let _ = std::fs::remove_dir_all(&root);
            Err(e)
        }
    }
}

fn extract(
    zip: &mut ZipArchive<BufReader<File>>,
    root: &Path,
    manifest: &Manifest,
    mut state: ProjectState,
    progress: &mut impl FnMut(f32),
) -> Result<Unpacked> {
    let mut names = Vec::new();
    let mut total = 0u64;
    for i in 0..zip.len() {
        let f = zip.by_index_raw(i)?;
        let name = f.name().to_string();
        if f.is_file() && ALLOWED.iter().any(|a| if a.ends_with('/') { name.starts_with(a) } else { name == *a }) {
            total += f.size();
            names.push((i, safe_rel(&name)?));
        }
    }
    let total = total.max(1);
    let mut done = 0u64;
    let mut buf = vec![0u8; 1 << 20];
    progress(0.0);
    for (i, rel) in names {
        let mut f = zip.by_index(i)?;
        let out = root.join(rel);
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut w = BufWriter::new(File::create(&out).with_context(|| format!("could not create {}", out.display()))?);
        loop {
            let n = f.read(&mut buf).with_context(|| format!("package corrupted at {}", f.name()))?;
            if n == 0 {
                break;
            }
            w.write_all(&buf[..n])?;
            done += n as u64;
            progress(done as f32 / total as f32 * 0.99);
        }
        w.flush()?;
    }

    // Package-relative paths → absolute paths on this machine.
    let abs = |rel: &str| -> Result<String> { Ok(root.join(safe_rel(rel)?).to_string_lossy().into_owned()) };
    let layout = Layout::new(root);
    for f in &mut state.frames {
        f.path = abs(&f.path)?;
    }
    if state.masks.is_some() && layout.masks().is_dir() {
        state.masks_dir = Some(layout.masks().to_string_lossy().into_owned());
    } else {
        state.masks = None;
        state.masks_dir = None;
    }
    if state.poses.is_some() && !layout.model().join("images.bin").is_file() {
        state.poses = None;
    }
    if state.poses.is_some() && state.final_ply.is_some() && layout.final_ply().is_file() {
        state.final_ply = Some(layout.final_ply().to_string_lossy().into_owned());
        if let Some(e) = &mut state.last_export {
            e.path = abs(&e.path)?;
        }
        state.last_export = state.last_export.filter(|e| Path::new(&e.path).is_file());
    } else {
        state.final_ply = None;
        state.last_psnr = None;
        state.last_export = None;
    }
    let (video, images) = match &manifest.source {
        Some(PackedSource::Video { file }) => (Some(abs(file)?), None),
        Some(PackedSource::Images { files }) => (None, Some(files.iter().map(|f| abs(f)).collect::<Result<Vec<_>>>()?)),
        None => (None, None),
    };
    state.video_path = match (&video, &images) {
        (Some(v), _) => Some(v.clone()),
        (_, Some(_)) => Some(root.join("source").join("images").to_string_lossy().into_owned()),
        _ => state.video_path,
    };
    layout.save(&state)?;
    progress(1.0);
    Ok(Unpacked {
        project_dir: root.to_string_lossy().into_owned(),
        video,
        images,
        session: manifest.session.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colmap::ModelStats;
    use crate::masks::MaskStats;

    #[test]
    fn pack_and_unpack_round_trip() {
        let tmp = std::env::temp_dir().join("splatforge_package_test");
        let _ = std::fs::remove_dir_all(&tmp);
        let layout = Layout::new(tmp.join("proj"));
        std::fs::create_dir_all(layout.images()).unwrap();
        std::fs::create_dir_all(layout.rejected()).unwrap();
        std::fs::create_dir_all(layout.masks()).unwrap();
        std::fs::create_dir_all(layout.model()).unwrap();
        let mut frames = Vec::new();
        for i in 0..3 {
            let rejected = i == 2;
            let dir = if rejected { layout.rejected() } else { layout.images() };
            let path = dir.join(format!("{i:05}.jpg"));
            std::fs::write(&path, format!("frame {i}")).unwrap();
            std::fs::write(layout.masks().join(format!("{i:05}.png")), b"m").unwrap();
            frames.push(FrameInfo { path: path.to_string_lossy().into_owned(), time: i as f64, sharpness: 1.0, source: None, rejected });
        }
        for f in MODEL_FILES {
            std::fs::write(layout.model().join(f), vec![7u8; 5000]).unwrap();
        }
        std::fs::write(layout.final_ply(), b"ply").unwrap();
        let video = tmp.join("orbit.mp4");
        std::fs::write(&video, b"mp4").unwrap();
        let stats = ModelStats { registered: 2, total: 2, points: 10, reprojection_error: None, mapper: "t".into(), inferred: 0, refined: 0 };
        let masks = MaskStats { count: 3, no_subject: 0, mean_coverage: 0.5, backend: "t".into(), key_color: None };
        layout
            .save(&ProjectState {
                video_path: Some(video.to_string_lossy().into_owned()),
                frames,
                masks: Some(masks),
                masks_dir: Some(layout.masks().to_string_lossy().into_owned()),
                poses: Some(stats),
                final_ply: Some(layout.final_ply().to_string_lossy().into_owned()),
                last_psnr: Some(30.0),
                last_export: None,
            })
            .unwrap();

        let req = |include: Include, dest: &str| PackRequest {
            project_dir: layout.root.to_string_lossy().into_owned(),
            dest: tmp.join(dest).to_string_lossy().into_owned(),
            include,
            source: Some(Source::Video { path: video.to_string_lossy().into_owned() }),
            session: Some(serde_json::json!({ "trim": { "start": 1 } })),
        };

        // Everything finished.
        let all = Include { masks: true, poses: true, splat: true };
        let packed = pack(&req(all, "all"), |_| {}).unwrap();
        assert!(packed.path.ends_with("all.gorbit"));
        let u = unpack(Path::new(&packed.path), |_| {}).unwrap();
        assert_eq!(u.project_dir, tmp.join("all").to_string_lossy());
        assert_eq!(u.video.as_deref(), Some(tmp.join("all/source/orbit.mp4").to_string_lossy().as_ref()));
        assert_eq!(u.session.unwrap()["trim"]["start"], 1);
        let s = Layout::new(&u.project_dir).load();
        assert_eq!(s.frames.len(), 3);
        assert!(s.frames[2].rejected && s.frames[2].path.ends_with("rejected/00002.jpg"));
        assert_eq!(std::fs::read_to_string(&s.frames[1].path).unwrap(), "frame 1");
        assert!(s.masks.is_some() && s.poses.is_some() && s.last_psnr == Some(30.0));
        assert_eq!(std::fs::read(s.final_ply.unwrap()).unwrap(), b"ply");
        assert_eq!(std::fs::read(Layout::new(&u.project_dir).model().join("points3D.bin")).unwrap(), vec![7u8; 5000]);

        // Mid-training: the old splat is left out; opening again doesn't overwrite.
        let packed = pack(&req(Include { masks: true, poses: true, splat: false }, "all.gorbit"), |_| {}).unwrap();
        let u = unpack(Path::new(&packed.path), |_| {}).unwrap();
        assert!(u.project_dir.ends_with("all (2)"));
        let s = Layout::new(&u.project_dir).load();
        assert!(s.poses.is_some() && s.final_ply.is_none() && s.last_psnr.is_none());
        assert!(!Path::new(&u.project_dir).join("splat.ply").exists());

        // Recomputing poses: they and the splat are left out.
        let packed = pack(&req(Include { masks: true, poses: false, splat: true }, "no_poses"), |_| {}).unwrap();
        let s = Layout::new(&unpack(Path::new(&packed.path), |_| {}).unwrap().project_dir).load();
        assert!(s.masks.is_some() && s.poses.is_none() && s.final_ply.is_none());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn rejects_unsafe_paths() {
        assert!(safe_rel("../x").is_err());
        assert!(safe_rel("/x").is_err());
        assert!(safe_rel("a/../../x").is_err());
        assert!(safe_rel("C:/x").is_err());
        assert!(safe_rel("dataset/images/00001.jpg").is_ok());
    }
}
