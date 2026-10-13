//! Locates the external binaries on macOS, Windows and Linux.
//!
//! GUI apps don't inherit the shell's PATH (on macOS) nor see PATH changes
//! made after login (on Windows), so besides PATH we look where the common
//! installers put executables, and in the app's own tools folder (filled by
//! `pnpm setup:tools`).

use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tools {
    pub ffmpeg: Option<PathBuf>,
    pub ffprobe: Option<PathBuf>,
    pub colmap: Option<PathBuf>,
    /// COLMAP built with CUDA: extraction, matching and bundle adjustment on the GPU.
    pub colmap_cuda: bool,
    pub brush: Option<PathBuf>,
    /// macOS: Vision-based masker (sidecar `splatforge_mask`).
    pub masker: Option<PathBuf>,
    /// Windows: masking models downloaded by the tools installer.
    pub birefnet_model: Option<PathBuf>,
    pub modnet_model: Option<PathBuf>,
    /// Windows and macOS (Apple Silicon): RMBG-1.4, downloaded on demand.
    pub rmbg_model: Option<PathBuf>,
    /// "macos" | "windows" | "linux"
    pub os: &'static str,
}

impl Tools {
    pub fn detect() -> Self {
        let colmap = find("colmap", "SPLATFORGE_COLMAP");
        let colmap_cuda = colmap.as_deref().is_some_and(colmap_has_cuda);
        Self {
            ffmpeg: find("ffmpeg", "SPLATFORGE_FFMPEG"),
            ffprobe: find("ffprobe", "SPLATFORGE_FFPROBE"),
            colmap,
            colmap_cuda,
            brush: find_sidecar("brush_app", "SPLATFORGE_BRUSH"),
            masker: if cfg!(target_os = "macos") { find_sidecar("splatforge_mask", "SPLATFORGE_MASKER") } else { None },
            birefnet_model: model_if(cfg!(windows), BIREFNET_FILE),
            modnet_model: model_if(cfg!(windows), MODNET_FILE),
            rmbg_model: model_if(ONNX_AVAILABLE, RMBG_FILE),
            os: std::env::consts::OS,
        }
    }

    pub fn require(path: &Option<PathBuf>, name: &str, hint: &str) -> anyhow::Result<PathBuf> {
        path.clone()
            .ok_or_else(|| anyhow::anyhow!("{name} not found. {hint}"))
    }
}

/// How to install each tool on this system (for error messages).
pub fn install_hint(tool: &str) -> &'static str {
    match (std::env::consts::OS, tool) {
        ("macos", "ffmpeg") => "Install with: brew install ffmpeg",
        ("macos", "colmap") => "Install with: brew install colmap",
        (_, "brush") => "Run `pnpm setup:tools` to download Brush.",
        ("windows" | "macos", "model") => "Use \"Install tools\" in the app to download the masking model.",
        ("windows", _) => "Run `pnpm setup:tools` to download the tools.",
        (_, "ffmpeg") => "Install ffmpeg with your system package manager.",
        _ => "Install COLMAP 4.x (https://colmap.github.io).",
    }
}

/// Folder where `pnpm setup:tools` installs the downloaded tools.
pub fn tools_root() -> Option<PathBuf> {
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    };
    base.map(|b| b.join("SplatForge").join("tools"))
}

pub const BIREFNET_FILE: &str = "birefnet_lite.onnx";
pub const MODNET_FILE: &str = "modnet.onnx";
pub const RMBG_FILE: &str = "rmbg14.onnx";

/// This build ships ONNX Runtime (Windows; macOS on Apple Silicon).
pub const ONNX_AVAILABLE: bool = cfg!(any(windows, all(target_os = "macos", target_arch = "aarch64")));

/// Where the installer saves the masking models.
pub fn models_dir() -> Option<PathBuf> {
    tools_root().map(|r| r.join("models"))
}

fn model_if(supported: bool, file: &str) -> Option<PathBuf> {
    if !supported {
        return None;
    }
    models_dir().map(|d| d.join(file)).filter(|p| p.is_file())
}

fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(root) = tools_root() {
        dirs.push(root.join("ffmpeg").join("bin"));
        dirs.push(root.join("colmap").join("bin"));
    }
    if cfg!(windows) {
        let env = |k: &str| std::env::var_os(k).map(PathBuf::from);
        if let Some(local) = env("LOCALAPPDATA") {
            dirs.push(local.join("Microsoft").join("WinGet").join("Links"));
        }
        if let Some(home) = env("USERPROFILE") {
            dirs.push(home.join("scoop").join("shims"));
        }
        if let Some(pd) = env("ProgramData") {
            dirs.push(pd.join("chocolatey").join("bin"));
        }
        if let Some(pf) = env("ProgramFiles") {
            dirs.push(pf.join("COLMAP").join("bin"));
            dirs.push(pf.join("ffmpeg").join("bin"));
        }
    } else {
        dirs.extend(["/opt/homebrew/bin", "/usr/local/bin"].map(PathBuf::from));
    }
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    dirs
}

#[cfg(unix)]
fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    p.metadata()
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(p: &Path) -> bool {
    p.is_file()
}

fn exe(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

fn find(name: &str, env_override: &str) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(env_override).map(PathBuf::from) {
        if is_executable(&p) {
            return Some(p);
        }
    }
    let file = exe(name);
    search_dirs()
        .into_iter()
        .map(|d| d.join(&file))
        .find(|p| is_executable(p))
}

/// Sidecars (`externalBin`) sit next to the app executable, both in dev and in the bundle.
fn find_sidecar(name: &str, env_override: &str) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(env_override).map(PathBuf::from) {
        if is_executable(&p) {
            return Some(p);
        }
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        let p = dir.join(exe(name));
        if is_executable(&p) {
            return Some(p);
        }
    }
    find(name, env_override)
}

/// The first line of `colmap help` says "with CUDA" or "without GPU support".
fn colmap_has_cuda(colmap: &Path) -> bool {
    let mut cmd = std::process::Command::new(colmap);
    cmd.arg("help").stdin(Stdio::null());
    crate::proc::hide_console_std(&mut cmd);
    cmd.output()
        .map(|o| {
            let text = [o.stdout, o.stderr].concat();
            String::from_utf8_lossy(&text).contains("with CUDA")
        })
        .unwrap_or(false)
}
