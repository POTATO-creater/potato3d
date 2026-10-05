//! Localiza os binários externos em macOS, Windows e Linux.
//!
//! Apps GUI não herdam o PATH do shell (no macOS) nem veem alterações de
//! PATH feitas depois do login (no Windows), então além do PATH procuramos
//! nos lugares onde os instaladores comuns colocam os executáveis, e na
//! pasta de ferramentas do próprio app (preenchida por `pnpm setup:tools`).

use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tools {
    pub ffmpeg: Option<PathBuf>,
    pub ffprobe: Option<PathBuf>,
    pub colmap: Option<PathBuf>,
    /// COLMAP compilado com CUDA: extração, matching e bundle adjustment na GPU.
    pub colmap_cuda: bool,
    pub brush: Option<PathBuf>,
    /// macOS: recortador Vision (sidecar `splatforge_mask`).
    pub masker: Option<PathBuf>,
    /// Windows: modelos de recorte baixados pelo instalador de ferramentas.
    pub birefnet_model: Option<PathBuf>,
    pub modnet_model: Option<PathBuf>,
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
            birefnet_model: model_if_windows(BIREFNET_FILE),
            modnet_model: model_if_windows(MODNET_FILE),
            os: std::env::consts::OS,
        }
    }

    pub fn require(path: &Option<PathBuf>, name: &str, hint: &str) -> anyhow::Result<PathBuf> {
        path.clone()
            .ok_or_else(|| anyhow::anyhow!("{name} não encontrado. {hint}"))
    }
}

/// Como instalar cada ferramenta neste sistema (mensagens de erro).
pub fn install_hint(tool: &str) -> &'static str {
    match (std::env::consts::OS, tool) {
        ("macos", "ffmpeg") => "Instale com: brew install ffmpeg",
        ("macos", "colmap") => "Instale com: brew install colmap",
        (_, "brush") => "Rode `pnpm setup:tools` para baixar o Brush.",
        ("windows", "model") => "Use \"Instalar ferramentas\" no app para baixar o modelo de recorte.",
        ("windows", _) => "Rode `pnpm setup:tools` para baixar as ferramentas.",
        (_, "ffmpeg") => "Instale o ffmpeg pelo gerenciador de pacotes do sistema.",
        _ => "Instale o COLMAP 4.x (https://colmap.github.io).",
    }
}

/// Pasta onde `pnpm setup:tools` instala as ferramentas baixadas.
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

/// Onde o instalador grava os modelos de recorte (Windows).
pub fn models_dir() -> Option<PathBuf> {
    tools_root().map(|r| r.join("models"))
}

fn model_if_windows(file: &str) -> Option<PathBuf> {
    if !cfg!(windows) {
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

/// Sidecars (`externalBin`) ficam junto do executável do app, em dev e no bundle.
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

/// A primeira linha de `colmap help` diz "with CUDA" ou "without GPU support".
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
