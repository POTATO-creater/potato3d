//! Instalação das ferramentas externas pelo próprio app (primeiro uso).
//!
//! No Windows baixa o COLMAP (com CUDA se houver GPU NVIDIA) e o ffmpeg para
//! a pasta de ferramentas do app, conferindo o SHA-256. O Brush já vem
//! dentro do instalador (sidecar). No macOS/Linux as ferramentas vêm do
//! gerenciador de pacotes, então o plano é vazio.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::proc::Cancelled;
use crate::tools::{tools_root, Tools};

const COLMAP_TAG: &str = "4.2.1";
// SHA-256 publicados na release 4.2.1 do COLMAP.
const COLMAP_CUDA: (&str, &str, u64) = (
    "colmap-x64-windows-cuda.zip",
    "e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881",
    414_666_626,
);
const COLMAP_CPU: (&str, &str, u64) = (
    "colmap-x64-windows-nocuda.zip",
    "c493d88cb4a43f21afba42cb55bf786e5434a0420e73db0ba1454c5d8b3686ae",
    127_731_595,
);
/// Builds estáveis do ffmpeg (LGPL) do BtbN. A tag "latest" é refeita
/// diariamente, então o arquivo e o hash são resolvidos pela API do GitHub.
const FFMPEG_RELEASE: &str = "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Package {
    /// Subpasta em `tools_root()`; o pacote deve conter `bin/<exe>`.
    pub id: String,
    pub label: String,
    pub file: String,
    pub url: String,
    pub sha256: String,
    pub size: u64,
    pub exe: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Plan {
    pub packages: Vec<Package>,
    pub total_bytes: u64,
    pub cuda: bool,
    /// Explicação quando não há nada a baixar por aqui (macOS/Linux).
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum InstallEvent {
    Started { id: String, label: String, total: u64 },
    Progress { id: String, done: u64, total: u64 },
    Extracting { id: String },
    Done { id: String },
    Finished,
    Failed { error: String },
    Cancelled,
}

fn has_nvidia_gpu() -> bool {
    let mut cmd = std::process::Command::new("nvidia-smi");
    cmd.arg("-L")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    crate::proc::hide_console_std(&mut cmd);
    cmd.status().is_ok_and(|s| s.success())
}

fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent(concat!("SplatForge/", env!("CARGO_PKG_VERSION")))
        .build()?)
}

async fn ffmpeg_package(client: &reqwest::Client) -> Result<Package> {
    let release: serde_json::Value = client
        .get(FFMPEG_RELEASE)
        .send()
        .await?
        .error_for_status()
        .context("não foi possível consultar a release do ffmpeg no GitHub")?
        .json()
        .await?;
    // ffmpeg-n<versão>-latest-win64-lgpl-<versão>.zip; pega a maior versão.
    let mut best: Option<(Vec<u32>, &serde_json::Value)> = None;
    for a in release["assets"].as_array().into_iter().flatten() {
        let name = a["name"].as_str().unwrap_or_default();
        let Some(ver) = name
            .strip_prefix("ffmpeg-n")
            .and_then(|r| r.split_once("-latest-win64-lgpl-"))
            .filter(|(_, tail)| tail.ends_with(".zip") && !tail.contains("shared"))
            .map(|(v, _)| v.split('.').filter_map(|p| p.parse().ok()).collect::<Vec<u32>>())
        else {
            continue;
        };
        if best.as_ref().is_none_or(|(b, _)| ver > *b) {
            best = Some((ver, a));
        }
    }
    let (_, a) = best.context("nenhum build do ffmpeg para Windows encontrado")?;
    let sha256 = a["digest"]
        .as_str()
        .and_then(|d| d.strip_prefix("sha256:"))
        .context("o GitHub não publicou o SHA-256 do ffmpeg; abortando por segurança")?;
    Ok(Package {
        id: "ffmpeg".into(),
        label: "ffmpeg".into(),
        file: a["name"].as_str().unwrap_or_default().into(),
        url: a["browser_download_url"].as_str().unwrap_or_default().into(),
        sha256: sha256.into(),
        size: a["size"].as_u64().unwrap_or(0),
        exe: "ffmpeg.exe".into(),
    })
}

/// O que falta instalar neste computador.
pub async fn plan() -> Result<Plan> {
    if !cfg!(windows) {
        let note = if cfg!(target_os = "macos") {
            "No macOS instale pelo Homebrew: brew install ffmpeg colmap"
        } else {
            "No Linux instale ffmpeg e COLMAP 4.x pelo gerenciador de pacotes."
        };
        return Ok(Plan { packages: vec![], total_bytes: 0, cuda: false, note: Some(note.into()) });
    }
    let tools = Tools::detect();
    let cuda = has_nvidia_gpu();
    let client = client()?;
    let mut packages = Vec::new();
    if tools.colmap.is_none() {
        let (file, sha, size) = if cuda { COLMAP_CUDA } else { COLMAP_CPU };
        packages.push(Package {
            id: "colmap".into(),
            label: format!("COLMAP {COLMAP_TAG}{}", if cuda { " (CUDA)" } else { "" }),
            file: file.into(),
            url: format!("https://github.com/colmap/colmap/releases/download/{COLMAP_TAG}/{file}"),
            sha256: sha.into(),
            size,
            exe: "colmap.exe".into(),
        });
    }
    if tools.ffmpeg.is_none() || tools.ffprobe.is_none() {
        packages.push(ffmpeg_package(&client).await?);
    }
    let total_bytes = packages.iter().map(|p| p.size).sum();
    Ok(Plan { packages, total_bytes, cuda, note: None })
}

pub async fn install(plan: Plan, cancel: CancellationToken, emit: impl Fn(InstallEvent) + Send + Sync + 'static) {
    let emit = std::sync::Arc::new(emit);
    let result = async {
        let root = tools_root().context("pasta de dados do usuário não encontrada")?;
        std::fs::create_dir_all(&root)?;
        let client = client()?;
        for pkg in &plan.packages {
            install_one(&client, &root, pkg, &cancel, emit.as_ref()).await?;
        }
        anyhow::Ok(())
    }
    .await;
    emit(match result {
        Ok(()) => InstallEvent::Finished,
        Err(e) if crate::proc::is_cancelled(&e) => InstallEvent::Cancelled,
        Err(e) => InstallEvent::Failed { error: format!("{e:#}") },
    });
}

async fn install_one(
    client: &reqwest::Client,
    root: &Path,
    pkg: &Package,
    cancel: &CancellationToken,
    emit: &(dyn Fn(InstallEvent) + Send + Sync),
) -> Result<()> {
    let staging = root.join(format!(".tmp-{}", pkg.id));
    let result = download_and_place(client, root, &staging, pkg, cancel, emit).await;
    // Nunca deixa download parcial para trás, com sucesso ou não.
    let _ = std::fs::remove_dir_all(&staging);
    result
}

async fn download_and_place(
    client: &reqwest::Client,
    root: &Path,
    staging: &Path,
    pkg: &Package,
    cancel: &CancellationToken,
    emit: &(dyn Fn(InstallEvent) + Send + Sync),
) -> Result<()> {
    let id = pkg.id.clone();
    emit(InstallEvent::Started { id: id.clone(), label: pkg.label.clone(), total: pkg.size });

    if staging.exists() {
        std::fs::remove_dir_all(staging)?;
    }
    std::fs::create_dir_all(staging)?;
    let archive = staging.join(&pkg.file);

    // Download com SHA-256 calculado durante a transferência.
    let res = client.get(&pkg.url).send().await?.error_for_status()
        .with_context(|| format!("download de {} falhou", pkg.label))?;
    let total = res.content_length().unwrap_or(pkg.size);
    let mut file = std::io::BufWriter::new(std::fs::File::create(&archive)?);
    let mut hasher = Sha256::new();
    let mut done = 0u64;
    let mut last_emit = 0u64;
    let mut stream = res.bytes_stream();
    loop {
        let chunk = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(Cancelled.into()),
            c = stream.next() => c,
        };
        let Some(chunk) = chunk else { break };
        let chunk = chunk.context("conexão interrompida durante o download")?;
        hasher.update(&chunk);
        file.write_all(&chunk)?;
        done += chunk.len() as u64;
        if done - last_emit >= 2 << 20 || done == total {
            last_emit = done;
            emit(InstallEvent::Progress { id: id.clone(), done, total });
        }
    }
    file.flush()?;
    drop(file);

    let got = format!("{:x}", hasher.finalize());
    if !got.eq_ignore_ascii_case(&pkg.sha256) {
        bail!("SHA-256 de {} não confere (esperado {}, obtido {got})", pkg.file, pkg.sha256);
    }

    // Extração (bloqueante) e troca atômica da pasta final.
    emit(InstallEvent::Extracting { id: id.clone() });
    let (root, staging, pkg2) = (root.to_path_buf(), staging.to_path_buf(), pkg.clone());
    tokio::task::spawn_blocking(move || extract_and_place(&root, &staging, &archive, &pkg2)).await??;
    emit(InstallEvent::Done { id });
    Ok(())
}

fn extract_and_place(root: &Path, staging: &Path, archive: &Path, pkg: &Package) -> Result<()> {
    let out = staging.join("x");
    let mut zip = zip::ZipArchive::new(std::fs::File::open(archive)?)?;
    zip.extract(&out).context("não foi possível extrair o pacote")?;
    std::fs::remove_file(archive)?;

    let exe = find_file(&out, &pkg.exe).with_context(|| format!("{} não encontrado no pacote", pkg.exe))?;
    // .../<pacote>/bin/<exe> → <pacote>
    let pkg_root = exe.parent().and_then(Path::parent).context("estrutura inesperada do pacote")?;
    let target = root.join(&pkg.id);
    if target.exists() {
        std::fs::remove_dir_all(&target)?;
    }
    std::fs::rename(pkg_root, &target)?;
    Ok(())
}

fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if let Some(hit) = find_file(&p, name) {
                return Some(hit);
            }
        } else if p.file_name().is_some_and(|n| n.to_string_lossy().eq_ignore_ascii_case(name)) {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Baixa o COLMAP de verdade (128 MB):
    /// `cargo test installs_real_package -- --ignored --nocapture`
    #[tokio::test]
    #[ignore]
    async fn installs_real_package() {
        let (file, sha, size) = COLMAP_CPU;
        let pkg = Package {
            id: "colmap".into(),
            label: "COLMAP".into(),
            file: file.into(),
            url: format!("https://github.com/colmap/colmap/releases/download/{COLMAP_TAG}/{file}"),
            sha256: sha.into(),
            size,
            exe: "colmap.exe".into(),
        };
        let root = std::env::temp_dir().join("splatforge_install_test");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let events = std::sync::Mutex::new(Vec::new());
        install_one(&client().unwrap(), &root, &pkg, &CancellationToken::new(), &|e| {
            events.lock().unwrap().push(e)
        })
        .await
        .unwrap();
        assert!(root.join("colmap/bin/colmap.exe").is_file());
        assert!(!root.join(".tmp-colmap").exists());
        let ev = events.into_inner().unwrap();
        assert!(matches!(ev.last(), Some(InstallEvent::Done { .. })));
        println!("{} eventos, último progresso: {:?}", ev.len(), ev.iter().rev().find(|e| matches!(e, InstallEvent::Progress { .. })));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Também usa a rede: `cargo test rejects_wrong_checksum -- --ignored`
    #[tokio::test]
    #[ignore]
    async fn rejects_wrong_checksum() {
        let (file, _, size) = COLMAP_CPU;
        let pkg = Package {
            id: "x".into(),
            label: "x".into(),
            file: "small.txt".into(),
            // Arquivo pequeno qualquer; o hash propositalmente não confere.
            url: "https://raw.githubusercontent.com/colmap/colmap/4.2.1/README.md".into(),
            sha256: "0".repeat(64),
            size,
            exe: file.into(),
        };
        let root = std::env::temp_dir().join("splatforge_install_badsum");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let err = install_one(&client().unwrap(), &root, &pkg, &CancellationToken::new(), &|_| {})
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("não confere"), "{err:#}");
        assert!(!root.join(".tmp-x").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
