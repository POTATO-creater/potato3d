//! Etapa opcional "Recorte": isola o sujeito em cada frame.
//!
//! - macOS: framework Vision (sidecar Swift `splatforge_mask`), ~40 ms/frame
//!   na GPU de um M4, sem modelo para baixar.
//! - Windows: BiRefNet-lite (MIT) com ONNX Runtime + DirectML, ou CPU se não
//!   houver GPU DirectX 12. O modelo é baixado pelo instalador de ferramentas.
//!
//! As máscaras (cinza, 255 = sujeito) são usadas de dois jeitos:
//! - treino: o Brush recebe PNGs RGBA e aprende o fundo como transparente,
//!   então o splat sai só com o sujeito;
//! - COLMAP (opcional): ignora features do fundo.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskOptions {
    pub enabled: bool,
    /// Também passa as máscaras ao COLMAP. Ajuda quando o fundo "derrete";
    /// atrapalha em sujeitos com pouca textura (ex.: nuca), por isso é opcional.
    pub for_poses: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskStats {
    pub count: u32,
    /// Frames em que nenhum sujeito foi detectado (máscara cheia).
    pub no_subject: u32,
    /// Fração média da imagem ocupada pelo sujeito.
    pub mean_coverage: f32,
    pub backend: String,
}

fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("png")))
        .collect();
    v.sort();
    Ok(v)
}

pub async fn run(
    tools: &crate::tools::Tools,
    images: &Path,
    masks: &Path,
    cancel: &CancellationToken,
    mut on_progress: impl FnMut(f32, &str),
    mut on_log: impl FnMut(&str),
) -> Result<MaskStats> {
    if masks.exists() {
        std::fs::remove_dir_all(masks)?;
    }
    std::fs::create_dir_all(masks)?;
    let total = list_images(images)?.len();
    if total == 0 {
        bail!("não há frames para recortar");
    }
    #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
    let mut no_subject = 0u32;

    #[cfg(target_os = "macos")]
    let backend = {
        let bin = crate::tools::Tools::require(&tools.masker, "Recortador (Vision)", "Recompile o app no macOS 14+.")?;
        let mut cmd = tokio::process::Command::new(bin);
        cmd.arg(images).arg(masks);
        crate::proc::run_streaming(cmd, cancel, |line| {
            if let Some((i, n)) = line
                .strip_prefix("progress ")
                .and_then(|r| r.split_once('/'))
                .and_then(|(i, n)| Some((i.parse::<u32>().ok()?, n.parse::<u32>().ok()?)))
            {
                on_progress(i as f32 / n.max(1) as f32, &format!("Recortando {i}/{n}"));
            } else if let Some(name) = line.strip_prefix("nosubject ") {
                no_subject += 1;
                on_log(&format!("nenhum sujeito detectado em {name}; frame mantido inteiro"));
            } else {
                on_log(line);
            }
        })
        .await
        .context("falha no recorte (Vision)")?;
        "Vision · GPU".to_string()
    };

    #[cfg(windows)]
    let backend = {
        let model = crate::tools::Tools::require(&tools.mask_model, "Modelo de recorte (BiRefNet)", crate::tools::install_hint("model"))?;
        let (images, masks, cancel2) = (images.to_path_buf(), masks.to_path_buf(), cancel.clone());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(u32, u32, Option<String>)>();
        let worker = tokio::task::spawn_blocking(move || onnx::run(&model, &images, &masks, &cancel2, tx));
        while let Some((i, n, log)) = rx.recv().await {
            match log {
                Some(l) => on_log(&l),
                None => on_progress(i as f32 / n.max(1) as f32, &format!("Recortando {i}/{n}")),
            }
        }
        worker.await??
    };

    #[cfg(not(any(target_os = "macos", windows)))]
    let backend: String = {
        let _ = (tools, cancel, &mut on_progress, &mut on_log);
        bail!("o recorte automático ainda não está disponível neste sistema");
    };

    // Estatísticas (em paralelo): cobertura média do sujeito.
    let files = list_images(masks)?;
    if files.len() != total {
        bail!("o recorte gerou {} máscaras para {total} frames", files.len());
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
    })
}

/// Monta o dataset de treino com fundo transparente: PNGs RGBA (alpha =
/// máscara) e uma cópia do modelo COLMAP com os nomes trocados para `.png`.
/// O Brush procura cada imagem pelo nome registrado no `images.bin`.
pub fn build_alpha_dataset(dataset: &Path, masks: &Path, out: &Path) -> Result<()> {
    if out.exists() {
        std::fs::remove_dir_all(out)?;
    }
    let out_images = out.join("images");
    let out_model = out.join("sparse").join("0");
    std::fs::create_dir_all(&out_images)?;
    std::fs::create_dir_all(&out_model)?;

    list_images(&dataset.join("images"))?.par_iter().try_for_each(|img_path| -> Result<()> {
        let stem = img_path.file_stem().context("nome de frame inválido")?;
        let mask_path = masks.join(stem).with_extension("png");
        let mut rgba = image::open(img_path)?.into_rgba8();
        let mask = image::open(&mask_path)
            .with_context(|| format!("máscara ausente: {}", mask_path.display()))?
            .into_luma8();
        let mask = if mask.dimensions() == rgba.dimensions() {
            mask
        } else {
            image::imageops::resize(&mask, rgba.width(), rgba.height(), image::imageops::FilterType::Triangle)
        };
        for (px, m) in rgba.pixels_mut().zip(mask.pixels()) {
            px[3] = m[0];
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
    })
}

/// O COLMAP espera `<nome_da_imagem>.png` (ex.: `00000.jpg.png`), preto = ignorar.
pub fn write_colmap_masks(images: &Path, masks: &Path, out: &Path) -> Result<()> {
    if out.exists() {
        std::fs::remove_dir_all(out)?;
    }
    std::fs::create_dir_all(out)?;
    for img in list_images(images)? {
        let name = img.file_name().context("nome inválido")?.to_string_lossy().into_owned();
        let stem = img.file_stem().context("nome inválido")?;
        let src = masks.join(stem).with_extension("png");
        let dst = out.join(format!("{name}.png"));
        if std::fs::hard_link(&src, &dst).is_err() {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

// ---------- BiRefNet: pré/pós-processamento (compartilhado e testável) ----------

pub const MODEL_SIZE: u32 = 1024;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Imagem → tensor NCHW 1×3×1024×1024 normalizado (ImageNet).
pub fn preprocess(img: &image::RgbImage) -> Vec<f32> {
    let s = MODEL_SIZE;
    let small = image::imageops::resize(img, s, s, image::imageops::FilterType::Triangle);
    let plane = (s * s) as usize;
    let mut out = vec![0f32; 3 * plane];
    for (i, px) in small.pixels().enumerate() {
        for c in 0..3 {
            out[c * plane + i] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    out
}

/// Logits 1024×1024 → máscara do tamanho original.
pub fn postprocess(logits: &[f32], width: u32, height: u32) -> image::GrayImage {
    let s = MODEL_SIZE;
    let bytes: Vec<u8> = logits.iter().map(|&v| (255.0 / (1.0 + (-v).exp())).round() as u8).collect();
    let m = image::GrayImage::from_raw(s, s, bytes).expect("saída do modelo com tamanho inesperado");
    image::imageops::resize(&m, width, height, image::imageops::FilterType::Triangle)
}

#[cfg(windows)]
mod onnx {
    use super::*;
    use ort::{ep, session::Session, value::Tensor};

    type Tx = tokio::sync::mpsc::UnboundedSender<(u32, u32, Option<String>)>;

    /// Os erros do construtor carregam o próprio builder (não é `Send`).
    fn ort_err<R>(e: ort::Error<R>) -> anyhow::Error {
        anyhow::anyhow!("{e}")
    }

    fn session(model: &Path, gpu: bool) -> Result<Session> {
        // DirectML exige execução sequencial e sem padrão de memória.
        let mut b = Session::builder()
            .map_err(ort_err)?
            .with_parallel_execution(false)
            .map_err(ort_err)?
            .with_memory_pattern(false)
            .map_err(ort_err)?;
        if gpu {
            b = b
                .with_execution_providers([ep::DirectML::default().build().error_on_failure()])
                .map_err(ort_err)?;
        }
        b.commit_from_file(model).map_err(ort_err)
    }

    pub fn run(model: &Path, images: &Path, masks: &Path, cancel: &CancellationToken, tx: Tx) -> Result<String> {
        let (mut sess, backend) = match session(model, true) {
            Ok(s) => (s, "BiRefNet · DirectML"),
            Err(e) => {
                let _ = tx.send((0, 0, Some(format!("DirectML indisponível ({e:#}); usando CPU"))));
                (session(model, false)?, "BiRefNet · CPU")
            }
        };
        let input = sess.inputs()[0].name().to_string();
        let files = list_images(images)?;
        let n = files.len() as u32;
        // Pré-processa o próximo frame em paralelo enquanto a GPU trabalha.
        let load = |p: &PathBuf| -> Result<(u32, u32, Vec<f32>)> {
            let img = image::open(p)?.into_rgb8();
            Ok((img.width(), img.height(), preprocess(&img)))
        };
        let mut next = files.first().map(load).transpose()?;
        for (i, path) in files.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(crate::proc::Cancelled.into());
            }
            let (w, h, data) = next.take().context("frame não carregado")?;
            let following = files.get(i + 1).cloned();
            let (prefetch, logits) = rayon::join(
                || following.as_ref().map(load).transpose(),
                || -> Result<Vec<f32>> {
                    let s = MODEL_SIZE as usize;
                    let t = Tensor::from_array(([1usize, 3, s, s], data))?;
                    let out = sess.run(ort::inputs![input.as_str() => t])?;
                    let (_, v) = out[out.len() - 1].try_extract_tensor::<f32>()?;
                    Ok(v.to_vec())
                },
            );
            next = prefetch?;
            let mask = postprocess(&logits?, w, h);
            mask.save(masks.join(path.file_stem().context("nome inválido")?).with_extension("png"))?;
            let _ = tx.send((i as u32 + 1, n, None));
        }
        Ok(backend.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_and_post_processing_shapes() {
        let img = image::RgbImage::from_pixel(64, 48, image::Rgb([124, 116, 104]));
        let t = preprocess(&img);
        assert_eq!(t.len(), 3 * (MODEL_SIZE * MODEL_SIZE) as usize);
        // Cor ≈ média do ImageNet → valores perto de zero.
        assert!(t.iter().all(|v| v.abs() < 0.05));
        let logits = vec![10.0f32; (MODEL_SIZE * MODEL_SIZE) as usize];
        let m = postprocess(&logits, 64, 48);
        assert_eq!(m.dimensions(), (64, 48));
        assert!(m.pixels().all(|p| p[0] == 255));
    }
}
