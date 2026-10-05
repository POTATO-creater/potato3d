//! Etapa opcional "Recorte": isola o sujeito em cada frame.
//!
//! Métodos:
//! - macOS, automático: framework Vision (sidecar Swift `splatforge_mask`),
//!   ~40 ms/frame na GPU de um M4, qualquer sujeito, sem modelo para baixar.
//! - Windows, pessoas: MODNet (Apache 2.0), rede convolucional leve — 144 ms
//!   na CPU de um M4 contra 7,9 s do BiRefNet, com qualidade equivalente em
//!   pessoas (inclusive de costas).
//! - Windows, qualquer objeto: BiRefNet-lite (MIT). Mais preciso fora de
//!   pessoas, mas as convoluções deformáveis viram ~16 mil operações no ONNX
//!   e ficam lentas até na GPU (~4 s/frame numa RTX 4070 Ti Super).
//! - Cor sólida (chroma key): distância RGB à cor de fundo, com tolerância,
//!   suavização e preenchimento a partir das bordas. Instantâneo.
//!
//! As máscaras (cinza, 255 = sujeito) viram o alpha dos PNGs de treino: o
//! Brush aprende o fundo como transparente e o splat sai só com o sujeito.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{bail, Context, Result};
use image::{GrayImage, RgbImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum MaskMethod {
    /// macOS: Vision. Windows: MODNet.
    #[default]
    Auto,
    /// Windows: MODNet (rápido, só pessoas).
    People,
    /// Windows: BiRefNet (qualquer objeto, lento).
    General,
    /// Chroma key por cor sólida.
    Chroma,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ChromaOptions {
    /// Detecta a cor do fundo pela mediana das bordas de cada frame.
    pub auto_color: bool,
    pub color: [u8; 3],
    /// 0–100: distância RGB considerada fundo.
    pub tolerance: f32,
    /// 0–100: faixa de transição suave depois da tolerância.
    pub softness: f32,
    /// Só remove regiões ligadas à borda (não fura o sujeito).
    pub contiguous: bool,
    /// Tira o reflexo da cor de fundo nas bordas do sujeito.
    pub despill: bool,
}

impl Default for ChromaOptions {
    fn default() -> Self {
        // Mais conservador que o usual em editores: um furo semitransparente no
        // sujeito vira buraco no splat (18/8 já comia tons claros de azul).
        Self { auto_color: true, color: [0, 255, 0], tolerance: 12.0, softness: 6.0, contiguous: true, despill: true }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskOptions {
    pub enabled: bool,
    /// Também passa as máscaras ao COLMAP. Ajuda quando o fundo "derrete";
    /// atrapalha em sujeitos com pouca textura (ex.: nuca), por isso é opcional.
    pub for_poses: bool,
    #[serde(default)]
    pub method: MaskMethod,
    #[serde(default)]
    pub chroma: ChromaOptions,
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
    /// Chroma key: cor de fundo usada (mediana entre os frames).
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

/// Mensagens de um trabalho bloqueante: progresso `(i, n)` ou uma linha de log.
type Tx = tokio::sync::mpsc::UnboundedSender<Msg>;
enum Msg {
    Progress(u32, u32),
    Log(String),
}

/// Roda `work` numa thread de bloqueio repassando progresso e log.
async fn drive<T: Send + 'static>(
    work: impl FnOnce(Tx) -> Result<T> + Send + 'static,
    on_progress: &mut impl FnMut(f32, &str),
    on_log: &mut impl FnMut(&str),
) -> Result<T> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = tokio::task::spawn_blocking(move || work(tx));
    while let Some(msg) = rx.recv().await {
        match msg {
            Msg::Progress(i, n) => on_progress(i as f32 / n.max(1) as f32, &format!("Recortando {i}/{n}")),
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

    let mut no_subject = 0u32;
    let mut key_color = None;
    let backend = if opts.method == MaskMethod::Chroma {
        let (images, masks, chroma, cancel) = (images.to_path_buf(), masks.to_path_buf(), opts.chroma.clone(), cancel.clone());
        let key = drive(move |tx| chroma::run(&images, &masks, &chroma, &cancel, &tx), &mut on_progress, &mut on_log).await?;
        key_color = Some(key);
        "Cor sólida".to_string()
    } else {
        run_ai(tools, opts.method, images, masks, cancel, &mut on_progress, &mut on_log, &mut no_subject).await?
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
        key_color,
    })
}

// Cada sistema usa só parte dos argumentos.
#[allow(clippy::too_many_arguments, unused_variables)]
async fn run_ai(
    tools: &crate::tools::Tools,
    method: MaskMethod,
    images: &Path,
    masks: &Path,
    cancel: &CancellationToken,
    on_progress: &mut impl FnMut(f32, &str),
    on_log: &mut impl FnMut(&str),
    no_subject: &mut u32,
) -> Result<String> {
    #[cfg(target_os = "macos")]
    {
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
                *no_subject += 1;
                on_log(&format!("nenhum sujeito detectado em {name}; frame mantido inteiro"));
            } else {
                on_log(line);
            }
        })
        .await
        .context("falha no recorte (Vision)")?;
        Ok("Vision · GPU".to_string())
    }

    #[cfg(windows)]
    {
        let kind = if method == MaskMethod::General { ModelKind::BiRefNet } else { ModelKind::Modnet };
        let model = match kind {
            ModelKind::BiRefNet => &tools.birefnet_model,
            ModelKind::Modnet => &tools.modnet_model,
        };
        let model = crate::tools::Tools::require(model, kind.label(), crate::tools::install_hint("model"))?;
        let (images, masks, cancel) = (images.to_path_buf(), masks.to_path_buf(), cancel.clone());
        drive(move |tx| onnx::run(&model, kind, &images, &masks, &cancel, &tx), on_progress, on_log).await
    }

    #[cfg(not(any(target_os = "macos", windows)))]
    {
        let _ = (tools, method, images, masks, cancel, on_progress, on_log, no_subject);
        bail!("o recorte por IA ainda não está disponível neste sistema; use a cor sólida");
    }
}

/// Remove o reflexo da cor de fundo ("spill") na borda do recorte: o canal
/// dominante da cor-chave é limitado ao maior dos outros dois. Só na borda
/// (alpha < 255 ou até `EDGE_PX` do fundo) para não desbotar partes do
/// sujeito que tenham a mesma cor do fundo.
#[derive(Debug, Clone, Copy)]
pub struct Despill {
    pub key: [u8; 3],
    /// Distância RGB até onde o pixel ainda é considerado contaminado.
    pub reach: f32,
}

impl Despill {
    pub fn from_chroma(key: [u8; 3], c: &ChromaOptions) -> Option<Self> {
        let (max, min) = (*key.iter().max().unwrap() as i32, *key.iter().min().unwrap() as i32);
        // Só faz sentido para cores saturadas (verde, azul, magenta…).
        (c.despill && max - min > 60).then(|| Self { key, reach: (c.tolerance + 2.0 * c.softness) * 4.42 })
    }

    fn apply(&self, px: &mut [u8]) {
        let c = (0..3).max_by_key(|&i| self.key[i]).unwrap();
        if chroma::dist(px, self.key) > self.reach {
            return;
        }
        let limit = (0..3).filter(|&i| i != c).map(|i| px[i]).max().unwrap();
        px[c] = px[c].min(limit);
    }
}

/// Monta o dataset de treino com fundo transparente: PNGs RGBA (alpha =
/// máscara) e uma cópia do modelo COLMAP com os nomes trocados para `.png`.
/// O Brush procura cada imagem pelo nome registrado no `images.bin`.
const EDGE_PX: i32 = 2;

/// Pixels a até `r` px de um pixel de fundo (máscara < 128): dilatação
/// separável (linhas, depois colunas).
fn near_background(mask: &GrayImage, r: i32) -> Vec<bool> {
    let (w, h) = (mask.width() as i32, mask.height() as i32);
    let bg: Vec<bool> = mask.as_raw().iter().map(|&m| m < 128).collect();
    let mut rows = vec![false; bg.len()];
    for y in 0..h {
        for x in 0..w {
            rows[(y * w + x) as usize] = (x - r..=x + r).any(|xx| xx >= 0 && xx < w && bg[(y * w + xx) as usize]);
        }
    }
    let mut out = vec![false; bg.len()];
    for y in 0..h {
        for x in 0..w {
            out[(y * w + x) as usize] = (y - r..=y + r).any(|yy| yy >= 0 && yy < h && rows[(yy * w + x) as usize]);
        }
    }
    out
}

pub fn build_alpha_dataset(dataset: &Path, masks: &Path, out: &Path, despill: Option<Despill>) -> Result<()> {
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
        let edge = despill.map(|_| near_background(&mask, EDGE_PX));
        for (i, (px, m)) in rgba.pixels_mut().zip(mask.pixels()).enumerate() {
            px[3] = m[0];
            if let (Some(d), Some(edge)) = (despill, &edge) {
                if m[0] > 0 && (m[0] < 255 || edge[i]) {
                    d.apply(&mut px.0[..3]);
                }
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

// ---------- Chroma key ----------

pub mod chroma {
    use super::*;

    pub fn dist(px: &[u8], key: [u8; 3]) -> f32 {
        let d: [f32; 3] = std::array::from_fn(|i| px[i] as f32 - key[i] as f32);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    /// Mediana por canal dos pixels da borda (robusta a um canto "sujo").
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

    /// Máscara (255 = sujeito) por distância à cor-chave.
    pub fn mask(img: &RgbImage, key: [u8; 3], o: &ChromaOptions) -> GrayImage {
        let (w, h) = img.dimensions();
        let n = (w * h) as usize;
        let tol = o.tolerance.clamp(0.0, 100.0) * 4.42; // 0–100 → 0–442 (distância RGB máxima)
        let soft = (o.softness.clamp(0.0, 100.0) * 4.42).max(1.0);
        let d: Vec<f32> = img.as_raw().par_chunks_exact(3).map(|p| dist(p, key)).collect();

        // Com "contíguo", só conta como fundo o que se liga à borda.
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

    /// Recorta todos os frames; devolve a cor-chave mediana usada.
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
                mask(&img, key, o).save(masks.join(p.file_stem().context("nome inválido")?).with_extension("png"))?;
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
                "cor de fundo detectada (mediana): #{:02x}{:02x}{:02x}",
                key[0], key[1], key[2]
            )));
        }
        Ok(key)
    }
}

// ---------- Modelos ONNX: pré/pós-processamento (compartilhado e testável) ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// BiRefNet-lite: entrada fixa 1024², normalização ImageNet, saída em logits.
    BiRefNet,
    /// MODNet: menor lado 512 (múltiplos de 32), normalização [-1, 1], saída 0–1.
    Modnet,
}

impl ModelKind {
    pub fn label(self) -> &'static str {
        match self {
            ModelKind::BiRefNet => "BiRefNet",
            ModelKind::Modnet => "MODNet",
        }
    }

    /// Tamanho (largura, altura) de entrada do modelo para um frame.
    pub fn input_size(self, w: u32, h: u32) -> (u32, u32) {
        match self {
            ModelKind::BiRefNet => (1024, 1024),
            ModelKind::Modnet => {
                let s = 512.0 / w.min(h) as f32;
                let r = |v: u32| (((v as f32 * s) / 32.0).round() as u32 * 32).max(32);
                (r(w), r(h))
            }
        }
    }

    /// Imagem → tensor NCHW e seu formato.
    pub fn preprocess(self, img: &RgbImage) -> (Vec<f32>, [usize; 4]) {
        let (iw, ih) = self.input_size(img.width(), img.height());
        let small = image::imageops::resize(img, iw, ih, image::imageops::FilterType::Triangle);
        let (mean, std) = match self {
            ModelKind::BiRefNet => ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ModelKind::Modnet => ([0.5; 3], [0.5; 3]),
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

    /// Saída do modelo → máscara do tamanho original.
    pub fn postprocess(self, out: &[f32], (iw, ih): (u32, u32), width: u32, height: u32) -> GrayImage {
        let to_u8 = |v: f32| match self {
            ModelKind::BiRefNet => (255.0 / (1.0 + (-v).exp())).round() as u8,
            ModelKind::Modnet => (v.clamp(0.0, 1.0) * 255.0).round() as u8,
        };
        let bytes: Vec<u8> = out.iter().map(|&v| to_u8(v)).collect();
        let m = GrayImage::from_raw(iw, ih, bytes).expect("saída do modelo com tamanho inesperado");
        image::imageops::resize(&m, width, height, image::imageops::FilterType::Triangle)
    }
}

#[cfg(windows)]
mod onnx {
    use super::*;
    use ort::{ep, session::Session, value::Tensor};

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

    pub fn run(model: &Path, kind: ModelKind, images: &Path, masks: &Path, cancel: &CancellationToken, tx: &Tx) -> Result<String> {
        let (mut sess, device) = match session(model, true) {
            Ok(s) => (s, "DirectML"),
            Err(e) => {
                let _ = tx.send(Msg::Log(format!("DirectML indisponível ({e:#}); usando CPU")));
                (session(model, false)?, "CPU")
            }
        };
        let input = sess.inputs()[0].name().to_string();
        let files = list_images(images)?;
        let n = files.len() as u32;
        // Pré-processa o próximo frame em paralelo enquanto a GPU trabalha.
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
            let (w, h, data, shape) = next.take().context("frame não carregado")?;
            let following = files.get(i + 1).cloned();
            let (prefetch, out) = rayon::join(
                || following.as_ref().map(load).transpose(),
                || -> Result<Vec<f32>> {
                    let t = Tensor::from_array((shape, data))?;
                    let out = sess.run(ort::inputs![input.as_str() => t])?;
                    // BiRefNet: última saída é a final; MODNet tem uma só.
                    let (_, v) = out[out.len() - 1].try_extract_tensor::<f32>()?;
                    Ok(v.to_vec())
                },
            );
            next = prefetch?;
            let mask = kind.postprocess(&out?, (shape[3] as u32, shape[2] as u32), w, h);
            mask.save(masks.join(path.file_stem().context("nome inválido")?).with_extension("png"))?;
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
        // Cor ≈ média do ImageNet → valores perto de zero.
        assert!(t.iter().all(|v| v.abs() < 0.05));
        let m = ModelKind::BiRefNet.postprocess(&vec![10.0; 1024 * 1024], (1024, 1024), 64, 48);
        assert_eq!(m.dimensions(), (64, 48));
        assert!(m.pixels().all(|p| p[0] == 255));

        // MODNet: menor lado 512, múltiplos de 32 (960×1280 → 512×672).
        assert_eq!(ModelKind::Modnet.input_size(960, 1280), (512, 672));
        let (_, shape) = ModelKind::Modnet.preprocess(&RgbImage::new(960, 1280));
        assert_eq!(shape, [1, 3, 672, 512]);
        let m = ModelKind::Modnet.postprocess(&vec![0.5; 512 * 672], (512, 672), 960, 1280);
        assert!(m.pixels().all(|p| p[0] == 128));
    }

    /// Fundo verde com um quadrado vermelho no meio e um "furo" verde dentro
    /// dele: com "contíguo", o furo não é removido.
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
        assert_eq!(m.get_pixel(0, 0)[0], 0, "fundo removido");
        assert_eq!(m.get_pixel(15, 15)[0], 255, "sujeito mantido");
        assert_eq!(m.get_pixel(20, 20)[0], 255, "furo interno preservado (contíguo)");
        let m = chroma::mask(&img, key, &ChromaOptions { contiguous: false, ..o.clone() });
        assert_eq!(m.get_pixel(20, 20)[0], 0, "sem contíguo, o furo some");

        // Despill: pixel esverdeado perto da chave perde o excesso de verde.
        let d = Despill::from_chroma(key, &o).unwrap();
        let mask_edge = near_background(&m, 2);
        assert!(mask_edge[(10 * 40 + 10) as usize], "canto do quadrado encosta no fundo");
        assert!(!mask_edge[(20 * 40 + 15) as usize], "interior longe do fundo");
        let mut px = [90u8, 170, 80];
        d.apply(&mut px);
        assert_eq!(px, [90, 90, 80]);
        // Cor bem diferente da chave fica intacta.
        let mut skin = [200u8, 150, 120];
        d.apply(&mut skin);
        assert_eq!(skin, [200, 150, 120]);
    }
}
