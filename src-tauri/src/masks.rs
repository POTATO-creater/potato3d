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
//! - RMBG-1.4 (BRIA, não comercial), Windows e macOS em Apple Silicon: IS-Net
//!   de convoluções simples, bordas melhores que o Vision em objetos.
//!   ~0,4 s/frame com CoreML na GPU de um M4; baixado só se escolhido.
//! - Cor sólida (chroma key): distância RGB à cor de fundo, com tolerância,
//!   suavização e preenchimento a partir das bordas. Instantâneo.
//!
//! As máscaras (cinza, 255 = sujeito) viram o alpha dos PNGs de treino: o
//! Brush aprende o fundo como transparente e o splat sai só com o sujeito.
//! Com fundo verde/azul, o despill tira do sujeito a luz rebatida pelo fundo.

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
    /// Windows: MODNet (rápido, só pessoas).
    People,
    /// Windows: BiRefNet (qualquer objeto, lento).
    General,
    /// RMBG-1.4 (qualquer objeto; Windows e macOS em Apple Silicon).
    Rmbg,
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
}

impl Default for ChromaOptions {
    fn default() -> Self {
        // Mais conservador que o usual em editores: um furo semitransparente no
        // sujeito vira buraco no splat (18/8 já comia tons claros de azul).
        Self { auto_color: true, color: [0, 255, 0], tolerance: 12.0, softness: 6.0, contiguous: true }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MaskOptions {
    pub enabled: bool,
    /// Também passa as máscaras ao COLMAP. Ajuda quando o fundo "derrete";
    /// atrapalha em sujeitos com pouca textura (ex.: nuca), por isso é opcional.
    pub for_poses: bool,
    pub method: MaskMethod,
    pub chroma: ChromaOptions,
    /// Tira do sujeito o reflexo de um fundo verde ou azul (qualquer método).
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
    on_progress: &mut impl FnMut(f32, crate::msg::Msg),
    on_log: &mut impl FnMut(&str),
    no_subject: &mut u32,
) -> Result<String> {
    if method == MaskMethod::Rmbg && !crate::tools::ONNX_AVAILABLE {
        bail!("o RMBG-1.4 precisa de Windows ou de um Mac com Apple Silicon");
    }

    // Modelos ONNX: tudo no Windows; só o RMBG no macOS (o resto é Vision).
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
        let bin = crate::tools::Tools::require(&tools.masker, "Recortador (Vision)", "Recompile o app no macOS 14+.")?;
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
                on_log(&format!("nenhum sujeito detectado em {name}; frame mantido inteiro"));
            } else {
                on_log(line);
            }
        })
        .await
        .context("falha no recorte (Vision)")?;
        Ok("Vision · GPU".to_string())
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (tools, images, masks, cancel, on_progress, on_log, no_subject);
        bail!("o recorte por IA ainda não está disponível neste sistema; use a cor sólida");
    }
}

/// Supressão do reflexo da cor do fundo ("spill"): a luz que uma tela verde
/// rebate no sujeito. O canal dominante da cor-chave é limitado ao maior dos
/// outros dois em todo o sujeito (não só na borda: o verde rebatido cobre
/// superfícies inteiras, como a base de uma bola cinza) e a luminância tirada
/// volta como cinza, para o sujeito não escurecer. Limitar pelo maior (e não
/// pela média) dos outros canais preserva amarelos e cianos.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Despill {
    /// Canal da cor-chave: 1 = verde, 2 = azul.
    channel: usize,
    /// 0–1.
    strength: f32,
}

/// Peso de cada canal na luminância (Rec. 709).
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

impl Despill {
    /// Só para fundos verdes ou azuis bem saturados: num fundo bege ou cinza o
    /// "reflexo" seria a própria cor do sujeito (pele, madeira…).
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

/// Lê a máscara de um frame já no tamanho dele.
fn load_mask(masks: &Path, img_path: &Path, (w, h): (u32, u32)) -> Result<GrayImage> {
    let stem = img_path.file_stem().context("nome de frame inválido")?;
    let mask_path = masks.join(stem).with_extension("png");
    let mask = image::open(&mask_path)
        .with_context(|| format!("máscara ausente: {}", mask_path.display()))?
        .into_luma8();
    Ok(if mask.dimensions() == (w, h) {
        mask
    } else {
        image::imageops::resize(&mask, w, h, image::imageops::FilterType::Triangle)
    })
}

/// Cor do fundo: mediana por canal dos pixels que a máscara diz serem fundo,
/// numa amostra de até 24 frames. Vale para qualquer método de recorte e é
/// uma só para o vídeo todo (cores coerentes entre as vistas).
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

/// Monta o dataset de treino com fundo transparente: PNGs RGBA (alpha =
/// máscara) e uma cópia do modelo COLMAP com os nomes trocados para `.png`.
/// O Brush procura cada imagem pelo nome registrado no `images.bin`.
/// `despill`: intensidade 0–100; devolve a cor de fundo quando aplicou.
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
        let stem = img_path.file_stem().context("nome de frame inválido")?;
        let mut rgba = image::open(img_path)?.into_rgba8();
        let mask = load_mask(masks, img_path, rgba.dimensions())?;
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

    /// Distância RGB até a cor-chave, aceitando a chave mais clara (até 1,6x):
    /// telas costumam ser mais iluminadas no meio que nas bordas, de onde vem
    /// a cor detectada. Mais escura não: a base de um objeto à sombra, tingida
    /// pelo reflexo verde, seria confundida com fundo.
    pub fn dist(px: &[u8], key: [u8; 3]) -> f32 {
        let k: [f32; 3] = key.map(|v| v as f32);
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let p = [px[0] as f32, px[1] as f32, px[2] as f32];
        let s = (dot(p, k) / dot(k, k).max(1.0)).clamp(1.0, 1.6);
        let d: [f32; 3] = std::array::from_fn(|i| p[i] - s * k[i]);
        dot(d, d).sqrt()
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
    /// RMBG-1.4: entrada fixa 1024², `x - 0.5`, saída 0–1 esticada entre o
    /// mínimo e o máximo do frame (como no `utilities.py` da BRIA).
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

    /// Tamanho (largura, altura) de entrada do modelo para um frame.
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

    /// Imagem → tensor NCHW e seu formato.
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

    /// Saída do modelo → máscara do tamanho original.
    pub fn postprocess(self, out: &[f32], (iw, ih): (u32, u32), width: u32, height: u32) -> GrayImage {
        // RMBG: estica o resultado entre o mínimo e o máximo do frame, salvo
        // quando quase não há sujeito (esticar só amplificaria ruído).
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
        let m = GrayImage::from_raw(iw, ih, bytes).expect("saída do modelo com tamanho inesperado");
        image::imageops::resize(&m, width, height, image::imageops::FilterType::Triangle)
    }
}

#[cfg(any(windows, all(target_os = "macos", target_arch = "aarch64")))]
mod onnx {
    use super::*;
    use ort::{ep, session::Session, value::Tensor};

    /// Os erros do construtor carregam o próprio builder (não é `Send`).
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

    /// GPU sem o Neural Engine: compilar redes grandes para o ANE já travou o
    /// `ANECompilerService` do sistema por mais de uma hora (BiRefNet).
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
        // DirectML exige execução sequencial e sem padrão de memória. Lote
        // fixo em 1: com ele dinâmico o CoreML só aceita parte do grafo e o
        // RMBG fica 3x mais lento.
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
                let _ = tx.send(Msg::Log(format!("{GPU} indisponível ({e:#}); usando CPU")));
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
                    // BiRefNet: última saída é a final; MODNet e RMBG têm uma só.
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

        // Tela mais iluminada que a chave ainda é fundo; um objeto escuro
        // tingido de verde (base de uma bola à sombra) não.
        assert!(chroma::dist(&[30, 250, 45], [20, 200, 30]) < 15.0);
        assert!(chroma::dist(&[30, 80, 40], [30, 200, 40]) > 100.0);

    }

    /// Despill: a luz verde rebatida numa superfície cinza some no sujeito
    /// inteiro, sem escurecer; amarelo e pele ficam intactos.
    #[test]
    fn despill_removes_green_spill_everywhere() {
        let d = Despill::new([30, 200, 40], 100.0).expect("tela verde");
        // Base de uma bola cinza iluminada pelo chão verde.
        let mut ball = [30u8, 80, 40];
        d.apply(&mut ball);
        assert!(ball[1] <= ball[0].max(ball[2]), "sem verde dominante: {ball:?}");
        let luma = |p: [u8; 3]| (0..3).map(|i| p[i] as f32 * LUMA[i]).sum::<f32>();
        assert!(luma(ball) > 50.0, "luminância preservada: {ball:?}");
        for keep in [[200u8, 150, 120], [220, 220, 30], [60, 60, 60]] {
            let mut p = keep;
            d.apply(&mut p);
            assert_eq!(p, keep);
        }
        // Meia intensidade tira metade do excesso.
        let mut half = [40u8, 140, 40];
        Despill::new([30, 200, 40], 50.0).unwrap().apply(&mut half);
        assert!(half[1] > half[0] && half[1] < 140, "{half:?}");

        // Azul também; fundos pouco saturados ou avermelhados não.
        assert_eq!(Despill::new([20, 60, 210], 100.0).map(|d| d.channel), Some(2));
        assert!(Despill::new([150, 190, 150], 100.0).is_none(), "verde-claro de parede");
        assert!(Despill::new([210, 120, 60], 100.0).is_none(), "fundo laranja");
        assert!(Despill::new([30, 200, 40], 0.0).is_none());
    }

    #[test]
    fn rmbg_pre_and_post_processing() {
        let (t, shape) = ModelKind::Rmbg.preprocess(&RgbImage::from_pixel(64, 48, image::Rgb([255, 0, 128])));
        assert_eq!(shape, [1, 3, 1024, 1024]);
        let plane = 1024 * 1024;
        assert!((t[0] - 0.5).abs() < 1e-3 && (t[plane] + 0.5).abs() < 1e-3 && t[2 * plane].abs() < 0.01);
        // Saída esticada entre mínimo e máximo do frame.
        let mut out = vec![0.1f32; plane];
        out[..plane / 2].fill(0.9);
        let m = ModelKind::Rmbg.postprocess(&out, (1024, 1024), 1024, 1024);
        assert_eq!((m.get_pixel(0, 0)[0], m.get_pixel(0, 1023)[0]), (255, 0));
        // Sem sujeito: nada é esticado.
        let m = ModelKind::Rmbg.postprocess(&vec![0.02; plane], (1024, 1024), 64, 48);
        assert!(m.pixels().all(|p| p[0] == 5));
    }

    /// Recorte real com um modelo ONNX, medindo o tempo:
    /// `SPLATFORGE_TEST_MODEL=rmbg14.onnx SPLATFORGE_TEST_FRAMES=<pasta>
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
        println!("{backend}: {n} frames em {:.1} s ({:.0} ms/frame) → {}", t0.elapsed().as_secs_f32(),
            t0.elapsed().as_secs_f32() * 1000.0 / n as f32, out.display());
    }

    /// Cor de fundo pela máscara e dataset com despill aplicado.
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
