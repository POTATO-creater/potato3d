//! SOG v2 writer (PlayCanvas): an uncompressed ZIP with `meta.json` and
//! lossless WebP textures. Port of WebGaussian's `sog-encode.ts`, which in
//! turn follows splat-transform's `write-sog.ts`: Morton order, 1D quantizer
//! via dynamic programming (256-entry codebooks), log-transformed 16-bit
//! positions, "smallest three" quaternions and opacity in the `sh0` alpha.
//!
//! Unlike SHARP, Brush trains SH bands. They are stored as in
//! splat-transform: a palette of up to 64k SH vectors (k-means), quantized
//! into a 256-entry codebook, plus a 16-bit label per gaussian. The k-means is
//! two-level (256 coarse groups, each subdivided in proportion to its
//! population) to fit on the CPU: splat-transform uses the GPU for this.

use std::io::{Seek, Write};
use std::path::Path;

use anyhow::{bail, Result};
use rayon::prelude::*;
use serde_json::json;

use super::ply::{sh_coeffs, Splats};
use crate::msg;
use crate::msg::Msg;

const KMEANS_ITERATIONS: usize = 8;
const COARSE_CLUSTERS: usize = 256;
const COARSE_SAMPLE: usize = 65_536;

pub fn write(path: &Path, s: &Splats, mut progress: impl FnMut(f32, &Msg)) -> Result<u64> {
    let n = s.count;
    if n == 0 {
        bail!("no gaussians to export");
    }
    let width = (((n as f64).sqrt() / 4.0).ceil() as usize * 4).max(4);
    let height = ((n as f64 / width as f64 / 4.0).ceil() as usize * 4).max(4);
    if width > 16383 || height > 16383 {
        bail!("too many gaussians for SOG ({n})");
    }
    let texels = width * height;

    progress(0.0, &msg!("export.sorting"));
    let order = morton_order(&s.pos);

    // ---- quantize scales and colors (256-entry codebooks)
    progress(0.1, &msg!("export.quantizing"));
    let column = |src: &[f32], stride: usize, a: usize| -> Vec<f32> { src.iter().skip(a).step_by(stride).copied().collect() };
    let (scales_q, sh0_q) = rayon::join(
        || quantize_1d(&[column(&s.scale, 3, 0), column(&s.scale, 3, 1), column(&s.scale, 3, 2)], 256),
        || quantize_1d(&[column(&s.dc, 3, 0), column(&s.dc, 3, 1), column(&s.dc, 3, 2)], 256),
    );

    // ---- SH palette
    let sh = if s.sh_degree > 0 {
        progress(0.25, &msg!("export.clustering"));
        Some(encode_sh(s))
    } else {
        None
    };

    progress(0.6, &msg!("export.textures"));
    // ---- means: log, 16 bits split across two textures
    let log_t = |v: f32| (v as f64).signum() * ((v as f64).abs() + 1.0).ln();
    let mut mins = [f64::INFINITY; 3];
    let mut maxs = [f64::NEG_INFINITY; 3];
    for p in s.pos.chunks_exact(3) {
        for a in 0..3 {
            mins[a] = mins[a].min(log_t(p[a]));
            maxs[a] = maxs[a].max(log_t(p[a]));
        }
    }
    let mut means_l = vec![0u8; texels * 4];
    let mut means_u = vec![0u8; texels * 4];
    let mut quats = vec![0u8; texels * 4];
    let mut scales = vec![0u8; texels * 4];
    let mut sh0 = vec![0u8; texels * 4];
    let mut sh_labels = vec![0u8; if sh.is_some() { texels * 4 } else { 0 }];
    for (t, &g) in order.iter().enumerate() {
        let g = g as usize;
        let o = t * 4;
        for a in 0..3 {
            let range = maxs[a] - mins[a];
            let v = (65535.0 * (log_t(s.pos[g * 3 + a]) - mins[a]) / if range == 0.0 { 1.0 } else { range }) as u32;
            means_l[o + a] = (v & 0xff) as u8;
            means_u[o + a] = ((v >> 8) & 0xff) as u8;
        }
        means_l[o + 3] = 0xff;
        means_u[o + 3] = 0xff;

        quats[o..o + 4].copy_from_slice(&pack_quat(&s.rot[g * 4..g * 4 + 4]));

        for a in 0..3 {
            scales[o + a] = scales_q.labels[a][g];
            sh0[o + a] = sh0_q.labels[a][g];
        }
        scales[o + 3] = 0xff;
        let opacity = 1.0 / (1.0 + (-s.opacity[g] as f64).exp());
        sh0[o + 3] = (opacity * 255.0).clamp(0.0, 255.0) as u8;

        if let Some(sh) = &sh {
            let label = sh.labels[g];
            sh_labels[o] = (label & 0xff) as u8;
            sh_labels[o + 1] = (label >> 8) as u8;
            sh_labels[o + 3] = 0xff;
        }
    }

    let mut textures: Vec<(&str, Vec<u8>, usize, usize)> = vec![
        ("means_l.webp", means_l, width, height),
        ("means_u.webp", means_u, width, height),
        ("quats.webp", quats, width, height),
        ("scales.webp", scales, width, height),
        ("sh0.webp", sh0, width, height),
    ];
    let mut meta = json!({
        "version": 2,
        "asset": { "generator": concat!("Gausify Orbit ", env!("CARGO_PKG_VERSION")) },
        "count": n,
        "means": { "mins": mins, "maxs": maxs, "files": ["means_l.webp", "means_u.webp"] },
        "scales": { "codebook": scales_q.centroids, "files": ["scales.webp"] },
        "quats": { "files": ["quats.webp"] },
        "sh0": { "codebook": sh0_q.centroids, "files": ["sh0.webp"] },
    });
    if let Some(sh) = sh {
        let coeffs = sh_coeffs(s.sh_degree);
        textures.push(("shN_centroids.webp", sh.centroids_rgba, 64 * coeffs, sh.count.div_ceil(64)));
        textures.push(("shN_labels.webp", sh_labels, width, height));
        meta["shN"] = json!({
            "count": sh.count,
            "bands": s.sh_degree,
            "codebook": sh.codebook,
            "files": ["shN_centroids.webp", "shN_labels.webp"],
        });
    }

    progress(0.7, &msg!("export.compressing"));
    let encoded: Vec<Vec<u8>> = textures
        .par_iter()
        .map(|(_, rgba, w, h)| webp_lossless(rgba, *w, *h))
        .collect::<Result<_>>()?;

    progress(0.95, &msg!("export.writingSog"));
    let file = std::fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for ((name, ..), bytes) in textures.iter().zip(&encoded) {
        zip.start_file(*name, opts)?;
        zip.write_all(bytes)?;
    }
    zip.start_file("meta.json", opts)?;
    zip.write_all(serde_json::to_string(&meta)?.as_bytes())?;
    let mut file = zip.finish()?;
    Ok(file.stream_position()?)
}

/// Normalized quaternion in 4 bytes: the three smallest components in RGB and
/// the index of the largest (dropped) one in A (252 + i).
fn pack_quat(q: &[f32]) -> [u8; 4] {
    let mut q = [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64];
    let l = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let l = if l > 0.0 { l } else { 1.0 };
    q.iter_mut().for_each(|v| *v /= l);
    let mut max = 0;
    for i in 1..4 {
        if q[i].abs() > q[max].abs() {
            max = i;
        }
    }
    let sign = if q[max] < 0.0 { -1.0 } else { 1.0 } * std::f64::consts::SQRT_2;
    q.iter_mut().for_each(|v| *v *= sign);
    let idx = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]][max];
    let b = |v: f64| (255.0 * (v * 0.5 + 0.5)).clamp(0.0, 255.0) as u8;
    [b(q[idx[0]]), b(q[idx[1]]), b(q[idx[2]]), 252 + max as u8]
}

// ---------------------------------------------------------------------------
// Morton order (spatial locality → WebP compresses better).
// ---------------------------------------------------------------------------

fn morton_order(pos: &[f32]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..(pos.len() / 3) as u32).collect();
    sort_morton(pos, &mut idx);
    idx
}

fn part1by2(mut v: u32) -> u32 {
    v &= 0x3ff;
    v = (v ^ (v << 16)) & 0xff00_00ff;
    v = (v ^ (v << 8)) & 0x0300_f00f;
    v = (v ^ (v << 4)) & 0x030c_30c3;
    v = (v ^ (v << 2)) & 0x0924_9249;
    v
}

/// Port of `sortMortonInterleaved`: 10-bit codes per axis; groups sharing a
/// code with more than 256 elements are refined recursively.
fn sort_morton(pos: &[f32], range: &mut [u32]) {
    if range.is_empty() {
        return;
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for &i in range.iter() {
        for a in 0..3 {
            let v = pos[i as usize * 3 + a];
            mn[a] = mn[a].min(v);
            mx[a] = mx[a].max(v);
        }
    }
    let len = [0, 1, 2].map(|a| mx[a] - mn[a]);
    if len.iter().any(|l| !l.is_finite()) || len.iter().all(|&l| l == 0.0) {
        return;
    }
    let mul = len.map(|l| if l == 0.0 { 0.0 } else { 1024.0 / l });
    let code = |i: u32| {
        let c = [0, 1, 2].map(|a| ((pos[i as usize * 3 + a] - mn[a]) * mul[a]).min(1023.0) as u32);
        (part1by2(c[2]) << 2) + (part1by2(c[1]) << 1) + part1by2(c[0])
    };
    let mut keyed: Vec<(u32, u32)> = range.iter().map(|&i| (code(i), i)).collect();
    keyed.sort_by_key(|&(k, _)| k);
    for (dst, &(_, i)) in range.iter_mut().zip(&keyed) {
        *dst = i;
    }
    let mut start = 0;
    while start < keyed.len() {
        let mut end = start + 1;
        while end < keyed.len() && keyed[end].0 == keyed[start].0 {
            end += 1;
        }
        if end - start > 256 {
            sort_morton(pos, &mut range[start..end]);
        }
        start = end;
    }
}

// ---------------------------------------------------------------------------
// Optimal 1D quantizer (dynamic programming over a histogram), direct port of
// `quantize1dColumns` (k = 256, alpha = 0.5). The columns share one codebook.
// ---------------------------------------------------------------------------

pub struct Quantized {
    pub centroids: Vec<f32>,
    pub labels: Vec<Vec<u8>>,
}

pub fn quantize_1d(columns: &[Vec<f32>], k: usize) -> Quantized {
    let rows = columns.first().map_or(0, Vec::len);
    let data: Vec<f32> = columns.concat();
    let n = data.len();
    if n == 0 {
        return Quantized { centroids: vec![0.0; k], labels: columns.iter().map(|_| vec![0; rows]).collect() };
    }
    let mut sorted = data.clone();
    sorted.par_sort_unstable_by(f32::total_cmp);
    let (vmin, vmax) = (sorted[0] as f64, sorted[n - 1] as f64);
    if vmax - vmin < 1e-20 {
        return Quantized { centroids: vec![vmin as f32; k], labels: columns.iter().map(|_| vec![0; rows]).collect() };
    }

    let h = n.min(1024);
    let range = vmax - vmin;
    let iqr = sorted[n * 3 / 4] as f64 - sorted[n / 4] as f64;
    let beta = (1.0 - iqr / range).clamp(0.5, 0.999);
    let mut counts = vec![0f64; h];
    let mut sums = vec![0f64; h];
    for (i, &v) in sorted.iter().enumerate() {
        let uniform = (v as f64 - vmin) / range;
        let quantile = i as f64 / n as f64;
        let bin = ((h as f64 * (beta * quantile + (1.0 - beta) * uniform)) as usize).min(h - 1);
        counts[bin] += 1.0;
        sums[bin] += v as f64;
    }
    let centers: Vec<f64> = (0..h)
        .map(|i| if counts[i] > 0.0 { sums[i] / counts[i] } else { vmin + (i as f64 + 0.5) / h as f64 * range })
        .collect();
    let weights: Vec<f64> = counts.iter().map(|&c| if c > 0.0 { c.powf(0.5) } else { 0.0 }).collect();
    let mut pw = vec![0f64; h + 1];
    let mut pwx = vec![0f64; h + 1];
    let mut pwxx = vec![0f64; h + 1];
    for i in 0..h {
        pw[i + 1] = pw[i] + weights[i];
        pwx[i + 1] = pwx[i] + weights[i] * centers[i];
        pwxx[i + 1] = pwxx[i] + weights[i] * centers[i] * centers[i];
    }
    let cost = |a: usize, b: usize| {
        let w = pw[b + 1] - pw[a];
        if w <= 0.0 {
            return 0.0;
        }
        let wx = pwx[b + 1] - pwx[a];
        pwxx[b + 1] - pwxx[a] - wx * wx / w
    };
    let mean = |a: usize, b: usize| {
        let w = pw[b + 1] - pw[a];
        if w <= 0.0 { (centers[a] + centers[b]) * 0.5 } else { (pwx[b + 1] - pwx[a]) / w }
    };

    let eff_k = k.min(counts.iter().filter(|&&c| c > 0.0).count());
    const INF: f64 = 1e30;
    let mut prev: Vec<f64> = (0..h).map(|j| cost(0, j)).collect();
    let mut curr = vec![INF; h];
    let mut split: Vec<Vec<i32>> = vec![Vec::new(); eff_k + 1];
    split[1] = vec![-1; h];
    for m in 2..=eff_k {
        curr.fill(INF);
        let mut sm = vec![0i32; h];
        for j in m - 1..h {
            let (mut best, mut best_s) = (INF, m - 2);
            for s in m - 2..j {
                let c = prev[s] + cost(s + 1, j);
                if c < best {
                    best = c;
                    best_s = s;
                }
            }
            curr[j] = best;
            sm[j] = best_s as i32;
        }
        split[m] = sm;
        std::mem::swap(&mut prev, &mut curr);
    }

    let mut cents = vec![0f32; eff_k];
    let mut j = h as i32 - 1;
    for m in (1..=eff_k).rev() {
        let s = if m > 1 { split[m][j as usize] } else { -1 };
        cents[m - 1] = mean((s + 1) as usize, j as usize) as f32;
        j = s;
    }
    cents.sort_by(f32::total_cmp);
    let mut centroids = cents.clone();
    centroids.resize(k, *cents.last().unwrap());

    let label = |v: f32| -> u8 {
        let (mut lo, mut hi) = (0usize, k - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if v < (centroids[mid] + centroids[mid + 1]) * 0.5 {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo as u8
    };
    let labels = columns.iter().map(|c| c.par_iter().map(|&v| label(v)).collect()).collect();
    Quantized { centroids, labels }
}

// ---------------------------------------------------------------------------
// SH bands: palette via two-level k-means.
// ---------------------------------------------------------------------------

struct ShPalette {
    count: usize,
    labels: Vec<u16>,
    codebook: Vec<f32>,
    /// `shN_centroids` texture: 64 centroids per row, `coeffs` texels each.
    centroids_rgba: Vec<u8>,
}

fn encode_sh(s: &Splats) -> ShPalette {
    let n = s.count;
    let d = s.rest_len();
    let coeffs = sh_coeffs(s.sh_degree);
    // Same palette size as splat-transform.
    let palette = ((2f64.powf((n as f64 / 1024.0).log2().floor()).min(64.0) * 1024.0) as usize).clamp(1, n.min(65_536));
    let (centroids, labels) = kmeans_two_level(&s.rest, n, d, palette);
    let count = centroids.len() / d;

    // Single codebook over every palette column.
    let cols: Vec<Vec<f32>> = (0..d).map(|j| (0..count).map(|i| centroids[i * d + j]).collect()).collect();
    let q = quantize_1d(&cols, 256);
    let rows = count.div_ceil(64);
    let mut rgba = vec![0u8; 64 * coeffs * rows * 4];
    for i in 0..count {
        for j in 0..coeffs {
            let o = (i * coeffs + j) * 4;
            rgba[o] = q.labels[j][i];
            rgba[o + 1] = q.labels[coeffs + j][i];
            rgba[o + 2] = q.labels[2 * coeffs + j][i];
            rgba[o + 3] = 0xff;
        }
    }
    ShPalette { count, labels, codebook: q.centroids, centroids_rgba: rgba }
}

fn nearest(point: &[f32], centroids: &[f32], d: usize) -> (u32, f32) {
    let mut best = (0u32, f32::INFINITY);
    for (c, cent) in centroids.chunks_exact(d).enumerate() {
        let mut dist = 0f32;
        for k in 0..d {
            let t = point[k] - cent[k];
            dist += t * t;
            if dist >= best.1 {
                break;
            }
        }
        if dist < best.1 {
            best = (c as u32, dist);
        }
    }
    best
}

/// Lloyd over the `members` points of `data`, with `k` initial centroids
/// evenly spaced through the list. Returns the centroids and each member's
/// local label.
fn lloyd(data: &[f32], d: usize, members: &[u32], k: usize, iterations: usize) -> (Vec<f32>, Vec<u32>) {
    let k = k.min(members.len()).max(1);
    let mut cent: Vec<f32> = (0..k)
        .flat_map(|c| {
            let m = members[c * members.len() / k] as usize;
            data[m * d..(m + 1) * d].iter().copied()
        })
        .collect();
    let mut labels = vec![0u32; members.len()];
    for it in 0..iterations {
        let changed: usize = members
            .iter()
            .zip(labels.iter_mut())
            .map(|(&m, l)| {
                let (c, _) = nearest(&data[m as usize * d..(m as usize + 1) * d], &cent, d);
                let ch = (*l != c) as usize;
                *l = c;
                ch
            })
            .sum();
        if it > 0 && changed == 0 {
            break;
        }
        let mut sums = vec![0f64; k * d];
        let mut counts = vec![0u32; k];
        for (&m, &l) in members.iter().zip(&labels) {
            counts[l as usize] += 1;
            for j in 0..d {
                sums[l as usize * d + j] += data[m as usize * d + j] as f64;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    cent[c * d + j] = (sums[c * d + j] / counts[c] as f64) as f32;
                }
            } else {
                // Empty centroid: restart at an arbitrary point of the group.
                let m = members[(c * 7919 + it * 104_729) % members.len()] as usize;
                cent[c * d..(c + 1) * d].copy_from_slice(&data[m * d..(m + 1) * d]);
            }
        }
    }
    (cent, labels)
}

/// Hierarchical k-means: coarse groups on a sample, every point assigned to
/// them, then each group subdivided with centroids proportional to its
/// population (sum = `k`).
fn kmeans_two_level(data: &[f32], n: usize, d: usize, k: usize) -> (Vec<f32>, Vec<u16>) {
    let coarse_k = COARSE_CLUSTERS.min(k);
    let step = (n / COARSE_SAMPLE).max(1);
    let sample: Vec<u32> = (0..n as u32).step_by(step).collect();
    let (coarse, _) = lloyd(data, d, &sample, coarse_k, KMEANS_ITERATIONS);
    let coarse_k = coarse.len() / d;

    let assign: Vec<u32> = (0..n).into_par_iter().map(|i| nearest(&data[i * d..(i + 1) * d], &coarse, d).0).collect();
    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); coarse_k];
    for (i, &g) in assign.iter().enumerate() {
        groups[g as usize].push(i as u32);
    }
    groups.retain(|g| !g.is_empty());

    // Distribute the k centroids proportionally (at least 1 per group).
    let mut alloc: Vec<usize> = groups.iter().map(|g| ((g.len() * k) as f64 / n as f64).floor().max(1.0) as usize).collect();
    let mut total: usize = alloc.iter().sum();
    let mut by_size: Vec<usize> = (0..groups.len()).collect();
    by_size.sort_by_key(|&g| std::cmp::Reverse(groups[g].len()));
    let mut cursor = 0;
    while total < k && !by_size.is_empty() {
        let g = by_size[cursor % by_size.len()];
        if alloc[g] < groups[g].len() {
            alloc[g] += 1;
            total += 1;
        }
        cursor += 1;
        if cursor > by_size.len() * 4 && total < k {
            break;
        }
    }
    while total > k {
        let g = by_size[cursor % by_size.len()];
        if alloc[g] > 1 {
            alloc[g] -= 1;
            total -= 1;
        }
        cursor += 1;
    }

    let fine: Vec<(Vec<f32>, Vec<u32>)> = groups
        .par_iter()
        .zip(alloc.par_iter())
        .map(|(members, &kc)| lloyd(data, d, members, kc, KMEANS_ITERATIONS))
        .collect();

    let mut centroids = Vec::with_capacity(k * d);
    let mut labels = vec![0u16; n];
    for (members, (cent, local)) in groups.iter().zip(fine) {
        let base = (centroids.len() / d) as u32;
        for (&m, &l) in members.iter().zip(&local) {
            labels[m as usize] = (base + l) as u16;
        }
        centroids.extend(cent);
    }
    (centroids, labels)
}

// ---------------------------------------------------------------------------
// Lossless WebP.
// ---------------------------------------------------------------------------

/// libwebp lossless effort (0–9). On a real Brush splat, 9 gave files only
/// 0.4% smaller than cwebp's default, and much more slowly.
const WEBP_EFFORT: i32 = 6;

/// Like splat-transform's `webp_encode_lossless_rgba_level`: lossless preset
/// and `exact`, so the encoder doesn't alter the RGB of texels with alpha 0
/// (opacity rounded to zero in `sh0`).
fn webp_lossless(rgba: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
    use libwebp_sys::*;
    // SAFETY: structs are initialized by the libwebp functions; the writer
    // buffer is freed on every path.
    unsafe {
        let mut config = WebPConfig::new().map_err(|_| anyhow::anyhow!("WebPConfigInit failed"))?;
        if WebPConfigLosslessPreset(&mut config, WEBP_EFFORT) == 0 {
            bail!("WebPConfigLosslessPreset failed");
        }
        config.exact = 1;
        let mut pic = WebPPicture::new().map_err(|_| anyhow::anyhow!("WebPPictureInit failed"))?;
        pic.use_argb = 1;
        pic.width = w as i32;
        pic.height = h as i32;
        if WebPPictureImportRGBA(&mut pic, rgba.as_ptr(), (w * 4) as i32) == 0 {
            WebPPictureFree(&mut pic);
            bail!("WebPPictureImportRGBA failed");
        }
        let mut writer = std::mem::zeroed::<WebPMemoryWriter>();
        WebPMemoryWriterInit(&mut writer);
        pic.writer = Some(WebPMemoryWrite);
        pic.custom_ptr = &mut writer as *mut _ as *mut std::ffi::c_void;
        let ok = WebPEncode(&config, &mut pic);
        let code = pic.error_code;
        WebPPictureFree(&mut pic);
        let out = if ok != 0 && writer.size > 0 { Some(std::slice::from_raw_parts(writer.mem, writer.size).to_vec()) } else { None };
        WebPMemoryWriterClear(&mut writer);
        out.ok_or_else(|| anyhow::anyhow!("WebP encoding failed ({code:?})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::ply::tests::sample;

    #[test]
    fn quantizer_is_monotonic_and_close() {
        let col: Vec<f32> = (0..10_000).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
        let q = quantize_1d(std::slice::from_ref(&col), 256);
        assert!(q.centroids.windows(2).all(|w| w[0] <= w[1]));
        let err = col.iter().zip(&q.labels[0]).map(|(v, &l)| (v - q.centroids[l as usize]).abs()).fold(0f32, f32::max);
        assert!(err < 0.1, "max error {err}");
    }

    #[test]
    fn quat_packing_roundtrip() {
        let q = [0.3f32, -0.5, 0.6, 0.2];
        let p = pack_quat(&q);
        let l = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let max = (p[3] - 252) as usize;
        assert_eq!(max, 2);
        // Decode like the reader does: (b/255 − 0.5)·2/√2.
        let dec = |b: u8| (b as f32 / 255.0 - 0.5) * 2.0 / std::f32::consts::SQRT_2;
        let rest = [0, 1, 3].map(|i| q[i] / l * q[2].signum());
        for (k, &r) in rest.iter().enumerate() {
            assert!((dec(p[k]) - r).abs() < 0.01);
        }
    }

    #[test]
    fn kmeans_labels_in_range() {
        let s = sample(5_000, 1);
        let (cent, labels) = kmeans_two_level(&s.rest, s.count, 9, 512);
        let count = cent.len() / 9;
        assert!(count <= 512 && count > 256, "{count}");
        assert!(labels.iter().all(|&l| (l as usize) < count));
    }

    #[test]
    fn writes_a_loadable_bundle() {
        let dir = std::env::temp_dir().join(format!("splatforge-sog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.sog");
        let size = write(&p, &sample(3_000, 1), |_, _| {}).unwrap();
        assert!(size > 0);
        let mut zip = zip::ZipArchive::new(std::fs::File::open(&p).unwrap()).unwrap();
        let mut meta = String::new();
        std::io::Read::read_to_string(&mut zip.by_name("meta.json").unwrap(), &mut meta).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&meta).unwrap();
        assert_eq!(meta["count"], 3000);
        assert_eq!(meta["shN"]["bands"], 1);
        assert_eq!(meta["scales"]["codebook"].as_array().unwrap().len(), 256);
        assert!(zip.by_name("shN_labels.webp").is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }
}
