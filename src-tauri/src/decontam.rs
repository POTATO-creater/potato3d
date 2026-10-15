//! Edge decontamination of the training images.
//!
//! A soft mask edge keeps the photo's color, which is a mix of subject and
//! background: `I = m·F + (1 − m)·B`. Brush premultiplies by the alpha, so the
//! background leaks into the training target and the splat learns a light
//! fringe around hair and outlines (measured on a white-wall video: edges 25%
//! brighter than the interior next to them; 0.6% after this pass).
//!
//! Recovers the subject color `F = (I − (1 − a)·B) / a`, with `B` estimated
//! from the background pixels nearby. The mixing ratio `a` is the training
//! alpha (the eroded mask), not the raw matte: mattes leak a little past the
//! real outline, so the raw value underestimates the background share — on a
//! white-wall video it left half of the fringe (1.13 vs 1.00). Where `a` is too
//! small for the solve to be stable, `F` is the color of the solid subject
//! nearby instead (little of it reaches the target anyway: it is weighted by `a`).
//!
//! Both fields are normalized averages over a grid of `BLOCK`² pixel blocks,
//! blurred, so the pass is O(pixels). Same algorithm as
//! `web/src/pipeline/decontam.ts`.

use image::{GrayImage, RgbaImage};

/// Grid cell, in pixels.
const BLOCK: u32 = 4;
/// Box blur radius on the grid (cells) and number of passes: ~±24 px support.
const BLUR_RADIUS: i64 = 3;
const BLUR_PASSES: usize = 2;
/// Mask values (0–255) counted as clear background, and the solid subject: only
/// a full mask, near-solid edge pixels are still mixed with the background.
const BACKGROUND_MAX: u8 = 5;
const SOLID: u8 = 255;
/// Below this alpha the unmixing is unstable: use the solid color nearby.
const UNMIX_MIN: f32 = 0.25;

/// Normalized average color of the pixels selected by `pick`, as a blurred
/// grid. Cells with no selected pixel nearby are `None`.
struct Field {
    gw: u32,
    gh: u32,
    color: Vec<Option<[f32; 3]>>,
}

impl Field {
    fn new(img: &RgbaImage, mask: &GrayImage, pick: impl Fn(u8) -> bool) -> Field {
        let (w, h) = img.dimensions();
        let (gw, gh) = (w.div_ceil(BLOCK), h.div_ceil(BLOCK));
        let n = (gw * gh) as usize;
        let mut sum = vec![[0f32; 4]; n]; // rgb·weight, weight
        for (x, y, px) in img.enumerate_pixels() {
            if pick(mask.get_pixel(x, y)[0]) {
                let s = &mut sum[((y / BLOCK) * gw + x / BLOCK) as usize];
                for c in 0..3 {
                    s[c] += px[c] as f32;
                }
                s[3] += 1.0;
            }
        }
        for _ in 0..BLUR_PASSES {
            sum = box_blur(&sum, gw as i64, gh as i64, 1, 0);
            sum = box_blur(&sum, gw as i64, gh as i64, 0, 1);
        }
        let color = sum
            .iter()
            .map(|s| (s[3] > 1e-3).then(|| [s[0] / s[3], s[1] / s[3], s[2] / s[3]]))
            .collect();
        Field { gw, gh, color }
    }

    fn at(&self, x: u32, y: u32) -> Option<[f32; 3]> {
        self.color[((y / BLOCK).min(self.gh - 1) * self.gw + (x / BLOCK).min(self.gw - 1)) as usize]
    }
}

/// Box blur along (dx, dy) with radius `BLUR_RADIUS`, edges clamped.
fn box_blur(src: &[[f32; 4]], w: i64, h: i64, dx: i64, dy: i64) -> Vec<[f32; 4]> {
    let mut out = vec![[0f32; 4]; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0f32; 4];
            for k in -BLUR_RADIUS..=BLUR_RADIUS {
                let sx = (x + k * dx).clamp(0, w - 1);
                let sy = (y + k * dy).clamp(0, h - 1);
                let s = src[(sy * w + sx) as usize];
                for c in 0..4 {
                    acc[c] += s[c];
                }
            }
            out[(y * w + x) as usize] = acc;
        }
    }
    out
}

/// Replaces the color of the pixels whose training alpha is partial
/// (`0 < alpha < 255`) with the subject color unmixed from the background.
/// `mask` is the raw matte (finds clear background and solid subject); `alpha`
/// is the training alpha (the eroded mask), also used as the mixing ratio.
/// Pixels with alpha 0 or 255 are untouched.
pub fn decontaminate(img: &mut RgbaImage, mask: &GrayImage, alpha: &GrayImage) {
    let background = Field::new(img, mask, |m| m <= BACKGROUND_MAX);
    let solid = Field::new(img, mask, |m| m == SOLID);
    for (x, y, px) in img.enumerate_pixels_mut() {
        let a = alpha.get_pixel(x, y)[0];
        if a == 0 || a == 255 {
            continue;
        }
        let m = a as f32 / 255.0;
        let i = [px[0] as f32, px[1] as f32, px[2] as f32];
        let unmix = |b: [f32; 3]| {
            let mm = m.max(UNMIX_MIN);
            std::array::from_fn::<f32, 3, _>(|c| ((i[c] - (1.0 - mm) * b[c]) / mm).clamp(0.0, 255.0))
        };
        // Below UNMIX_MIN the solve amplifies noise: take the solid neighbor.
        let f = match (background.at(x, y), solid.at(x, y)) {
            (_, Some(s)) if m < UNMIX_MIN => s,
            (Some(b), _) => unmix(b),
            _ => continue,
        };
        for c in 0..3 {
            px[c] = f[c].round().clamp(0.0, 255.0) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};

    /// Red subject on a white wall, soft 8 px edge; the matte leaks 1 px past the
    /// real outline (its erosion is the true mixing ratio). The photo's edge
    /// pixels are pink; after the pass they are red again, while the solid
    /// interior and the transparent background are untouched.
    #[test]
    fn unmixes_edges_back_to_the_subject_color() {
        let (w, h) = (96u32, 64u32);
        let (f, b) = ([200f32, 30.0, 40.0], [240f32, 240.0, 235.0]);
        let mask = GrayImage::from_fn(w, h, |x, _| {
            Luma([((x as f32 - 40.0) / 8.0 * 255.0).clamp(0.0, 255.0) as u8])
        });
        let alpha = crate::masks::erode(&mask, 1);
        let mut img = RgbaImage::from_fn(w, h, |x, y| {
            let m = alpha.get_pixel(x, y)[0] as f32 / 255.0;
            let c = |k: usize| (m * f[k] + (1.0 - m) * b[k]).round() as u8;
            Rgba([c(0), c(1), c(2), 255])
        });
        let before = img.clone();
        decontaminate(&mut img, &mask, &alpha);

        let dist = |p: &Rgba<u8>| (0..3).map(|c| (p[c] as f32 - f[c]).powi(2)).sum::<f32>().sqrt();
        for x in 41..48 {
            let a = alpha.get_pixel(x, 30)[0];
            if a > 0 && a < 255 {
                let (old, new) = (dist(before.get_pixel(x, 30)), dist(img.get_pixel(x, 30)));
                assert!(new < 12.0, "x={x}: edge still mixed, {:?}", img.get_pixel(x, 30));
                assert!(new < old, "x={x}: no improvement");
            }
        }
        assert_eq!(img.get_pixel(80, 30), before.get_pixel(80, 30), "solid interior untouched");
        assert_eq!(img.get_pixel(5, 30), before.get_pixel(5, 30), "background untouched");
    }

    /// No background anywhere (subject fills the frame): nothing to unmix from,
    /// colors stay.
    #[test]
    fn without_background_keeps_colors() {
        let mask = GrayImage::from_pixel(32, 32, Luma([200]));
        let alpha = GrayImage::from_pixel(32, 32, Luma([200]));
        let mut img = RgbaImage::from_pixel(32, 32, Rgba([10, 120, 230, 255]));
        let before = img.clone();
        decontaminate(&mut img, &mask, &alpha);
        assert_eq!(img, before);
    }
}
