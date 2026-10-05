//! Lê o modelo binário do COLMAP para enquadrar o visualizador: centro do
//! objeto, eixo "para cima" da órbita e a trajetória das câmeras.

use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::Result;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneFrame {
    /// Mediana dos pontos esparsos (robusta a outliers do fundo).
    pub center: [f64; 3],
    /// Média do "para cima" das câmeras, normalizada.
    pub up: [f64; 3],
    /// Distância mediana câmera → centro.
    pub radius: f64,
    /// Centros das câmeras registradas, na ordem dos frames.
    pub cameras: Vec<[f64; 3]>,
    /// Índice do frame de cada câmera (lacunas = frames que o COLMAP descartou).
    pub frame_indices: Vec<u32>,
}

struct Bin<R: Read>(R);

impl<R: Read> Bin<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; N];
        self.0.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.bytes()?))
    }
    fn skip(&mut self, n: u64) -> Result<()> {
        std::io::copy(&mut (&mut self.0).take(n), &mut std::io::sink())?;
        Ok(())
    }
    fn cstr(&mut self) -> Result<String> {
        let mut s = Vec::new();
        loop {
            let [b] = self.bytes::<1>()?;
            if b == 0 {
                break;
            }
            s.push(b);
        }
        Ok(String::from_utf8_lossy(&s).into_owned())
    }
}

fn open(p: &Path) -> Result<Bin<BufReader<std::fs::File>>> {
    Ok(Bin(BufReader::new(std::fs::File::open(p)?)))
}

/// Rotação mundo→câmera a partir do quatérnio (w, x, y, z) do COLMAP.
fn rotation(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [w, x, y, z] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

pub fn read(model: &Path) -> Result<SceneFrame> {
    // images.bin: centro = -Rᵀt; "para cima" da câmera COLMAP = -Y → Rᵀ(0,-1,0).
    let mut r = open(&model.join("images.bin"))?;
    let n = r.u64()?;
    let mut cams: Vec<(String, [f64; 3])> = Vec::with_capacity(n as usize);
    let mut up = [0.0; 3];
    for _ in 0..n {
        let _id = r.u32()?;
        let q = [r.f64()?, r.f64()?, r.f64()?, r.f64()?];
        let t = [r.f64()?, r.f64()?, r.f64()?];
        let _cam = r.u32()?;
        let name = r.cstr()?;
        let n2d = r.u64()?;
        r.skip(n2d * 24)?; // x f64, y f64, point3D_id i64
        let m = rotation(q);
        let c = [
            -(m[0][0] * t[0] + m[1][0] * t[1] + m[2][0] * t[2]),
            -(m[0][1] * t[0] + m[1][1] * t[1] + m[2][1] * t[2]),
            -(m[0][2] * t[0] + m[1][2] * t[1] + m[2][2] * t[2]),
        ];
        for k in 0..3 {
            up[k] -= m[1][k];
        }
        cams.push((name, c));
    }
    cams.sort_by(|a, b| a.0.cmp(&b.0));
    // Os frames são gravados como 00000.jpg, 00001.jpg, …
    let frame_indices: Vec<u32> = cams
        .iter()
        .enumerate()
        .map(|(i, (name, _))| {
            Path::new(name)
                .file_stem()
                .and_then(|s| s.to_str()?.parse().ok())
                .unwrap_or(i as u32)
        })
        .collect();
    let cameras: Vec<[f64; 3]> = cams.into_iter().map(|(_, c)| c).collect();
    let len = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt().max(1e-9);
    let up = [up[0] / len, up[1] / len, up[2] / len];

    // points3D.bin: só as coordenadas importam.
    let mut r = open(&model.join("points3D.bin"))?;
    let np = r.u64()?;
    let (mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..np {
        let _id = r.u64()?;
        xs.push(r.f64()?);
        ys.push(r.f64()?);
        zs.push(r.f64()?);
        r.skip(3 + 8)?; // rgb, erro
        let track = r.u64()?;
        r.skip(track * 8)?;
    }
    let center = if xs.is_empty() {
        let n = cameras.len().max(1) as f64;
        let s = cameras.iter().fold([0.0; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]]);
        [s[0] / n, s[1] / n, s[2] / n]
    } else {
        [median(xs), median(ys), median(zs)]
    };

    let radius = median(
        cameras
            .iter()
            .map(|c| ((c[0] - center[0]).powi(2) + (c[1] - center[1]).powi(2) + (c[2] - center[2]).powi(2)).sqrt())
            .collect(),
    );

    Ok(SceneFrame { center, up, radius, cameras, frame_indices })
}

/// Copia `images.bin` trocando o nome de cada imagem (o resto é preservado
/// byte a byte).
pub fn rewrite_image_names(src: &Path, dst: &Path, rename: impl Fn(&str) -> String) -> Result<()> {
    use std::io::Write;
    let mut r = open(src)?;
    let mut w = std::io::BufWriter::new(std::fs::File::create(dst)?);
    let n = r.u64()?;
    w.write_all(&n.to_le_bytes())?;
    for _ in 0..n {
        // image_id u32 + qvec 4×f64 + tvec 3×f64 + camera_id u32
        let head: [u8; 4 + 32 + 24 + 4] = r.bytes()?;
        w.write_all(&head)?;
        let name = r.cstr()?;
        w.write_all(rename(&name).as_bytes())?;
        w.write_all(&[0])?;
        let n2d = r.u64()?;
        w.write_all(&n2d.to_le_bytes())?;
        std::io::copy(&mut (&mut r.0).take(n2d * 24), &mut w)?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Roda com `SPLATFORGE_TEST_MODEL=<dataset>/sparse/0 cargo test -- --ignored`.
    /// Numa órbita a altura constante, todas as câmeras devem ficar na mesma
    /// "altura" ao longo do eixo `up` estimado.
    #[test]
    #[ignore]
    fn orbit_cameras_share_height() {
        let model = std::env::var("SPLATFORGE_TEST_MODEL").expect("SPLATFORGE_TEST_MODEL");
        let f = super::read(std::path::Path::new(&model)).unwrap();
        let h: Vec<f64> = f
            .cameras
            .iter()
            .map(|c| (0..3).map(|k| (c[k] - f.center[k]) * f.up[k]).sum())
            .collect();
        let mean = h.iter().sum::<f64>() / h.len() as f64;
        let spread = h.iter().map(|x| (x - mean).abs()).fold(0.0, f64::max);
        println!("radius={:.3} height={mean:.3} spread={spread:.4} n={}", f.radius, h.len());
        assert!(spread < 0.05 * f.radius, "spread {spread} vs radius {}", f.radius);
    }
}
