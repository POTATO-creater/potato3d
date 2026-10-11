//! Lê o modelo binário do COLMAP para enquadrar o visualizador: centro do
//! objeto, eixo "para cima" da órbita e a trajetória das câmeras.

use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result};
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
    /// Lente do vídeo, para o visualizador enquadrar como a filmagem.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lens: Option<Lens>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Lens {
    /// Campo de visão vertical, em radianos.
    pub fov_y: f64,
    /// Largura / altura.
    pub aspect: f64,
}

/// Lente da primeira câmera do cameras.bin.
fn read_lens(path: &Path) -> Result<Option<Lens>> {
    let mut r = open(path)?;
    if r.u64()? == 0 {
        return Ok(None);
    }
    let _id = r.u32()?;
    let model = r.u32()?;
    let (w, h) = (r.u64()? as f64, r.u64()? as f64);
    let (f0, f1) = (r.f64()?, r.f64()?);
    // Modelos com uma focal só (f, cx, cy, …); os demais começam com fx, fy.
    let fy = if matches!(model, 0 | 2 | 3 | 8 | 9) { f0 } else { f1 };
    if !(w > 0.0 && h > 0.0 && fy > 0.0) {
        return Ok(None);
    }
    Ok(Some(Lens { fov_y: 2.0 * (h / 2.0 / fy).atan(), aspect: w / h }))
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

    let lens = read_lens(&model.join("cameras.bin")).ok().flatten();

    Ok(SceneFrame { center, up, radius, cameras, frame_indices, lens })
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

type V3 = [f64; 3];
type M3 = [[f64; 3]; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: V3) -> V3 {
    let l = dot(a, a).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}
fn mul(a: &M3, b: &M3) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}
fn apply(m: &M3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// Menor rotação que leva o vetor unitário `a` em `b` (Rodrigues).
fn rotation_between(a: V3, b: V3) -> M3 {
    let v = cross(a, b);
    let k = [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]];
    let k2 = mul(&k, &k);
    let s = 1.0 / (1.0 + dot(a, b));
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (i == j) as u8 as f64 + k[i][j] + k2[i][j] * s;
        }
    }
    m
}

/// Quatérnio (w, x, y, z) de uma matriz de rotação (Shepperd).
fn quaternion(m: &M3) -> [f64; 4] {
    let tr = m[0][0] + m[1][1] + m[2][2];
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]
    };
    if q[0] < 0.0 {
        [-q[0], -q[1], -q[2], -q[3]]
    } else {
        q
    }
}

/// Força a órbita plana: projeta o centro de cada câmera no plano da órbita
/// e gira a câmera para que o sujeito continue no mesmo pixel. Subir a câmera
/// e incliná-la para baixo dão imagens quase iguais de um objeto compacto, e
/// essa é a troca que o COLMAP às vezes faz errado com pouca textura. Grava o
/// modelo corrigido em `out`; os pontos 3D ficam como estão e devem passar
/// por um bundle adjustment com as poses fixas. Retorna o maior desvio de
/// altura removido, como fração do raio da órbita.
pub fn flatten_orbit(model: &Path, out: &Path) -> Result<f64> {
    use std::io::Write;
    let frame = read(model)?;
    let target = frame.center;

    let mut r = open(&model.join("images.bin"))?;
    let n = r.u64()?;
    let mut poses: Vec<(u32, String, [f64; 4], V3)> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let id = r.u32()?;
        let q = [r.f64()?, r.f64()?, r.f64()?, r.f64()?];
        let t = [r.f64()?, r.f64()?, r.f64()?];
        let _cam = r.u32()?;
        let name = r.cstr()?;
        let n2d = r.u64()?;
        r.skip(n2d * 24)?;
        poses.push((id, name, q, t));
    }
    if poses.len() < 3 {
        anyhow::bail!("câmeras demais fora do modelo para ajustar um plano");
    }
    poses.sort_by(|a, b| a.1.cmp(&b.1));
    let centers: Vec<V3> = poses
        .iter()
        .map(|(_, _, q, t)| {
            let m = rotation(*q);
            [-dot([m[0][0], m[1][0], m[2][0]], *t), -dot([m[0][1], m[1][1], m[2][1]], *t), -dot([m[0][2], m[1][2], m[2][2]], *t)]
        })
        .collect();
    let k = centers.len() as f64;
    let mean = centers.iter().fold([0.0; 3], |a, c| [a[0] + c[0] / k, a[1] + c[1] / k, a[2] + c[2] / k]);
    // Normal de Newell: soma dos produtos vetoriais de câmeras consecutivas.
    // Vale para arcos parciais e não precisa de decomposição em autovalores.
    let mut normal = [0.0; 3];
    for w in centers.windows(2) {
        let c = cross(sub(w[0], mean), sub(w[1], mean));
        normal = [normal[0] + c[0], normal[1] + c[1], normal[2] + c[2]];
    }
    let area = dot(normal, normal).sqrt();
    let spread: f64 = centers.iter().map(|c| dot(sub(*c, mean), sub(*c, mean))).sum::<f64>() / k;
    if area < 1e-3 * spread * k {
        anyhow::bail!("as câmeras quase não formam um arco; não dá para definir o plano da órbita");
    }
    let normal = unit(normal);

    let mut fixed = std::collections::HashMap::new();
    let mut max_h = 0f64;
    for ((id, _, q, _), c) in poses.iter().zip(&centers) {
        let h = dot(sub(*c, mean), normal);
        max_h = max_h.max(h.abs());
        let c2 = sub(*c, [normal[0] * h, normal[1] * h, normal[2] * h]);
        let rot = mul(&rotation(*q), &rotation_between(unit(sub(target, c2)), unit(sub(target, *c))));
        let t = apply(&rot, c2);
        fixed.insert(*id, (quaternion(&rot), [-t[0], -t[1], -t[2]]));
    }
    let pose_bytes = |(q, t): &([f64; 4], V3)| -> Vec<u8> {
        q.iter().chain(t.iter()).flat_map(|v| v.to_le_bytes()).collect()
    };

    std::fs::create_dir_all(out)?;
    for f in ["cameras.bin", "rigs.bin", "points3D.bin"] {
        std::fs::copy(model.join(f), out.join(f))?;
    }

    let mut r = open(&model.join("images.bin"))?;
    let mut w = std::io::BufWriter::new(std::fs::File::create(out.join("images.bin"))?);
    let n = r.u64()?;
    w.write_all(&n.to_le_bytes())?;
    for _ in 0..n {
        let id = r.u32()?;
        r.skip(56)?;
        w.write_all(&id.to_le_bytes())?;
        w.write_all(&pose_bytes(&fixed[&id]))?;
        let rest = r.u32()?; // camera_id
        w.write_all(&rest.to_le_bytes())?;
        let name = r.cstr()?;
        w.write_all(name.as_bytes())?;
        w.write_all(&[0])?;
        let n2d = r.u64()?;
        w.write_all(&n2d.to_le_bytes())?;
        std::io::copy(&mut (&mut r.0).take(n2d * 24), &mut w)?;
    }
    w.flush()?;

    // No COLMAP 4 a pose que vale é a do frame (rig_from_world); com uma
    // câmera por rig ela é a própria pose da imagem.
    let mut r = open(&model.join("frames.bin"))?;
    let mut w = std::io::BufWriter::new(std::fs::File::create(out.join("frames.bin"))?);
    let n = r.u64()?;
    w.write_all(&n.to_le_bytes())?;
    for _ in 0..n {
        let ids: [u8; 8] = r.bytes()?; // frame_id + rig_id
        r.skip(56)?;
        let nd = r.u32()?;
        let mut data = Vec::with_capacity(nd as usize);
        for _ in 0..nd {
            let kind = i32::from_le_bytes(r.bytes()?);
            let sensor = r.u32()?;
            let data_id = r.u64()?;
            data.push((kind, sensor, data_id));
        }
        let [(0, _, image_id)] = data[..] else {
            anyhow::bail!("frame com mais de um sensor (rig); órbita plana não suportada");
        };
        let pose = fixed.get(&(image_id as u32)).context("frame sem imagem correspondente")?;
        w.write_all(&ids)?;
        w.write_all(&pose_bytes(pose))?;
        w.write_all(&nd.to_le_bytes())?;
        for (kind, sensor, data_id) in data {
            w.write_all(&kind.to_le_bytes())?;
            w.write_all(&sensor.to_le_bytes())?;
            w.write_all(&data_id.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(max_h / frame.radius.max(1e-12))
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

    /// Mesmo modelo de teste: achatar de novo um modelo já achatado não
    /// encontra mais desvio de altura. Grava em `SPLATFORGE_FLAT_OUT` se definido.
    #[test]
    #[ignore]
    fn flatten_orbit_levels_cameras() {
        let model = std::env::var("SPLATFORGE_TEST_MODEL").expect("SPLATFORGE_TEST_MODEL");
        let out = std::env::var("SPLATFORGE_FLAT_OUT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("splatforge_flat_test"));
        let dev = super::flatten_orbit(std::path::Path::new(&model), &out).unwrap();
        let again = super::flatten_orbit(&out, &out.with_extension("again")).unwrap();
        println!("desvio corrigido={:.3}% raio; residual={again:.2e}", dev * 100.0);
        assert!(again < 1e-9);
    }
}
