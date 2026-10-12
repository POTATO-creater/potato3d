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

fn transpose(m: &M3) -> M3 {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Rotação de `th` em torno do eixo unitário `n` (Rodrigues).
fn about(n: V3, th: f64) -> M3 {
    let k = [[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]];
    let k2 = mul(&k, &k);
    let (s, c) = th.sin_cos();
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (i == j) as u8 as f64 + s * k[i][j] + (1.0 - c) * k2[i][j];
        }
    }
    m
}

fn slerp(a: [f64; 4], b: [f64; 4], u: f64) -> [f64; 4] {
    let mut d: f64 = (0..4).map(|k| a[k] * b[k]).sum();
    let sign = if d < 0.0 { -1.0 } else { 1.0 };
    d *= sign;
    let (mut wa, mut wb) = (1.0 - u, u * sign);
    if d < 0.9995 {
        let th = d.acos();
        wa = ((1.0 - u) * th).sin() / th.sin();
        wb = (u * th).sin() / th.sin() * sign;
    }
    let q: [f64; 4] = std::array::from_fn(|k| wa * a[k] + wb * b[k]);
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    q.map(|v| v / n)
}

/// Hermite cúbico com tangentes de Catmull-Rom não uniformes: segue a
/// velocidade dos vizinhos dos dois lados da lacuna.
fn hermite(s: &[f64], y: &[f64], a: usize, b: usize, at: f64) -> f64 {
    let slope = |i: usize, j: usize| (y[j] - y[i]) / (s[j] - s[i]);
    let ma = if a > 0 { slope(a - 1, b) } else { slope(a, b) };
    let mb = if b + 1 < s.len() { slope(a, b + 1) } else { slope(a, b) };
    let h = s[b] - s[a];
    let u = (at - s[a]) / h;
    let (u2, u3) = (u * u, u * u * u);
    (2.0 * u3 - 3.0 * u2 + 1.0) * y[a] + (u3 - 2.0 * u2 + u) * h * ma + (-2.0 * u3 + 3.0 * u2) * y[b] + (u3 - u2) * h * mb
}

/// Inclinação (mínimos quadrados) de y em s.
fn fit_slope(s: &[f64], y: &[f64]) -> f64 {
    let n = s.len() as f64;
    let (ms, my) = (s.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let num: f64 = s.iter().zip(y).map(|(a, b)| (a - ms) * (b - my)).sum();
    let den: f64 = s.iter().map(|a| (a - ms).powi(2)).sum();
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Instante de cada imagem, na ordem dos nomes; se faltar ou não crescer
/// (frames de imagens soltas), o número no nome (00012.jpg → 12), que guarda
/// os buracos dos frames recusados. Mesma regra de web/src/sfm/index.ts.
fn orbit_times(names: &[String], times: &std::collections::HashMap<String, f64>) -> Vec<f64> {
    let increasing = |v: &[f64]| v.windows(2).all(|w| w[1] > w[0]);
    let t: Option<Vec<f64>> = names.iter().map(|n| times.get(n).copied()).collect();
    if let Some(t) = t.filter(|t| increasing(t)) {
        return t;
    }
    let idx: Option<Vec<f64>> = names
        .iter()
        .map(|n| Path::new(n).file_stem()?.to_str()?.parse::<f64>().ok())
        .collect();
    idx.filter(|v| increasing(v)).unwrap_or_else(|| (0..names.len()).map(|i| i as f64).collect())
}

/// Correspondências 2D (pixels) → 3D de um frame, pelos matches com os
/// vizinhos registrados.
pub type Correspondences = Vec<([f64; 2], [f64; 3])>;

pub struct FillOptions<'a> {
    /// Órbita plana: a pose refinada volta para o plano da órbita.
    pub flat: bool,
    /// Por nome do frame não registrado; os que não têm ficam só inferidos.
    pub correspondences: &'a std::collections::HashMap<String, Correspondences>,
}

#[derive(Debug, Default)]
pub struct Filled {
    /// Frames que ganharam pose, na ordem dos nomes.
    pub inferred: Vec<String>,
    /// Desses, os refinados contra os pontos 3D, com o número de inliers.
    pub refined: Vec<(String, usize)>,
}

/// Focais e centro (pixels) da primeira câmera do cameras.bin. A distorção
/// radial dos modelos que a têm é ignorada (o app pede câmeras pinhole).
struct Intrinsics {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
}

fn read_intrinsics(path: &Path) -> Result<Intrinsics> {
    let mut r = open(path)?;
    anyhow::ensure!(r.u64()? > 0, "cameras.bin vazio");
    let _id = r.u32()?;
    let model = r.u32()?;
    let (_w, _h) = (r.u64()?, r.u64()?);
    let (a, b, c) = (r.f64()?, r.f64()?, r.f64()?);
    Ok(match model {
        // SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL: f, cx, cy, …
        0 | 2 | 3 => Intrinsics { fx: a, fy: a, cx: b, cy: c },
        // PINHOLE, OPENCV, …: fx, fy, cx, cy, …
        _ => Intrinsics { fx: a, fy: b, cx: c, cy: r.f64()? },
    })
}

/// Erro de reprojeção (pixels) de uma correspondência na pose (r, t).
fn reproj(k: &Intrinsics, r: &M3, t: V3, (uv, x): &([f64; 2], V3)) -> f64 {
    let p = add(apply(r, *x), t);
    if p[2] <= 1e-9 {
        return f64::INFINITY;
    }
    (k.fx * p[0] / p[2] + k.cx - uv[0]).hypot(k.fy * p[1] / p[2] + k.cy - uv[1])
}

/// Resolve o sistema 6×6 `a x = b` (eliminação com pivô parcial).
fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for col in 0..6 {
        let piv = (col..6).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..6 {
            let f = a[row][col] / a[col][col];
            for k in col..6 {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 6];
    for row in (0..6).rev() {
        x[row] = (b[row] - (row + 1..6).map(|k| a[row][k] * x[k]).sum::<f64>()) / a[row][row];
    }
    Some(x)
}

/// Mínimo de inliers (≤ 4 px) para aceitar uma pose refinada.
const REFINE_MIN_INLIERS: usize = 15;

/// Refina a pose inferida (mundo→câmera `m`, centro `c`) contra as
/// correspondências, partindo dela: a cada rodada só entram as que a pose
/// atual projeta perto (limiar de 32 px caindo até 4 px), num Gauss-Newton
/// amortecido com perda de Huber. Assim, matches errados longe da pose
/// inferida não puxam a câmera. O centro inferido entra como prior com
/// desvio `sigma` (mesmas unidades do modelo): onde os pontos vistos deixam
/// uma direção mal amarrada, vale a trajetória; onde amarram bem, os pontos.
/// Devolve a pose e os inliers finais, ou None se sobrarem poucos. Mesmo
/// algoritmo de web/src/sfm/orbit.ts: refinePose.
fn refine_pose(m: M3, c: V3, sigma: f64, corr: &[([f64; 2], V3)], k: &Intrinsics) -> Option<(M3, V3, usize)> {
    let mut r = m;
    let mut t = scale(apply(&m, c), -1.0);
    // Custo em px²: o prior vira (σ_px/σ)²·|centro − c|².
    let prior = (KEYPOINT_SIGMA / sigma.max(1e-12)).powi(2);
    let center = |r: &M3, t: V3| scale(apply(&transpose(r), t), -1.0);
    for thr in [32.0, 16.0, 8.0, 4.0] {
        let sel: Vec<&([f64; 2], V3)> = corr.iter().filter(|x| reproj(k, &r, t, x) < thr).collect();
        if sel.len() < REFINE_MIN_INLIERS {
            return None;
        }
        let delta = thr / 2.0;
        let cost = |r: &M3, t: V3| -> f64 {
            let d = sub(center(r, t), c);
            prior * dot(d, d)
                + sel
                    .iter()
                    .map(|x| {
                        let e = reproj(k, r, t, x).min(1e3);
                        if e <= delta { e * e } else { 2.0 * delta * e - delta * delta }
                    })
                    .sum::<f64>()
        };
        let mut lambda = 1e-3;
        let mut current = cost(&r, t);
        for _ in 0..100 {
            let (mut h, mut g) = ([[0.0; 6]; 6], [0.0; 6]);
            for (uv, x) in &sel {
                let rx = apply(&r, *x);
                let p = add(rx, t);
                if p[2] <= 1e-9 {
                    continue;
                }
                let iz = 1.0 / p[2];
                let e = [k.fx * p[0] * iz + k.cx - uv[0], k.fy * p[1] * iz + k.cy - uv[1]];
                let norm = e[0].hypot(e[1]);
                let w = if norm <= delta { 1.0 } else { delta / norm };
                // d(pixel)/d(p) · [−[RX]× | I], com p = exp(δr)·RX + t + δt.
                let jp = [[k.fx * iz, 0.0, -k.fx * p[0] * iz * iz], [0.0, k.fy * iz, -k.fy * p[1] * iz * iz]];
                let skew = [[0.0, rx[2], -rx[1]], [-rx[2], 0.0, rx[0]], [rx[1], -rx[0], 0.0]];
                let mut j = [[0.0; 6]; 2];
                for a in 0..2 {
                    for b in 0..3 {
                        j[a][b] = (0..3).map(|q| jp[a][q] * skew[q][b]).sum();
                        j[a][b + 3] = jp[a][b];
                    }
                }
                for a in 0..6 {
                    g[a] += w * (j[0][a] * e[0] + j[1][a] * e[1]);
                    for b in 0..6 {
                        h[a][b] += w * (j[0][a] * j[0][b] + j[1][a] * j[1][b]);
                    }
                }
            }
            // Prior: centro = −Rᵀt; d(centro) = −Rᵀ[t]×·δr − Rᵀ·δt.
            let rt = transpose(&r);
            let d = sub(center(&r, t), c);
            let tx = [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]];
            let rtx = mul(&rt, &tx);
            let jc: [[f64; 6]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| if b < 3 { -rtx[a][b] } else { -rt[a][b - 3] }));
            for a in 0..6 {
                g[a] += prior * (0..3).map(|q| jc[q][a] * d[q]).sum::<f64>();
                for b in 0..6 {
                    h[a][b] += prior * (0..3).map(|q| jc[q][a] * jc[q][b]).sum::<f64>();
                }
            }
            let mut damped = h;
            for a in 0..6 {
                damped[a][a] += lambda * h[a][a].max(1e-9);
            }
            let Some(d) = solve6(damped, g.map(|v| -v)) else { break };
            let w = [d[0], d[1], d[2]];
            let angle = dot(w, w).sqrt();
            let r2 = if angle > 1e-12 { mul(&about(scale(w, 1.0 / angle), angle), &r) } else { r };
            let t2 = add(t, [d[3], d[4], d[5]]);
            let next = cost(&r2, t2);
            if next < current {
                let done = current - next < 1e-9 * current.max(1e-12);
                (r, t, current) = (r2, t2, next);
                lambda = (lambda / 10.0).max(1e-9);
                if done {
                    break;
                }
            } else {
                lambda *= 10.0;
            }
        }
    }
    let inliers = corr.iter().filter(|x| reproj(k, &r, t, x) < 4.0).count();
    (inliers >= REFINE_MIN_INLIERS).then(|| (r, scale(apply(&transpose(&r), t), -1.0), inliers))
}

/// Ângulo, raio e altura da órbita no instante `t`: Hermite entre os
/// registrados vizinhos; antes do primeiro ou depois do último, extrapola o
/// ângulo com a velocidade dos 4 da ponta e usa o raio e a altura médios deles.
fn orbit_at(s: &[f64], th: &[f64], rad: &[f64], hgt: &[f64], t: f64) -> [f64; 3] {
    match s.iter().position(|&v| v > t) {
        Some(b) if b > 0 => [th, rad, hgt].map(|y| hermite(s, y, b - 1, b, t)),
        b => {
            let end = if b == Some(0) { 0 } else { s.len() - 1 };
            let ks: Vec<usize> = if end == 0 { (0..4.min(s.len())).collect() } else { (end.saturating_sub(3)..=end).collect() };
            let pick = |y: &[f64]| ks.iter().map(|&k| y[k]).collect::<Vec<f64>>();
            let mean = |y: &[f64]| ks.iter().map(|&k| y[k]).sum::<f64>() / ks.len() as f64;
            [th[end] + fit_slope(&pick(s), &pick(th)) * (t - s[end]), mean(rad), mean(hgt)]
        }
    }
}

/// Ruído típico de um ponto 2D casado (pixels): pesa as correspondências
/// contra a confiança na trajetória (`refine_pose`).
const KEYPOINT_SIGMA: f64 = 0.5;

/// Ângulo (radianos) da rotação relativa entre duas orientações.
fn rotation_angle(a: &M3, b: &M3) -> f64 {
    let m = mul(a, &transpose(b));
    ((m[0][0] + m[1][1] + m[2][2] - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
}

/// Dá pose aos frames de `images` que o modelo não registrou, a partir dos
/// vizinhos: em coordenadas da órbita (ângulo em torno do eixo, raio e
/// altura), por Hermite cúbico no instante de cada frame, ou seja, seguindo a
/// velocidade e o espaçamento da câmera; a orientação relativa à órbita vem
/// por slerp. Nas pontas, extrapola com a velocidade angular dos últimos
/// frames. Sem arco definido, interpola os centros em linha reta. Depois,
/// a pose inferida é refinada contra os pontos 3D que o frame enxerga
/// (`refine_pose`), se houver correspondências suficientes e o resultado não
/// se afastar demais dela. As imagens novas entram sem pontos 2D (só o treino
/// as usa). Mesmo algoritmo de web/src/sfm/orbit.ts: fillMissing.
pub fn fill_missing(
    model: &Path,
    images: &Path,
    times: &std::collections::HashMap<String, f64>,
    opts: &FillOptions,
) -> Result<Filled> {
    let mut names: Vec<String> = std::fs::read_dir(images)?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| {
            let ext = Path::new(n).extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
            matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "webp")
        })
        .collect();
    names.sort();

    // images.bin inteiro: as entradas existentes são copiadas byte a byte.
    let images_bin = std::fs::read(model.join("images.bin"))?;
    let mut r = Bin(&images_bin[..]);
    let count = r.u64()?;
    let (mut known, mut max_id, mut camera_id) = (std::collections::HashMap::new(), 0u32, 1u32);
    for _ in 0..count {
        let id = r.u32()?;
        let q = [r.f64()?, r.f64()?, r.f64()?, r.f64()?];
        let t = [r.f64()?, r.f64()?, r.f64()?];
        camera_id = r.u32()?;
        let name = r.cstr()?;
        let n2d = r.u64()?;
        r.skip(n2d * 24)?;
        max_id = max_id.max(id);
        known.insert(name, (q, t));
    }
    let missing: Vec<usize> = (0..names.len()).filter(|&i| !known.contains_key(&names[i])).collect();
    if missing.is_empty() {
        return Ok(Filled::default());
    }
    let reg: Vec<usize> = (0..names.len()).filter(|&i| known.contains_key(&names[i])).collect();
    if reg.len() < 3 {
        anyhow::bail!("poucas câmeras registradas para inferir as outras");
    }
    let at = orbit_times(&names, times);
    let s: Vec<f64> = reg.iter().map(|&i| at[i]).collect();
    let rots: Vec<M3> = reg.iter().map(|&i| rotation(known[&names[i]].0)).collect();
    let centers: Vec<V3> = reg
        .iter()
        .zip(&rots)
        .map(|(&i, m)| scale(apply(&transpose(m), known[&names[i]].1), -1.0))
        .collect();
    let frame = read(model)?;
    let o = frame.center;
    let intrinsics = read_intrinsics(&model.join("cameras.bin")).ok();
    let mut refined = Vec::new();

    // Normal de Newell (como em flatten_orbit); sem arco, interpolação linear.
    let k = centers.len() as f64;
    let mean = scale(centers.iter().fold([0.0; 3], |a, c| add(a, *c)), 1.0 / k);
    let mut normal = [0.0; 3];
    for w in centers.windows(2) {
        normal = add(normal, cross(sub(w[0], mean), sub(w[1], mean)));
    }
    let spread: f64 = centers.iter().map(|c| dot(sub(*c, mean), sub(*c, mean))).sum::<f64>() / k;
    let linear = dot(normal, normal).sqrt() < 1e-3 * spread * k;
    let n = unit(normal);

    // Base do plano: e1 aponta para a primeira câmera.
    let d0 = sub(centers[0], o);
    let e1 = unit(sub(d0, scale(n, dot(d0, n))));
    let e2 = cross(n, e1);
    let (mut th, mut rad, mut hgt): (Vec<f64>, Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new(), Vec::new());
    for c in &centers {
        let d = sub(*c, o);
        let h = dot(d, n);
        let p = sub(d, scale(n, h));
        let mut a = dot(p, e2).atan2(dot(p, e1));
        // Desenrola: passo de menos de meia volta entre registrados vizinhos.
        if let Some(&prev) = th.last() {
            a += std::f64::consts::TAU * ((prev - a) / std::f64::consts::TAU).round();
        }
        th.push(a);
        rad.push(dot(p, p).sqrt());
        hgt.push(h);
    }
    // Orientação "na órbita": a câmera girada de volta para θ = 0.
    let mut q0: Vec<[f64; 4]> = Vec::new();
    for (m, a) in rots.iter().zip(&th) {
        let mut q = quaternion(&if linear { *m } else { mul(m, &about(n, *a)) });
        if let Some(prev) = q0.last() {
            if (0..4).map(|k| prev[k] * q[k]).sum::<f64>() < 0.0 {
                q = q.map(|v| -v);
            }
        }
        q0.push(q);
    }

    // Confiança na trajetória, medida nos próprios registrados: cada um
    // previsto sem ele (no meio, pelos vizinhos; nas pontas, extrapolado).
    let at_center = |[theta, r, h]: [f64; 3]| add(o, add(add(scale(e1, r * theta.cos()), scale(e2, r * theta.sin())), scale(n, h)));
    let without = |k: usize| -> [Vec<f64>; 4] {
        [&s, &th, &rad, &hgt].map(|y| y.iter().enumerate().filter(|&(j, _)| j != k).map(|(_, v)| *v).collect())
    };
    let loo = |k: usize| {
        let [s2, th2, rad2, hgt2] = without(k);
        let d = sub(at_center(orbit_at(&s2, &th2, &rad2, &hgt2, s[k])), centers[k]);
        dot(d, d).sqrt()
    };
    let last = reg.len() - 1;
    let step = median(s.windows(2).map(|w| w[1] - w[0]).collect());
    let (err_mid, err_end) = if linear || reg.len() < 6 {
        (0.05 * frame.radius, 0.05 * frame.radius)
    } else {
        (median((1..last).map(loo).collect()), loo(0).max(loo(last)))
    };
    let floor = 1e-4 * frame.radius;
    // A partir do erro medido com um frame só de fora: no meio, cresce com
    // a lacuna (≈ linear numa órbita suave); na ponta, com o quadrado da
    // distância extrapolada.
    let sigma_at = |t: f64| -> f64 {
        match s.iter().position(|&v| v > t) {
            Some(b) if b > 0 => err_mid * ((s[b] - s[b - 1]) / (2.0 * step)).max(1.0),
            b => {
                let end = if b == Some(0) { 0 } else { last };
                err_end * ((t - s[end]).abs() / step).max(1.0).powi(2)
            }
        }
        .max(floor)
    };

    let mut added: Vec<(String, M3, V3)> = Vec::new();
    for &i in &missing {
        let t = at[i];
        let b = s.iter().position(|&v| v > t);
        let (q, lin) = match b {
            Some(b) if b > 0 => {
                let u = (t - s[b - 1]) / (s[b] - s[b - 1]);
                (slerp(q0[b - 1], q0[b], u), add(centers[b - 1], scale(sub(centers[b], centers[b - 1]), u)))
            }
            _ => {
                // Antes do primeiro ou depois do último registrado.
                let end = if b == Some(0) { 0 } else { last };
                let ks = if end == 0 { (0, 3.min(last)) } else { (last.saturating_sub(3), last) };
                let v = scale(sub(centers[ks.1], centers[ks.0]), 1.0 / (s[ks.1] - s[ks.0]));
                (q0[end], add(centers[end], scale(v, t - s[end])))
            }
        };
        let orbit = orbit_at(&s, &th, &rad, &hgt, t);
        let h = orbit[2];
        let m = rotation(q);
        let (mut m, mut c) = if linear { (m, lin) } else { (mul(&m, &transpose(&about(n, orbit[0]))), at_center(orbit)) };
        let corr = opts.correspondences.get(&names[i]);
        if let Some((k, corr)) = intrinsics.as_ref().zip(corr) {
            match refine_pose(m, c, sigma_at(t), corr, k) {
                // Longe demais da trajetória: matches ruins, fica a inferida.
                Some((m2, c2, inliers))
                    if dot(sub(c2, c), sub(c2, c)).sqrt() <= 0.2 * frame.radius
                        && rotation_angle(&m2, &m) <= 15f64.to_radians() =>
                {
                    (m, c) = (m2, c2);
                    if opts.flat && !linear {
                        // De volta ao plano, com o sujeito no mesmo pixel.
                        let flat = sub(c, scale(n, dot(sub(c, o), n) - h));
                        m = mul(&m, &rotation_between(unit(sub(o, flat)), unit(sub(o, c))));
                        c = flat;
                    }
                    refined.push((names[i].clone(), inliers));
                }
                _ => {}
            }
        }
        added.push((names[i].clone(), m, c));
    }

    // images.bin: entradas antigas + as novas, sem pontos 2D.
    let pose_bytes = |m: &M3, c: V3| -> Vec<u8> {
        let t = scale(apply(m, c), -1.0);
        quaternion(m).iter().chain(t.iter()).flat_map(|v| v.to_le_bytes()).collect()
    };
    let mut out = Vec::with_capacity(images_bin.len() + added.len() * 128);
    out.extend_from_slice(&(count + added.len() as u64).to_le_bytes());
    out.extend_from_slice(&images_bin[8..]);
    for (k, (name, m, c)) in added.iter().enumerate() {
        out.extend_from_slice(&(max_id + 1 + k as u32).to_le_bytes());
        out.extend(pose_bytes(m, *c));
        out.extend_from_slice(&camera_id.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(&0u64.to_le_bytes());
    }

    // COLMAP 4: cada imagem nova também precisa do seu frame (rig de uma câmera).
    let frames_path = model.join("frames.bin");
    let frames_out = if frames_path.exists() {
        let bin = std::fs::read(&frames_path)?;
        let mut r = Bin(&bin[..]);
        let nf = r.u64()?;
        let (mut max_frame, mut rig) = (0u32, None);
        for _ in 0..nf {
            let id = r.u32()?;
            let rig_id = r.u32()?;
            r.skip(56)?;
            let nd = r.u32()?;
            r.skip(nd as u64 * 16)?;
            max_frame = max_frame.max(id);
            rig.get_or_insert(rig_id);
        }
        let mut out = Vec::with_capacity(bin.len() + added.len() * 96);
        out.extend_from_slice(&(nf + added.len() as u64).to_le_bytes());
        out.extend_from_slice(&bin[8..]);
        for (k, (_, m, c)) in added.iter().enumerate() {
            out.extend_from_slice(&(max_frame + 1 + k as u32).to_le_bytes());
            out.extend_from_slice(&rig.unwrap_or(1).to_le_bytes());
            out.extend(pose_bytes(m, *c));
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&0i32.to_le_bytes()); // sensor: câmera
            out.extend_from_slice(&camera_id.to_le_bytes());
            out.extend_from_slice(&((max_id + 1 + k as u32) as u64).to_le_bytes());
        }
        Some(out)
    } else {
        None
    };

    std::fs::write(model.join("images.bin"), out)?;
    if let Some(out) = frames_out {
        std::fs::write(frames_path, out)?;
    }
    Ok(Filled { inferred: added.into_iter().map(|(name, _, _)| name).collect(), refined })
}

/// Pose (mundo→câmera, centro) de cada imagem do modelo, pelo nome.
#[cfg(test)]
pub(crate) fn read_poses(model: &Path) -> Result<std::collections::HashMap<String, (M3, V3)>> {
    let mut out = std::collections::HashMap::new();
    let mut r = open(&model.join("images.bin"))?;
    for _ in 0..r.u64()? {
        let _id = r.u32()?;
        let q = [r.f64()?, r.f64()?, r.f64()?, r.f64()?];
        let t = [r.f64()?, r.f64()?, r.f64()?];
        let _cam = r.u32()?;
        let name = r.cstr()?;
        let n2d = r.u64()?;
        r.skip(n2d * 24)?;
        let m = rotation(q);
        out.insert(name, (m, scale(apply(&transpose(&m), t), -1.0)));
    }
    Ok(out)
}

/// Erro de centro (fração de `radius`) e de rotação (graus) entre duas poses.
#[cfg(test)]
pub(crate) fn pose_error(a: &(M3, V3), b: &(M3, V3), radius: f64) -> (f64, f64) {
    (dot(sub(a.1, b.1), sub(a.1, b.1)).sqrt() / radius, rotation_angle(&a.0, &b.0).to_degrees())
}

/// Cópia do images.bin sem as imagens de `names` (simula câmeras não registradas).
#[cfg(test)]
pub(crate) fn drop_images(src: &Path, dst: &Path, names: &[String]) -> Result<()> {
    let mut r = open(src)?;
    let mut entries = Vec::new();
    for _ in 0..r.u64()? {
        let head: [u8; 64] = r.bytes()?;
        let name = r.cstr()?;
        let n2d = r.u64()?;
        let mut pts = vec![0u8; n2d as usize * 24];
        r.0.read_exact(&mut pts)?;
        if !names.contains(&name) {
            entries.push((head, name, n2d, pts));
        }
    }
    let mut out = (entries.len() as u64).to_le_bytes().to_vec();
    for (head, name, n2d, pts) in entries {
        out.extend_from_slice(&head);
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(&n2d.to_le_bytes());
        out.extend(pts);
    }
    std::fs::write(dst, out)?;
    Ok(())
}

/// Para cada imagem do modelo, o id do ponto 3D de cada ponto 2D (−1 = sem
/// ponto), e as coordenadas dos pontos 3D.
pub type Tracks = (std::collections::HashMap<String, Vec<i64>>, std::collections::HashMap<u64, V3>);

pub fn read_tracks(model: &Path) -> Result<Tracks> {
    let mut images = std::collections::HashMap::new();
    let mut r = open(&model.join("images.bin"))?;
    for _ in 0..r.u64()? {
        r.skip(4 + 56 + 4)?; // id, pose, camera_id
        let name = r.cstr()?;
        let n2d = r.u64()?;
        let mut ids = Vec::with_capacity(n2d as usize);
        for _ in 0..n2d {
            r.skip(16)?; // x, y
            ids.push(i64::from_le_bytes(r.bytes()?));
        }
        images.insert(name, ids);
    }
    let mut points = std::collections::HashMap::new();
    let mut r = open(&model.join("points3D.bin"))?;
    for _ in 0..r.u64()? {
        let id = r.u64()?;
        points.insert(id, [r.f64()?, r.f64()?, r.f64()?]);
        r.skip(3 + 8)?;
        let track = r.u64()?;
        r.skip(track * 8)?;
    }
    Ok((images, points))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Câmera em `c` olhando para a origem, "para cima" ≈ +z.
    fn look_at(c: V3) -> M3 {
        let z = unit(scale(c, -1.0));
        let x = unit(cross(z, [0.0, 0.0, -1.0]));
        [x, cross(z, x), z]
    }

    /// Mesma órbita de web/src/sfm/orbit.test.ts: 96 frames a 24 fps,
    /// velocidade angular variando, raio 4, altura 1.
    fn truth(i: usize) -> V3 {
        let s = i as f64 / 24.0;
        let th = std::f64::consts::TAU * 0.85 * (s / 4.0) * (0.8 + 0.2 * (s / 4.0));
        [4.0 * th.cos(), 4.0 * th.sin(), 1.0]
    }

    #[test]
    fn refine_pose_recovers_from_a_rough_guess() {
        let k = Intrinsics { fx: 800.0, fy: 800.0, cx: 480.0, cy: 360.0 };
        let c_true = truth(30);
        let m_true = look_at(c_true);
        // Pontos espalhados perto da origem, vistos pela câmera verdadeira,
        // com ruído de ~0,5 px e 20% de matches errados.
        let mut seed = 12345u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let t_true = scale(apply(&m_true, c_true), -1.0);
        let corr: Vec<([f64; 2], V3)> = (0..200)
            .map(|i| {
                let x = [rnd() * 2.0, rnd() * 2.0, rnd() * 2.0];
                let p = add(apply(&m_true, x), t_true);
                let mut uv = [k.fx * p[0] / p[2] + k.cx + rnd(), k.fy * p[1] / p[2] + k.cy + rnd()];
                if i % 5 == 0 {
                    uv = [uv[0] + rnd() * 400.0, uv[1] + rnd() * 400.0];
                }
                (uv, x)
            })
            .collect();
        // Palpite a 3% do raio e ~1,5° da verdade.
        let c0 = add(c_true, [0.08, -0.06, 0.05]);
        let m0 = mul(&about(unit([1.0, 2.0, 0.5]), 1.5f64.to_radians()), &look_at(c0));
        // Prior fraco (palpite incerto): valem os pontos.
        let (m, c, inliers) = refine_pose(m0, c0, 1.0, &corr, &k).expect("refina");
        let err = dot(sub(c, c_true), sub(c, c_true)).sqrt() / 4.0;
        assert!(err < 0.002, "centro a {:.3}% do raio", err * 100.0);
        assert!(rotation_angle(&m, &m_true) < 0.1f64.to_radians());
        assert!(inliers >= 150, "{inliers} inliers");
        // Sem correspondências boas suficientes, não inventa.
        assert!(refine_pose(m0, c0, 1.0, &corr[..10], &k).is_none());
    }

    #[test]
    fn fill_missing_follows_the_orbit() {
        let root = std::env::temp_dir().join("splatforge_fill_test");
        let _ = std::fs::remove_dir_all(&root);
        let (model, images) = (root.join("sparse"), root.join("images"));
        std::fs::create_dir_all(&model).unwrap();
        std::fs::create_dir_all(&images).unwrap();
        let lost = [0usize, 1, 20, 21, 22, 23, 50, 94, 95];
        let mut times = std::collections::HashMap::new();
        let (mut img, mut frames, mut count) = (Vec::new(), Vec::new(), 0u64);
        for i in 0..96 {
            let name = format!("{i:05}.jpg");
            std::fs::write(images.join(&name), b"x").unwrap();
            times.insert(name.clone(), i as f64 / 24.0);
            if lost.contains(&i) {
                continue;
            }
            let m = look_at(truth(i));
            let t = scale(apply(&m, truth(i)), -1.0);
            let pose: Vec<u8> = quaternion(&m).iter().chain(t.iter()).flat_map(|v| v.to_le_bytes()).collect();
            count += 1;
            img.extend_from_slice(&(i as u32 + 1).to_le_bytes());
            img.extend(&pose);
            img.extend_from_slice(&1u32.to_le_bytes());
            img.extend_from_slice(name.as_bytes());
            img.push(0);
            img.extend_from_slice(&0u64.to_le_bytes());
            frames.extend_from_slice(&(i as u32 + 1).to_le_bytes());
            frames.extend_from_slice(&1u32.to_le_bytes());
            frames.extend(&pose);
            frames.extend_from_slice(&1u32.to_le_bytes());
            frames.extend_from_slice(&0i32.to_le_bytes());
            frames.extend_from_slice(&1u32.to_le_bytes());
            frames.extend_from_slice(&(i as u64 + 1).to_le_bytes());
        }
        let with_count = |body: Vec<u8>| [count.to_le_bytes().to_vec(), body].concat();
        std::fs::write(model.join("images.bin"), with_count(img)).unwrap();
        std::fs::write(model.join("frames.bin"), with_count(frames)).unwrap();
        // Um ponto na origem: é para onde as câmeras olham.
        let mut pts = 1u64.to_le_bytes().to_vec();
        pts.extend_from_slice(&1u64.to_le_bytes());
        pts.extend([0f64; 3].iter().flat_map(|v| v.to_le_bytes()));
        pts.extend([0u8; 3]);
        pts.extend_from_slice(&0f64.to_le_bytes());
        pts.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(model.join("points3D.bin"), pts).unwrap();

        let none = std::collections::HashMap::new();
        let opts = FillOptions { flat: false, correspondences: &none };
        let filled = fill_missing(&model, &images, &times, &opts).unwrap().inferred;
        assert_eq!(filled.len(), lost.len());
        let frame = read(&model).unwrap();
        assert_eq!(frame.cameras.len(), 96);
        for &i in &lost {
            let c = frame.cameras[i];
            let err = dot(sub(c, truth(i)), sub(c, truth(i))).sqrt() / 4.0;
            assert!(err < 0.005, "frame {i}: centro a {:.3}% do raio", err * 100.0);
        }
        // frames.bin ganhou um frame por imagem nova.
        let fb = std::fs::read(model.join("frames.bin")).unwrap();
        assert_eq!(u64::from_le_bytes(fb[..8].try_into().unwrap()), 96);
        assert_eq!(fb.len(), 8 + 96 * (8 + 56 + 4 + 16));
        // Rodar de novo não muda nada.
        assert!(fill_missing(&model, &images, &times, &opts).unwrap().inferred.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

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
