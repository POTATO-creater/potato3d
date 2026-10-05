//! Leitura e escrita do .ply de Gaussian Splatting (layout do INRIA).
//!
//! O Brush grava as propriedades numa ordem própria (f_dc, f_rest, opacity,
//! rot, scale, xyz), então a leitura localiza cada coluna pelo nome. A escrita
//! usa a ordem do INRIA, que todo visualizador entende.

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};

/// Gaussianas em colunas (SoA), com os valores crus do .ply: escala em log,
/// opacidade em logit e cor como coeficientes SH.
#[derive(Debug, Clone, Default)]
pub struct Splats {
    pub count: usize,
    /// Grau SH das bandas `f_rest` (0–3).
    pub sh_degree: u32,
    /// x, y, z
    pub pos: Vec<f32>,
    /// log(σ) por eixo
    pub scale: Vec<f32>,
    /// (w, x, y, z), não necessariamente normalizado
    pub rot: Vec<f32>,
    /// logit da opacidade
    pub opacity: Vec<f32>,
    /// f_dc_0..2
    pub dc: Vec<f32>,
    /// f_rest_* por gaussiana, na ordem do .ply (canal maior: todos os
    /// coeficientes do R, depois G, depois B).
    pub rest: Vec<f32>,
}

/// Coeficientes SH por canal além do DC, para cada grau.
pub fn sh_coeffs(degree: u32) -> usize {
    [0, 3, 8, 15][degree.min(3) as usize]
}

impl Splats {
    pub fn rest_len(&self) -> usize {
        sh_coeffs(self.sh_degree) * 3
    }

    /// Mantém só as gaussianas de `keep`, na ordem dada.
    pub fn select(&self, keep: &[u32]) -> Splats {
        let rl = self.rest_len();
        let mut out = Splats {
            count: keep.len(),
            sh_degree: self.sh_degree,
            pos: Vec::with_capacity(keep.len() * 3),
            scale: Vec::with_capacity(keep.len() * 3),
            rot: Vec::with_capacity(keep.len() * 4),
            opacity: Vec::with_capacity(keep.len()),
            dc: Vec::with_capacity(keep.len() * 3),
            rest: Vec::with_capacity(keep.len() * rl),
        };
        for &i in keep {
            let i = i as usize;
            out.pos.extend_from_slice(&self.pos[i * 3..i * 3 + 3]);
            out.scale.extend_from_slice(&self.scale[i * 3..i * 3 + 3]);
            out.rot.extend_from_slice(&self.rot[i * 4..i * 4 + 4]);
            out.opacity.push(self.opacity[i]);
            out.dc.extend_from_slice(&self.dc[i * 3..i * 3 + 3]);
            out.rest.extend_from_slice(&self.rest[i * rl..(i + 1) * rl]);
        }
        out
    }

    /// Descarta as bandas SH acima de `degree`.
    pub fn truncate_sh(&mut self, degree: u32) {
        if degree >= self.sh_degree {
            return;
        }
        let (old, new) = (sh_coeffs(self.sh_degree), sh_coeffs(degree));
        let mut rest = Vec::with_capacity(self.count * new * 3);
        for g in 0..self.count {
            let base = g * old * 3;
            for c in 0..3 {
                rest.extend_from_slice(&self.rest[base + c * old..base + c * old + new]);
            }
        }
        self.rest = rest;
        self.sh_degree = degree;
    }
}

#[derive(Clone, Copy)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn parse(s: &str) -> Option<Ty> {
        Some(match s {
            "char" | "int8" => Ty::I8,
            "uchar" | "uint8" => Ty::U8,
            "short" | "int16" => Ty::I16,
            "ushort" | "uint16" => Ty::U16,
            "int" | "int32" => Ty::I32,
            "uint" | "uint32" => Ty::U32,
            "float" | "float32" => Ty::F32,
            "double" | "float64" => Ty::F64,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            Ty::I8 | Ty::U8 => 1,
            Ty::I16 | Ty::U16 => 2,
            Ty::I32 | Ty::U32 | Ty::F32 => 4,
            Ty::F64 => 8,
        }
    }
    fn read(self, b: &[u8]) -> f32 {
        match self {
            Ty::I8 => b[0] as i8 as f32,
            Ty::U8 => b[0] as f32,
            Ty::I16 => i16::from_le_bytes([b[0], b[1]]) as f32,
            Ty::U16 => u16::from_le_bytes([b[0], b[1]]) as f32,
            Ty::I32 => i32::from_le_bytes(b[..4].try_into().unwrap()) as f32,
            Ty::U32 => u32::from_le_bytes(b[..4].try_into().unwrap()) as f32,
            Ty::F32 => f32::from_le_bytes(b[..4].try_into().unwrap()),
            Ty::F64 => f64::from_le_bytes(b[..8].try_into().unwrap()) as f32,
        }
    }
}

struct Prop {
    name: String,
    ty: Ty,
    offset: usize,
}

struct Header {
    count: usize,
    props: Vec<Prop>,
    stride: usize,
}

fn read_header(r: &mut impl BufRead, path: &Path) -> Result<Header> {
    let mut line = String::new();
    r.read_line(&mut line)?;
    if line.trim_end() != "ply" {
        bail!("{} não é um arquivo .ply", path.display());
    }
    let mut count = None;
    let mut props: Vec<Prop> = Vec::new();
    let mut stride = 0;
    let mut in_vertex = false;
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            bail!("cabeçalho do .ply sem end_header");
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.as_slice() {
            ["end_header"] => break,
            ["format", fmt, ..] if *fmt != "binary_little_endian" => {
                bail!("formato .ply não suportado: {fmt} (só binary_little_endian)")
            }
            ["element", name, n] => {
                if count.is_some() {
                    // Elementos depois do vértice não interessam.
                    in_vertex = false;
                } else if *name == "vertex" {
                    count = Some(n.parse::<usize>()?);
                    in_vertex = true;
                } else {
                    bail!(".ply com o elemento \"{name}\" antes dos vértices (compactado?) não é suportado");
                }
            }
            ["property", "list", ..] if in_vertex => bail!("propriedade em lista nos vértices do .ply"),
            ["property", ty, name] if in_vertex => {
                let ty = Ty::parse(ty).with_context(|| format!("tipo de propriedade desconhecido: {ty}"))?;
                props.push(Prop { name: name.to_string(), ty, offset: stride });
                stride += ty.size();
            }
            _ => {}
        }
    }
    Ok(Header { count: count.context(".ply sem elemento vertex")?, props, stride })
}

fn degree_from_rest(n: usize) -> Result<u32> {
    Ok(match n {
        0 => 0,
        9 => 1,
        24 => 2,
        45 => 3,
        n => bail!("{n} coeficientes f_rest não correspondem a nenhum grau SH"),
    })
}

/// Quantidade de gaussianas e grau SH, lendo só o cabeçalho.
pub fn info(path: &Path) -> Result<(usize, u32)> {
    let mut r = BufReader::new(std::fs::File::open(path).with_context(|| format!("abrindo {}", path.display()))?);
    let h = read_header(&mut r, path)?;
    let rest = h.props.iter().filter(|p| p.name.starts_with("f_rest_")).count();
    Ok((h.count, degree_from_rest(rest)?))
}

pub fn read(path: &Path) -> Result<Splats> {
    let file = std::fs::File::open(path).with_context(|| format!("abrindo {}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, file);
    let Header { count, props, stride } = read_header(&mut r, path)?;

    let col = |name: &str| -> Result<(Ty, usize)> {
        props
            .iter()
            .find(|p| p.name == name)
            .map(|p| (p.ty, p.offset))
            .with_context(|| format!(".ply sem a propriedade {name} (não parece um Gaussian Splat)"))
    };
    let pos = [col("x")?, col("y")?, col("z")?];
    let scale = [col("scale_0")?, col("scale_1")?, col("scale_2")?];
    let rot = [col("rot_0")?, col("rot_1")?, col("rot_2")?, col("rot_3")?];
    let opacity = col("opacity")?;
    let dc = [col("f_dc_0")?, col("f_dc_1")?, col("f_dc_2")?];
    let mut rest = Vec::new();
    while let Ok(c) = col(&format!("f_rest_{}", rest.len())) {
        rest.push(c);
    }
    let sh_degree = degree_from_rest(rest.len())?;

    let mut s = Splats {
        count,
        sh_degree,
        pos: Vec::with_capacity(count * 3),
        scale: Vec::with_capacity(count * 3),
        rot: Vec::with_capacity(count * 4),
        opacity: Vec::with_capacity(count),
        dc: Vec::with_capacity(count * 3),
        rest: Vec::with_capacity(count * rest.len()),
    };
    let get = |row: &[u8], (ty, off): (Ty, usize)| ty.read(&row[off..]);
    let mut row = vec![0u8; stride];
    for _ in 0..count {
        r.read_exact(&mut row).context(".ply truncado")?;
        s.pos.extend(pos.iter().map(|&c| get(&row, c)));
        s.scale.extend(scale.iter().map(|&c| get(&row, c)));
        s.rot.extend(rot.iter().map(|&c| get(&row, c)));
        s.opacity.push(get(&row, opacity));
        s.dc.extend(dc.iter().map(|&c| get(&row, c)));
        s.rest.extend(rest.iter().map(|&c| get(&row, c)));
    }
    Ok(s)
}

pub fn write(path: &Path, s: &Splats) -> Result<u64> {
    let mut w = BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
    let mut header = String::from("ply\nformat binary_little_endian 1.0\ncomment Exported from SplatForge\n");
    header += &format!("element vertex {}\n", s.count);
    let mut names: Vec<String> = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"].map(String::from).to_vec();
    names.extend((0..s.rest_len()).map(|i| format!("f_rest_{i}")));
    names.extend(["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"].map(String::from));
    for n in &names {
        header += &format!("property float {n}\n");
    }
    header += "end_header\n";
    w.write_all(header.as_bytes())?;

    let rl = s.rest_len();
    let mut row: Vec<f32> = Vec::with_capacity(names.len());
    for i in 0..s.count {
        row.clear();
        row.extend_from_slice(&s.pos[i * 3..i * 3 + 3]);
        row.extend_from_slice(&s.dc[i * 3..i * 3 + 3]);
        row.extend_from_slice(&s.rest[i * rl..(i + 1) * rl]);
        row.push(s.opacity[i]);
        row.extend_from_slice(&s.scale[i * 3..i * 3 + 3]);
        row.extend_from_slice(&s.rot[i * 4..i * 4 + 4]);
        for v in &row {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(w.get_ref().metadata()?.len())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn sample(n: usize, degree: u32) -> Splats {
        let rl = sh_coeffs(degree) * 3;
        // splitmix64 → valor em [-0,5, 0,5), determinístico por (i, k).
        let f = |i: usize, k: usize| {
            let mut z = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (k as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (z ^ (z >> 31)) as f32 / u64::MAX as f32 - 0.5
        };
        Splats {
            count: n,
            sh_degree: degree,
            pos: (0..n * 3).map(|k| f(k / 3, k) * 4.0).collect(),
            scale: (0..n * 3).map(|k| -4.0 + f(k / 3, k)).collect(),
            rot: (0..n * 4).map(|k| f(k / 4, k) + if k % 4 == 0 { 1.0 } else { 0.0 }).collect(),
            opacity: (0..n).map(|i| f(i, 1) * 8.0).collect(),
            dc: (0..n * 3).map(|k| f(k / 3, k)).collect(),
            rest: (0..n * rl).map(|k| f(k / rl.max(1), k) * 0.1).collect(),
        }
    }

    #[test]
    fn roundtrip_and_truncate() {
        let dir = std::env::temp_dir().join(format!("splatforge-ply-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.ply");
        let s = sample(100, 2);
        write(&p, &s).unwrap();
        let back = read(&p).unwrap();
        assert_eq!(back.count, 100);
        assert_eq!(back.sh_degree, 2);
        assert_eq!(back.pos, s.pos);
        assert_eq!(back.rest, s.rest);
        assert_eq!(back.rot, s.rot);
        assert_eq!(info(&p).unwrap(), (100, 2));

        let mut t = back.clone();
        t.truncate_sh(1);
        assert_eq!(t.rest_len(), 9);
        // Coeficiente 0 do canal G fica na posição 3 (antes estava em 8).
        assert_eq!(t.rest[3], back.rest[8]);
        assert_eq!(t.rest[9 + 6], back.rest[24 + 16]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
