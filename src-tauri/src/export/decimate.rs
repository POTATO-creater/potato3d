//! Decimação: reduz a quantidade de gaussianas fundindo as *parecidas*.
//!
//! Port do `sharp-decimate.ts` do WebGaussian, que segue o esquema geracional
//! por pares do splat-transform (PlayCanvas, MIT): a cada geração, cada splat
//! acha os vizinhos mais baratos de fundir (hash espacial → vizinhos mais
//! próximos → custo da aresta), grupos disjuntos (pares, fechados em cadeia
//! até 4 membros) são casados gulosamente em ordem de custo e cada grupo vira
//! uma gaussiana por casamento de momentos. Repete até chegar ao alvo.
//!
//! - Custo geométrico = aumento de entropia da fusão em forma fechada
//!   (½·logdet(Σm) − Σ pₖ·½·logdetₖ): penaliza distância e formas diferentes.
//! - Custo de cor = |Δf_dc|², o mesmo critério do splat-transform.
//! - Pesos e opacidade pelo modelo de "tinta": cada membro pesa
//!   `opacidade × área do elipsoide`, e a opacidade fundida redistribui a tinta
//!   total sobre a área do splat resultante (limitada a 1). Conservar a tinta
//!   mantém as superfícies fechadas; tirar a média das opacidades abre buracos
//!   translúcidos.
//! - Diferente do WebGaussian (SHARP só tem DC), as bandas SH também são
//!   fundidas, pela média ponderada pela tinta.

use rayon::prelude::*;

use super::ply::Splats;

/// Abaixo disso a decimação não compensa a perda.
const MIN_COUNT: usize = 10_000;
const EPS_COV: f64 = 1e-8;
/// Peso do termo de cor (coeficientes DC, como no splat-transform).
const COLOR_COST_SCALE: f64 = 1.0;
/// Vizinhos mais próximos avaliados por splat.
const NEIGHBOR_SCAN_CAP: usize = 4;
/// Melhores candidatos mantidos por splat para o casamento.
const CANDIDATES: usize = 2;
const SELECT_BUCKETS: usize = 1024;
/// Tamanho máximo do grupo: pares primeiro, depois fechamento em cadeia.
const GROUP_CAPS: [u32; 2] = [3, 4];
const MAX_GENERATIONS: usize = 24;
const MAX_GRID_RESOLUTION: i64 = 1024;
/// Entradas lidas por célula vizinha (limita células patológicas).
const PER_CELL_CAP: usize = 32;
const NONE: u32 = u32::MAX;

/// Área do elipsoide (aproximação de Knud Thomsen, p ≈ 1,6075).
fn ellipsoid_area(sx: f64, sy: f64, sz: f64) -> f64 {
    const P: f64 = 1.6075;
    let a = (sx * sy).powf(P);
    let b = (sx * sz).powf(P);
    let c = (sy * sz).powf(P);
    4.0 * std::f64::consts::PI * ((a + b + c) / 3.0).powf(1.0 / P)
}

fn sigmoid(x: f32) -> f64 {
    1.0 / (1.0 + (-x as f64).exp())
}

/// Conjunto de trabalho de uma geração, em valores lineares.
struct Work {
    n: usize,
    rl: usize,
    pos: Vec<f32>,
    /// σ por eixo (linear)
    scale: Vec<f32>,
    /// (w, x, y, z) normalizado
    rot: Vec<f32>,
    /// opacidade linear (0–1)
    opacity: Vec<f32>,
    dc: Vec<f32>,
    rest: Vec<f32>,
    /// Covariância no mundo, simétrica: xx, yy, zz, xy, xz, yz.
    cov: Vec<f64>,
    logdet: Vec<f64>,
    /// opacidade × área
    ink: Vec<f64>,
}

impl Work {
    fn with_capacity(n: usize, rl: usize) -> Work {
        Work {
            n: 0,
            rl,
            pos: Vec::with_capacity(n * 3),
            scale: Vec::with_capacity(n * 3),
            rot: Vec::with_capacity(n * 4),
            opacity: Vec::with_capacity(n),
            dc: Vec::with_capacity(n * 3),
            rest: Vec::with_capacity(n * rl),
            cov: Vec::with_capacity(n * 6),
            logdet: Vec::with_capacity(n),
            ink: Vec::with_capacity(n),
        }
    }

    fn from_splats(s: &Splats) -> Work {
        let rl = s.rest_len();
        let mut w = Work::with_capacity(s.count, rl);
        for i in 0..s.count {
            let q = &s.rot[i * 4..i * 4 + 4];
            let norm = (q.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-12);
            let rot = [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
            let scale = [s.scale[i * 3].exp(), s.scale[i * 3 + 1].exp(), s.scale[i * 3 + 2].exp()];
            w.push(
                &s.pos[i * 3..i * 3 + 3],
                scale,
                rot,
                sigmoid(s.opacity[i]) as f32,
                &s.dc[i * 3..i * 3 + 3],
                &s.rest[i * rl..(i + 1) * rl],
            );
        }
        w
    }

    /// Acrescenta um splat calculando covariância, logdet e tinta.
    fn push(&mut self, pos: &[f32], scale: [f32; 3], rot: [f32; 4], opacity: f32, dc: &[f32], rest: &[f32]) {
        let sx = (scale[0] as f64).max(1e-9);
        let sy = (scale[1] as f64).max(1e-9);
        let sz = (scale[2] as f64).max(1e-9);
        let (vx, vy, vz) = (sx * sx, sy * sy, sz * sz);
        let r = quat_to_matrix(rot.map(|v| v as f64));
        self.cov.extend_from_slice(&[
            r[0] * r[0] * vx + r[1] * r[1] * vy + r[2] * r[2] * vz,
            r[3] * r[3] * vx + r[4] * r[4] * vy + r[5] * r[5] * vz,
            r[6] * r[6] * vx + r[7] * r[7] * vy + r[8] * r[8] * vz,
            r[0] * r[3] * vx + r[1] * r[4] * vy + r[2] * r[5] * vz,
            r[0] * r[6] * vx + r[1] * r[7] * vy + r[2] * r[8] * vz,
            r[3] * r[6] * vx + r[4] * r[7] * vy + r[5] * r[8] * vz,
        ]);
        // det(R·S²·Rᵀ) = det(S²)
        self.logdet.push(vx.ln() + vy.ln() + vz.ln());
        self.ink.push(opacity as f64 * ellipsoid_area(sx, sy, sz) + 1e-30);
        self.pos.extend_from_slice(pos);
        self.scale.extend_from_slice(&scale);
        self.rot.extend_from_slice(&rot);
        self.opacity.push(opacity);
        self.dc.extend_from_slice(dc);
        self.rest.extend_from_slice(rest);
        self.n += 1;
    }

    fn copy_from(&mut self, src: &Work, i: usize) {
        self.pos.extend_from_slice(&src.pos[i * 3..i * 3 + 3]);
        self.scale.extend_from_slice(&src.scale[i * 3..i * 3 + 3]);
        self.rot.extend_from_slice(&src.rot[i * 4..i * 4 + 4]);
        self.opacity.push(src.opacity[i]);
        self.dc.extend_from_slice(&src.dc[i * 3..i * 3 + 3]);
        self.rest.extend_from_slice(&src.rest[i * src.rl..(i + 1) * src.rl]);
        self.cov.extend_from_slice(&src.cov[i * 6..i * 6 + 6]);
        self.logdet.push(src.logdet[i]);
        self.ink.push(src.ink[i]);
        self.n += 1;
    }

    fn into_splats(self, sh_degree: u32) -> Splats {
        let logit = |p: f32| {
            let p = (p as f64).clamp(1e-6, 1.0 - 1e-6);
            (p / (1.0 - p)).ln() as f32
        };
        Splats {
            count: self.n,
            sh_degree,
            scale: self.scale.iter().map(|s| s.max(1e-12).ln()).collect(),
            opacity: self.opacity.iter().map(|&p| logit(p)).collect(),
            pos: self.pos,
            rot: self.rot,
            dc: self.dc,
            rest: self.rest,
        }
    }
}

/// Funde gaussianas parecidas até restar ~`1 − reduction` delas.
/// `reduction` vai de 0 a 0,9; zero (ou entrada pequena) não faz nada.
pub fn decimate(input: &Splats, reduction: f32, mut progress: impl FnMut(f32)) -> Option<Splats> {
    let reduction = reduction.clamp(0.0, 0.9);
    if reduction <= 0.0 || input.count < MIN_COUNT {
        return None;
    }
    let target = ((input.count as f64 * (1.0 - reduction as f64)).round() as usize).max(1);
    let mut work = Work::from_splats(input);
    let to_remove = (work.n - target) as f32;

    // Uma geração travada (sem vizinhos ao alcance) amplia a busca.
    let mut reach = 1.0;
    let mut generation = 0;
    while generation < MAX_GENERATIONS && work.n > target {
        match run_generation(&work, target, reach) {
            Some(next) => {
                work = next;
                generation += 1;
                progress(1.0 - (work.n - target.min(work.n)) as f32 / to_remove);
            }
            None => {
                reach *= 2.0;
                if reach > 64.0 {
                    break;
                }
            }
        }
    }
    (work.n < input.count).then(|| work.into_splats(input.sh_degree))
}

fn run_generation(work: &Work, target: usize, reach: f64) -> Option<Work> {
    let (cost, idx) = build_candidates(work, reach);
    let sel = select_merges(work.n, &cost, &idx, work.n - target);
    (sel.removed > 0).then(|| apply_merges(work, &sel))
}

// ---------------------------------------------------------------------------
// Candidatos: hash espacial → vizinhos mais próximos → custo da aresta.
// ---------------------------------------------------------------------------

struct Grid {
    n: [i64; 3],
    min: [f64; 3],
    cell: f64,
    /// Índices dos splats agrupados por célula.
    entries: Vec<u32>,
    /// Início de cada célula densa em `entries` (+ sentinela).
    start: Vec<u32>,
    keys: Vec<i64>,
    ids: Vec<u32>,
    mask: usize,
    cell_count: usize,
}

fn hash_slot(key: i64, mask: usize) -> usize {
    ((key as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 20) as usize & mask
}

impl Grid {
    fn build(work: &Work, min: [f64; 3], max: [f64; 3], cell: f64) -> Grid {
        let n = work.n;
        let dims = [0, 1, 2].map(|a| (((max[a] - min[a] + 1e-9) / cell).ceil() as i64).clamp(1, MAX_GRID_RESOLUTION));
        let mut capacity = 1usize;
        while capacity < n * 2 {
            capacity <<= 1;
        }
        let mut keys = vec![-1i64; capacity];
        let mut ids = vec![0u32; capacity];
        let mask = capacity - 1;

        let mut grid = Grid { n: dims, min, cell, entries: Vec::new(), start: Vec::new(), keys: Vec::new(), ids: Vec::new(), mask, cell_count: 0 };
        let mut cell_of = vec![0u32; n];
        let mut cell_count = 0u32;
        for (i, c) in cell_of.iter_mut().enumerate() {
            let key = grid.key_of(&work.pos[i * 3..i * 3 + 3]);
            let mut slot = hash_slot(key, mask);
            loop {
                if keys[slot] == key {
                    *c = ids[slot];
                    break;
                }
                if keys[slot] == -1 {
                    keys[slot] = key;
                    ids[slot] = cell_count;
                    *c = cell_count;
                    cell_count += 1;
                    break;
                }
                slot = (slot + 1) & mask;
            }
        }
        let cells = cell_count as usize;
        let mut start = vec![0u32; cells + 1];
        for &c in &cell_of {
            start[c as usize + 1] += 1;
        }
        for c in 0..cells {
            start[c + 1] += start[c];
        }
        let mut cursor = start[..cells].to_vec();
        let mut entries = vec![0u32; n];
        for (i, &c) in cell_of.iter().enumerate() {
            entries[cursor[c as usize] as usize] = i as u32;
            cursor[c as usize] += 1;
        }
        grid.entries = entries;
        grid.start = start;
        grid.keys = keys;
        grid.ids = ids;
        grid.cell_count = cells;
        grid
    }

    fn coords(&self, p: &[f32]) -> [i64; 3] {
        [0, 1, 2].map(|a| (((p[a] as f64 - self.min[a]) / self.cell).floor() as i64).clamp(0, self.n[a] - 1))
    }

    fn key(&self, c: [i64; 3]) -> i64 {
        (c[0] * self.n[1] + c[1]) * self.n[2] + c[2]
    }

    fn key_of(&self, p: &[f32]) -> i64 {
        self.key(self.coords(p))
    }

    fn lookup(&self, key: i64) -> Option<usize> {
        let mut slot = hash_slot(key, self.mask);
        loop {
            match self.keys[slot] {
                k if k == key => return Some(self.ids[slot] as usize),
                -1 => return None,
                _ => slot = (slot + 1) & self.mask,
            }
        }
    }
}

fn build_candidates(work: &Work, reach: f64) -> (Vec<f64>, Vec<u32>) {
    let n = work.n;
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in work.pos.chunks_exact(3) {
        for a in 0..3 {
            min[a] = min[a].min(p[a] as f64);
            max[a] = max[a].max(p[a] as f64);
        }
    }
    let ext = [0, 1, 2].map(|a| (max[a] - min[a]).max(1e-9));

    // Célula com ocupação média baixa: o custo da varredura de 27 células é
    // ocupação × 27 por splat. Superfícies ocupam bem menos células do que o
    // palpite volumétrico supõe, daí as correções pela ocupação real.
    let min_cell = ext[0].max(ext[1]).max(ext[2]) / MAX_GRID_RESOLUTION as f64;
    let mut cell = ((ext[0] * ext[1] * ext[2] / n as f64).cbrt() * 2.0 * reach).max(min_cell);
    let mut grid = Grid::build(work, min, max, cell);
    for _ in 0..6 {
        if n as f64 / grid.cell_count as f64 <= 2.5 || cell <= min_cell {
            break;
        }
        cell = (cell / 1.6).max(min_cell);
        grid = Grid::build(work, min, max, cell);
    }

    let mut cost = vec![f64::INFINITY; n * CANDIDATES];
    let mut idx = vec![NONE; n * CANDIDATES];
    cost.par_chunks_mut(CANDIDATES)
        .zip(idx.par_chunks_mut(CANDIDATES))
        .enumerate()
        .for_each(|(i, (cost, idx))| {
            let p = &work.pos[i * 3..i * 3 + 3];
            let c = grid.coords(p);
            let mut nn_d = [f32::INFINITY; NEIGHBOR_SCAN_CAP];
            let mut nn_i = [NONE; NEIGHBOR_SCAN_CAP];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let q = [c[0] + dx, c[1] + dy, c[2] + dz];
                        if (0..3).any(|a| q[a] < 0 || q[a] >= grid.n[a]) {
                            continue;
                        }
                        let Some(cid) = grid.lookup(grid.key(q)) else { continue };
                        let s = grid.start[cid] as usize;
                        let e = (grid.start[cid + 1] as usize).min(s + PER_CELL_CAP);
                        for &j in &grid.entries[s..e] {
                            if j as usize == i {
                                continue;
                            }
                            let o = &work.pos[j as usize * 3..j as usize * 3 + 3];
                            let d2 = (o[0] - p[0]).powi(2) + (o[1] - p[1]).powi(2) + (o[2] - p[2]).powi(2);
                            insert_sorted(&mut nn_d, &mut nn_i, d2, j);
                        }
                    }
                }
            }
            for &j in nn_i.iter().filter(|&&j| j != NONE) {
                insert_sorted(cost, idx, edge_cost(work, i, j as usize), j);
            }
        });
    (cost, idx)
}

/// Insere `(v, j)` numa lista curta ordenada por `v`, descartando o pior.
fn insert_sorted<T: PartialOrd + Copy>(vals: &mut [T], ids: &mut [u32], v: T, j: u32) {
    let last = vals.len() - 1;
    // NaN também fica de fora.
    if v.partial_cmp(&vals[last]) != Some(std::cmp::Ordering::Less) {
        return;
    }
    let mut k = last;
    while k > 0 && vals[k - 1] > v {
        vals[k] = vals[k - 1];
        ids[k] = ids[k - 1];
        k -= 1;
    }
    vals[k] = v;
    ids[k] = j;
}

/// Custo de fundir (i, j): aumento de entropia + |Δdc|².
fn edge_cost(w: &Work, i: usize, j: usize) -> f64 {
    let (wi, wj) = (w.ink[i], w.ink[j]);
    let pi = wi / (wi + wj);
    let pj = 1.0 - pi;
    let pos = |k: usize, a: usize| w.pos[k * 3 + a] as f64;
    let m = [0, 1, 2].map(|a| pi * pos(i, a) + pj * pos(j, a));
    let di = [0, 1, 2].map(|a| pos(i, a) - m[a]);
    let dj = [0, 1, 2].map(|a| pos(j, a) - m[a]);
    let (ci, cj) = (&w.cov[i * 6..i * 6 + 6], &w.cov[j * 6..j * 6 + 6]);
    let sxx = pi * (ci[0] + di[0] * di[0]) + pj * (cj[0] + dj[0] * dj[0]) + EPS_COV;
    let syy = pi * (ci[1] + di[1] * di[1]) + pj * (cj[1] + dj[1] * dj[1]) + EPS_COV;
    let szz = pi * (ci[2] + di[2] * di[2]) + pj * (cj[2] + dj[2] * dj[2]) + EPS_COV;
    let sxy = pi * (ci[3] + di[0] * di[1]) + pj * (cj[3] + dj[0] * dj[1]);
    let sxz = pi * (ci[4] + di[0] * di[2]) + pj * (cj[4] + dj[0] * dj[2]);
    let syz = pi * (ci[5] + di[1] * di[2]) + pj * (cj[5] + dj[1] * dj[2]);
    let det = sxx * (syy * szz - syz * syz) - sxy * (sxy * szz - syz * sxz) + sxz * (sxy * syz - syy * sxz);
    let geo = 0.5 * (det.max(1e-30).ln() - pi * w.logdet[i] - pj * w.logdet[j]);
    let color: f64 = (0..3).map(|a| (w.dc[i * 3 + a] as f64 - w.dc[j * 3 + a] as f64).powi(2)).sum();
    geo + COLOR_COST_SCALE * color
}

// ---------------------------------------------------------------------------
// Seleção: casamento guloso disjunto em ordem de custo + fechamento em cadeia.
// ---------------------------------------------------------------------------

struct Selection {
    /// Grupo de cada splat, ou NONE para os que sobrevivem.
    group_of: Vec<u32>,
    /// Membros por grupo (CSR).
    offsets: Vec<u32>,
    members: Vec<u32>,
    removed: usize,
}

fn select_merges(n: usize, cost: &[f64], idx: &[u32], needed: usize) -> Selection {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &c in cost.iter().filter(|c| c.is_finite()) {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    let mut group_of = vec![NONE; n];
    if !lo.is_finite() {
        return Selection { group_of, offsets: vec![0], members: Vec::new(), removed: 0 };
    }
    let span = if hi > lo { hi - lo } else { 1.0 };
    let bucket = |c: f64| (((c - lo) / span * SELECT_BUCKETS as f64) as usize).min(SELECT_BUCKETS - 1);

    // Ordenação por contagem das entradas finitas, por faixa de custo.
    let mut counts = vec![0u32; SELECT_BUCKETS + 1];
    for &c in cost.iter().filter(|c| c.is_finite()) {
        counts[bucket(c) + 1] += 1;
    }
    for b in 0..SELECT_BUCKETS {
        counts[b + 1] += counts[b];
    }
    let mut order = vec![0u32; counts[SELECT_BUCKETS] as usize];
    let mut cursor = counts[..SELECT_BUCKETS].to_vec();
    for (e, &c) in cost.iter().enumerate() {
        if c.is_finite() {
            let b = bucket(c);
            order[cursor[b] as usize] = e as u32;
            cursor[b] += 1;
        }
    }

    let mut pairs: Vec<[u32; 2]> = Vec::new();
    let mut removed = 0;
    // Passada principal: casa pontas livres, do mais barato para o mais caro.
    for &e in &order {
        if removed >= needed {
            break;
        }
        let j = idx[e as usize];
        let i = e as usize / CANDIDATES;
        if j == NONE || group_of[i] != NONE || group_of[j as usize] != NONE {
            continue;
        }
        group_of[i] = pairs.len() as u32;
        group_of[j as usize] = pairs.len() as u32;
        pairs.push([i as u32, j]);
        removed += 1;
    }

    // Fechamento em cadeia: splats ainda livres entram no grupo de um
    // candidato, para alvos altos fecharem em menos gerações.
    let groups = pairs.len();
    let mut size = vec![2u32; groups];
    let mut joins: Vec<(u32, u32)> = Vec::new();
    for cap in GROUP_CAPS {
        for &e in &order {
            if removed >= needed {
                break;
            }
            let j = idx[e as usize];
            let i = e as usize / CANDIDATES;
            if j == NONE || group_of[i] != NONE {
                continue;
            }
            let g = group_of[j as usize];
            if g == NONE || size[g as usize] >= cap {
                continue;
            }
            group_of[i] = g;
            size[g as usize] += 1;
            joins.push((i as u32, g));
            removed += 1;
        }
    }

    let mut offsets = vec![0u32; groups + 1];
    for g in 0..groups {
        offsets[g + 1] = offsets[g] + size[g];
    }
    let mut members = vec![0u32; offsets[groups] as usize];
    let mut fill = vec![2u32; groups];
    for (g, p) in pairs.iter().enumerate() {
        members[offsets[g] as usize] = p[0];
        members[offsets[g] as usize + 1] = p[1];
    }
    for (i, g) in joins {
        members[(offsets[g as usize] + fill[g as usize]) as usize] = i;
        fill[g as usize] += 1;
    }
    Selection { group_of, offsets, members, removed }
}

// ---------------------------------------------------------------------------
// Fusão: casamento de momentos n-ário que conserva a tinta.
// ---------------------------------------------------------------------------

fn apply_merges(w: &Work, sel: &Selection) -> Work {
    let mut out = Work::with_capacity(w.n - sel.removed, w.rl);
    let mut emitted = vec![false; sel.offsets.len() - 1];
    let mut rest = vec![0f64; w.rl];
    for i in 0..w.n {
        let g = sel.group_of[i];
        if g == NONE {
            out.copy_from(w, i);
            continue;
        }
        if std::mem::replace(&mut emitted[g as usize], true) {
            continue;
        }
        let members = &sel.members[sel.offsets[g as usize] as usize..sel.offsets[g as usize + 1] as usize];
        merge_group(w, members, &mut out, &mut rest);
    }
    out
}

fn merge_group(w: &Work, members: &[u32], out: &mut Work, rest: &mut [f64]) {
    let total: f64 = members.iter().map(|&i| w.ink[i as usize]).sum();
    let weight = |i: usize| w.ink[i] / total;
    let mut m = [0f64; 3];
    for &i in members {
        let i = i as usize;
        for a in 0..3 {
            m[a] += weight(i) * w.pos[i * 3 + a] as f64;
        }
    }
    let mut s = [EPS_COV, EPS_COV, EPS_COV, 0.0, 0.0, 0.0];
    let mut dc = [0f64; 3];
    rest.fill(0.0);
    for &i in members {
        let i = i as usize;
        let p = weight(i);
        let d = [0, 1, 2].map(|a| w.pos[i * 3 + a] as f64 - m[a]);
        let c = &w.cov[i * 6..i * 6 + 6];
        s[0] += p * (c[0] + d[0] * d[0]);
        s[1] += p * (c[1] + d[1] * d[1]);
        s[2] += p * (c[2] + d[2] * d[2]);
        s[3] += p * (c[3] + d[0] * d[1]);
        s[4] += p * (c[4] + d[0] * d[2]);
        s[5] += p * (c[5] + d[1] * d[2]);
        for a in 0..3 {
            dc[a] += p * w.dc[i * 3 + a] as f64;
        }
        for (k, r) in rest.iter_mut().enumerate() {
            *r += p * w.rest[i * w.rl + k] as f64;
        }
    }

    let mut mat = [s[0], s[3], s[4], s[3], s[1], s[5], s[4], s[5], s[2]];
    let mut vecs = jacobi_eigen(&mut mat);
    let mut ev = [mat[0], mat[4], mat[8]];
    sort_eigen_desc(&mut ev, &mut vecs);
    ensure_proper_rotation(&mut vecs);
    let sc = ev.map(|v| v.max(1e-12).sqrt());
    let q = quat_from_matrix(&vecs);

    // Opacidade que conserva a tinta: a tinta do grupo espalhada pela área do
    // splat fundido, limitada a 1.
    let area = ellipsoid_area(sc[0], sc[1], sc[2]).max(1e-30);
    let opacity = (total / area).min(1.0);
    let rest32: Vec<f32> = rest.iter().map(|&v| v as f32).collect();
    out.push(
        &m.map(|v| v as f32),
        sc.map(|v| v as f32),
        q.map(|v| v as f32),
        opacity as f32,
        &dc.map(|v| v as f32),
        &rest32,
    );
}

// ---------------------------------------------------------------------------
// Álgebra 3×3 (linha maior), igual ao sharp-eigen.ts.
// ---------------------------------------------------------------------------

/// Matriz de rotação a partir de (w, x, y, z) normalizado.
fn quat_to_matrix([w, x, y, z]: [f64; 4]) -> [f64; 9] {
    [
        w * w + x * x - y * y - z * z,
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        w * w - x * x + y * y - z * z,
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        w * w - x * x - y * y + z * z,
    ]
}

/// Jacobi in-place: a diagonal de `m` vira os autovalores; as colunas do
/// retorno são os autovetores.
fn jacobi_eigen(m: &mut [f64; 9]) -> [f64; 9] {
    let mut v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for _ in 0..8 {
        if m[1].abs() + m[2].abs() + m[5].abs() < 1e-14 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            jacobi_rotate(m, &mut v, p, q);
        }
    }
    v
}

fn jacobi_rotate(m: &mut [f64; 9], v: &mut [f64; 9], p: usize, q: usize) {
    let (pp, qq, pq, qp) = (p * 3 + p, q * 3 + q, p * 3 + q, q * 3 + p);
    let (app, aqq, apq) = (m[pp], m[qq], m[pq]);
    if apq.abs() < 1e-18 {
        return;
    }
    let tau = (aqq - app) / (2.0 * apq);
    let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau * tau).sqrt()) } else { -1.0 / (-tau + (1.0 + tau * tau).sqrt()) };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    for k in 0..3 {
        if k == p || k == q {
            continue;
        }
        let (mkp, mkq) = (m[k * 3 + p], m[k * 3 + q]);
        let (nkp, nkq) = (c * mkp - s * mkq, s * mkp + c * mkq);
        m[k * 3 + p] = nkp;
        m[p * 3 + k] = nkp;
        m[k * 3 + q] = nkq;
        m[q * 3 + k] = nkq;
    }
    m[pp] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    m[qq] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    m[pq] = 0.0;
    m[qp] = 0.0;
    for k in 0..3 {
        let (vkp, vkq) = (v[k * 3 + p], v[k * 3 + q]);
        v[k * 3 + p] = c * vkp - s * vkq;
        v[k * 3 + q] = s * vkp + c * vkq;
    }
}

fn swap_columns(v: &mut [f64; 9], a: usize, b: usize) {
    for row in 0..3 {
        v.swap(row * 3 + a, row * 3 + b);
    }
}

fn sort_eigen_desc(ev: &mut [f64; 3], v: &mut [f64; 9]) {
    for (a, b) in [(0, 1), (1, 2), (0, 1)] {
        if ev[a] < ev[b] {
            ev.swap(a, b);
            swap_columns(v, a, b);
        }
    }
}

/// Inverte uma coluna se preciso para a base ser rotação própria (det = +1).
fn ensure_proper_rotation(r: &mut [f64; 9]) {
    let det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
    if det < 0.0 {
        r[2] = -r[2];
        r[5] = -r[5];
        r[8] = -r[8];
    }
}

fn quat_from_matrix(r: &[f64; 9]) -> [f64; 4] {
    let [r00, r01, r02, r10, r11, r12, r20, r21, r22] = *r;
    let trace = r00 + r11 + r22;
    let q = if trace > 0.0 {
        let s = 2.0 * (trace + 1.0).max(1e-12).sqrt();
        [0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s]
    } else if r00 > r11 && r00 > r22 {
        let s = 2.0 * (1.0 + r00 - r11 - r22).max(1e-12).sqrt();
        [(r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s]
    } else if r11 > r22 {
        let s = 2.0 * (1.0 + r11 - r00 - r22).max(1e-12).sqrt();
        [(r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s]
    } else {
        let s = 2.0 * (1.0 + r22 - r00 - r11).max(1e-12).sqrt();
        [(r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s]
    };
    let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    q.map(|v| v / norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::ply::tests::sample;

    #[test]
    fn hits_target_and_keeps_ink() {
        let s = sample(40_000, 1);
        let out = decimate(&s, 0.5, |_| {}).unwrap();
        assert!((out.count as f64 - 20_000.0).abs() < 400.0, "count {}", out.count);
        assert_eq!(out.rest.len(), out.count * 9);
        assert!(out.pos.iter().chain(&out.scale).chain(&out.opacity).all(|v| v.is_finite()));

        // A tinta total (opacidade × área) não deve encolher com a fusão
        // (só cresce quando a opacidade bate no teto de 1… e aí encolhe um pouco).
        let ink = |s: &Splats| -> f64 {
            (0..s.count)
                .map(|i| {
                    let sc = [0, 1, 2].map(|a| (s.scale[i * 3 + a] as f64).exp());
                    sigmoid(s.opacity[i]) * ellipsoid_area(sc[0], sc[1], sc[2])
                })
                .sum()
        };
        let (a, b) = (ink(&s), ink(&out));
        assert!(b > a * 0.8, "tinta {a} → {b}");
    }

    #[test]
    fn rotation_roundtrip() {
        let q = [0.8f64, 0.2, -0.4, 0.4];
        let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        let q = q.map(|v| v / n);
        let back = quat_from_matrix(&quat_to_matrix(q));
        let dot: f64 = q.iter().zip(&back).map(|(a, b)| a * b).sum();
        assert!((dot.abs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn small_input_untouched() {
        assert!(decimate(&sample(500, 0), 0.5, |_| {}).is_none());
    }
}
