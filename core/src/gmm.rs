//! Gaussian mixture models in 2D, fit with expectation–maximization, recorded
//! iteration by iteration.

use std::f64::consts::PI;

use wasm_bindgen::prelude::*;

use crate::grid::Grid;
use crate::kmeans::kmeans_init;

/// Per-component parameters: mean (2), covariance `[sxx, sxy, syy]`, weight.
pub const COMP_STRIDE: usize = 6;
const REG: f64 = 1e-4;

#[derive(Clone, Copy)]
struct Comp {
    mx: f64,
    my: f64,
    sxx: f64,
    sxy: f64,
    syy: f64,
    w: f64,
}

impl Comp {
    fn log_pdf(&self, x: f64, y: f64) -> f64 {
        let det = (self.sxx * self.syy - self.sxy * self.sxy).max(1e-300);
        let dx = x - self.mx;
        let dy = y - self.my;
        // (Σ⁻¹ = 1/det · [[syy, −sxy], [−sxy, sxx]])
        let q = (self.syy * dx * dx - 2.0 * self.sxy * dx * dy + self.sxx * dy * dy) / det;
        -0.5 * q - 0.5 * det.ln() - (2.0 * PI).ln()
    }
}

fn log_sum_exp(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

fn decode(params: &[f64]) -> Vec<Comp> {
    params
        .chunks_exact(COMP_STRIDE)
        .map(|c| Comp { mx: c[0], my: c[1], sxx: c[2], sxy: c[3], syy: c[4], w: c[5] })
        .collect()
}

fn encode(comps: &[Comp], out: &mut Vec<f64>) {
    for c in comps {
        out.extend_from_slice(&[c.mx, c.my, c.sxx, c.sxy, c.syy, c.w]);
    }
}

/// A complete EM run.
///
/// With `T = iterations`: `params` holds `T + 1` snapshots of `k` components
/// (stride [`COMP_STRIDE`]), `responsibilities` holds `T + 1` blocks of `n · k`
/// values (soft assignments computed from the matching snapshot), and
/// `log_likelihood` holds `T + 1` mean log-likelihoods.
#[wasm_bindgen]
pub struct GmmRun {
    params: Vec<f64>,
    responsibilities: Vec<f64>,
    log_likelihood: Vec<f64>,
    iterations: usize,
    converged: bool,
}

#[wasm_bindgen]
impl GmmRun {
    #[wasm_bindgen(getter)]
    pub fn params(&self) -> Vec<f64> {
        self.params.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn responsibilities(&self) -> Vec<f64> {
        self.responsibilities.clone()
    }
    #[wasm_bindgen(getter, js_name = logLikelihood)]
    pub fn log_likelihood(&self) -> Vec<f64> {
        self.log_likelihood.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> usize {
        self.iterations
    }
    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.converged
    }
}

/// E-step: responsibilities (n · k) and the mean log-likelihood.
fn e_step(pts: &[(f64, f64)], comps: &[Comp], resp: &mut Vec<f64>) -> f64 {
    let k = comps.len();
    resp.clear();
    let mut buf = vec![0.0; k];
    let mut ll = 0.0;
    for &(x, y) in pts {
        for (j, c) in comps.iter().enumerate() {
            buf[j] = c.w.max(1e-300).ln() + c.log_pdf(x, y);
        }
        let lse = log_sum_exp(&buf);
        ll += lse;
        resp.extend(buf.iter().map(|v| (v - lse).exp()));
    }
    ll / pts.len().max(1) as f64
}

/// Fit a `k`-component GMM to stride-`stride` points with EM. Means start at
/// k-means++ seeds, covariances at a fraction of the data covariance.
#[wasm_bindgen]
pub fn gmm_run(points: &[f64], stride: usize, k: usize, seed: u32, max_iter: usize) -> GmmRun {
    let stride = stride.max(2);
    let pts: Vec<(f64, f64)> = points.chunks_exact(stride).map(|p| (p[0], p[1])).collect();
    let n = pts.len().max(1) as f64;
    let k = k.max(1);
    let (mut mx, mut my) = (0.0, 0.0);
    for &(x, y) in &pts {
        mx += x;
        my += y;
    }
    mx /= n;
    my /= n;
    let (mut vx, mut vy) = (0.0, 0.0);
    for &(x, y) in &pts {
        vx += (x - mx).powi(2);
        vy += (y - my).powi(2);
    }
    let spread = ((vx + vy) / (2.0 * n)).max(1e-3) * 0.25;
    let seeds = kmeans_init(points, stride, k, 1, seed);
    let mut comps: Vec<Comp> = seeds
        .chunks_exact(2)
        .map(|m| Comp { mx: m[0], my: m[1], sxx: spread, sxy: 0.0, syy: spread, w: 1.0 / k as f64 })
        .collect();

    let mut params = Vec::new();
    let mut responsibilities = Vec::new();
    let mut log_likelihood = Vec::new();
    let mut resp = Vec::new();
    let mut iterations = 0;
    let mut converged = false;

    encode(&comps, &mut params);
    let mut ll = e_step(&pts, &comps, &mut resp);
    responsibilities.extend_from_slice(&resp);
    log_likelihood.push(ll);

    for _ in 0..max_iter {
        // M-step.
        for (j, c) in comps.iter_mut().enumerate() {
            let mut nk = 0.0;
            let (mut sx, mut sy) = (0.0, 0.0);
            for (i, &(x, y)) in pts.iter().enumerate() {
                let r = resp[i * k + j];
                nk += r;
                sx += r * x;
                sy += r * y;
            }
            if nk < 1e-9 {
                continue;
            }
            let (ux, uy) = (sx / nk, sy / nk);
            let (mut a, mut b, mut d) = (0.0, 0.0, 0.0);
            for (i, &(x, y)) in pts.iter().enumerate() {
                let r = resp[i * k + j];
                a += r * (x - ux) * (x - ux);
                b += r * (x - ux) * (y - uy);
                d += r * (y - uy) * (y - uy);
            }
            *c = Comp { mx: ux, my: uy, sxx: a / nk + REG, sxy: b / nk, syy: d / nk + REG, w: nk / n };
        }
        let next = e_step(&pts, &comps, &mut resp);
        iterations += 1;
        encode(&comps, &mut params);
        responsibilities.extend_from_slice(&resp);
        log_likelihood.push(next);
        if (next - ll).abs() < 1e-6 {
            converged = true;
            break;
        }
        ll = next;
    }
    GmmRun { params, responsibilities, log_likelihood, iterations, converged }
}

/// Mixture density and the most responsible component over a grid, for one
/// parameter snapshot (`k · COMP_STRIDE` values). Stride 2: `[component, density]`.
#[wasm_bindgen]
pub fn gmm_grid(params: &[f64], xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Vec<f64> {
    let comps = decode(params);
    let g = Grid::new(xmin, xmax, ymin, ymax, nx, ny);
    let mut out = Vec::with_capacity(g.len() * 2);
    for j in 0..g.ny {
        let y = g.y(j);
        for i in 0..g.nx {
            let x = g.x(i);
            let mut best = 0;
            let mut best_v = f64::NEG_INFINITY;
            let mut total = 0.0;
            for (c, comp) in comps.iter().enumerate() {
                let v = comp.w.max(1e-300).ln() + comp.log_pdf(x, y);
                total += v.exp();
                if v > best_v {
                    best_v = v;
                    best = c;
                }
            }
            out.push(best as f64);
            out.push(total);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_gaussian;

    fn mixture() -> Vec<f64> {
        let mut pts = make_gaussian(300, -0.5, 0.3, 0.35, 0.06, 0.6, 1);
        pts.extend(make_gaussian(300, 0.5, -0.2, 0.3, 0.07, -0.9, 2));
        pts
    }

    #[test]
    fn likelihood_never_decreases_and_recovers_shapes() {
        let pts = mixture();
        let run = gmm_run(&pts, 2, 2, 3, 300);
        for w in run.log_likelihood.windows(2) {
            assert!(w[1] >= w[0] - 1e-9, "EM must not decrease the likelihood");
        }
        let last = &run.params[run.iterations * 2 * COMP_STRIDE..];
        let comps = decode(last);
        // Each component should be strongly elongated (eigen ratio) like the truth.
        for c in comps {
            let tr = c.sxx + c.syy;
            let det = c.sxx * c.syy - c.sxy * c.sxy;
            let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
            let ratio = ((tr / 2.0 - disc) / (tr / 2.0 + disc)).sqrt();
            assert!(ratio < 0.4, "component not elongated: {ratio}");
            assert!((c.w - 0.5).abs() < 0.1);
        }
    }

    #[test]
    fn responsibilities_are_distributions() {
        let pts = mixture();
        let run = gmm_run(&pts, 2, 3, 1, 20);
        for r in run.responsibilities.chunks(3) {
            let s: f64 = r.iter().sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
        let g = gmm_grid(&run.params[..3 * COMP_STRIDE], -1.0, 1.0, -1.0, 1.0, 8, 8);
        assert_eq!(g.len(), 128);
    }
}
