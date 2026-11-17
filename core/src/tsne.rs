//! t-SNE (exact O(n²) gradient), stepped incrementally so the embedding can
//! be watched as it forms.

use wasm_bindgen::prelude::*;

use crate::rng::Rng;

const EXAGGERATION: f64 = 12.0;
const EXAGGERATION_ITERS: u32 = 100;

#[wasm_bindgen]
pub struct Tsne {
    n: usize,
    p: Vec<f64>,
    y: Vec<f64>,
    vel: Vec<f64>,
    gains: Vec<f64>,
    iter: u32,
    learning_rate: f64,
}

/// Conditional affinities for one row with precision β; returns (row, entropy).
fn row_affinities(d2: &[f64], i: usize, beta: f64, row: &mut [f64]) -> f64 {
    let mut sum = 0.0;
    for (j, r) in row.iter_mut().enumerate() {
        *r = if j == i { 0.0 } else { (-beta * d2[j]).exp() };
        sum += *r;
    }
    let sum = sum.max(1e-300);
    let mut h = 0.0;
    for (j, r) in row.iter_mut().enumerate() {
        *r /= sum;
        if j != i && *r > 1e-300 {
            h -= *r * r.ln();
        }
    }
    h
}

#[wasm_bindgen]
impl Tsne {
    /// `points` are `n` rows of `dim` values (any extra columns are ignored
    /// when `stride > dim`).
    #[wasm_bindgen(constructor)]
    pub fn new(points: &[f64], stride: usize, dim: usize, perplexity: f64, learning_rate: f64, seed: u32) -> Tsne {
        let stride = stride.max(dim).max(1);
        let n = points.len() / stride;
        let target = perplexity.clamp(1.0, (n.max(2) - 1) as f64).ln();
        let mut p = vec![0.0; n * n];
        let mut d2 = vec![0.0; n];
        let mut row = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                d2[j] = (0..dim).map(|k| (points[i * stride + k] - points[j * stride + k]).powi(2)).sum();
            }
            // Binary search the precision so the row's entropy matches log(perplexity).
            let (mut lo, mut hi, mut beta) = (0.0, f64::INFINITY, 1.0);
            for _ in 0..64 {
                let h = row_affinities(&d2, i, beta, &mut row);
                if (h - target).abs() < 1e-6 {
                    break;
                }
                if h > target {
                    lo = beta;
                    beta = if hi.is_finite() { 0.5 * (beta + hi) } else { beta * 2.0 };
                } else {
                    hi = beta;
                    beta = 0.5 * (beta + lo);
                }
            }
            row_affinities(&d2, i, beta, &mut row);
            p[i * n..(i + 1) * n].copy_from_slice(&row);
        }
        // Symmetrize: p_ij = (p_j|i + p_i|j) / 2n.
        for i in 0..n {
            for j in i + 1..n {
                let v = ((p[i * n + j] + p[j * n + i]) / (2.0 * n as f64)).max(1e-12);
                p[i * n + j] = v;
                p[j * n + i] = v;
            }
            p[i * n + i] = 0.0;
        }
        let mut rng = Rng::new(seed);
        let y = (0..n * 2).map(|_| 1e-2 * rng.normal()).collect();
        Tsne { n, p, y, vel: vec![0.0; n * 2], gains: vec![1.0; n * 2], iter: 0, learning_rate }
    }

    /// Run `iters` gradient steps (with early exaggeration, momentum and gains).
    pub fn step(&mut self, iters: u32) {
        let n = self.n;
        let mut num = vec![0.0; n * n];
        let mut grad = vec![0.0; n * 2];
        for _ in 0..iters {
            let exag = if self.iter < EXAGGERATION_ITERS { EXAGGERATION } else { 1.0 };
            let momentum = if self.iter < 250 { 0.5 } else { 0.8 };
            let mut sum_q = 0.0;
            for i in 0..n {
                for j in i + 1..n {
                    let dx = self.y[2 * i] - self.y[2 * j];
                    let dy = self.y[2 * i + 1] - self.y[2 * j + 1];
                    let v = 1.0 / (1.0 + dx * dx + dy * dy);
                    num[i * n + j] = v;
                    num[j * n + i] = v;
                    sum_q += 2.0 * v;
                }
            }
            let sum_q = sum_q.max(1e-300);
            grad.iter_mut().for_each(|g| *g = 0.0);
            for i in 0..n {
                let (mut gx, mut gy) = (0.0, 0.0);
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let v = num[i * n + j];
                    let mult = (exag * self.p[i * n + j] - v / sum_q) * v;
                    gx += mult * (self.y[2 * i] - self.y[2 * j]);
                    gy += mult * (self.y[2 * i + 1] - self.y[2 * j + 1]);
                }
                grad[2 * i] = 4.0 * gx;
                grad[2 * i + 1] = 4.0 * gy;
            }
            for k in 0..n * 2 {
                let same = (grad[k] > 0.0) == (self.vel[k] > 0.0);
                self.gains[k] = if same { (self.gains[k] * 0.8).max(0.01) } else { self.gains[k] + 0.2 };
                self.vel[k] = momentum * self.vel[k] - self.learning_rate * self.gains[k] * grad[k];
                self.y[k] += self.vel[k];
            }
            // Keep the embedding centered.
            let (mut mx, mut my) = (0.0, 0.0);
            for i in 0..n {
                mx += self.y[2 * i];
                my += self.y[2 * i + 1];
            }
            mx /= n.max(1) as f64;
            my /= n.max(1) as f64;
            for i in 0..n {
                self.y[2 * i] -= mx;
                self.y[2 * i + 1] -= my;
            }
            self.iter += 1;
        }
    }

    /// Current 2D embedding (stride 2).
    pub fn embedding(&self) -> Vec<f64> {
        self.y.clone()
    }

    /// KL(P ‖ Q), the cost being minimized.
    pub fn kl(&self) -> f64 {
        let n = self.n;
        let mut sum_q = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dx = self.y[2 * i] - self.y[2 * j];
                    let dy = self.y[2 * i + 1] - self.y[2 * j + 1];
                    sum_q += 1.0 / (1.0 + dx * dx + dy * dy);
                }
            }
        }
        let mut kl = 0.0;
        for i in 0..n {
            for j in 0..n {
                let p = self.p[i * n + j];
                if i != j && p > 1e-12 {
                    let dx = self.y[2 * i] - self.y[2 * j];
                    let dy = self.y[2 * i + 1] - self.y[2 * j + 1];
                    let q = (1.0 / (1.0 + dx * dx + dy * dy)) / sum_q;
                    kl += p * (p / q.max(1e-300)).ln();
                }
            }
        }
        kl
    }

    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> u32 {
        self.iter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_clusters_nd;

    #[test]
    fn affinities_are_a_distribution_and_clusters_separate() {
        // Three well-separated clusters in 8D.
        let dim = 8;
        let mut centers = vec![0.0; 3 * dim];
        for c in 0..3 {
            centers[c * dim + c] = 6.0;
        }
        let data = make_clusters_nd(90, &centers, &[0.5, 0.5, 0.5], dim, 1);
        let mut t = Tsne::new(&data, dim + 1, dim, 15.0, 100.0, 2);
        let total: f64 = t.p.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "sum P = {total}");
        let kl0 = {
            t.step(150);
            t.kl()
        };
        t.step(300);
        assert!(t.kl() <= kl0 + 1e-9);
        // Nearest neighbour in the embedding shares the true cluster.
        let y = t.embedding();
        let n = 90;
        let mut agree = 0;
        for i in 0..n {
            let mut best = (f64::INFINITY, 0);
            for j in 0..n {
                if i != j {
                    let d = (y[2 * i] - y[2 * j]).powi(2) + (y[2 * i + 1] - y[2 * j + 1]).powi(2);
                    if d < best.0 {
                        best = (d, j);
                    }
                }
            }
            if data[i * (dim + 1) + dim] == data[best.1 * (dim + 1) + dim] {
                agree += 1;
            }
        }
        assert!(agree as f64 / n as f64 > 0.97);
    }
}
