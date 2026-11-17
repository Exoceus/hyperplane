//! Support vector machines trained with SMO (sequential minimal
//! optimization), using LIBSVM's second-order working-set selection.
//!
//! Solves the soft-margin dual
//! `min ½ αᵀQα − Σα  s.t.  yᵀα = 0, 0 ≤ α ≤ C`, with `Q_ij = y_i y_j K(x_i, x_j)`.

use wasm_bindgen::prelude::*;

use crate::grid::Grid;

const TAU: f64 = 1e-12;

#[derive(Clone, Copy)]
struct Kernel {
    kind: u8,
    gamma: f64,
    degree: i32,
    coef0: f64,
}

impl Kernel {
    /// 0 = linear, 1 = RBF, 2 = polynomial.
    fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        match self.kind {
            1 => {
                let d2: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
                (-self.gamma * d2).exp()
            }
            2 => {
                let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                (self.gamma * dot + self.coef0).powi(self.degree)
            }
            _ => a.iter().zip(b).map(|(x, y)| x * y).sum(),
        }
    }
}

/// A trained SVM. The decision function is `f(x) = Σ αᵢ yᵢ K(xᵢ, x) − ρ`.
#[wasm_bindgen]
pub struct SvmModel {
    dim: usize,
    points: Vec<f64>,
    labels: Vec<f64>,
    alpha: Vec<f64>,
    rho: f64,
    c: f64,
    kernel: Kernel,
    iterations: usize,
}

#[wasm_bindgen]
impl SvmModel {
    /// Dual coefficients αᵢ, one per training point (0 = not a support vector).
    #[wasm_bindgen(getter)]
    pub fn alphas(&self) -> Vec<f64> {
        self.alpha.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn rho(&self) -> f64 {
        self.rho
    }

    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    #[wasm_bindgen(getter)]
    pub fn c(&self) -> f64 {
        self.c
    }

    /// Primal weights `w = Σ αᵢ yᵢ xᵢ` (meaningful for the linear kernel).
    pub fn weights(&self) -> Vec<f64> {
        let mut w = vec![0.0; self.dim];
        for (i, &a) in self.alpha.iter().enumerate() {
            if a > 0.0 {
                for k in 0..self.dim {
                    w[k] += a * self.labels[i] * self.points[i * self.dim + k];
                }
            }
        }
        w
    }

    /// Decision value at a point (given as `dim` coordinates).
    pub fn decision(&self, x: &[f64]) -> f64 {
        let mut f = -self.rho;
        for (i, &a) in self.alpha.iter().enumerate() {
            if a > 1e-12 {
                f += a * self.labels[i] * self.kernel.eval(&self.points[i * self.dim..(i + 1) * self.dim], x);
            }
        }
        f
    }

    /// Decision values over a grid (2D models only).
    pub fn decision_grid(&self, xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Vec<f64> {
        let sv: Vec<usize> = (0..self.alpha.len()).filter(|&i| self.alpha[i] > 1e-12).collect();
        let mut buf = [0.0; 2];
        Grid::new(xmin, xmax, ymin, ymax, nx, ny).map(|x, y| {
            buf[0] = x;
            buf[1] = y;
            let mut f = -self.rho;
            for &i in &sv {
                f += self.alpha[i] * self.labels[i] * self.kernel.eval(&self.points[i * self.dim..i * self.dim + 2], &buf);
            }
            f
        })
    }
}

/// Train an SVM on data with stride `dim + 1` (`dim` coordinates, then the
/// class: 1 → y = +1, anything else → y = −1).
///
/// `kernel`: 0 = linear, 1 = RBF `exp(−γ‖x−z‖²)`, 2 = polynomial `(γ x·z + coef0)^degree`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn svm_train(data: &[f64], dim: usize, c: f64, kernel: u8, gamma: f64, degree: u32, coef0: f64, max_iter: usize) -> SvmModel {
    let dim = dim.max(1);
    let stride = dim + 1;
    let n = data.len() / stride;
    let kernel = Kernel { kind: kernel, gamma, degree: degree.max(1) as i32, coef0 };
    let mut points = Vec::with_capacity(n * dim);
    let mut y = Vec::with_capacity(n);
    for p in data.chunks_exact(stride) {
        points.extend_from_slice(&p[..dim]);
        y.push(if p[dim] > 0.5 { 1.0 } else { -1.0 });
    }
    let c = c.max(1e-9);

    // Precompute the kernel matrix: n is small (hundreds of points).
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let v = kernel.eval(&points[i * dim..(i + 1) * dim], &points[j * dim..(j + 1) * dim]);
            k[i * n + j] = v;
            k[j * n + i] = v;
        }
    }
    let q = |i: usize, j: usize| y[i] * y[j] * k[i * n + j];

    let mut alpha = vec![0.0; n];
    let mut grad = vec![-1.0; n];
    let mut iterations = 0;
    let has_both = y.iter().any(|&v| v > 0.0) && y.iter().any(|&v| v < 0.0);

    while has_both && iterations < max_iter {
        // Working-set selection (Fan, Chen & Lin 2005).
        let mut gmax = f64::NEG_INFINITY;
        let mut i_sel = usize::MAX;
        for t in 0..n {
            let up = (y[t] > 0.0 && alpha[t] < c) || (y[t] < 0.0 && alpha[t] > 0.0);
            if up && -y[t] * grad[t] >= gmax {
                gmax = -y[t] * grad[t];
                i_sel = t;
            }
        }
        if i_sel == usize::MAX {
            break;
        }
        let i = i_sel;
        let mut gmin = f64::INFINITY;
        let mut obj_min = f64::INFINITY;
        let mut j_sel = usize::MAX;
        for t in 0..n {
            let low = (y[t] < 0.0 && alpha[t] < c) || (y[t] > 0.0 && alpha[t] > 0.0);
            if !low {
                continue;
            }
            let v = -y[t] * grad[t];
            gmin = gmin.min(v);
            let b = gmax - v;
            if b > 0.0 {
                let a = (k[i * n + i] + k[t * n + t] - 2.0 * k[i * n + t]).max(TAU);
                let obj = -(b * b) / a;
                if obj <= obj_min {
                    obj_min = obj;
                    j_sel = t;
                }
            }
        }
        if j_sel == usize::MAX || gmax - gmin < 1e-4 {
            break;
        }
        let j = j_sel;
        iterations += 1;

        let (old_i, old_j) = (alpha[i], alpha[j]);
        let quad = (k[i * n + i] + k[j * n + j] - 2.0 * k[i * n + j]).max(TAU);
        if y[i] != y[j] {
            let delta = (-grad[i] - grad[j]) / quad;
            let diff = alpha[i] - alpha[j];
            alpha[i] += delta;
            alpha[j] += delta;
            if diff > 0.0 {
                if alpha[j] < 0.0 {
                    alpha[j] = 0.0;
                    alpha[i] = diff;
                }
            } else if alpha[i] < 0.0 {
                alpha[i] = 0.0;
                alpha[j] = -diff;
            }
            if diff > 0.0 {
                if alpha[i] > c {
                    alpha[i] = c;
                    alpha[j] = c - diff;
                }
            } else if alpha[j] > c {
                alpha[j] = c;
                alpha[i] = c + diff;
            }
        } else {
            let delta = (grad[i] - grad[j]) / quad;
            let sum = alpha[i] + alpha[j];
            alpha[i] -= delta;
            alpha[j] += delta;
            if sum > c {
                if alpha[i] > c {
                    alpha[i] = c;
                    alpha[j] = sum - c;
                }
            } else if alpha[j] < 0.0 {
                alpha[j] = 0.0;
                alpha[i] = sum;
            }
            if sum > c {
                if alpha[j] > c {
                    alpha[j] = c;
                    alpha[i] = sum - c;
                }
            } else if alpha[i] < 0.0 {
                alpha[i] = 0.0;
                alpha[j] = sum;
            }
        }
        let (di, dj) = (alpha[i] - old_i, alpha[j] - old_j);
        for t in 0..n {
            grad[t] += q(t, i) * di + q(t, j) * dj;
        }
    }

    // Offset ρ: average over free support vectors, else midpoint of the bounds.
    let (mut ub, mut lb) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut free, mut sum_free) = (0usize, 0.0);
    for t in 0..n {
        let yg = y[t] * grad[t];
        if alpha[t] >= c - 1e-12 {
            if y[t] < 0.0 {
                ub = ub.min(yg);
            } else {
                lb = lb.max(yg);
            }
        } else if alpha[t] <= 1e-12 {
            if y[t] > 0.0 {
                ub = ub.min(yg);
            } else {
                lb = lb.max(yg);
            }
        } else {
            free += 1;
            sum_free += yg;
        }
    }
    let rho = if free > 0 {
        sum_free / free as f64
    } else if ub.is_finite() && lb.is_finite() {
        (ub + lb) / 2.0
    } else {
        0.0
    };

    SvmModel { dim, points, labels: y, alpha, rho, c, kernel, iterations }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_linear, make_shape};

    fn accuracy(m: &SvmModel, data: &[f64], dim: usize) -> f64 {
        let n = data.len() / (dim + 1);
        let ok = data
            .chunks(dim + 1)
            .filter(|p| (m.decision(&p[..dim]) >= 0.0) == (p[dim] > 0.5))
            .count();
        ok as f64 / n as f64
    }

    #[test]
    fn hard_margin_linear_finds_the_max_margin() {
        // Symmetric toy problem: the max-margin line is x = 0 with margin 1.
        let data = [-1.0, 0.0, 0.0, -2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, -1.0, 1.0];
        let m = svm_train(&data, 2, 1e6, 0, 1.0, 1, 0.0, 10_000);
        let w = m.weights();
        assert!((w[0] - 1.0).abs() < 1e-3 && w[1].abs() < 1e-3, "w = {w:?}");
        assert!(m.rho().abs() < 1e-3);
        // Exactly the two closest points are support vectors.
        let sv: Vec<usize> = (0..4).filter(|&i| m.alphas()[i] > 1e-6).collect();
        assert_eq!(sv, vec![0, 2]);
    }

    #[test]
    fn linear_separable_is_fit_perfectly_and_kkt_holds() {
        let data = make_linear(120, 0.6, 0.8, 0.1, 0.1, 0.0, 3);
        let c = 100.0;
        let m = svm_train(&data, 2, c, 0, 1.0, 1, 0.0, 100_000);
        assert_eq!(accuracy(&m, &data, 2), 1.0);
        // KKT: y f(x) ≥ 1 − tol for non-SVs, ≈ 1 for free SVs.
        for (i, p) in data.chunks(3).enumerate() {
            let yf = if p[2] > 0.5 { 1.0 } else { -1.0 } * m.decision(&p[..2]);
            let a = m.alphas()[i];
            if a < 1e-8 {
                assert!(yf > 1.0 - 1e-2, "point {i}: yf {yf}");
            } else if a < c - 1e-8 {
                assert!((yf - 1.0).abs() < 1e-2, "free SV {i}: yf {yf}");
            }
        }
        let sum: f64 = data.chunks(3).zip(m.alphas()).map(|(p, a)| if p[2] > 0.5 { a } else { -a }).sum();
        assert!(sum.abs() < 1e-8, "yᵀα = {sum}");
    }

    #[test]
    fn rbf_separates_circles_and_poly_separates_xor() {
        let circles = make_shape("circles", 200, 0.1, 1);
        let m = svm_train(&circles, 2, 10.0, 1, 2.0, 3, 1.0, 100_000);
        assert!(accuracy(&m, &circles, 2) > 0.97);
        let xor = make_shape("xor", 200, 0.05, 2);
        let m = svm_train(&xor, 2, 10.0, 2, 1.0, 2, 0.0, 100_000);
        assert!(accuracy(&m, &xor, 2) > 0.95);
    }

    #[test]
    fn works_in_three_dimensions() {
        // Circles lifted to (x, y, x² + y²) become linearly separable.
        let circles = make_shape("circles", 160, 0.05, 4);
        let lifted: Vec<f64> = circles.chunks(3).flat_map(|p| [p[0], p[1], p[0] * p[0] + p[1] * p[1], p[2]]).collect();
        let m = svm_train(&lifted, 3, 100.0, 0, 1.0, 1, 0.0, 100_000);
        assert_eq!(accuracy(&m, &lifted, 3), 1.0);
    }
}
