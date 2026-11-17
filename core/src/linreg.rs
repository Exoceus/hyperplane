//! Simple linear regression `ŷ = m·x + b` with mean-squared-error loss.
//!
//! The MSE of a line only depends on five averages of the data
//! (the *sufficient statistics*), so evaluating the whole loss surface over a
//! grid of `(m, b)` values is O(1) per cell after one pass over the data.

use wasm_bindgen::prelude::*;

use crate::grid::Grid;

struct Stats {
    sx: f64,
    sy: f64,
    sxx: f64,
    sxy: f64,
    syy: f64,
}

impl Stats {
    fn of(data: &[f64]) -> Stats {
        let n = (data.len() / 2).max(1) as f64;
        let mut s = Stats { sx: 0.0, sy: 0.0, sxx: 0.0, sxy: 0.0, syy: 0.0 };
        for p in data.chunks_exact(2) {
            let (x, y) = (p[0], p[1]);
            s.sx += x;
            s.sy += y;
            s.sxx += x * x;
            s.sxy += x * y;
            s.syy += y * y;
        }
        s.sx /= n;
        s.sy /= n;
        s.sxx /= n;
        s.sxy /= n;
        s.syy /= n;
        s
    }

    /// `L(m, b) = mean((m x + b − y)²)` expanded in terms of the averages.
    fn mse(&self, m: f64, b: f64) -> f64 {
        (m * m * self.sxx + b * b + self.syy + 2.0 * m * b * self.sx - 2.0 * m * self.sxy - 2.0 * b * self.sy)
            .max(0.0)
    }

    fn grad(&self, m: f64, b: f64) -> (f64, f64) {
        (
            2.0 * (m * self.sxx + b * self.sx - self.sxy),
            2.0 * (m * self.sx + b - self.sy),
        )
    }
}

/// Closed-form least-squares fit (the normal equations). Returns `[m, b]`.
#[wasm_bindgen]
pub fn linreg_fit(data: &[f64]) -> Vec<f64> {
    let s = Stats::of(data);
    let var_x = s.sxx - s.sx * s.sx;
    if var_x.abs() < 1e-12 {
        return vec![0.0, s.sy];
    }
    let m = (s.sxy - s.sx * s.sy) / var_x;
    vec![m, s.sy - m * s.sx]
}

/// Mean squared error of the line `(m, b)` on stride-2 data.
#[wasm_bindgen]
pub fn linreg_mse(data: &[f64], m: f64, b: f64) -> f64 {
    Stats::of(data).mse(m, b)
}

/// Gradient `[∂L/∂m, ∂L/∂b]` of the MSE at `(m, b)`.
#[wasm_bindgen]
pub fn linreg_grad(data: &[f64], m: f64, b: f64) -> Vec<f64> {
    let (gm, gb) = Stats::of(data).grad(m, b);
    vec![gm, gb]
}

/// MSE evaluated over a grid of parameters (`m` on the x-axis, `b` on the
/// y-axis).
#[wasm_bindgen]
pub fn linreg_loss_grid(data: &[f64], mmin: f64, mmax: f64, bmin: f64, bmax: f64, nx: usize, ny: usize) -> Vec<f64> {
    let s = Stats::of(data);
    Grid::new(mmin, mmax, bmin, bmax, nx, ny).map(|m, b| s.mse(m, b))
}

/// Full-batch gradient descent from `(m0, b0)`. Returns stride-3
/// `[m, b, loss]` for the start and each step. Stops early when the loss
/// explodes (divergence) so the UI can show it without overflowing.
#[wasm_bindgen]
pub fn linreg_gd(data: &[f64], m0: f64, b0: f64, lr: f64, steps: usize) -> Vec<f64> {
    let s = Stats::of(data);
    let (mut m, mut b) = (m0, b0);
    let mut out = Vec::with_capacity((steps + 1) * 3);
    out.extend_from_slice(&[m, b, s.mse(m, b)]);
    for _ in 0..steps {
        let (gm, gb) = s.grad(m, b);
        m -= lr * gm;
        b -= lr * gb;
        let loss = s.mse(m, b);
        out.extend_from_slice(&[m, b, loss]);
        if !loss.is_finite() || loss > 1e8 {
            break;
        }
    }
    out
}

/// Full-batch gradient descent for least squares with `d` features on data
/// with stride `d + 1` (features, then target). Starts at `init` (`d`
/// weights then the bias). Returns stride `d + 2`: `[w…, b, mse]` for the
/// start and every step.
#[wasm_bindgen]
pub fn lsq_gd(data: &[f64], d: usize, init: &[f64], lr: f64, steps: usize) -> Vec<f64> {
    let stride = d + 1;
    let n = (data.len() / stride).max(1) as f64;
    let mut p: Vec<f64> = init.iter().take(d + 1).cloned().collect();
    p.resize(d + 1, 0.0);
    let mse = |p: &[f64]| {
        data.chunks_exact(stride)
            .map(|r| {
                let pred: f64 = (0..d).map(|k| p[k] * r[k]).sum::<f64>() + p[d];
                (pred - r[d]).powi(2)
            })
            .sum::<f64>()
            / n
    };
    let mut out = Vec::with_capacity((steps + 1) * (d + 2));
    out.extend_from_slice(&p);
    out.push(mse(&p));
    let mut grad = vec![0.0; d + 1];
    for _ in 0..steps {
        grad.iter_mut().for_each(|g| *g = 0.0);
        for r in data.chunks_exact(stride) {
            let err = (0..d).map(|k| p[k] * r[k]).sum::<f64>() + p[d] - r[d];
            for k in 0..d {
                grad[k] += 2.0 * err * r[k] / n;
            }
            grad[d] += 2.0 * err / n;
        }
        for k in 0..=d {
            p[k] -= lr * grad[k];
        }
        let loss = mse(&p);
        out.extend_from_slice(&p);
        out.push(loss);
        if !loss.is_finite() || loss > 1e8 {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_regression;

    #[test]
    fn closed_form_matches_gradient_descent() {
        let data = make_regression(80, 0.7, -0.2, 0.15, 0.0, 9);
        let fit = linreg_fit(&data);
        let g = linreg_grad(&data, fit[0], fit[1]);
        assert!(g[0].abs() < 1e-9 && g[1].abs() < 1e-9);
        let traj = linreg_gd(&data, -2.0, 2.0, 0.3, 2000);
        let last = &traj[traj.len() - 3..];
        assert!((last[0] - fit[0]).abs() < 1e-6);
        assert!((last[1] - fit[1]).abs() < 1e-6);
    }

    #[test]
    fn plane_gradient_descent_recovers_the_plane() {
        let mut rng = crate::rng::Rng::new(5);
        let mut data = Vec::new();
        for _ in 0..200 {
            let (a, b) = (rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
            data.extend_from_slice(&[a, b, 0.6 * a - 0.4 * b + 0.2]);
        }
        let t = lsq_gd(&data, 2, &[0.0, 0.0, 0.0], 0.5, 500);
        let last = &t[t.len() - 4..];
        assert!((last[0] - 0.6).abs() < 1e-3 && (last[1] + 0.4).abs() < 1e-3 && (last[2] - 0.2).abs() < 1e-3, "{last:?}");
        assert!(last[3] < 1e-6);
    }

    #[test]
    fn mse_matches_direct_computation() {
        let data = make_regression(30, -1.2, 0.4, 0.3, 1.5, 2);
        let (m, b) = (0.3, -0.1);
        let direct: f64 =
            data.chunks(2).map(|p| (m * p[0] + b - p[1]).powi(2)).sum::<f64>() / 30.0;
        assert!((linreg_mse(&data, m, b) - direct).abs() < 1e-9);
    }
}
