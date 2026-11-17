//! Gaussian process regression in one dimension.

use std::f64::consts::PI;

use wasm_bindgen::prelude::*;

use crate::rng::Rng;

/// Kernel ids: 0 = RBF, 1 = Matérn 3/2, 2 = periodic (period 1), 3 = linear.
fn kernel(kind: u8, a: f64, b: f64, length: f64, variance: f64) -> f64 {
    let d = (a - b).abs();
    let l = length.max(1e-6);
    match kind {
        1 => {
            let r = 3f64.sqrt() * d / l;
            variance * (1.0 + r) * (-r).exp()
        }
        2 => {
            let s = (PI * d / 1.0).sin();
            variance * (-2.0 * s * s / (l * l)).exp()
        }
        3 => variance * (a * b + 0.1),
        _ => variance * (-0.5 * d * d / (l * l)).exp(),
    }
}

/// In-place Cholesky of a symmetric positive-definite matrix (lower triangle).
/// Returns false if it fails.
fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
        for k in j + 1..n {
            a[j * n + k] = 0.0;
        }
    }
    true
}

/// Cholesky with escalating jitter on the diagonal.
fn robust_cholesky(mat: &[f64], n: usize) -> Vec<f64> {
    let mut jitter = 0.0;
    loop {
        let mut a = mat.to_vec();
        for i in 0..n {
            a[i * n + i] += jitter;
        }
        if cholesky(&mut a, n) || jitter > 1.0 {
            return a;
        }
        jitter = if jitter == 0.0 { 1e-10 } else { jitter * 10.0 };
    }
}

fn forward(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in 0..n {
        let s: f64 = (0..i).map(|k| l[i * n + k] * x[k]).sum();
        x[i] = (b[i] - s) / l[i * n + i];
    }
    x
}

fn backward(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k * n + i] * x[k]).sum();
        x[i] = (b[i] - s) / l[i * n + i];
    }
    x
}

struct Posterior {
    mean: Vec<f64>,
    cov: Vec<f64>,
}

fn posterior(xs: &[f64], ys: &[f64], kind: u8, length: f64, variance: f64, noise: f64, grid: &[f64]) -> Posterior {
    let n = xs.len().min(ys.len());
    let m = grid.len();
    let mut cov = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            cov[i * m + j] = kernel(kind, grid[i], grid[j], length, variance);
        }
    }
    if n == 0 {
        return Posterior { mean: vec![0.0; m], cov };
    }
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            k[i * n + j] = kernel(kind, xs[i], xs[j], length, variance) + if i == j { noise } else { 0.0 };
        }
    }
    let l = robust_cholesky(&k, n);
    let alpha = backward(&l, n, &forward(&l, n, &ys[..n]));
    let mut mean = vec![0.0; m];
    // v_j = L⁻¹ k(X, x*_j)
    let mut v = vec![vec![0.0; n]; m];
    for j in 0..m {
        let ks: Vec<f64> = (0..n).map(|i| kernel(kind, xs[i], grid[j], length, variance)).collect();
        mean[j] = ks.iter().zip(&alpha).map(|(a, b)| a * b).sum();
        v[j] = forward(&l, n, &ks);
    }
    for i in 0..m {
        for j in i..m {
            let s: f64 = v[i].iter().zip(&v[j]).map(|(a, b)| a * b).sum();
            cov[i * m + j] -= s;
            cov[j * m + i] = cov[i * m + j];
        }
    }
    Posterior { mean, cov }
}

/// Posterior mean and standard deviation at each `grid` point: `[mean (m), std (m)]`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn gp_posterior(xs: &[f64], ys: &[f64], kind: u8, length: f64, variance: f64, noise: f64, grid: &[f64]) -> Vec<f64> {
    let p = posterior(xs, ys, kind, length, variance, noise, grid);
    let m = grid.len();
    let mut out = p.mean;
    out.extend((0..m).map(|i| p.cov[i * m + i].max(0.0).sqrt()));
    out
}

/// `count` functions drawn from the posterior on `grid`: `count` blocks of `m`.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn gp_samples(xs: &[f64], ys: &[f64], kind: u8, length: f64, variance: f64, noise: f64, grid: &[f64], count: usize, seed: u32) -> Vec<f64> {
    let p = posterior(xs, ys, kind, length, variance, noise, grid);
    let m = grid.len();
    let l = robust_cholesky(&p.cov, m);
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(count * m);
    for _ in 0..count {
        let z: Vec<f64> = (0..m).map(|_| rng.normal()).collect();
        for i in 0..m {
            let s: f64 = (0..=i).map(|k| l[i * m + k] * z[k]).sum();
            out.push(p.mean[i] + s);
        }
    }
    out
}

/// Log marginal likelihood `log p(y | X, θ)`.
#[wasm_bindgen]
pub fn gp_log_marginal(xs: &[f64], ys: &[f64], kind: u8, length: f64, variance: f64, noise: f64) -> f64 {
    let n = xs.len().min(ys.len());
    if n == 0 {
        return 0.0;
    }
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            k[i * n + j] = kernel(kind, xs[i], xs[j], length, variance) + if i == j { noise } else { 0.0 };
        }
    }
    let l = robust_cholesky(&k, n);
    let alpha = backward(&l, n, &forward(&l, n, &ys[..n]));
    let fit: f64 = ys[..n].iter().zip(&alpha).map(|(a, b)| a * b).sum();
    let logdet: f64 = (0..n).map(|i| l[i * n + i].ln()).sum::<f64>() * 2.0;
    -0.5 * fit - 0.5 * logdet - 0.5 * n as f64 * (2.0 * PI).ln()
}

/// Maximize the marginal likelihood over length scale and variance (grid
/// search in log space, then a local refinement). Returns `[length, variance]`.
#[wasm_bindgen]
pub fn gp_fit_hyper(xs: &[f64], ys: &[f64], kind: u8, noise: f64) -> Vec<f64> {
    let mut best = (f64::NEG_INFINITY, 0.3, 1.0);
    for i in 0..40 {
        let length = 10f64.powf(-1.7 + 2.2 * i as f64 / 39.0);
        for j in 0..30 {
            let variance = 10f64.powf(-1.5 + 2.5 * j as f64 / 29.0);
            let v = gp_log_marginal(xs, ys, kind, length, variance, noise);
            if v > best.0 {
                best = (v, length, variance);
            }
        }
    }
    let (mut l, mut s) = (best.1, best.2);
    let mut step = 0.15;
    for _ in 0..40 {
        let mut improved = false;
        for (dl, ds) in [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)] {
            let (nl, ns) = (l * 10f64.powf(dl), s * 10f64.powf(ds));
            let v = gp_log_marginal(xs, ys, kind, nl, ns, noise);
            if v > best.0 {
                best = (v, nl, ns);
                l = nl;
                s = ns;
                improved = true;
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    vec![l, s]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_noise_free_data_with_zero_variance_there() {
        let xs = [-0.5, 0.0, 0.6];
        let ys = [0.3, -0.2, 0.8];
        let out = gp_posterior(&xs, &ys, 0, 0.3, 1.0, 1e-8, &xs);
        for i in 0..3 {
            assert!((out[i] - ys[i]).abs() < 1e-4, "mean {} vs {}", out[i], ys[i]);
            assert!(out[3 + i] < 1e-3, "std {}", out[3 + i]);
        }
        // Far away: back to the prior.
        let far = gp_posterior(&xs, &ys, 0, 0.3, 1.0, 1e-8, &[5.0]);
        assert!(far[0].abs() < 1e-6 && (far[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn samples_pass_near_observations() {
        let xs = [0.0];
        let ys = [1.0];
        let s = gp_samples(&xs, &ys, 0, 0.3, 1.0, 1e-6, &[0.0, 0.5], 20, 3);
        for c in s.chunks(2) {
            assert!((c[0] - 1.0).abs() < 0.05);
        }
    }

    #[test]
    fn marginal_likelihood_prefers_the_true_length_scale() {
        let mut rng = Rng::new(9);
        let xs: Vec<f64> = (0..40).map(|_| rng.range(-1.0, 1.0)).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (6.0 * x).sin()).collect();
        let fit = gp_fit_hyper(&xs, &ys, 0, 1e-4);
        assert!(fit[0] > 0.1 && fit[0] < 0.5, "length {}", fit[0]);
        assert!(gp_log_marginal(&xs, &ys, 0, fit[0], fit[1], 1e-4) > gp_log_marginal(&xs, &ys, 0, 0.02, 1.0, 1e-4));
    }
}
