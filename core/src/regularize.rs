//! Polynomial regression with ridge (L2) and lasso (L1) regularization.
//!
//! Features are Legendre polynomials `P₀ … P_d` of `x ∈ [-1, 1]`, which are
//! far better conditioned than raw powers, so high degrees stay numerically
//! sane while still overfitting exactly the way raw polynomials do.

use wasm_bindgen::prelude::*;

use crate::rng::Rng;

fn legendre(x: f64, degree: usize, out: &mut Vec<f64>) {
    out.clear();
    out.push(1.0);
    if degree >= 1 {
        out.push(x);
    }
    for k in 1..degree {
        let next = ((2 * k + 1) as f64 * x * out[k] - k as f64 * out[k - 1]) / (k + 1) as f64;
        out.push(next);
    }
}

fn design(data: &[f64], degree: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rows = Vec::with_capacity(data.len() / 2);
    let mut ys = Vec::with_capacity(data.len() / 2);
    let mut buf = Vec::new();
    for p in data.chunks_exact(2) {
        legendre(p[0], degree, &mut buf);
        rows.push(buf.clone());
        ys.push(p[1]);
    }
    (rows, ys)
}

/// Solve `A x = b` (dense, small) by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap_or(col);
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        if d.abs() < 1e-14 {
            continue;
        }
        for r in col + 1..n {
            let f = a[r][col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|c| a[r][c] * x[c]).sum();
        x[r] = if a[r][r].abs() < 1e-14 { 0.0 } else { (b[r] - s) / a[r][r] };
    }
    x
}

fn ridge(rows: &[Vec<f64>], ys: &[f64], lambda: f64) -> Vec<f64> {
    let p = rows.first().map_or(1, |r| r.len());
    let n = rows.len().max(1) as f64;
    let mut a = vec![vec![0.0; p]; p];
    let mut b = vec![0.0; p];
    for (r, &y) in rows.iter().zip(ys) {
        for i in 0..p {
            b[i] += r[i] * y / n;
            for j in 0..p {
                a[i][j] += r[i] * r[j] / n;
            }
        }
    }
    // Tiny jitter keeps the system solvable when there are more features than points.
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += if i == 0 { 1e-12 } else { lambda + 1e-10 };
    }
    solve(a, b)
}

fn soft(v: f64, t: f64) -> f64 {
    if v > t {
        v - t
    } else if v < -t {
        v + t
    } else {
        0.0
    }
}

fn lasso(rows: &[Vec<f64>], ys: &[f64], lambda: f64) -> Vec<f64> {
    let p = rows.first().map_or(1, |r| r.len());
    let n = rows.len().max(1) as f64;
    let mut w = vec![0.0; p];
    let mut resid: Vec<f64> = ys.to_vec();
    let z: Vec<f64> = (0..p).map(|k| rows.iter().map(|r| r[k] * r[k]).sum::<f64>() / n).collect();
    for _ in 0..2000 {
        let mut change: f64 = 0.0;
        for k in 0..p {
            if z[k] < 1e-14 {
                continue;
            }
            let rho = rows.iter().zip(&resid).map(|(r, e)| r[k] * e).sum::<f64>() / n + z[k] * w[k];
            let next = if k == 0 { rho / z[k] } else { soft(rho, lambda) / z[k] };
            let delta = next - w[k];
            if delta != 0.0 {
                for (r, e) in rows.iter().zip(resid.iter_mut()) {
                    *e -= r[k] * delta;
                }
                w[k] = next;
                change = change.max(delta.abs());
            }
        }
        if change < 1e-9 {
            break;
        }
    }
    w
}

fn fit(data: &[f64], degree: usize, lambda: f64, penalty: u8) -> Vec<f64> {
    let (rows, ys) = design(data, degree);
    if rows.is_empty() {
        return vec![0.0; degree + 1];
    }
    if penalty == 1 && lambda > 0.0 { lasso(&rows, &ys, lambda) } else { ridge(&rows, &ys, lambda) }
}

fn predict(coefs: &[f64], x: f64, buf: &mut Vec<f64>) -> f64 {
    legendre(x, coefs.len() - 1, buf);
    coefs.iter().zip(buf.iter()).map(|(c, f)| c * f).sum()
}

fn mse(coefs: &[f64], data: &[f64]) -> f64 {
    let mut buf = Vec::new();
    let n = (data.len() / 2).max(1) as f64;
    data.chunks_exact(2).map(|p| (predict(coefs, p[0], &mut buf) - p[1]).powi(2)).sum::<f64>() / n
}

/// Noisy samples of a smooth curve, `x ∈ [-1, 1]`. Stride 2.
#[wasm_bindgen]
pub fn make_curve(n: usize, noise: f64, seed: u32) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let x = rng.range(-1.0, 1.0);
        out.push(x);
        out.push(0.65 * (2.8 * x).sin() + 0.25 * x + noise * rng.normal());
    }
    out
}

/// Fit a degree-`degree` polynomial. `penalty`: 0 = ridge (L2), 1 = lasso (L1).
/// Minimizes `mean squared error + λ·penalty` (lasso uses ½·MSE, the usual convention).
/// Returns Legendre coefficients `c₀ … c_d` (c₀ is not penalized).
#[wasm_bindgen]
pub fn poly_fit(data: &[f64], degree: usize, lambda: f64, penalty: u8) -> Vec<f64> {
    fit(data, degree, lambda, penalty)
}

/// Evaluate a fitted polynomial at `n` evenly spaced points on `[xmin, xmax]`. Stride 2.
#[wasm_bindgen]
pub fn poly_curve(coefs: &[f64], xmin: f64, xmax: f64, n: usize) -> Vec<f64> {
    let mut buf = Vec::new();
    let n = n.max(2);
    (0..n)
        .flat_map(|i| {
            let x = xmin + (xmax - xmin) * i as f64 / (n - 1) as f64;
            [x, if coefs.is_empty() { 0.0 } else { predict(coefs, x, &mut buf) }]
        })
        .collect()
}

/// Train and test MSE for degrees `0 … max_degree`. Stride 2.
#[wasm_bindgen]
pub fn poly_error_curve(train: &[f64], test: &[f64], max_degree: usize, lambda: f64, penalty: u8) -> Vec<f64> {
    (0..=max_degree)
        .flat_map(|d| {
            let c = fit(train, d, lambda, penalty);
            [mse(&c, train), mse(&c, test)]
        })
        .collect()
}

/// Coefficients along a path of λ values: `lambdas.len()` blocks of `degree + 1`.
#[wasm_bindgen]
pub fn poly_lambda_path(train: &[f64], degree: usize, lambdas: &[f64], penalty: u8) -> Vec<f64> {
    lambdas.iter().flat_map(|&l| fit(train, degree, l, penalty)).collect()
}

/// Minimize the quadratic `(w − c)ᵀ A (w − c)` subject to `‖w‖ ≤ t`, where the
/// norm is L2 (`norm = 2`) or L1 (`norm = 1`). Returns `[w₁, w₂]`.
#[wasm_bindgen]
pub fn constrained_quadratic(a11: f64, a12: f64, a22: f64, cx: f64, cy: f64, t: f64, norm: u8) -> Vec<f64> {
    let f = |x: f64, y: f64| {
        let (dx, dy) = (x - cx, y - cy);
        a11 * dx * dx + 2.0 * a12 * dx * dy + a22 * dy * dy
    };
    let inside = if norm == 1 { cx.abs() + cy.abs() <= t } else { cx.hypot(cy) <= t };
    if inside {
        return vec![cx, cy];
    }
    if norm == 1 {
        // The optimum lies on an edge (or vertex) of the diamond.
        let verts = [(t, 0.0), (0.0, t), (-t, 0.0), (0.0, -t)];
        let mut best = (f64::INFINITY, 0.0, 0.0);
        for i in 0..4 {
            let (p, q) = (verts[i], verts[(i + 1) % 4]);
            let (ux, uy) = (q.0 - p.0, q.1 - p.1);
            // f(p + s u) is quadratic in s: minimize on [0, 1].
            let (dx, dy) = (p.0 - cx, p.1 - cy);
            let qa = a11 * ux * ux + 2.0 * a12 * ux * uy + a22 * uy * uy;
            let qb = 2.0 * (a11 * dx * ux + a12 * (dx * uy + dy * ux) + a22 * dy * uy);
            let s = if qa > 1e-15 { (-qb / (2.0 * qa)).clamp(0.0, 1.0) } else { 0.0 };
            let (x, y) = (p.0 + s * ux, p.1 + s * uy);
            let v = f(x, y);
            if v < best.0 {
                best = (v, x, y);
            }
        }
        vec![best.1, best.2]
    } else {
        // Search the circle, then refine with golden-section search.
        let g = |th: f64| f(t * th.cos(), t * th.sin());
        let steps = 720;
        let mut best = 0.0;
        for i in 0..steps {
            let th = i as f64 / steps as f64 * std::f64::consts::TAU;
            if g(th) < g(best) {
                best = th;
            }
        }
        let h = std::f64::consts::TAU / steps as f64;
        let (mut lo, mut hi) = (best - h, best + h);
        let phi = 0.618_033_988_75;
        for _ in 0..60 {
            let m1 = hi - phi * (hi - lo);
            let m2 = lo + phi * (hi - lo);
            if g(m1) < g(m2) {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        let th = 0.5 * (lo + hi);
        vec![t * th.cos(), t * th.sin()]
    }
}

/// Train a degree-`degree` polynomial by gradient descent with momentum and
/// record `frames` snapshots on a geometric schedule of steps (1 … max_steps,
/// plus step 0). Stride `degree + 4`: `[step, train_mse, test_mse, c₀ … c_d]`.
/// Early on only the smooth, high-variance directions are learned; the
/// wiggly ones come much later, which is why stopping early regularizes.
#[wasm_bindgen]
pub fn poly_gd_trace(train: &[f64], test: &[f64], degree: usize, lr: f64, max_steps: usize, frames: usize) -> Vec<f64> {
    let (rows, ys) = design(train, degree);
    let p = degree + 1;
    let n = rows.len().max(1) as f64;
    let mut c = vec![0.0; p];
    let mut v = vec![0.0; p];
    let mut out = Vec::new();
    let frames = frames.max(2);
    let schedule: Vec<usize> = (0..frames)
        .map(|f| if f == 0 { 0 } else { ((max_steps as f64).powf(f as f64 / (frames - 1) as f64)).round() as usize })
        .collect();
    let mut next = 0;
    let mut record = |step: usize, c: &[f64], out: &mut Vec<f64>| {
        out.push(step as f64);
        out.push(mse(c, train));
        out.push(mse(c, test));
        out.extend_from_slice(c);
    };
    let mut grad = vec![0.0; p];
    for step in 0..=max_steps {
        while next < schedule.len() && schedule[next] == step {
            record(step, &c, &mut out);
            next += 1;
        }
        if next >= schedule.len() {
            break;
        }
        grad.iter_mut().for_each(|g| *g = 0.0);
        for (r, &y) in rows.iter().zip(&ys) {
            let err: f64 = r.iter().zip(&c).map(|(a, b)| a * b).sum::<f64>() - y;
            for k in 0..p {
                grad[k] += 2.0 * err * r[k] / n;
            }
        }
        for k in 0..p {
            v[k] = 0.9 * v[k] - lr * grad[k];
            c[k] += v[k];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_degree_overfits_and_ridge_helps() {
        let train = make_curve(14, 0.15, 1);
        let test = make_curve(400, 0.15, 2);
        let curve = poly_error_curve(&train, &test, 13, 0.0, 0);
        let (train_hi, test_hi) = (curve[2 * 13], curve[2 * 13 + 1]);
        let best_test = curve.chunks(2).map(|c| c[1]).fold(f64::INFINITY, f64::min);
        assert!(train_hi < 1e-6, "degree 13 through 14 points should interpolate: {train_hi}");
        assert!(test_hi > 5.0 * best_test, "high degree should generalize badly");
        let reg = poly_fit(&train, 13, 1e-3, 0);
        assert!(mse(&reg, &test) < test_hi);
    }

    #[test]
    fn gradient_descent_overfits_late_so_early_stopping_helps() {
        let train = make_curve(14, 0.18, 7);
        let test = make_curve(300, 0.18, 108);
        let t = poly_gd_trace(&train, &test, 12, 0.05, 200_000, 60);
        let stride = 12 + 4;
        let frames: Vec<&[f64]> = t.chunks(stride).collect();
        let best = frames.iter().map(|f| f[2]).fold(f64::INFINITY, f64::min);
        let last = frames.last().unwrap();
        println!("best test {best}, final test {}, final train {}", last[2], last[1]);
        assert!(last[1] < frames[5][1], "training error keeps falling");
        assert!(last[2] > 1.3 * best, "test error should rise after its minimum");
    }

    #[test]
    fn lasso_produces_exact_zeros() {
        let train = make_curve(60, 0.1, 3);
        let c = poly_fit(&train, 12, 0.02, 1);
        let zeros = c[1..].iter().filter(|v| **v == 0.0).count();
        assert!(zeros >= 4, "expected sparse coefficients, got {c:?}");
        let r = poly_fit(&train, 12, 0.02, 0);
        assert!(r[1..].iter().all(|v| *v != 0.0));
    }

    #[test]
    fn l1_constraint_lands_on_a_vertex() {
        // Optimum far along the x axis with mild correlation: the L1 solution is the vertex (t, 0).
        let w = constrained_quadratic(1.0, 0.2, 1.0, 2.0, 0.4, 1.0, 1);
        assert!((w[0] - 1.0).abs() < 1e-9 && w[1].abs() < 1e-9, "{w:?}");
        let w2 = constrained_quadratic(1.0, 0.2, 1.0, 2.0, 0.4, 1.0, 2);
        assert!((w2[0].hypot(w2[1]) - 1.0).abs() < 1e-6 && w2[1] > 0.0);
    }
}
