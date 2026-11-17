//! Gaussian naive Bayes (diagonal covariances) and its full-covariance
//! cousin, quadratic discriminant analysis, for 2D data.

use std::f64::consts::PI;

use wasm_bindgen::prelude::*;

use crate::grid::Grid;

/// Parameters per class, stride [`CLASS_STRIDE`]:
/// `[prior, mean_x, mean_y, var_x, cov_xy, var_y]`.
pub const CLASS_STRIDE: usize = 6;

/// Fit class-conditional Gaussians to the first `n_use` points of stride-3
/// data. With `full = false` (naive Bayes) the covariance is diagonal.
/// `smoothing` is added to the variances for stability.
#[wasm_bindgen]
pub fn gaussian_bayes_fit(data: &[f64], n_use: usize, classes: usize, full: bool, smoothing: f64) -> Vec<f64> {
    let classes = classes.max(1);
    let mut out = vec![0.0; classes * CLASS_STRIDE];
    let pts: Vec<&[f64]> = data.chunks_exact(3).take(n_use).collect();
    let total = pts.len().max(1) as f64;
    for c in 0..classes {
        let members: Vec<(f64, f64)> = pts.iter().filter(|p| p[2] as usize == c).map(|p| (p[0], p[1])).collect();
        let n = members.len();
        let prior = (n as f64 + 1.0) / (total + classes as f64);
        let (mut mx, mut my) = (0.0, 0.0);
        for &(x, y) in &members {
            mx += x;
            my += y;
        }
        let nf = n.max(1) as f64;
        mx /= nf;
        my /= nf;
        let (mut vx, mut vxy, mut vy) = (0.0, 0.0, 0.0);
        for &(x, y) in &members {
            vx += (x - mx).powi(2);
            vxy += (x - mx) * (y - my);
            vy += (y - my).powi(2);
        }
        // With fewer than two points, fall back to a broad default.
        let (vx, vxy, vy) = if n >= 2 { (vx / nf, vxy / nf, vy / nf) } else { (0.25, 0.0, 0.25) };
        let o = c * CLASS_STRIDE;
        out[o..o + CLASS_STRIDE].copy_from_slice(&[prior, mx, my, vx + smoothing, if full { vxy } else { 0.0 }, vy + smoothing]);
    }
    out
}

fn log_joint(p: &[f64], x: f64, y: f64) -> f64 {
    let (prior, mx, my, a, b, d) = (p[0], p[1], p[2], p[3], p[4], p[5]);
    let det = (a * d - b * b).max(1e-300);
    let (dx, dy) = (x - mx, y - my);
    let q = (d * dx * dx - 2.0 * b * dx * dy + a * dy * dy) / det;
    prior.max(1e-300).ln() - 0.5 * q - 0.5 * det.ln() - (2.0 * PI).ln()
}

/// Posterior over a grid: stride 2 `[most probable class, its posterior]`.
#[wasm_bindgen]
pub fn gaussian_bayes_grid(params: &[f64], xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Vec<f64> {
    let k = params.len() / CLASS_STRIDE;
    let g = Grid::new(xmin, xmax, ymin, ymax, nx, ny);
    let mut out = Vec::with_capacity(g.len() * 2);
    let mut buf = vec![0.0; k];
    for j in 0..g.ny {
        let y = g.y(j);
        for i in 0..g.nx {
            let x = g.x(i);
            let mut best = 0;
            for c in 0..k {
                buf[c] = log_joint(&params[c * CLASS_STRIDE..(c + 1) * CLASS_STRIDE], x, y);
                if buf[c] > buf[best] {
                    best = c;
                }
            }
            let m = buf[best];
            let z: f64 = buf.iter().map(|v| (v - m).exp()).sum();
            out.push(best as f64);
            out.push(1.0 / z);
        }
    }
    out
}

/// Prior-weighted density `p(x, y | c) · P(c)` of one class over a grid.
#[wasm_bindgen]
pub fn gaussian_bayes_density(params: &[f64], class: usize, xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Vec<f64> {
    let p = &params[class * CLASS_STRIDE..(class + 1) * CLASS_STRIDE];
    Grid::new(xmin, xmax, ymin, ymax, nx, ny).map(|x, y| log_joint(p, x, y).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_two_blobs;

    #[test]
    fn recovers_means_and_classifies() {
        let data = make_two_blobs(400, 1.4, 0.2, 3);
        let p = gaussian_bayes_fit(&data, 400, 2, false, 1e-6);
        assert!((p[1] + 0.56).abs() < 0.05 && (p[7] - 0.56).abs() < 0.05, "means {p:?}");
        assert!((p[3] - 0.04).abs() < 0.01, "variance {}", p[3]);
        let g = gaussian_bayes_grid(&p, -1.0, 1.0, -1.0, 1.0, 2, 2);
        assert_eq!(g[0], 0.0); // top-left → class 0 side? (x < 0)
        assert_eq!(g[2 * 3], 1.0); // bottom-right … class 1 is at +x
        for c in g.chunks(2) {
            assert!(c[1] >= 0.5 && c[1] <= 1.0);
        }
    }

    #[test]
    fn full_covariance_captures_correlation() {
        // Points along y = x: naive Bayes forces b = 0, QDA does not.
        let data: Vec<f64> = (0..50).flat_map(|i| {
            let t = i as f64 / 25.0 - 1.0;
            [t, t + 0.01 * (i % 3) as f64, 0.0]
        }).collect();
        let naive = gaussian_bayes_fit(&data, 50, 1, false, 0.0);
        let full = gaussian_bayes_fit(&data, 50, 1, true, 0.0);
        assert_eq!(naive[4], 0.0);
        assert!(full[4] > 0.3);
    }
}
