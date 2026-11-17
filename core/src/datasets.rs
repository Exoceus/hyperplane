//! Synthetic dataset generators. All live roughly inside `[-1, 1]²` so that
//! every lesson can share the same scale and sensible learning rates.

use std::f64::consts::PI;

use wasm_bindgen::prelude::*;

use crate::rng::Rng;

fn push(out: &mut Vec<f64>, x: f64, y: f64, class: usize) {
    out.push(x);
    out.push(y);
    out.push(class as f64);
}

/// Points uniformly scattered in `[-1, 1]²`, labelled by which side of the
/// line `w1·x + w2·y + b = 0` they fall on (class 1 where it is positive).
///
/// * `margin` – points closer than this (Euclidean distance) to the line are
///   rejected, guaranteeing a gap of `2·margin` between the classes.
/// * `flip` – probability of flipping each label (makes data non-separable).
///
/// Returns stride-3 `[x, y, class]`.
#[wasm_bindgen]
pub fn make_linear(n: usize, w1: f64, w2: f64, b: f64, margin: f64, flip: f64, seed: u32) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let norm = (w1 * w1 + w2 * w2).sqrt().max(1e-12);
    let mut out = Vec::with_capacity(n * 3);
    let mut attempts = 0usize;
    while out.len() < n * 3 && attempts < n * 2000 {
        attempts += 1;
        let x = rng.range(-1.0, 1.0);
        let y = rng.range(-1.0, 1.0);
        let s = (w1 * x + w2 * y + b) / norm;
        if s.abs() < margin {
            continue;
        }
        let mut class = if s >= 0.0 { 1 } else { 0 };
        if rng.uniform() < flip {
            class = 1 - class;
        }
        push(&mut out, x, y, class);
    }
    out
}

/// `k` isotropic Gaussian blobs with standard deviation `spread`, with
/// centers spread out over the plane. Class = blob index. Stride 3.
#[wasm_bindgen]
pub fn make_blobs(n: usize, k: usize, spread: f64, seed: u32) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let k = k.max(1);
    // Place centers with a best-candidate sampler so they don't overlap.
    let mut centers: Vec<(f64, f64)> = Vec::with_capacity(k);
    for _ in 0..k {
        let mut best = (0.0, 0.0);
        let mut best_d = -1.0;
        for _ in 0..30 {
            let c = (rng.range(-0.65, 0.65), rng.range(-0.65, 0.65));
            let d = centers
                .iter()
                .map(|p| (p.0 - c.0).powi(2) + (p.1 - c.1).powi(2))
                .fold(f64::INFINITY, f64::min);
            if d > best_d {
                best_d = d;
                best = c;
            }
        }
        centers.push(best);
    }
    let mut out = Vec::with_capacity(n * 3);
    for idx in 0..n {
        let c = idx % k;
        let (cx, cy) = centers[c];
        push(&mut out, cx + spread * rng.normal(), cy + spread * rng.normal(), c);
    }
    out
}

/// Two Gaussian blobs at fixed, mirrored positions; `separation` controls how
/// far apart they are (0 = on top of each other). Stride 3.
#[wasm_bindgen]
pub fn make_two_blobs(n: usize, separation: f64, spread: f64, seed: u32) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let (dx, dy) = (0.8 * separation, 0.6 * separation);
    let mut out = Vec::with_capacity(n * 3);
    for idx in 0..n {
        let c = idx % 2;
        let sign = if c == 1 { 1.0 } else { -1.0 };
        push(
            &mut out,
            sign * dx * 0.5 + spread * rng.normal(),
            sign * dy * 0.5 + spread * rng.normal(),
            c,
        );
    }
    out
}

/// Classic toy shapes for non-linear classification. Stride 3.
/// `kind`: `"circles" | "moons" | "xor" | "spiral" | "blobs"`.
#[wasm_bindgen]
pub fn make_shape(kind: &str, n: usize, noise: f64, seed: u32) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n * 3);
    match kind {
        "circles" => {
            for idx in 0..n {
                let c = idx % 2;
                let r = if c == 1 { 0.32 } else { 0.78 };
                let t = rng.range(0.0, 2.0 * PI);
                let rr = r + noise * 0.5 * rng.normal();
                push(&mut out, rr * t.cos(), rr * t.sin(), c);
            }
        }
        "moons" => {
            for idx in 0..n {
                let c = idx % 2;
                let t = rng.range(0.0, PI);
                let (x, y) = if c == 0 {
                    (t.cos(), t.sin())
                } else {
                    (1.0 - t.cos(), 0.5 - t.sin())
                };
                let x = (x - 0.5) * 0.62 + noise * 0.4 * rng.normal();
                let y = (y - 0.25) * 0.62 + noise * 0.4 * rng.normal();
                push(&mut out, x, y, c);
            }
        }
        "xor" => {
            for _ in 0..n {
                let mut x = rng.range(-1.0, 1.0);
                let mut y = rng.range(-1.0, 1.0);
                x += 0.06 * x.signum();
                y += 0.06 * y.signum();
                let c = if x * y > 0.0 { 1 } else { 0 };
                let x = (x * 0.9 + noise * 0.4 * rng.normal()).clamp(-1.05, 1.05);
                let y = (y * 0.9 + noise * 0.4 * rng.normal()).clamp(-1.05, 1.05);
                push(&mut out, x, y, c);
            }
        }
        "spiral" => {
            for idx in 0..n {
                let c = idx % 2;
                let t = rng.uniform().sqrt();
                let r = 0.08 + 0.85 * t;
                let a = 3.0 * PI * t + c as f64 * PI;
                let x = r * a.cos() + noise * 0.25 * rng.normal();
                let y = r * a.sin() + noise * 0.25 * rng.normal();
                push(&mut out, x, y, c);
            }
        }
        _ => return make_two_blobs(n, 1.0, 0.18 + noise * 0.3, seed),
    }
    out
}

/// Noisy line `y = slope·x + intercept + ε`, with `x` uniform in
/// `[shift - 1, shift + 1]`. Stride 2 `[x, y]`.
#[wasm_bindgen]
pub fn make_regression(n: usize, slope: f64, intercept: f64, noise: f64, shift: f64, seed: u32) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let x = shift + rng.range(-1.0, 1.0);
        out.push(x);
        out.push(slope * x + intercept + noise * rng.normal());
    }
    out
}

/// Correlated Gaussian cloud centered at `(cx, cy)` with standard deviations
/// `sx`, `sy` along axes rotated by `angle` radians. Stride 2.
#[wasm_bindgen]
pub fn make_gaussian(n: usize, cx: f64, cy: f64, sx: f64, sy: f64, angle: f64, seed: u32) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let (s, c) = angle.sin_cos();
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let u = sx * rng.normal();
        let v = sy * rng.normal();
        out.push(cx + c * u - s * v);
        out.push(cy + s * u + c * v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_respects_margin_and_side() {
        let d = make_linear(200, 0.6, -0.8, 0.1, 0.1, 0.0, 3);
        assert_eq!(d.len(), 600);
        for p in d.chunks(3) {
            let s = 0.6 * p[0] - 0.8 * p[1] + 0.1;
            assert!(s.abs() >= 0.1 - 1e-9);
            assert_eq!(p[2] == 1.0, s >= 0.0);
        }
    }

    #[test]
    fn shapes_have_requested_size_and_two_classes() {
        for kind in ["circles", "moons", "xor", "spiral", "blobs"] {
            let d = make_shape(kind, 101, 0.1, 1);
            assert_eq!(d.len(), 303, "{kind}");
            assert!(d.chunks(3).any(|p| p[2] == 0.0));
            assert!(d.chunks(3).any(|p| p[2] == 1.0));
            assert!(d.chunks(3).all(|p| p[0].abs() < 2.0 && p[1].abs() < 2.0));
        }
    }
}
