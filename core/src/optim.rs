//! Test surfaces and first-order optimizers (GD, momentum, RMSProp, Adam).

use wasm_bindgen::prelude::*;

use crate::grid::Grid;

// ---------------------------------------------------------------------------
// 2D surfaces
// ---------------------------------------------------------------------------

/// Surface ids:
/// 0 = tilted elongated bowl, 1 = Rosenbrock valley, 2 = Himmelblau (four
/// minima), 3 = saddle with two minima, 4 = egg crate (many local minima).
pub fn surface(id: u8, x: f64, y: f64) -> (f64, f64, f64) {
    match id {
        0 => {
            let (s, c) = 0.5f64.sin_cos();
            let u = x * c + y * s;
            let v = -x * s + y * c;
            let f = 0.5 * (u * u + 10.0 * v * v);
            (f, u * c - 10.0 * v * s, u * s + 10.0 * v * c)
        }
        1 => {
            let f = (1.0 - x).powi(2) + 10.0 * (y - x * x).powi(2);
            (f, -2.0 * (1.0 - x) - 40.0 * x * (y - x * x), 20.0 * (y - x * x))
        }
        2 => {
            let (xx, yy) = (2.0 * x, 2.0 * y);
            let a = xx * xx + yy - 11.0;
            let b = xx + yy * yy - 7.0;
            let f = (a * a + b * b) / 50.0;
            let dx = 4.0 * xx * a + 2.0 * b;
            let dy = 2.0 * a + 4.0 * yy * b;
            (f, dx / 25.0, dy / 25.0)
        }
        3 => {
            let f = 0.5 * x * x + 0.25 * y.powi(4) - 0.5 * y * y;
            (f, x, y.powi(3) - y)
        }
        _ => {
            let k = 2.6;
            let a = 0.35;
            let f = (x * x + y * y) / 6.0 - a * (k * x).cos() * (k * y).cos();
            (
                f,
                x / 3.0 + a * k * (k * x).sin() * (k * y).cos(),
                y / 3.0 + a * k * (k * x).cos() * (k * y).sin(),
            )
        }
    }
}

/// `[f, ∂f/∂x, ∂f/∂y]` at a point.
#[wasm_bindgen]
pub fn surface_eval(id: u8, x: f64, y: f64) -> Vec<f64> {
    let (f, gx, gy) = surface(id, x, y);
    vec![f, gx, gy]
}

/// Surface values over a grid.
#[wasm_bindgen]
pub fn surface_grid(id: u8, xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Vec<f64> {
    Grid::new(xmin, xmax, ymin, ymax, nx, ny).map(|x, y| surface(id, x, y).0)
}

/// Optimizer ids: 0 = gradient descent, 1 = momentum (heavy ball),
/// 2 = RMSProp, 3 = Adam.
///
/// Returns stride-3 `[x, y, f]` for the start point and every step. Stops
/// early if the iterate escapes far away or becomes non-finite.
#[wasm_bindgen]
pub fn optimize(surface_id: u8, optimizer: u8, x0: f64, y0: f64, lr: f64, steps: usize) -> Vec<f64> {
    let (beta, rho, beta1, beta2, eps) = (0.9, 0.9, 0.9, 0.999, 1e-8);
    let (mut x, mut y) = (x0, y0);
    let (mut vx, mut vy) = (0.0, 0.0); // momentum / first moment
    let (mut sx, mut sy) = (0.0, 0.0); // second moment
    let mut out = Vec::with_capacity((steps + 1) * 3);
    out.extend_from_slice(&[x, y, surface(surface_id, x, y).0]);
    for t in 1..=steps {
        let (_, gx, gy) = surface(surface_id, x, y);
        match optimizer {
            0 => {
                x -= lr * gx;
                y -= lr * gy;
            }
            1 => {
                vx = beta * vx + gx;
                vy = beta * vy + gy;
                x -= lr * vx;
                y -= lr * vy;
            }
            2 => {
                sx = rho * sx + (1.0 - rho) * gx * gx;
                sy = rho * sy + (1.0 - rho) * gy * gy;
                x -= lr * gx / (sx.sqrt() + eps);
                y -= lr * gy / (sy.sqrt() + eps);
            }
            _ => {
                vx = beta1 * vx + (1.0 - beta1) * gx;
                vy = beta1 * vy + (1.0 - beta1) * gy;
                sx = beta2 * sx + (1.0 - beta2) * gx * gx;
                sy = beta2 * sy + (1.0 - beta2) * gy * gy;
                let c1 = 1.0 - beta1.powi(t as i32);
                let c2 = 1.0 - beta2.powi(t as i32);
                x -= lr * (vx / c1) / ((sx / c2).sqrt() + eps);
                y -= lr * (vy / c1) / ((sy / c2).sqrt() + eps);
            }
        }
        let f = surface(surface_id, x, y).0;
        out.extend_from_slice(&[x, y, f]);
        if !(x.is_finite() && y.is_finite()) || x.abs() > 1e3 || y.abs() > 1e3 {
            break;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 1D functions
// ---------------------------------------------------------------------------

/// 1D function ids: 0 = parabola `x²`, 1 = lopsided double well,
/// 2 = plateau `1 − 1/(1+x²)` (vanishing gradients far from the minimum).
pub fn func1d(id: u8, x: f64) -> (f64, f64) {
    match id {
        0 => (x * x, 2.0 * x),
        1 => (
            0.5 * x.powi(4) - x * x + 0.3 * x + 0.6,
            2.0 * x.powi(3) - 2.0 * x + 0.3,
        ),
        _ => {
            let d = 1.0 + x * x;
            (1.0 - 1.0 / d, 2.0 * x / (d * d))
        }
    }
}

/// `[f, f']` at `x`.
#[wasm_bindgen]
pub fn func1d_eval(id: u8, x: f64) -> Vec<f64> {
    let (f, d) = func1d(id, x);
    vec![f, d]
}

/// `n` samples of `f` on `[xmin, xmax]` (endpoints included), stride 2.
#[wasm_bindgen]
pub fn func1d_curve(id: u8, xmin: f64, xmax: f64, n: usize) -> Vec<f64> {
    let n = n.max(2);
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let x = xmin + (xmax - xmin) * i as f64 / (n - 1) as f64;
        out.push(x);
        out.push(func1d(id, x).0);
    }
    out
}

/// Plain gradient descent in 1D. Stride 2 `[x, f]`, start included.
#[wasm_bindgen]
pub fn gd1d(id: u8, x0: f64, lr: f64, steps: usize) -> Vec<f64> {
    let mut x = x0;
    let mut out = Vec::with_capacity((steps + 1) * 2);
    out.extend_from_slice(&[x, func1d(id, x).0]);
    for _ in 0..steps {
        x -= lr * func1d(id, x).1;
        out.extend_from_slice(&[x, func1d(id, x).0]);
        if !x.is_finite() || x.abs() > 1e3 {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric_grad(id: u8, x: f64, y: f64) -> (f64, f64) {
        let h = 1e-6;
        let fx = (surface(id, x + h, y).0 - surface(id, x - h, y).0) / (2.0 * h);
        let fy = (surface(id, x, y + h).0 - surface(id, x, y - h).0) / (2.0 * h);
        (fx, fy)
    }

    #[test]
    fn analytic_gradients_match_numeric() {
        for id in 0..5u8 {
            for &(x, y) in &[(0.3, -0.7), (-1.1, 0.4), (1.7, 1.2)] {
                let (_, gx, gy) = surface(id, x, y);
                let (nx, ny) = numeric_grad(id, x, y);
                assert!((gx - nx).abs() < 1e-4 * (1.0 + nx.abs()), "id {id} gx {gx} vs {nx}");
                assert!((gy - ny).abs() < 1e-4 * (1.0 + ny.abs()), "id {id} gy {gy} vs {ny}");
            }
        }
        for id in 0..3u8 {
            let x = 0.77;
            let h = 1e-6;
            let num = (func1d(id, x + h).0 - func1d(id, x - h).0) / (2.0 * h);
            assert!((func1d(id, x).1 - num).abs() < 1e-5);
        }
    }

    #[test]
    fn optimizers_reach_bowl_minimum() {
        for (opt, lr) in [(0u8, 0.15), (1, 0.03), (2, 0.005), (3, 0.1)] {
            let t = optimize(0, opt, 1.8, -1.2, lr, 2000);
            let f = t[t.len() - 1];
            assert!(f < 1e-3, "optimizer {opt} ended at f = {f}");
        }
    }

    #[test]
    fn gd_diverges_with_large_step_on_parabola() {
        let t = gd1d(0, 1.0, 1.1, 500);
        assert!(t[t.len() - 2].abs() > 100.0);
        let t = gd1d(0, 1.0, 0.4, 200);
        assert!(t[t.len() - 2].abs() < 1e-6);
    }
}
