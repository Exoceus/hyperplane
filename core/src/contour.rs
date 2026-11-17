//! Marching squares: turn a grid of values into iso-lines.

use wasm_bindgen::prelude::*;

use crate::grid::Grid;

/// Contour segments of `values` (a grid laid out as described in
/// [`Grid`]) at each requested level.
///
/// Returns stride-5 `[x1, y1, x2, y2, level_index]` in data coordinates.
#[wasm_bindgen]
pub fn contour_segments(
    values: &[f64],
    nx: usize,
    ny: usize,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    levels: &[f64],
) -> Vec<f64> {
    let g = Grid::new(xmin, xmax, ymin, ymax, nx, ny);
    let mut out = Vec::new();
    if nx < 2 || ny < 2 || values.len() < nx * ny {
        return out;
    }
    let v = |i: usize, j: usize| values[j * nx + i];

    for (li, &level) in levels.iter().enumerate() {
        let lf = li as f64;
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                // Corners: a = top-left, b = top-right, c = bottom-right, d = bottom-left.
                let (a, b, c, d) = (v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1));
                if !(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite()) {
                    continue;
                }
                let case = ((a > level) as u8) << 3
                    | ((b > level) as u8) << 2
                    | ((c > level) as u8) << 1
                    | (d > level) as u8;
                if case == 0 || case == 15 {
                    continue;
                }
                let (x0, x1, y0, y1) = (g.x(i), g.x(i + 1), g.y(j), g.y(j + 1));
                let lerp = |p: f64, q: f64| {
                    let den = q - p;
                    if den.abs() < 1e-300 { 0.5 } else { ((level - p) / den).clamp(0.0, 1.0) }
                };
                let top = (x0 + (x1 - x0) * lerp(a, b), y0);
                let right = (x1, y0 + (y1 - y0) * lerp(b, c));
                let bottom = (x0 + (x1 - x0) * lerp(d, c), y1);
                let left = (x0, y0 + (y1 - y0) * lerp(a, d));
                let mut seg = |p: (f64, f64), q: (f64, f64)| {
                    out.extend_from_slice(&[p.0, p.1, q.0, q.1, lf]);
                };
                let center_above = (a + b + c + d) * 0.25 > level;
                match case {
                    1 | 14 => seg(left, bottom),
                    2 | 13 => seg(bottom, right),
                    3 | 12 => seg(left, right),
                    4 | 11 => seg(top, right),
                    6 | 9 => seg(top, bottom),
                    7 | 8 => seg(left, top),
                    5 => {
                        if center_above {
                            seg(left, top);
                            seg(bottom, right);
                        } else {
                            seg(top, right);
                            seg(left, bottom);
                        }
                    }
                    10 => {
                        if center_above {
                            seg(top, right);
                            seg(left, bottom);
                        } else {
                            seg(left, top);
                            seg(bottom, right);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_contour_points_lie_on_circle() {
        let g = Grid::new(-1.0, 1.0, -1.0, 1.0, 64, 64);
        let vals = g.map(|x, y| x * x + y * y);
        let segs = contour_segments(&vals, 64, 64, -1.0, 1.0, -1.0, 1.0, &[0.25]);
        assert!(segs.len() > 5 * 40);
        for s in segs.chunks(5) {
            for (x, y) in [(s[0], s[1]), (s[2], s[3])] {
                let r = (x * x + y * y).sqrt();
                assert!((r - 0.5).abs() < 0.02, "r = {r}");
            }
        }
    }
}
