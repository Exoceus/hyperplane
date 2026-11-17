//! k-means clustering (Lloyd's algorithm), recorded phase by phase.

use wasm_bindgen::prelude::*;

use crate::grid::Grid;
use crate::rng::Rng;

#[inline]
fn d2(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    (ax - bx).powi(2) + (ay - by).powi(2)
}

fn nearest(x: f64, y: f64, centroids: &[f64]) -> (usize, f64) {
    let mut best = (0usize, f64::INFINITY);
    for (c, p) in centroids.chunks_exact(2).enumerate() {
        let d = d2(x, y, p[0], p[1]);
        if d < best.1 {
            best = (c, d);
        }
    }
    best
}

fn coords(points: &[f64], stride: usize) -> Vec<(f64, f64)> {
    points.chunks_exact(stride.max(2)).map(|p| (p[0], p[1])).collect()
}

/// Initial centroids (stride 2, `k` of them).
///
/// `method`: 0 = Forgy (k distinct random data points), 1 = k-means++,
/// 2 = uniformly random positions in the data's bounding box.
#[wasm_bindgen]
pub fn kmeans_init(points: &[f64], stride: usize, k: usize, method: u8, seed: u32) -> Vec<f64> {
    let pts = coords(points, stride);
    let mut rng = Rng::new(seed);
    let n = pts.len();
    let mut out = Vec::with_capacity(k * 2);
    if n == 0 {
        for _ in 0..k {
            out.push(rng.range(-0.8, 0.8));
            out.push(rng.range(-0.8, 0.8));
        }
        return out;
    }
    match method {
        1 => {
            let first = pts[rng.below(n)];
            out.extend_from_slice(&[first.0, first.1]);
            let mut dist: Vec<f64> = pts.iter().map(|p| d2(p.0, p.1, first.0, first.1)).collect();
            while out.len() < k * 2 {
                let total: f64 = dist.iter().sum();
                let idx = if total <= 0.0 {
                    rng.below(n)
                } else {
                    let mut r = rng.uniform() * total;
                    let mut chosen = n - 1;
                    for (i, d) in dist.iter().enumerate() {
                        r -= d;
                        if r <= 0.0 {
                            chosen = i;
                            break;
                        }
                    }
                    chosen
                };
                let c = pts[idx];
                out.extend_from_slice(&[c.0, c.1]);
                for (i, p) in pts.iter().enumerate() {
                    dist[i] = dist[i].min(d2(p.0, p.1, c.0, c.1));
                }
            }
        }
        2 => {
            let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
            for p in &pts {
                x0 = x0.min(p.0);
                x1 = x1.max(p.0);
                y0 = y0.min(p.1);
                y1 = y1.max(p.1);
            }
            for _ in 0..k {
                out.push(rng.range(x0, x1));
                out.push(rng.range(y0, y1));
            }
        }
        _ => {
            let mut idx: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut idx);
            for c in 0..k {
                let p = pts[idx[c % n]];
                out.extend_from_slice(&[p.0, p.1]);
            }
        }
    }
    out
}

/// A complete run of Lloyd's algorithm.
///
/// With `T = iterations`:
/// - `centroids`: `(T + 1) · k · 2` values, `c₀ … c_T` where `c_{t+1}` is the
///   mean of the points assigned in step `t`;
/// - `assignments`: `T · n` cluster indices, `a₀ … a_{T−1}` where `a_t` assigns
///   each point to its nearest centroid in `c_t`;
/// - `inertia`: `2T` values interleaving `J(a_t, c_t)` and `J(a_t, c_{t+1})`
///   (the objective after each assign and each update phase).
#[wasm_bindgen]
pub struct KMeansRun {
    centroids: Vec<f64>,
    assignments: Vec<f64>,
    inertia: Vec<f64>,
    iterations: usize,
    converged: bool,
}

#[wasm_bindgen]
impl KMeansRun {
    #[wasm_bindgen(getter)]
    pub fn centroids(&self) -> Vec<f64> {
        self.centroids.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn assignments(&self) -> Vec<f64> {
        self.assignments.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn inertia(&self) -> Vec<f64> {
        self.inertia.clone()
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

fn lloyd(pts: &[(f64, f64)], init: &[f64], max_iter: usize) -> KMeansRun {
    let k = init.len() / 2;
    let n = pts.len();
    let mut c = init[..k * 2].to_vec();
    let mut centroids = c.clone();
    let mut assignments = Vec::new();
    let mut inertia = Vec::new();
    let mut prev: Option<Vec<usize>> = None;
    let mut converged = false;
    let mut iterations = 0;

    for _ in 0..max_iter.max(1) {
        // Assign.
        let mut a = vec![0usize; n];
        let mut j_assign = 0.0;
        for (i, p) in pts.iter().enumerate() {
            let (best, d) = nearest(p.0, p.1, &c);
            a[i] = best;
            j_assign += d;
        }
        // Update.
        let mut sums = vec![(0.0, 0.0, 0usize); k];
        for (i, p) in pts.iter().enumerate() {
            let s = &mut sums[a[i]];
            s.0 += p.0;
            s.1 += p.1;
            s.2 += 1;
        }
        for (ci, s) in sums.iter().enumerate() {
            if s.2 > 0 {
                c[2 * ci] = s.0 / s.2 as f64;
                c[2 * ci + 1] = s.1 / s.2 as f64;
            }
        }
        let j_update: f64 = pts.iter().enumerate().map(|(i, p)| d2(p.0, p.1, c[2 * a[i]], c[2 * a[i] + 1])).sum();

        assignments.extend(a.iter().map(|&x| x as f64));
        centroids.extend_from_slice(&c);
        inertia.push(j_assign);
        inertia.push(j_update);
        iterations += 1;

        if prev.as_ref() == Some(&a) {
            converged = true;
            break;
        }
        prev = Some(a);
    }
    KMeansRun { centroids, assignments, inertia, iterations, converged }
}

/// Run Lloyd's algorithm from the given initial centroids (stride 2).
#[wasm_bindgen]
pub fn kmeans_run(points: &[f64], stride: usize, init: &[f64], max_iter: usize) -> KMeansRun {
    lloyd(&coords(points, stride), init, max_iter)
}

/// Index of the nearest centroid for every grid sample (Voronoi regions).
#[wasm_bindgen]
pub fn kmeans_regions(centroids: &[f64], xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Vec<f64> {
    Grid::new(xmin, xmax, ymin, ymax, nx, ny).map(|x, y| nearest(x, y, centroids).0 as f64)
}

/// Best (lowest) final inertia over `restarts` k-means++ runs for each
/// `k = 1 … kmax`.
#[wasm_bindgen]
pub fn kmeans_elbow(points: &[f64], stride: usize, kmax: usize, restarts: usize, seed: u32) -> Vec<f64> {
    let pts = coords(points, stride);
    let mut out = Vec::with_capacity(kmax);
    for k in 1..=kmax {
        let mut best = f64::INFINITY;
        for r in 0..restarts.max(1) {
            let init = kmeans_init(points, stride, k, 1, seed.wrapping_add((k * 1000 + r) as u32));
            let run = lloyd(&pts, &init, 100);
            best = best.min(*run.inertia.last().unwrap_or(&0.0));
        }
        out.push(best);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_blobs;

    #[test]
    fn inertia_never_increases_and_converges() {
        let data = make_blobs(300, 4, 0.1, 8);
        let init = kmeans_init(&data, 3, 4, 2, 1);
        let run = kmeans_run(&data, 3, &init, 100);
        assert!(run.converged);
        for w in run.inertia.windows(2) {
            assert!(w[1] <= w[0] + 1e-9);
        }
        assert_eq!(run.centroids.len(), (run.iterations + 1) * 8);
        assert_eq!(run.assignments.len(), run.iterations * 300);
    }

    #[test]
    fn elbow_is_decreasing() {
        let data = make_blobs(200, 3, 0.08, 2);
        let e = kmeans_elbow(&data, 3, 6, 3, 1);
        for w in e.windows(2) {
            assert!(w[1] <= w[0] + 1e-9);
        }
        // Big drop until the true k, then small.
        assert!(e[2] < 0.2 * e[0]);
    }
}
