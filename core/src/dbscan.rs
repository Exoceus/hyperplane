//! DBSCAN: density-based clustering, recorded in the order points join
//! clusters so the expansion can be animated.

use wasm_bindgen::prelude::*;

fn coords(points: &[f64], stride: usize) -> Vec<(f64, f64)> {
    points.chunks_exact(stride.max(2)).map(|p| (p[0], p[1])).collect()
}

fn neighbours(pts: &[(f64, f64)], i: usize, eps2: f64) -> Vec<usize> {
    let (x, y) = pts[i];
    (0..pts.len()).filter(|&j| (pts[j].0 - x).powi(2) + (pts[j].1 - y).powi(2) <= eps2).collect()
}

/// Result of a DBSCAN run.
///
/// * `labels[i]`: cluster id (0, 1, …) or −1 for noise.
/// * `kinds[i]`: 2 = core, 1 = border, 0 = noise.
/// * `order`: stride 3 `[point, cluster, parent]` in the order points joined
///   a cluster (`parent` = the core point that reached it, −1 for a seed).
#[wasm_bindgen]
pub struct DbscanRun {
    labels: Vec<f64>,
    kinds: Vec<f64>,
    order: Vec<f64>,
    clusters: usize,
}

#[wasm_bindgen]
impl DbscanRun {
    #[wasm_bindgen(getter)]
    pub fn labels(&self) -> Vec<f64> {
        self.labels.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn kinds(&self) -> Vec<f64> {
        self.kinds.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn order(&self) -> Vec<f64> {
        self.order.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn clusters(&self) -> usize {
        self.clusters
    }
}

/// Cluster points (stride `stride`, first two values are x, y). A point is a
/// *core* point when at least `min_pts` points (itself included) lie within
/// `eps`; clusters are the connected components of core points plus the
/// border points they reach.
#[wasm_bindgen]
pub fn dbscan(points: &[f64], stride: usize, eps: f64, min_pts: usize) -> DbscanRun {
    let pts = coords(points, stride);
    let n = pts.len();
    let eps2 = eps * eps;
    let neigh: Vec<Vec<usize>> = (0..n).map(|i| neighbours(&pts, i, eps2)).collect();
    let core: Vec<bool> = neigh.iter().map(|v| v.len() >= min_pts.max(1)).collect();
    let mut labels = vec![-1i64; n];
    let mut order = Vec::new();
    let mut cluster = 0i64;

    for seed in 0..n {
        if labels[seed] >= 0 || !core[seed] {
            continue;
        }
        labels[seed] = cluster;
        order.extend_from_slice(&[seed as f64, cluster as f64, -1.0]);
        let mut queue = std::collections::VecDeque::from([seed]);
        while let Some(p) = queue.pop_front() {
            if !core[p] {
                continue;
            }
            for &q in &neigh[p] {
                if labels[q] < 0 {
                    labels[q] = cluster;
                    order.extend_from_slice(&[q as f64, cluster as f64, p as f64]);
                    queue.push_back(q);
                }
            }
        }
        cluster += 1;
    }

    let kinds = (0..n)
        .map(|i| if core[i] { 2.0 } else if labels[i] >= 0 { 1.0 } else { 0.0 })
        .collect();
    DbscanRun { labels: labels.iter().map(|&l| l as f64).collect(), kinds, order, clusters: cluster as usize }
}

/// Distance from every point to its `k`-th nearest neighbour (itself excluded),
/// sorted descending: the classic "k-distance" plot for choosing ε.
#[wasm_bindgen]
pub fn k_distance(points: &[f64], stride: usize, k: usize) -> Vec<f64> {
    let pts = coords(points, stride);
    let k = k.max(1);
    let mut out: Vec<f64> = (0..pts.len())
        .map(|i| {
            let mut d: Vec<f64> = (0..pts.len())
                .filter(|&j| j != i)
                .map(|j| ((pts[j].0 - pts[i].0).powi(2) + (pts[j].1 - pts[i].1).powi(2)).sqrt())
                .collect();
            if d.is_empty() {
                return 0.0;
            }
            let idx = (k - 1).min(d.len() - 1);
            d.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
            d[idx]
        })
        .collect();
    out.sort_by(|a, b| b.total_cmp(a));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_shape;

    #[test]
    fn separates_moons_that_kmeans_cannot() {
        let data = make_shape("moons", 300, 0.04, 2);
        let run = dbscan(&data, 3, 0.12, 5);
        assert_eq!(run.clusters, 2);
        // Each found cluster should be (almost) a single true moon.
        for c in 0..2 {
            let members: Vec<usize> = (0..300).filter(|&i| run.labels[i] == c as f64).collect();
            let ones = members.iter().filter(|&&i| data[3 * i + 2] > 0.5).count();
            let purity = ones.max(members.len() - ones) as f64 / members.len() as f64;
            assert!(purity > 0.98, "cluster {c} purity {purity}");
        }
    }

    #[test]
    fn isolated_points_are_noise_and_order_is_complete() {
        let mut data = make_shape("circles", 200, 0.03, 1);
        data.extend_from_slice(&[1.8, 1.8, 0.0]);
        let run = dbscan(&data, 3, 0.15, 4);
        assert_eq!(run.labels[200], -1.0);
        assert_eq!(run.kinds[200], 0.0);
        let assigned = run.labels.iter().filter(|&&l| l >= 0.0).count();
        assert_eq!(run.order.len() / 3, assigned);
    }

    #[test]
    fn k_distance_is_sorted() {
        let data = make_shape("moons", 100, 0.1, 3);
        let d = k_distance(&data, 3, 4);
        for w in d.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }
}
