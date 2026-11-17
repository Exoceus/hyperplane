//! k-nearest-neighbour classification.

use wasm_bindgen::prelude::*;

use crate::grid::Grid;

/// Metric ids: 0 = Euclidean (L2), 1 = Manhattan (L1), 2 = Chebyshev (L∞).
#[inline]
fn dist(metric: u8, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let (dx, dy) = ((ax - bx).abs(), (ay - by).abs());
    match metric {
        1 => dx + dy,
        2 => dx.max(dy),
        _ => (dx * dx + dy * dy).sqrt(),
    }
}

/// The `k` nearest training points to `(qx, qy)`, closest first, as
/// `(distance, index)`. `scratch` is reused between calls to avoid
/// allocation.
fn k_nearest(data: &[f64], metric: u8, qx: f64, qy: f64, k: usize, scratch: &mut Vec<(f64, u32)>) {
    scratch.clear();
    for (i, p) in data.chunks_exact(3).enumerate() {
        scratch.push((dist(metric, qx, qy, p[0], p[1]), i as u32));
    }
    let k = k.min(scratch.len());
    if k == 0 {
        return;
    }
    if k < scratch.len() {
        scratch.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
        scratch.truncate(k);
    }
    scratch.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
}

/// Majority vote among (sorted) neighbours. Ties go to the class whose
/// nearest member is closest. Returns `(class, vote_fraction)`.
fn vote(data: &[f64], neighbours: &[(f64, u32)], votes: &mut [u32]) -> (usize, f64) {
    votes.iter_mut().for_each(|v| *v = 0);
    for &(_, i) in neighbours {
        let c = data[3 * i as usize + 2] as usize;
        if c < votes.len() {
            votes[c] += 1;
        }
    }
    let max = *votes.iter().max().unwrap_or(&0);
    let winner = neighbours
        .iter()
        .map(|&(_, i)| data[3 * i as usize + 2] as usize)
        .find(|&c| c < votes.len() && votes[c] == max)
        .unwrap_or(0);
    let frac = if neighbours.is_empty() { 0.0 } else { max as f64 / neighbours.len() as f64 };
    (winner, frac)
}

fn num_classes(data: &[f64]) -> usize {
    data.chunks_exact(3).map(|p| p[2] as usize + 1).max().unwrap_or(1)
}

/// Classify every grid sample. Stride 2 `[class, vote_fraction]`.
#[wasm_bindgen]
pub fn knn_grid(data: &[f64], k: usize, metric: u8, xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Vec<f64> {
    let g = Grid::new(xmin, xmax, ymin, ymax, nx, ny);
    let mut out = Vec::with_capacity(g.len() * 2);
    let mut scratch = Vec::with_capacity(data.len() / 3);
    let mut votes = vec![0u32; num_classes(data)];
    for j in 0..g.ny {
        let y = g.y(j);
        for i in 0..g.nx {
            k_nearest(data, metric, g.x(i), y, k, &mut scratch);
            let (c, f) = vote(data, &scratch, &mut votes);
            out.push(c as f64);
            out.push(f);
        }
    }
    out
}

/// The `k` nearest neighbours of a query: `[i₁ … i_k, d₁ … d_k]`, sorted by
/// distance.
#[wasm_bindgen]
pub fn knn_query(data: &[f64], qx: f64, qy: f64, k: usize, metric: u8) -> Vec<f64> {
    let mut scratch = Vec::new();
    k_nearest(data, metric, qx, qy, k, &mut scratch);
    let mut out: Vec<f64> = scratch.iter().map(|&(_, i)| i as f64).collect();
    out.extend(scratch.iter().map(|&(d, _)| d));
    out
}

/// Accuracy as a function of k for `k = 1 … kmax`: stride 2
/// `[train_accuracy, test_accuracy]`. Training accuracy classifies the
/// training points themselves (each point is its own nearest neighbour —
/// which is exactly why k = 1 looks perfect on training data).
#[wasm_bindgen]
pub fn knn_accuracy_curve(train: &[f64], test: &[f64], kmax: usize, metric: u8) -> Vec<f64> {
    let classes = num_classes(train).max(num_classes(test));
    let mut acc = vec![[0usize; 2]; kmax];
    let mut scratch = Vec::with_capacity(train.len() / 3);
    let mut votes = vec![0u32; classes];
    for (set_idx, set) in [train, test].iter().enumerate() {
        for p in set.chunks_exact(3) {
            k_nearest(train, metric, p[0], p[1], kmax, &mut scratch);
            for k in 1..=kmax.min(scratch.len()) {
                let (c, _) = vote(train, &scratch[..k], &mut votes);
                if c == p[2] as usize {
                    acc[k - 1][set_idx] += 1;
                }
            }
        }
    }
    let (ntr, nte) = ((train.len() / 3).max(1) as f64, (test.len() / 3).max(1) as f64);
    acc.iter().flat_map(|a| [a[0] as f64 / ntr, a[1] as f64 / nte]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours_sorted_and_correct() {
        let data = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.2, 0.1, 0.0, 5.0, 5.0, 1.0];
        let q = knn_query(&data, 0.1, 0.0, 2, 0);
        assert_eq!(&q[..2], &[0.0, 2.0]);
        assert!(q[2] <= q[3]);
    }

    #[test]
    fn k1_has_perfect_training_accuracy() {
        let train = crate::datasets::make_shape("moons", 120, 0.2, 1);
        let test = crate::datasets::make_shape("moons", 120, 0.2, 2);
        let curve = knn_accuracy_curve(&train, &test, 15, 0);
        assert_eq!(curve[0], 1.0);
        assert!(curve[1] > 0.8);
    }

    #[test]
    fn grid_votes_in_range() {
        let data = crate::datasets::make_blobs(90, 3, 0.1, 3);
        let g = knn_grid(&data, 5, 1, -1.0, 1.0, -1.0, 1.0, 20, 20);
        assert_eq!(g.len(), 800);
        for c in g.chunks(2) {
            assert!(c[0] < 3.0 && c[1] > 0.0 && c[1] <= 1.0);
        }
    }
}
