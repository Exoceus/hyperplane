//! CART decision trees and random forests for 2D, multi-class data.

use wasm_bindgen::prelude::*;

use crate::grid::Grid;
use crate::rng::Rng;

/// Stride of a serialized node.
pub const NODE_STRIDE: usize = 9;

/// A node, serialized as
/// `[feature, threshold, left, right, majority, impurity, n, depth, parent]`.
/// `feature` is 0 (x) or 1 (y) for a split, −1 for a leaf. Points with
/// `coord < threshold` go left. Nodes are stored in breadth-first order, which
/// is also the order the splits were made.
#[derive(Clone, Debug)]
struct Node {
    feature: i32,
    threshold: f64,
    left: i32,
    right: i32,
    majority: usize,
    impurity: f64,
    n: usize,
    depth: usize,
    parent: i32,
}

#[derive(Clone, Copy)]
struct Params {
    max_depth: usize,
    min_leaf: usize,
    criterion: u8,
    /// Random forest: consider a single random feature per split.
    random_feature: bool,
}

fn impurity(counts: &[usize], n: usize, criterion: u8) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    if criterion == 1 {
        -counts.iter().filter(|&&c| c > 0).map(|&c| {
            let p = c as f64 / nf;
            p * p.log2()
        }).sum::<f64>()
    } else {
        1.0 - counts.iter().map(|&c| (c as f64 / nf).powi(2)).sum::<f64>()
    }
}

fn argmax(counts: &[usize]) -> usize {
    let mut best = 0;
    for (i, &c) in counts.iter().enumerate() {
        if c > counts[best] {
            best = i;
        }
    }
    best
}

/// Best split of `idx` on `feature`: (weighted child impurity, threshold).
fn best_split(pts: &[(f64, f64, usize)], idx: &mut [usize], feature: usize, classes: usize, p: Params) -> Option<(f64, f64)> {
    let key = |i: usize| if feature == 0 { pts[i].0 } else { pts[i].1 };
    idx.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
    let n = idx.len();
    let mut left = vec![0usize; classes];
    let mut right = vec![0usize; classes];
    for &i in idx.iter() {
        right[pts[i].2] += 1;
    }
    let mut best: Option<(f64, f64)> = None;
    for k in 1..n {
        let c = pts[idx[k - 1]].2;
        left[c] += 1;
        right[c] -= 1;
        let (a, b) = (key(idx[k - 1]), key(idx[k]));
        if b - a < 1e-12 || k < p.min_leaf || n - k < p.min_leaf {
            continue;
        }
        let w = (k as f64 * impurity(&left, k, p.criterion) + (n - k) as f64 * impurity(&right, n - k, p.criterion)) / n as f64;
        if best.map_or(true, |(bw, _)| w < bw - 1e-12) {
            best = Some((w, 0.5 * (a + b)));
        }
    }
    best
}

fn build(pts: &[(f64, f64, usize)], root: Vec<usize>, classes: usize, p: Params, rng: &mut Rng) -> Vec<Node> {
    let mut nodes: Vec<Node> = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    let make = |idx: &[usize], depth: usize, parent: i32| {
        let mut counts = vec![0usize; classes];
        for &i in idx {
            counts[pts[i].2] += 1;
        }
        Node {
            feature: -1,
            threshold: 0.0,
            left: -1,
            right: -1,
            majority: argmax(&counts),
            impurity: impurity(&counts, idx.len(), p.criterion),
            n: idx.len(),
            depth,
            parent,
        }
    };
    nodes.push(make(&root, 0, -1));
    members.push(root);
    let mut head = 0;
    while head < nodes.len() {
        let id = head;
        head += 1;
        let node = nodes[id].clone();
        if node.depth >= p.max_depth || node.impurity < 1e-12 || node.n < 2 * p.min_leaf.max(1) {
            continue;
        }
        let mut idx = members[id].clone();
        let features: Vec<usize> = if p.random_feature { vec![rng.below(2)] } else { vec![0, 1] };
        let mut best: Option<(f64, f64, usize)> = None;
        for &f in &features {
            if let Some((w, t)) = best_split(pts, &mut idx, f, classes, p) {
                if best.map_or(true, |(bw, _, _)| w < bw - 1e-12) {
                    best = Some((w, t, f));
                }
            }
        }
        let Some((w, t, f)) = best else { continue };
        // Below the root, stop when a split doesn't reduce impurity at all.
        // (The root may still make a useless-looking split, as on XOR.)
        if node.depth > 0 && !p.random_feature && w > node.impurity - 1e-12 {
            continue;
        }
        let (l, r): (Vec<usize>, Vec<usize>) = members[id].iter().partition(|&&i| if f == 0 { pts[i].0 < t } else { pts[i].1 < t });
        if l.is_empty() || r.is_empty() {
            continue;
        }
        let li = nodes.len() as i32;
        nodes.push(make(&l, node.depth + 1, id as i32));
        members.push(l);
        let ri = nodes.len() as i32;
        nodes.push(make(&r, node.depth + 1, id as i32));
        members.push(r);
        let n = &mut nodes[id];
        n.feature = f as i32;
        n.threshold = t;
        n.left = li;
        n.right = ri;
    }
    nodes
}

fn parse(data: &[f64]) -> (Vec<(f64, f64, usize)>, usize) {
    let pts: Vec<(f64, f64, usize)> = data.chunks_exact(3).map(|p| (p[0], p[1], p[2].max(0.0) as usize)).collect();
    let classes = pts.iter().map(|p| p.2 + 1).max().unwrap_or(1);
    (pts, classes)
}

fn predict(nodes: &[Node], x: f64, y: f64) -> usize {
    let mut id = 0usize;
    loop {
        let n = &nodes[id];
        if n.feature < 0 {
            return n.majority;
        }
        let v = if n.feature == 0 { x } else { y };
        id = if v < n.threshold { n.left as usize } else { n.right as usize };
    }
}

fn serialize(nodes: &[Node]) -> Vec<f64> {
    nodes
        .iter()
        .flat_map(|n| {
            [n.feature as f64, n.threshold, n.left as f64, n.right as f64, n.majority as f64, n.impurity, n.n as f64, n.depth as f64, n.parent as f64]
        })
        .collect()
}

/// Grow a CART tree on stride-3 data. `criterion`: 0 = Gini, 1 = entropy.
/// Returns nodes with stride [`NODE_STRIDE`] in breadth-first (split) order.
#[wasm_bindgen]
pub fn tree_build(data: &[f64], max_depth: usize, min_leaf: usize, criterion: u8) -> Vec<f64> {
    let (pts, classes) = parse(data);
    if pts.is_empty() {
        return Vec::new();
    }
    let p = Params { max_depth, min_leaf: min_leaf.max(1), criterion, random_feature: false };
    let mut rng = Rng::new(0);
    serialize(&build(&pts, (0..pts.len()).collect(), classes, p, &mut rng))
}

/// Train and test accuracy of a single tree for every depth `1 … max_depth`.
/// Stride 2: `[train, test]`.
#[wasm_bindgen]
pub fn tree_depth_curve(train: &[f64], test: &[f64], max_depth: usize, min_leaf: usize, criterion: u8) -> Vec<f64> {
    let (pts, classes) = parse(train);
    let (tpts, _) = parse(test);
    let mut rng = Rng::new(0);
    let mut out = Vec::with_capacity(max_depth * 2);
    for d in 1..=max_depth {
        let p = Params { max_depth: d, min_leaf: min_leaf.max(1), criterion, random_feature: false };
        let nodes = build(&pts, (0..pts.len()).collect(), classes, p, &mut rng);
        let acc = |set: &[(f64, f64, usize)]| set.iter().filter(|q| predict(&nodes, q.0, q.1) == q.2).count() as f64 / set.len().max(1) as f64;
        out.push(acc(&pts));
        out.push(acc(&tpts));
    }
    out
}

/// A random forest's vote over a grid: stride 2 `[class, vote_fraction]`.
/// Each tree sees a bootstrap sample and one random feature per split.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn forest_grid(data: &[f64], trees: usize, max_depth: usize, min_leaf: usize, seed: u32, xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Vec<f64> {
    let (pts, classes) = parse(data);
    let g = Grid::new(xmin, xmax, ymin, ymax, nx, ny);
    if pts.is_empty() {
        return vec![0.0; g.len() * 2];
    }
    let mut rng = Rng::new(seed);
    let p = Params { max_depth, min_leaf: min_leaf.max(1), criterion: 0, random_feature: true };
    let forest: Vec<Vec<Node>> = (0..trees.max(1))
        .map(|_| {
            let sample: Vec<usize> = (0..pts.len()).map(|_| rng.below(pts.len())).collect();
            build(&pts, sample, classes, p, &mut rng)
        })
        .collect();
    let mut votes = vec![0usize; classes];
    let mut out = Vec::with_capacity(g.len() * 2);
    for j in 0..g.ny {
        let y = g.y(j);
        for i in 0..g.nx {
            let x = g.x(i);
            votes.iter_mut().for_each(|v| *v = 0);
            for t in &forest {
                votes[predict(t, x, y)] += 1;
            }
            let c = argmax(&votes);
            out.push(c as f64);
            out.push(votes[c] as f64 / forest.len() as f64);
        }
    }
    out
}

/// Every tree of a random forest evaluated over a grid: `trees` blocks of
/// `nx · ny` class indices (u8), so the UI can add trees one at a time.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn forest_tree_grids(data: &[f64], trees: usize, max_depth: usize, min_leaf: usize, seed: u32, xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Vec<u8> {
    let (pts, classes) = parse(data);
    let g = Grid::new(xmin, xmax, ymin, ymax, nx, ny);
    let mut out = Vec::with_capacity(trees * g.len());
    if pts.is_empty() {
        return vec![0; trees * g.len()];
    }
    let mut rng = Rng::new(seed);
    let p = Params { max_depth, min_leaf: min_leaf.max(1), criterion: 0, random_feature: true };
    for _ in 0..trees {
        let sample: Vec<usize> = (0..pts.len()).map(|_| rng.below(pts.len())).collect();
        let tree = build(&pts, sample, classes, p, &mut rng);
        for j in 0..g.ny {
            let y = g.y(j);
            for i in 0..g.nx {
                out.push(predict(&tree, g.x(i), y) as u8);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_blobs, make_shape};

    #[test]
    fn deep_tree_fits_training_data() {
        let data = make_shape("moons", 200, 0.2, 1);
        let nodes = tree_build(&data, 30, 1, 0);
        let parsed: Vec<Node> = nodes
            .chunks(NODE_STRIDE)
            .map(|n| Node {
                feature: n[0] as i32,
                threshold: n[1],
                left: n[2] as i32,
                right: n[3] as i32,
                majority: n[4] as usize,
                impurity: n[5],
                n: n[6] as usize,
                depth: n[7] as usize,
                parent: n[8] as i32,
            })
            .collect();
        let acc = data.chunks(3).filter(|p| predict(&parsed, p[0], p[1]) == p[2] as usize).count();
        assert_eq!(acc, 200);
        // Breadth-first: depth never decreases along the array; children follow parents.
        for w in parsed.windows(2) {
            assert!(w[1].depth >= w[0].depth);
        }
        for (i, n) in parsed.iter().enumerate().skip(1) {
            assert!((n.parent as usize) < i);
        }
    }

    #[test]
    fn depth_one_is_a_single_split_and_curve_overfits() {
        let data = make_blobs(150, 3, 0.15, 2);
        let nodes = tree_build(&data, 1, 1, 0);
        assert_eq!(nodes.len() / NODE_STRIDE, 3);
        let train = make_shape("moons", 150, 0.35, 3);
        let test = make_shape("moons", 150, 0.35, 4);
        let curve = tree_depth_curve(&train, &test, 14, 1, 0);
        let last_train = curve[curve.len() - 2];
        assert!(last_train > 0.99);
        let best_test = curve.chunks(2).map(|c| c[1]).fold(0.0, f64::max);
        assert!(best_test > 0.8);
    }

    #[test]
    fn tree_grids_agree_with_forest_vote() {
        let data = make_shape("moons", 120, 0.2, 5);
        let grids = forest_tree_grids(&data, 9, 6, 1, 3, -1.0, 1.0, -1.0, 1.0, 10, 10);
        let vote = forest_grid(&data, 9, 6, 1, 3, -1.0, 1.0, -1.0, 1.0, 10, 10);
        for cell in 0..100 {
            let ones = (0..9).filter(|t| grids[t * 100 + cell] == 1).count();
            let winner = if ones * 2 > 9 { 1.0 } else { 0.0 };
            assert_eq!(vote[2 * cell], winner);
        }
    }

    #[test]
    fn forest_votes_are_fractions() {
        let data = make_shape("xor", 120, 0.1, 5);
        let g = forest_grid(&data, 25, 6, 1, 7, -1.0, 1.0, -1.0, 1.0, 12, 12);
        assert_eq!(g.len(), 288);
        for c in g.chunks(2) {
            assert!(c[1] > 0.0 && c[1] <= 1.0);
        }
    }
}
