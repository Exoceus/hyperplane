//! Logistic regression trained with full-batch gradient descent on the
//! (optionally L2-regularized) binary cross-entropy.

use wasm_bindgen::prelude::*;

use crate::grid::Grid;

#[inline]
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable `-log σ(z)`.
#[inline]
fn neg_log_sigmoid(z: f64) -> f64 {
    if z > 0.0 { (-z).exp().ln_1p() } else { -z + z.exp().ln_1p() }
}

/// Mean cross-entropy (plus `l2/2·|w|²`) and accuracy for weights `params`
/// (`d` weights followed by the bias) on data with stride `d + 1` whose last
/// entry is the 0/1 label.
fn loss_and_acc(data: &[f64], params: &[f64], l2: f64) -> (f64, f64) {
    let d = params.len() - 1;
    let n = data.len() / (d + 1);
    if n == 0 {
        return (0.0, 1.0);
    }
    let mut loss = 0.0;
    let mut correct = 0usize;
    for p in data.chunks_exact(d + 1) {
        let z: f64 = (0..d).map(|k| params[k] * p[k]).sum::<f64>() + params[d];
        let y = p[d];
        loss += if y > 0.5 { neg_log_sigmoid(z) } else { neg_log_sigmoid(-z) };
        if (z >= 0.0) == (y > 0.5) {
            correct += 1;
        }
    }
    let reg: f64 = 0.5 * l2 * params[..d].iter().map(|w| w * w).sum::<f64>();
    (loss / n as f64 + reg, correct as f64 / n as f64)
}

/// Gradient descent from `init` (`d` weights then bias) on stride-`d+1`
/// data. Returns stride-`d+3` records `[w…, b, loss, accuracy]`: the start
/// plus one record every `record_every` steps (and always the final step).
#[wasm_bindgen]
pub fn logreg_train(data: &[f64], init: &[f64], lr: f64, l2: f64, steps: usize, record_every: usize) -> Vec<f64> {
    let d = init.len().saturating_sub(1);
    let stride = d + 1;
    let n = data.len() / stride;
    let mut params = init.to_vec();
    let mut grad = vec![0.0; d + 1];
    let record_every = record_every.max(1);
    let mut out = Vec::new();
    let record = |params: &[f64], out: &mut Vec<f64>| {
        let (loss, acc) = loss_and_acc(data, params, l2);
        out.extend_from_slice(params);
        out.push(loss);
        out.push(acc);
    };
    record(&params, &mut out);
    if n == 0 {
        return out;
    }
    for step in 1..=steps {
        grad.iter_mut().for_each(|g| *g = 0.0);
        for p in data.chunks_exact(stride) {
            let z: f64 = (0..d).map(|k| params[k] * p[k]).sum::<f64>() + params[d];
            let err = sigmoid(z) - p[d];
            for k in 0..d {
                grad[k] += err * p[k];
            }
            grad[d] += err;
        }
        for k in 0..=d {
            let reg = if k < d { l2 * params[k] } else { 0.0 };
            params[k] -= lr * (grad[k] / n as f64 + reg);
        }
        if step % record_every == 0 || step == steps {
            record(&params, &mut out);
        }
    }
    out
}

/// `[loss, accuracy]` of the given parameters.
#[wasm_bindgen]
pub fn logreg_loss(data: &[f64], params: &[f64], l2: f64) -> Vec<f64> {
    let (l, a) = loss_and_acc(data, params, l2);
    vec![l, a]
}

/// `P(class 1) = σ(w1·x + w2·y + b)` over a grid.
#[wasm_bindgen]
pub fn logreg_grid(w1: f64, w2: f64, b: f64, xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Vec<f64> {
    Grid::new(xmin, xmax, ymin, ymax, nx, ny).map(|x, y| sigmoid(w1 * x + w2 * y + b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_two_blobs;

    #[test]
    fn learns_two_blobs_and_loss_decreases() {
        let data = make_two_blobs(200, 1.0, 0.2, 4);
        let traj = logreg_train(&data, &[0.0, 0.0, 0.0], 1.0, 0.0, 300, 10);
        let stride = 5;
        let first_loss = traj[3];
        let last = &traj[traj.len() - stride..];
        assert!(last[3] < first_loss * 0.3);
        assert!(last[4] > 0.95);
        for w in traj.chunks(stride).collect::<Vec<_>>().windows(2) {
            assert!(w[1][3] <= w[0][3] + 1e-12, "loss must not increase for small lr");
        }
    }

    #[test]
    fn one_dimensional_works() {
        let data = [-1.0, 0.0, -0.5, 0.0, 0.4, 1.0, 1.2, 1.0, 0.1, 0.0, -0.2, 1.0];
        let traj = logreg_train(&data, &[0.0, 0.0], 0.5, 0.0, 500, 500);
        assert_eq!(traj.len(), 2 * 4);
        assert!(traj[4] > 0.0, "weight should be positive");
    }
}
