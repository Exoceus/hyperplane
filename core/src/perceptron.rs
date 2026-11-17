//! Rosenblatt's perceptron, recorded update-by-update so the UI can replay
//! (and smoothly animate) every single correction it makes.

use wasm_bindgen::prelude::*;

/// Result of a perceptron run.
///
/// `history` has stride 5: `[w1, w2, b, point_index, epoch]`. Record 0 is the
/// initial state (`point_index = -1`, `epoch = 0`); every following record is
/// the state *after* an update triggered by the point at `point_index`.
#[wasm_bindgen]
pub struct PerceptronRun {
    history: Vec<f64>,
    converged: bool,
    epochs: u32,
}

#[wasm_bindgen]
impl PerceptronRun {
    #[wasm_bindgen(getter)]
    pub fn history(&self) -> Vec<f64> {
        self.history.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Number of full passes over the data that were performed.
    #[wasm_bindgen(getter)]
    pub fn epochs(&self) -> u32 {
        self.epochs
    }
}

/// Train a perceptron on stride-3 data (class 1 → y = +1, class 0 → y = −1).
///
/// The update on a mistake (`y·(w·x + b) ≤ 0`) is the classic
/// `w ← w + y·x`, `b ← b + y` (the bias update is skipped when
/// `use_bias` is false, which forces the boundary through the origin).
/// Points are visited in the order given, cycling until a full clean pass
/// (converged), `max_epochs` passes, or `max_updates` updates.
#[wasm_bindgen]
pub fn perceptron_train(
    data: &[f64],
    w1: f64,
    w2: f64,
    b: f64,
    use_bias: bool,
    max_epochs: u32,
    max_updates: u32,
) -> PerceptronRun {
    let n = data.len() / 3;
    let (mut w1, mut w2, mut b) = (w1, w2, if use_bias { b } else { 0.0 });
    let mut history = vec![w1, w2, b, -1.0, 0.0];
    let mut updates = 0u32;

    for epoch in 1..=max_epochs {
        let mut mistakes = 0;
        for i in 0..n {
            let (x, y_coord) = (data[3 * i], data[3 * i + 1]);
            let y = if data[3 * i + 2] > 0.5 { 1.0 } else { -1.0 };
            if y * (w1 * x + w2 * y_coord + b) <= 0.0 {
                mistakes += 1;
                w1 += y * x;
                w2 += y * y_coord;
                if use_bias {
                    b += y;
                }
                history.extend_from_slice(&[w1, w2, b, i as f64, epoch as f64]);
                updates += 1;
                if updates >= max_updates {
                    return PerceptronRun { history, converged: false, epochs: epoch };
                }
            }
        }
        if mistakes == 0 {
            return PerceptronRun { history, converged: true, epochs: epoch };
        }
    }
    PerceptronRun { history, converged: false, epochs: max_epochs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_linear;

    #[test]
    fn converges_on_separable_data_within_novikoff_bound() {
        let (tw1, tw2, tb) = (0.7, -0.4, 0.15);
        let data = make_linear(150, tw1, tw2, tb, 0.08, 0.0, 11);
        let run = perceptron_train(&data, 0.0, 0.0, 0.0, true, 10_000, 100_000);
        assert!(run.converged);
        let h = run.history();
        let last = &h[h.len() - 5..];
        for p in data.chunks(3) {
            let y = if p[2] > 0.5 { 1.0 } else { -1.0 };
            assert!(y * (last[0] * p[0] + last[1] * p[1] + last[2]) > 0.0);
        }
        // Novikoff: mistakes ≤ (R/γ)² in the augmented space (x, y, 1).
        let norm = (tw1 * tw1 + tw2 * tw2 + tb * tb).sqrt();
        let mut gamma = f64::INFINITY;
        let mut r: f64 = 0.0;
        for p in data.chunks(3) {
            let y = if p[2] > 0.5 { 1.0 } else { -1.0 };
            gamma = gamma.min(y * (tw1 * p[0] + tw2 * p[1] + tb) / norm);
            r = r.max((p[0] * p[0] + p[1] * p[1] + 1.0).sqrt());
        }
        let mistakes = h.len() / 5 - 1;
        assert!((mistakes as f64) <= (r / gamma).powi(2));
    }

    #[test]
    fn stops_on_non_separable_data() {
        let data = make_linear(60, 1.0, 1.0, 0.0, 0.0, 0.25, 5);
        let run = perceptron_train(&data, 0.0, 0.0, 0.0, true, 50, 100_000);
        assert!(!run.converged);
        assert_eq!(run.epochs(), 50);
    }
}
