//! A small fully-connected neural network for 2D binary classification,
//! trained with backpropagation. Small enough to train live at 60 fps while
//! exposing every neuron's activation for visualization.

use wasm_bindgen::prelude::*;

use crate::grid::Grid;
use crate::logreg::sigmoid;
use crate::rng::Rng;

/// Hidden activation ids: 0 = tanh, 1 = ReLU, 2 = sigmoid, 3 = linear.
#[inline]
fn activate(kind: u8, z: f64) -> f64 {
    match kind {
        0 => z.tanh(),
        1 => z.max(0.0),
        2 => sigmoid(z),
        _ => z,
    }
}

/// Derivative of the activation, given the pre-activation `z` and output `a`.
#[inline]
fn activate_prime(kind: u8, z: f64, a: f64) -> f64 {
    match kind {
        0 => 1.0 - a * a,
        1 => {
            if z > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        2 => a * (1.0 - a),
        _ => 1.0,
    }
}

#[wasm_bindgen]
pub struct Mlp {
    sizes: Vec<usize>,
    act: u8,
    /// `w[l]` maps layer `l` (size `sizes[l]`) to layer `l+1`; row-major
    /// `[out][in]`.
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    gw: Vec<Vec<f64>>,
    gb: Vec<Vec<f64>>,
    mw: Vec<Vec<f64>>,
    vw: Vec<Vec<f64>>,
    mb: Vec<Vec<f64>>,
    vb: Vec<Vec<f64>>,
    adam_t: i32,
    steps: u32,
    rng: Rng,
    order: Vec<usize>,
    cursor: usize,
    // Scratch buffers for a single forward/backward pass.
    acts: Vec<Vec<f64>>,
    zs: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
}

impl Mlp {
    fn layers(&self) -> usize {
        self.sizes.len() - 1
    }

    /// Forward pass writing into the provided buffers.
    fn forward_into(&self, x: f64, y: f64, acts: &mut [Vec<f64>], zs: &mut [Vec<f64>]) {
        acts[0][0] = x;
        acts[0][1] = y;
        let last = self.layers() - 1;
        for l in 0..self.layers() {
            let (n_in, n_out) = (self.sizes[l], self.sizes[l + 1]);
            for o in 0..n_out {
                let row = &self.w[l][o * n_in..(o + 1) * n_in];
                let mut z = self.b[l][o];
                for i in 0..n_in {
                    z += row[i] * acts[l][i];
                }
                zs[l][o] = z;
                acts[l + 1][o] = if l == last { sigmoid(z) } else { activate(self.act, z) };
            }
        }
    }

    fn buffers(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let acts = self.sizes.iter().map(|&s| vec![0.0; s]).collect();
        let zs = self.sizes[1..].iter().map(|&s| vec![0.0; s]).collect();
        (acts, zs)
    }

    fn output(&self, x: f64, y: f64, acts: &mut [Vec<f64>], zs: &mut [Vec<f64>]) -> f64 {
        self.forward_into(x, y, acts, zs);
        acts[self.layers()][0]
    }

    /// Accumulate gradients for one example (forward + backprop).
    fn accumulate(&mut self, x: f64, y: f64, label: f64) {
        let mut acts = std::mem::take(&mut self.acts);
        let mut zs = std::mem::take(&mut self.zs);
        self.forward_into(x, y, &mut acts, &mut zs);
        let layers = self.layers();
        // Sigmoid + cross-entropy: dL/dz_out = p − y.
        self.deltas[layers - 1][0] = acts[layers][0] - label;
        for l in (0..layers).rev() {
            let (n_in, n_out) = (self.sizes[l], self.sizes[l + 1]);
            for o in 0..n_out {
                let d = self.deltas[l][o];
                self.gb[l][o] += d;
                let grow = &mut self.gw[l][o * n_in..(o + 1) * n_in];
                for i in 0..n_in {
                    grow[i] += d * acts[l][i];
                }
            }
            if l > 0 {
                for i in 0..n_in {
                    let mut s = 0.0;
                    for o in 0..n_out {
                        s += self.w[l][o * n_in + i] * self.deltas[l][o];
                    }
                    self.deltas[l - 1][i] = s * activate_prime(self.act, zs[l - 1][i], acts[l][i]);
                }
            }
        }
        self.acts = acts;
        self.zs = zs;
    }
}

#[wasm_bindgen]
impl Mlp {
    /// `sizes` lists every layer width including the 2 inputs and the single
    /// output, e.g. `[2, 8, 8, 1]`.
    #[wasm_bindgen(constructor)]
    pub fn new(sizes: Vec<u32>, activation: u8, seed: u32) -> Mlp {
        let mut sizes: Vec<usize> = sizes.iter().map(|&s| (s as usize).max(1)).collect();
        if sizes.len() < 2 {
            sizes = vec![2, 1];
        }
        sizes[0] = 2;
        *sizes.last_mut().unwrap() = 1;
        let mut rng = Rng::new(seed);
        let gain = if activation == 1 { 2f64.sqrt() } else { 1.0 };
        let mut w = Vec::new();
        let mut b = Vec::new();
        for l in 0..sizes.len() - 1 {
            let (n_in, n_out) = (sizes[l], sizes[l + 1]);
            let scale = gain / (n_in as f64).sqrt();
            w.push((0..n_in * n_out).map(|_| scale * rng.normal()).collect::<Vec<_>>());
            b.push((0..n_out).map(|_| 0.1 * rng.normal()).collect::<Vec<_>>());
        }
        let zeros_like = |v: &Vec<Vec<f64>>| v.iter().map(|x| vec![0.0; x.len()]).collect::<Vec<_>>();
        let mut net = Mlp {
            gw: zeros_like(&w),
            gb: zeros_like(&b),
            mw: zeros_like(&w),
            vw: zeros_like(&w),
            mb: zeros_like(&b),
            vb: zeros_like(&b),
            deltas: zeros_like(&b),
            sizes,
            act: activation,
            w,
            b,
            adam_t: 0,
            steps: 0,
            rng,
            order: Vec::new(),
            cursor: 0,
            acts: Vec::new(),
            zs: Vec::new(),
        };
        let (acts, zs) = net.buffers();
        net.acts = acts;
        net.zs = zs;
        net
    }

    /// Run `steps` optimization steps on stride-3 data `[x, y, label]`.
    ///
    /// `optimizer`: 0 = SGD, 1 = Adam. `batch_size` 0 means full batch.
    /// Returns the loss on the full dataset afterwards.
    pub fn train(&mut self, data: &[f64], steps: u32, lr: f64, l2: f64, optimizer: u8, batch_size: usize) -> f64 {
        let n = data.len() / 3;
        if n == 0 {
            return 0.0;
        }
        if self.order.len() != n {
            self.order = (0..n).collect();
            self.rng.shuffle(&mut self.order);
            self.cursor = 0;
        }
        let bs = if batch_size == 0 || batch_size >= n { n } else { batch_size };
        for _ in 0..steps {
            for g in self.gw.iter_mut().chain(self.gb.iter_mut()) {
                g.iter_mut().for_each(|v| *v = 0.0);
            }
            for k in 0..bs {
                let idx = if bs == n {
                    k
                } else {
                    if self.cursor >= n {
                        let mut order = std::mem::take(&mut self.order);
                        self.rng.shuffle(&mut order);
                        self.order = order;
                        self.cursor = 0;
                    }
                    self.cursor += 1;
                    self.order[self.cursor - 1]
                };
                let p = &data[3 * idx..3 * idx + 3];
                self.accumulate(p[0], p[1], if p[2] > 0.5 { 1.0 } else { 0.0 });
            }
            let inv = 1.0 / bs as f64;
            self.adam_t += 1;
            let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8);
            let c1 = 1.0 - b1.powi(self.adam_t);
            let c2 = 1.0 - b2.powi(self.adam_t);
            for l in 0..self.layers() {
                for (k, wv) in self.w[l].iter_mut().enumerate() {
                    let g = self.gw[l][k] * inv + l2 * *wv;
                    if optimizer == 1 {
                        self.mw[l][k] = b1 * self.mw[l][k] + (1.0 - b1) * g;
                        self.vw[l][k] = b2 * self.vw[l][k] + (1.0 - b2) * g * g;
                        *wv -= lr * (self.mw[l][k] / c1) / ((self.vw[l][k] / c2).sqrt() + eps);
                    } else {
                        *wv -= lr * g;
                    }
                }
                for (k, bv) in self.b[l].iter_mut().enumerate() {
                    let g = self.gb[l][k] * inv;
                    if optimizer == 1 {
                        self.mb[l][k] = b1 * self.mb[l][k] + (1.0 - b1) * g;
                        self.vb[l][k] = b2 * self.vb[l][k] + (1.0 - b2) * g * g;
                        *bv -= lr * (self.mb[l][k] / c1) / ((self.vb[l][k] / c2).sqrt() + eps);
                    } else {
                        *bv -= lr * g;
                    }
                }
            }
            self.steps += 1;
        }
        self.loss(data)
    }

    /// Mean binary cross-entropy on stride-3 data.
    pub fn loss(&self, data: &[f64]) -> f64 {
        let (mut acts, mut zs) = self.buffers();
        let n = data.len() / 3;
        if n == 0 {
            return 0.0;
        }
        let mut total = 0.0;
        for p in data.chunks_exact(3) {
            let q = self.output(p[0], p[1], &mut acts, &mut zs).clamp(1e-12, 1.0 - 1e-12);
            total -= if p[2] > 0.5 { q.ln() } else { (1.0 - q).ln() };
        }
        total / n as f64
    }

    /// Fraction of points classified correctly (threshold 0.5).
    pub fn accuracy(&self, data: &[f64]) -> f64 {
        let (mut acts, mut zs) = self.buffers();
        let n = data.len() / 3;
        if n == 0 {
            return 1.0;
        }
        let correct = data
            .chunks_exact(3)
            .filter(|p| (self.output(p[0], p[1], &mut acts, &mut zs) >= 0.5) == (p[2] > 0.5))
            .count();
        correct as f64 / n as f64
    }

    /// `P(class 1)` at a single point.
    pub fn predict(&self, x: f64, y: f64) -> f64 {
        let (mut acts, mut zs) = self.buffers();
        self.output(x, y, &mut acts, &mut zs)
    }

    /// `P(class 1)` over a grid.
    pub fn predict_grid(&self, xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Vec<f64> {
        let (mut acts, mut zs) = self.buffers();
        Grid::new(xmin, xmax, ymin, ymax, nx, ny).map(|x, y| self.output(x, y, &mut acts, &mut zs))
    }

    /// Activation of every neuron over a grid. Neurons are ordered layer by
    /// layer (hidden layers first, the output neuron last); each contributes
    /// one contiguous block of `nx · ny` values.
    pub fn neuron_grids(&self, xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Vec<f64> {
        let g = Grid::new(xmin, xmax, ymin, ymax, nx, ny);
        let (mut acts, mut zs) = self.buffers();
        let total: usize = self.sizes[1..].iter().sum();
        let cells = g.len();
        let mut out = vec![0.0; total * cells];
        for j in 0..g.ny {
            for i in 0..g.nx {
                self.forward_into(g.x(i), g.y(j), &mut acts, &mut zs);
                let cell = j * g.nx + i;
                let mut neuron = 0;
                for layer in acts.iter().skip(1) {
                    for &a in layer {
                        out[neuron * cells + cell] = a;
                        neuron += 1;
                    }
                }
            }
        }
        out
    }

    /// Representation of stride-2 points at `layer` (0 = input, the last
    /// layer = output probability). Stride = that layer's width.
    pub fn layer_outputs(&self, points: &[f64], layer: usize) -> Vec<f64> {
        let layer = layer.min(self.layers());
        let (mut acts, mut zs) = self.buffers();
        let mut out = Vec::with_capacity(points.len() / 2 * self.sizes[layer]);
        for p in points.chunks_exact(2) {
            self.forward_into(p[0], p[1], &mut acts, &mut zs);
            out.extend_from_slice(&acts[layer]);
        }
        out
    }

    /// All parameters: for each layer, its weights (row-major `[out][in]`)
    /// followed by its biases.
    pub fn params(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for l in 0..self.layers() {
            out.extend_from_slice(&self.w[l]);
            out.extend_from_slice(&self.b[l]);
        }
        out
    }

    #[wasm_bindgen(getter)]
    pub fn steps(&self) -> u32 {
        self.steps
    }

    #[wasm_bindgen(getter, js_name = numParams)]
    pub fn num_params(&self) -> usize {
        self.w.iter().map(|w| w.len()).sum::<usize>() + self.b.iter().map(|b| b.len()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_shape;

    #[test]
    fn gradients_match_finite_differences() {
        let data = make_shape("moons", 20, 0.1, 3);
        let mut net = Mlp::new(vec![2, 3, 2, 1], 0, 5);
        for g in net.gw.iter_mut().chain(net.gb.iter_mut()) {
            g.iter_mut().for_each(|v| *v = 0.0);
        }
        for p in data.chunks(3) {
            net.accumulate(p[0], p[1], p[2]);
        }
        let n = 20.0;
        let h = 1e-6;
        for l in 0..net.layers() {
            for k in 0..net.w[l].len() {
                let orig = net.w[l][k];
                net.w[l][k] = orig + h;
                let lp = net.loss(&data);
                net.w[l][k] = orig - h;
                let lm = net.loss(&data);
                net.w[l][k] = orig;
                let numeric = (lp - lm) / (2.0 * h);
                let analytic = net.gw[l][k] / n;
                assert!((numeric - analytic).abs() < 1e-6, "layer {l} w{k}: {numeric} vs {analytic}");
            }
        }
    }

    #[test]
    fn learns_xor_and_circles() {
        for (kind, sizes, act) in [("xor", vec![2, 8, 1], 0u8), ("circles", vec![2, 8, 8, 1], 1u8)] {
            let data = make_shape(kind, 200, 0.05, 1);
            let mut net = Mlp::new(sizes, act, 2);
            for _ in 0..30 {
                net.train(&data, 50, 0.03, 0.0, 1, 0);
            }
            assert!(net.accuracy(&data) > 0.95, "{kind}: {}", net.accuracy(&data));
        }
    }

    #[test]
    fn narrow_two_unit_networks_can_untangle_moons() {
        // The "bending space" figure relies on 2-wide tanh layers learning
        // moons well. Check that most seeds succeed.
        let data = make_shape("moons", 200, 0.1, 4);
        let mut ok = 0;
        for seed in 0..10 {
            let mut net = Mlp::new(vec![2, 2, 2, 2, 1], 0, seed);
            for _ in 0..40 {
                net.train(&data, 50, 0.03, 0.0, 1, 0);
            }
            if net.accuracy(&data) > 0.95 {
                ok += 1;
            }
        }
        assert!(ok >= 6, "only {ok}/10 seeds learned moons");
    }
}


