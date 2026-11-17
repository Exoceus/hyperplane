//! Multi-armed Bernoulli bandits: four strategies racing on the same arms.

use wasm_bindgen::prelude::*;

use crate::rng::Rng;

/// Strategy ids, in output order.
pub const GREEDY: usize = 0;
pub const EPS_GREEDY: usize = 1;
pub const UCB: usize = 2;
pub const THOMPSON: usize = 3;
const STRATEGIES: usize = 4;

/// Gamma(shape, 1) via Marsaglia–Tsang (shape ≥ 1 here).
fn gamma(rng: &mut Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        return gamma(rng, shape + 1.0) * rng.uniform().max(1e-300).powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rng.normal();
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.uniform();
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

fn beta(rng: &mut Rng, a: f64, b: f64) -> f64 {
    let x = gamma(rng, a);
    let y = gamma(rng, b);
    x / (x + y)
}

#[derive(Clone)]
struct Arms {
    counts: Vec<u32>,
    wins: Vec<f64>,
    last: usize,
    regret: f64,
}

#[wasm_bindgen]
pub struct Bandit {
    means: Vec<f64>,
    states: Vec<Arms>,
    rng: Rng,
    epsilon: f64,
    ucb_c: f64,
    t: u32,
    regret: Vec<f64>,
}

impl Bandit {
    fn choose(&mut self, s: usize) -> usize {
        let k = self.means.len();
        let st = &self.states[s];
        // Every strategy except Thompson tries each arm once first.
        if s != THOMPSON {
            if let Some(a) = (0..k).find(|&a| st.counts[a] == 0) {
                return a;
            }
        }
        let mean = |a: usize| st.wins[a] / st.counts[a].max(1) as f64;
        let argmax = |f: &dyn Fn(usize) -> f64| (0..k).max_by(|&i, &j| f(i).total_cmp(&f(j))).unwrap_or(0);
        match s {
            EPS_GREEDY if self.rng.uniform() < self.epsilon => self.rng.below(k),
            UCB => {
                let total: f64 = st.counts.iter().map(|&c| c as f64).sum::<f64>().max(1.0);
                argmax(&|a| mean(a) + self.ucb_c * (2.0 * total.ln() / st.counts[a] as f64).sqrt())
            }
            THOMPSON => {
                let draws: Vec<f64> = (0..k)
                    .map(|a| {
                        let wins = self.states[s].wins[a];
                        let losses = self.states[s].counts[a] as f64 - wins;
                        beta(&mut self.rng, 1.0 + wins, 1.0 + losses)
                    })
                    .collect();
                (0..k).max_by(|&i, &j| draws[i].total_cmp(&draws[j])).unwrap_or(0)
            }
            _ => argmax(&mean),
        }
    }
}

#[wasm_bindgen]
impl Bandit {
    /// Arms pay 1 with probability `means[a]`, else 0.
    #[wasm_bindgen(constructor)]
    pub fn new(means: Vec<f64>, seed: u32) -> Bandit {
        let k = means.len().max(1);
        let arms = Arms { counts: vec![0; k], wins: vec![0.0; k], last: 0, regret: 0.0 };
        Bandit { means, states: vec![arms; STRATEGIES], rng: Rng::new(seed), epsilon: 0.1, ucb_c: 1.0, t: 0, regret: Vec::new() }
    }

    pub fn set_params(&mut self, epsilon: f64, ucb_c: f64) {
        self.epsilon = epsilon;
        self.ucb_c = ucb_c;
    }

    /// Every strategy pulls one arm, `n` times over.
    pub fn step(&mut self, n: u32) {
        let best = self.means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for _ in 0..n {
            for s in 0..STRATEGIES {
                let a = self.choose(s);
                let reward = if self.rng.uniform() < self.means[a] { 1.0 } else { 0.0 };
                let st = &mut self.states[s];
                st.counts[a] += 1;
                st.wins[a] += reward;
                st.last = a;
                st.regret += best - self.means[a];
                self.regret.push(st.regret);
            }
            self.t += 1;
        }
    }

    /// Pulls per arm, strategy-major: `4 · k`.
    pub fn counts(&self) -> Vec<f64> {
        self.states.iter().flat_map(|s| s.counts.iter().map(|&c| c as f64)).collect()
    }

    /// Wins per arm, strategy-major: `4 · k`.
    pub fn wins(&self) -> Vec<f64> {
        self.states.iter().flat_map(|s| s.wins.clone()).collect()
    }

    /// The arm each strategy pulled last.
    pub fn last(&self) -> Vec<f64> {
        self.states.iter().map(|s| s.last as f64).collect()
    }

    /// Cumulative (pseudo-)regret after every pull: stride 4.
    pub fn regret(&self) -> Vec<f64> {
        self.regret.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn t(&self) -> u32 {
        self.t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_sampler_has_the_right_mean() {
        let mut rng = Rng::new(1);
        let n = 20_000;
        let m: f64 = (0..n).map(|_| beta(&mut rng, 3.0, 7.0)).sum::<f64>() / n as f64;
        assert!((m - 0.3).abs() < 0.01, "mean {m}");
    }

    #[test]
    fn smart_strategies_find_the_best_arm() {
        let mut b = Bandit::new(vec![0.2, 0.5, 0.75], 4);
        b.step(3000);
        let counts = b.counts();
        for s in [UCB, THOMPSON] {
            let best = counts[s * 3 + 2];
            assert!(best > 0.85 * 3000.0, "strategy {s} pulled the best arm {best} times");
        }
        // Regret grows sublinearly for Thompson: late slope is tiny.
        let r = b.regret();
        let late = r[(2999) * 4 + THOMPSON] - r[(1999) * 4 + THOMPSON];
        assert!(late < 40.0, "Thompson regret over the last 1000 pulls: {late}");
    }
}
