//! Tabular reinforcement learning in a gridworld: Q-learning (model-free)
//! and value iteration (planning with a known model).
//!
//! Cells are row-major, row 0 at the top. Cell codes:
//! 0 = empty, 1 = wall, 2 = goal (+1, terminal), 3 = pit (−1, terminal),
//! 4 = start, 5 = small goal (+0.3, terminal).
//! Actions: 0 = up, 1 = right, 2 = down, 3 = left. With probability `slip`
//! the agent instead moves in one of the two perpendicular directions.

use wasm_bindgen::prelude::*;

use crate::rng::Rng;

const DX: [i32; 4] = [0, 1, 0, -1];
const DY: [i32; 4] = [-1, 0, 1, 0];

fn terminal_reward(code: u8) -> Option<f64> {
    match code {
        2 => Some(1.0),
        3 => Some(-1.0),
        5 => Some(0.3),
        _ => None,
    }
}

struct World {
    w: usize,
    h: usize,
    cells: Vec<u8>,
}

impl World {
    fn move_to(&self, s: usize, a: usize) -> usize {
        let (x, y) = ((s % self.w) as i32, (s / self.w) as i32);
        let (nx, ny) = (x + DX[a], y + DY[a]);
        if nx < 0 || ny < 0 || nx >= self.w as i32 || ny >= self.h as i32 {
            return s;
        }
        let n = ny as usize * self.w + nx as usize;
        if self.cells[n] == 1 { s } else { n }
    }

    /// Possible outcomes of taking `a` in `s`: (probability, next state).
    fn outcomes(&self, s: usize, a: usize, slip: f64) -> [(f64, usize); 3] {
        [
            (1.0 - slip, self.move_to(s, a)),
            (slip / 2.0, self.move_to(s, (a + 1) % 4)),
            (slip / 2.0, self.move_to(s, (a + 3) % 4)),
        ]
    }

    fn reward(&self, next: usize, step_reward: f64) -> (f64, bool) {
        match terminal_reward(self.cells[next]) {
            Some(r) => (r, true),
            None => (step_reward, false),
        }
    }

    fn is_terminal(&self, s: usize) -> bool {
        terminal_reward(self.cells[s]).is_some()
    }
}

/// A gridworld with a Q-learning agent living in it.
#[wasm_bindgen]
pub struct GridWorld {
    world: World,
    q: Vec<f64>,
    visits: Vec<u32>,
    pos: usize,
    rng: Rng,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    step_reward: f64,
    slip: f64,
    max_steps: u32,
    episode: u32,
    steps: u32,
    ep_return: f64,
    discount: f64,
    returns: Vec<f64>,
    lengths: Vec<f64>,
    trail: Vec<u32>,
}

impl GridWorld {
    fn start_cell(&mut self) -> usize {
        if let Some(i) = self.world.cells.iter().position(|&c| c == 4) {
            return i;
        }
        let free: Vec<usize> = (0..self.world.cells.len()).filter(|&i| self.world.cells[i] == 0).collect();
        if free.is_empty() { 0 } else { free[self.rng.below(free.len())] }
    }

    fn greedy(&mut self, s: usize) -> usize {
        let q = &self.q[s * 4..s * 4 + 4];
        let best = q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ties: Vec<usize> = (0..4).filter(|&a| q[a] >= best - 1e-12).collect();
        ties[self.rng.below(ties.len())]
    }

    fn begin_episode(&mut self) {
        self.pos = self.start_cell();
        self.steps = 0;
        self.ep_return = 0.0;
        self.discount = 1.0;
        self.trail.clear();
        self.trail.push(self.pos as u32);
    }

    fn end_episode(&mut self) {
        self.returns.push(self.ep_return);
        self.lengths.push(self.steps as f64);
        self.episode += 1;
        self.begin_episode();
    }

    /// One environment step with a Q-learning update. Returns `[from, action, to, reward, done]`.
    fn advance(&mut self) -> [f64; 5] {
        let s = self.pos;
        let a = if self.rng.uniform() < self.epsilon { self.rng.below(4) } else { self.greedy(s) };
        let roll = self.rng.uniform();
        let outs = self.world.outcomes(s, a, self.slip);
        let next = if roll < outs[0].0 { outs[0].1 } else if roll < outs[0].0 + outs[1].0 { outs[1].1 } else { outs[2].1 };
        let (r, done) = self.world.reward(next, self.step_reward);
        let future = if done { 0.0 } else { self.q[next * 4..next * 4 + 4].iter().cloned().fold(f64::NEG_INFINITY, f64::max) };
        let target = r + self.gamma * future;
        self.q[s * 4 + a] += self.alpha * (target - self.q[s * 4 + a]);
        self.visits[s] += 1;
        self.ep_return += self.discount * r;
        self.discount *= self.gamma;
        self.steps += 1;
        self.pos = next;
        self.trail.push(next as u32);
        let out = [s as f64, a as f64, next as f64, r, if done { 1.0 } else { 0.0 }];
        if done || self.steps >= self.max_steps {
            self.end_episode();
        }
        out
    }
}

#[wasm_bindgen]
impl GridWorld {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, cells: Vec<u8>, seed: u32) -> GridWorld {
        let n = width * height;
        let mut cells = cells;
        cells.resize(n, 0);
        let mut g = GridWorld {
            world: World { w: width, h: height, cells },
            q: vec![0.0; n * 4],
            visits: vec![0; n],
            pos: 0,
            rng: Rng::new(seed),
            alpha: 0.5,
            gamma: 0.95,
            epsilon: 0.1,
            step_reward: -0.01,
            slip: 0.0,
            max_steps: 300,
            episode: 0,
            steps: 0,
            ep_return: 0.0,
            discount: 1.0,
            returns: Vec::new(),
            lengths: Vec::new(),
            trail: Vec::new(),
        };
        g.begin_episode();
        g
    }

    pub fn set_params(&mut self, alpha: f64, gamma: f64, epsilon: f64, step_reward: f64, slip: f64) {
        self.alpha = alpha;
        self.gamma = gamma;
        self.epsilon = epsilon;
        self.step_reward = step_reward;
        self.slip = slip.clamp(0.0, 1.0);
    }

    /// Change one cell (e.g. the user drew a wall). Its Q-values are reset.
    pub fn set_cell(&mut self, index: usize, code: u8) {
        if index >= self.world.cells.len() {
            return;
        }
        if code == 4 {
            for c in self.world.cells.iter_mut() {
                if *c == 4 {
                    *c = 0;
                }
            }
        }
        self.world.cells[index] = code;
        for a in 0..4 {
            self.q[index * 4 + a] = 0.0;
        }
        if self.world.cells[self.pos] == 1 || code == 4 {
            self.begin_episode();
        }
    }

    /// Forget everything learned.
    pub fn reset(&mut self) {
        self.q.iter_mut().for_each(|v| *v = 0.0);
        self.visits.iter_mut().for_each(|v| *v = 0);
        self.returns.clear();
        self.lengths.clear();
        self.episode = 0;
        self.begin_episode();
    }

    /// Take one step. Returns `[from, action, to, reward, done]`.
    pub fn step(&mut self) -> Vec<f64> {
        self.advance().to_vec()
    }

    /// Run `n` whole episodes as fast as possible.
    pub fn train(&mut self, episodes: u32) {
        let target = self.episode + episodes;
        let mut guard = 0u64;
        while self.episode < target && guard < 5_000_000 {
            self.advance();
            guard += 1;
        }
    }

    pub fn q(&self) -> Vec<f64> {
        self.q.clone()
    }
    pub fn cells(&self) -> Vec<u8> {
        self.world.cells.clone()
    }
    pub fn visits(&self) -> Vec<u32> {
        self.visits.clone()
    }
    pub fn trail(&self) -> Vec<u32> {
        self.trail.clone()
    }
    /// Discounted return of every finished episode.
    pub fn returns(&self) -> Vec<f64> {
        self.returns.clone()
    }
    /// Length (steps) of every finished episode.
    pub fn lengths(&self) -> Vec<f64> {
        self.lengths.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn pos(&self) -> usize {
        self.pos
    }
    #[wasm_bindgen(getter)]
    pub fn episode(&self) -> u32 {
        self.episode
    }
}

/// Value iteration with the true model. Returns Q (cells · 4) after `sweeps`
/// synchronous Bellman backups starting from zero.
#[wasm_bindgen]
pub fn value_iteration(width: usize, height: usize, cells: &[u8], gamma: f64, step_reward: f64, slip: f64, sweeps: usize) -> Vec<f64> {
    let world = World { w: width, h: height, cells: cells.to_vec() };
    let n = width * height;
    let mut v = vec![0.0; n];
    let mut q = vec![0.0; n * 4];
    for _ in 0..sweeps {
        for s in 0..n {
            if world.cells[s] == 1 || world.is_terminal(s) {
                continue;
            }
            for a in 0..4 {
                q[s * 4 + a] = world
                    .outcomes(s, a, slip)
                    .iter()
                    .map(|&(p, next)| {
                        let (r, done) = world.reward(next, step_reward);
                        p * (r + if done { 0.0 } else { gamma * v[next] })
                    })
                    .sum();
            }
        }
        for s in 0..n {
            if world.cells[s] != 1 && !world.is_terminal(s) {
                v[s] = q[s * 4..s * 4 + 4].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            }
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 5×3 open room: start bottom-left, goal top-right.
    fn room() -> (usize, usize, Vec<u8>) {
        let mut cells = vec![0u8; 15];
        cells[10] = 4;
        cells[4] = 2;
        (5, 3, cells)
    }

    fn greedy_path_reaches_goal(w: usize, cells: &[u8], q: &[f64], start: usize) -> bool {
        let world = World { w, h: cells.len() / w, cells: cells.to_vec() };
        let mut s = start;
        for _ in 0..50 {
            if world.cells[s] == 2 {
                return true;
            }
            let qs = &q[s * 4..s * 4 + 4];
            let a = (0..4).max_by(|&i, &j| qs[i].total_cmp(&qs[j])).unwrap();
            s = world.move_to(s, a);
        }
        false
    }

    #[test]
    fn q_learning_learns_a_path_to_the_goal() {
        let (w, h, cells) = room();
        let mut g = GridWorld::new(w, h, cells.clone(), 1);
        g.set_params(0.5, 0.95, 0.2, -0.01, 0.0);
        g.train(300);
        assert!(greedy_path_reaches_goal(w, &cells, &g.q(), 10));
        let lengths = g.lengths();
        let late: f64 = lengths[lengths.len() - 50..].iter().sum::<f64>() / 50.0;
        assert!(late < 12.0, "late episodes still long: {late}");
    }

    #[test]
    fn value_iteration_values_decay_with_distance() {
        let (w, h, cells) = room();
        let q = value_iteration(w, h, &cells, 0.9, 0.0, 0.0, 50);
        let v = |s: usize| q[s * 4..s * 4 + 4].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Next to the goal: 1.0; one further: 0.9; etc.
        assert!((v(3) - 1.0).abs() < 1e-9);
        assert!((v(2) - 0.9).abs() < 1e-9);
        assert!(v(10) < v(2));
        assert!(greedy_path_reaches_goal(w, &cells, &q, 10));
    }

    #[test]
    fn walls_block_movement() {
        let world = World { w: 3, h: 1, cells: vec![0, 1, 0] };
        assert_eq!(world.move_to(0, 1), 0);
        assert_eq!(world.move_to(0, 3), 0);
    }
}
