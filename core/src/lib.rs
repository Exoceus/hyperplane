use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct DataPoint {
    x1: f64,
    x2: f64,
    label: i32, // +1 or -1
}

#[wasm_bindgen]
pub struct LinearlySeparableDatasetGenerator {
    n: usize,
}

#[wasm_bindgen]
impl LinearlySeparableDatasetGenerator {
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    #[wasm_bindgen(js_name = generateRandom)]
    pub fn generate_random(&self, seed: Option<u64>) -> Vec<DataPoint> {
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        // Define a random line: w1*x1 + w2*x2 + b = 0
        let w1: f64 = rng.gen_range(-1.0..1.0);
        let w2: f64 = rng.gen_range(-1.0..1.0);
        let b: f64 = rng.gen_range(-1.0..1.0);

        self.generate_with_params(&mut rng, w1, w2, b)
    }

    #[wasm_bindgen(js_name = generateFixed)]
    pub fn generate_fixed(&self, w1: f64, w2: f64, b: f64) -> Vec<DataPoint> {
        let mut rng: rand::prelude::StdRng = rand::rngs::StdRng::from_entropy();
        self.generate_with_params(&mut rng, w1, w2, b)
    }

    fn generate_with_params(
        &self,
        rng: &mut rand::rngs::StdRng,
        w1: f64,
        w2: f64,
        b: f64,
    ) -> Vec<DataPoint> {
        let mut dataset = Vec::new();

        for _ in 0..self.n {
            let x1: f64 = rng.gen_range(-10.0..10.0);
            let x2: f64 = rng.gen_range(-10.0..10.0);

            // Determine label based on which side of the line the point is on
            let value = w1 * x1 + w2 * x2 + b;
            let label = if value >= 0.0 { 1 } else { -1 };

            dataset.push(DataPoint { x1, x2, label });
        }

        dataset
    }
}
