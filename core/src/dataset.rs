use rand::Rng;
use rand::SeedableRng;
use wasm_bindgen::prelude::*;

use crate::structs;

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
    pub fn generate_random(&self, seed: Option<u64>) -> Vec<structs::DataPoint> {
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
    pub fn generate_fixed(&self, w1: f64, w2: f64, b: f64) -> Vec<structs::DataPoint> {
        let mut rng: rand::prelude::StdRng = rand::rngs::StdRng::from_entropy();
        self.generate_with_params(&mut rng, w1, w2, b)
    }

    fn generate_with_params(
        &self,
        rng: &mut rand::rngs::StdRng,
        w1: f64,
        w2: f64,
        b: f64,
    ) -> Vec<structs::DataPoint> {
        let mut dataset = Vec::new();

        for _ in 0..self.n {
            let x1: f64 = rng.gen_range(-10.0..10.0);
            let x2: f64 = rng.gen_range(-10.0..10.0);

            // Determine label based on which side of the line the point is on
            let value = w1 * x1 + w2 * x2 + b;
            let label = if value >= 0.0 { 1 } else { -1 };

            dataset.push(structs::DataPoint::new(x1, x2, label));
        }

        dataset
    }
}
