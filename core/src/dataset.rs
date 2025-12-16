use rand::Rng;
use rand::SeedableRng;
use wasm_bindgen::prelude::*;

use crate::structs;

#[wasm_bindgen]
pub struct LinearlySeparableDatasetGenerator {
    n: usize,
    dim: usize,
}

#[wasm_bindgen]
impl LinearlySeparableDatasetGenerator {
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize, dim: usize) -> Self {
        Self { n, dim }
    }

    #[wasm_bindgen(js_name = generateRandom)]
    pub fn generate_random(&self, seed: Option<u64>) -> Vec<structs::LabelledDataPoint> {
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        // Define a random line: <coefficients, X> + b = 0
        let coefficients = (1..=self.dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: f64 = rng.gen_range(-1.0..1.0);

        self.generate_with_params(&mut rng, &coefficients, b)
    }

    #[wasm_bindgen(js_name = generateFixed)]
    pub fn generate_fixed(
        &self,
        coefficients: Vec<f64>,
        b: f64,
    ) -> Vec<structs::LabelledDataPoint> {
        let mut rng: rand::prelude::StdRng = rand::rngs::StdRng::from_entropy();
        self.generate_with_params(&mut rng, &coefficients, b)
    }

    fn generate_with_params(
        &self,
        rng: &mut rand::rngs::StdRng,
        coefficients: &Vec<f64>,
        b: f64,
    ) -> Vec<structs::LabelledDataPoint> {
        let mut dataset = Vec::new();

        for _ in 0..self.n {
            let datapoint = structs::DataPoint::new(
                (1..=self.dim).map(|_| rng.gen_range(-10.0..10.0)).collect(),
            ); // generate random point

            // Determine label based on which side of the line the point is on
            let value = datapoint.dot(coefficients) + b;
            let label = if value >= 0.0 { 1 } else { -1 };

            dataset.push(structs::LabelledDataPoint::new(datapoint, label));
        }

        dataset
    }
}
