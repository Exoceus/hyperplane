use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct DataPoint {
    features: Vec<f64>,
}

impl DataPoint {
    pub fn new(features: Vec<f64>) -> Self {
        Self { features }
    }

    pub fn dot(&self, weights: &Vec<f64>) -> f64 {
        assert_eq!(
            self.features.len(),
            weights.len(),
            "Dimension mismatch in dot product. DataPoint has dimension {}, but weights have dimension {}.",
            self.features.len(),
            weights.len()
        );

        self.features
            .iter()
            .zip(weights.iter())
            .map(|(x, w)| x * w)
            .sum()
    }
}
#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct LabelledDataPoint {
    data_point: DataPoint,
    label: i32,
}

impl LabelledDataPoint {
    pub fn new(data_point: DataPoint, label: i32) -> Self {
        Self { data_point, label }
    }
}
