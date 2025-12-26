use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct DataPoint {
    pub features: Vec<f64>,
}

impl DataPoint {
    pub fn new(features: Vec<f64>) -> Self {
        Self { features }
    }
}
#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct LabelledDataPoint {
    pub data_point: DataPoint,
    pub label: i32,
}

impl LabelledDataPoint {
    pub fn new(data_point: DataPoint, label: i32) -> Self {
        Self { data_point, label }
    }
}
