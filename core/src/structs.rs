use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct DataPoint {
    x1: f64,
    x2: f64,
    label: i32, // +1 or -1
}

impl DataPoint {
    pub fn new(x1: f64, x2: f64, label: i32) -> Self {
        Self { x1, x2, label }
    }
}
