use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use crate::structs;
use crate::utils;

#[wasm_bindgen]
pub struct Perceptron {
    margin: f64,
    weights: Vec<f64>,
    bias: f64,
    max_epochs: usize,
}

#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct PerceptronStep {
    epoch: usize,
    weights: Vec<f64>,
    bias: f64,
}

#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct PerceptronResult {
    pub steps: Vec<PerceptronStep>,
    pub converged: bool,
}

#[wasm_bindgen]
impl Perceptron {
    #[wasm_bindgen(constructor)]
    pub fn new(margin: f64, weights: Vec<f64>, bias: f64, max_epochs: usize) -> Self {
        Self {
            margin,
            weights,
            bias,
            max_epochs,
        }
    }

    #[wasm_bindgen]
    pub fn train(&mut self, data: Vec<structs::LabelledDataPoint>) -> PerceptronResult {
        let mut steps = Vec::new();

        steps.push(PerceptronStep {
            epoch: 0,
            weights: self.weights.clone(),
            bias: self.bias.clone(),
        });

        for epoch in 1..=self.max_epochs {
            let mut encountered_mislabelled_point = false;

            for i in 0..data.len() {
                let data_point = &data[i];

                let prediction =
                    utils::dot(&self.weights, &data_point.data_point.features) + self.bias;

                if data_point.label as f64 * prediction <= self.margin {
                    encountered_mislabelled_point = true;

                    self.weights = utils::add(
                        &self.weights,
                        &utils::scale(data_point.label as f64, &data_point.data_point.features),
                    );

                    self.bias = self.bias + data_point.label as f64;

                    steps.push(PerceptronStep {
                        epoch: epoch,
                        weights: self.weights.clone(),
                        bias: self.bias.clone(),
                    });
                }
                // no update necessary if correct prediction
            }

            if !encountered_mislabelled_point {
                return PerceptronResult {
                    steps: steps,
                    converged: true,
                };
            }
        }

        return PerceptronResult {
            steps: steps,
            converged: false,
        };
    }

    pub fn prediction(&self, data_point: &structs::DataPoint) -> f64 {
        utils::dot(&self.weights, &data_point.features) + self.bias
    }
}
