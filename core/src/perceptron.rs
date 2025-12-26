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
pub struct PerceptronSteps {
    weights: Vec<f64>,
    bias: f64,
}

impl Perceptron {
    pub fn new(margin: f64, weights: Vec<f64>, bias: f64, max_epochs: usize) -> Self {
        Self {
            margin,
            weights,
            bias,
            max_epochs,
        }
    }

    pub fn train(&mut self, data: &[structs::LabelledDataPoint]) -> Vec<PerceptronSteps> {
        let mut steps = Vec::new();

        steps.push(PerceptronSteps {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
        });

        for _ in 1..=self.max_epochs {
            let mut encountered_mislabelled_point = false;

            for i in 0..data.len() {
                let data_point = &data[i];

                let prediction =
                    utils::dot(&self.weights, &data_point.data_point.features) + self.bias;

                if data_point.label as f64 * prediction <= self.margin {
                    encountered_mislabelled_point = true;

                    self.weights = utils::add(
                        &self.weights,
                        &utils::scale(data_point.label as f64, &self.weights),
                    );

                    self.bias = self.bias + data_point.label as f64;

                    steps.push(PerceptronSteps {
                        weights: self.weights.clone(),
                        bias: self.bias.clone(),
                    });
                }
                // no update necessary if correct prediction
            }

            if !encountered_mislabelled_point {
                break;
            }
        }

        steps
    }

    pub fn prediction(&self, data_point: &structs::DataPoint) -> f64 {
        utils::dot(&self.weights, &data_point.features) + self.bias
    }
}
