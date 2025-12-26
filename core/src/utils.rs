pub fn dot(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "Dimension mismatch in dot product. Vector a has dimension {}, but vector b has dimension {}.",
        a.len(),
        b.len(),
    );

    a.iter().zip(b.iter()).map(|(a_i, b_i)| a_i * b_i).sum()
}

pub fn scale(scaler: f64, vec: &Vec<f64>) -> Vec<f64> {
    vec.iter().map(|vec_i| vec_i * scaler).collect()
}

pub fn add(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(a_i, b_i)| a_i + b_i).collect()
}
