//! Principal component analysis via the covariance matrix and the Jacobi
//! eigenvalue algorithm (exact for small, symmetric matrices).

use wasm_bindgen::prelude::*;

/// Eigen-decomposition of a symmetric `d × d` matrix (row-major).
/// Returns eigenvalues (descending) and matching unit eigenvectors as rows.
pub fn symmetric_eigen(mat: &[f64], d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = mat.to_vec();
    let mut v = vec![0.0; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..d).flat_map(|i| (0..d).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i * d + j].powi(2)).sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..d {
            for q in p + 1..d {
                let apq = a[p * d + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..d {
                    let akp = a[k * d + p];
                    let akq = a[k * d + q];
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                for k in 0..d {
                    let apk = a[p * d + k];
                    let aqk = a[q * d + k];
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
                for k in 0..d {
                    let vkp = v[k * d + p];
                    let vkq = v[k * d + q];
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| a[j * d + j].total_cmp(&a[i * d + i]));
    let values = order.iter().map(|&i| a[i * d + i]).collect();
    let mut vectors = Vec::with_capacity(d * d);
    for &i in &order {
        // Column i of V is the eigenvector; canonical sign: first nonzero > 0.
        let mut col: Vec<f64> = (0..d).map(|k| v[k * d + i]).collect();
        if col.iter().find(|x| x.abs() > 1e-12).copied().unwrap_or(1.0) < 0.0 {
            col.iter_mut().for_each(|x| *x = -*x);
        }
        vectors.extend(col);
    }
    (values, vectors)
}

/// PCA of `d`-dimensional points (flat, stride `d`).
///
/// Returns `[mean (d), covariance (d·d), eigenvalues (d, descending),
/// eigenvectors (d·d, one per row)]`. Uses the population covariance (1/n).
#[wasm_bindgen]
pub fn pca(points: &[f64], d: usize) -> Vec<f64> {
    let d = d.max(1);
    let n = points.len() / d;
    let mut mean = vec![0.0; d];
    for p in points.chunks_exact(d) {
        for k in 0..d {
            mean[k] += p[k];
        }
    }
    mean.iter_mut().for_each(|m| *m /= n.max(1) as f64);
    let mut cov = vec![0.0; d * d];
    for p in points.chunks_exact(d) {
        for i in 0..d {
            for j in 0..d {
                cov[i * d + j] += (p[i] - mean[i]) * (p[j] - mean[j]);
            }
        }
    }
    cov.iter_mut().for_each(|c| *c /= n.max(1) as f64);
    let (values, vectors) = symmetric_eigen(&cov, d);
    let mut out = mean;
    out.extend(cov);
    out.extend(values);
    out.extend(vectors);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_gaussian;

    #[test]
    fn recovers_principal_axis() {
        let angle = 0.6f64;
        let pts = make_gaussian(4000, 0.2, -0.1, 0.5, 0.1, angle, 3);
        let r = pca(&pts, 2);
        let (l1, l2) = (r[6], r[7]);
        assert!(l1 > l2);
        assert!((l1.sqrt() - 0.5).abs() < 0.03);
        let (vx, vy) = (r[8], r[9]);
        let cos = (vx * angle.cos() + vy * angle.sin()).abs();
        assert!(cos > 0.999, "cos {cos}");
        // Eigenvectors orthonormal.
        let dot = r[8] * r[10] + r[9] * r[11];
        assert!(dot.abs() < 1e-9);
    }

    #[test]
    fn three_dimensional_eigen() {
        let m = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 1.0];
        let (vals, vecs) = symmetric_eigen(&m, 3);
        for e in 0..3 {
            let v = &vecs[e * 3..e * 3 + 3];
            for i in 0..3 {
                let mv: f64 = (0..3).map(|j| m[i * 3 + j] * v[j]).sum();
                assert!((mv - vals[e] * v[i]).abs() < 1e-9);
            }
        }
    }
}
