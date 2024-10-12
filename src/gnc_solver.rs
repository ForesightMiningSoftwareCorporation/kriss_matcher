use std::f64::INFINITY;

use log::{info, warn};
use nalgebra::{DMatrix, DVector, Matrix3, Matrix3xX, Vector3};
use nalgebra_lapack::SVD;

use crate::point::Point3D;

pub struct GNCSolverParams {
    pub gnc_factor: f64,
    pub noise_bound: f64,
    pub max_iterations: usize,
    pub cost_threshold: f64,
}

fn convert_points_to_matrix(points: &Vec<Point3D>) -> Matrix3xX<f64> {
    let matrix = points.into_iter().map(|p| p.to_vec()).flatten().collect();
    Matrix3xX::from_vec(matrix)
}

// TODO: rewrite to use entire TEASER++ solver

// rust adaptation of https://web.archive.org/web/20241009152000/
// https://github.com/MIT-SPARK/TEASER-plusplus/blob/
// 9ca20d9b52fcb631e7f8c9e3cc55c5ba131cc4e6/teaser/src/registration.cc#L730-L832
pub fn solve_rotation(
    params: &GNCSolverParams,
    source_points: &Vec<Point3D>,
    target_points: &Vec<Point3D>,
) -> (Matrix3<f64>, Vector3<f64>, Vec<bool>) {
    let source = convert_points_to_matrix(&source_points);
    let target = convert_points_to_matrix(&target_points);

    let match_size = source_points.len();
    let mut mu = 1.0;

    let mut prev_cost = INFINITY;
    let mut noise_bound_sq = params.noise_bound.powi(2);

    if noise_bound_sq < 1e-16 {
        noise_bound_sq = 1e-2
    }

    let mut weights: DVector<f64> = DVector::from_vec(vec![1.0 as f64; match_size]);
    let mut rotation = Matrix3::<f64>::identity();
    let mut translation = Vector3::<f64>::zeros();
    for i in 0..=params.max_iterations {
        if true {
            (rotation, translation) = svd_solve_rotation_translation(&source, &target, &weights);
        } else {
            rotation = svd_solve_rotation(&source, &target, &weights);
        }

        let diffs = (&target - (rotation * &source + translation)).map(|elm| elm.powi(2));
        let residuals_sq = diffs.row_sum().transpose();
        if i == 0 {
            let max_residual = residuals_sq
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            mu = 1.0 / (2.0 * max_residual / noise_bound_sq - 1.0);
            if mu <= 0.0 {
                warn!(
                    "GNC-TLS terminated because maximum residual at initialization is very small."
                );
                break;
            }
        }
        let th1 = (mu + 1.0) / mu * noise_bound_sq;
        let th2 = mu / (mu + 1.0) * noise_bound_sq;
        let mut cost = 0.0;
        for j in 0..match_size {
            let residual_sq_j = residuals_sq[j];
            cost += weights[j] * residual_sq_j;
            if residual_sq_j >= th1 {
                weights[j] = 0.0;
            } else if residual_sq_j <= th2 {
                weights[j] = 1.0;
            } else {
                weights[j] = (noise_bound_sq * mu * (mu + 1.0) / residual_sq_j).sqrt() - mu;
                assert!(weights[j] >= 0.0 && weights[j] <= 1.0);
            }
        }
        let cost_diff = (cost - prev_cost).abs();
        mu *= params.gnc_factor;
        prev_cost = cost;
        if cost_diff < params.cost_threshold {
            info!("GNC-TLS solver terminated due to cost convergence. Cost diff: {cost_diff}, iteration: {i}");
            break;
        }
    }
    let mut inliers = vec![false; weights.ncols()];
    for j in 0..weights.ncols() {
        inliers[j] = weights[j] >= 0.5;
    }
    (rotation, translation, inliers)
}

fn svd_solve_rotation(
    source: &Matrix3xX<f64>,
    target: &Matrix3xX<f64>,
    weights: &DVector<f64>,
) -> Matrix3<f64> {
    let w_diag = DMatrix::from_diagonal(&weights);

    let h = source * w_diag * target.transpose();

    let svd_solution = SVD::new(h);
    match svd_solution {
        Some(svd) => {
            let u = svd.u;
            let mut v = svd.vt.transpose();
            if u.determinant() * v.determinant() < 0.0 {
                v.set_column(2, &(-v.column(2).clone_owned()));
            }
            let result = v * u.transpose();
            result
        }
        None => {
            panic!("Unable to solve SVD for GNC");
        }
    }
}

// adaptation of https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
fn svd_solve_rotation_translation(
    source: &Matrix3xX<f64>,
    target: &Matrix3xX<f64>,
    weights: &DVector<f64>,
) -> (Matrix3<f64>, Vector3<f64>) {
    assert_eq!(source.ncols(), target.ncols());

    let n = source.ncols();
    assert_eq!(weights.len(), n);

    let weights_sum = weights.sum();

    if weights_sum <= 1e-16 {
        panic!("Unexpectedly small sum of weights");
    }
    let source_centroid = (source * weights) / weights_sum;
    let target_centroid = (target * weights) / weights_sum;

    let source_centered = source - source_centroid;
    let target_centered = target - target_centroid;

    let w_diag = DMatrix::from_diagonal(&weights);

    let h = source_centered * w_diag * target_centered.transpose();

    let svd_solution = SVD::new(h);
    match svd_solution {
        Some(svd) => {
            let u = svd.u;
            let mut v = svd.vt.transpose();
            if u.determinant() * v.determinant() < 0.0 {
                v.set_column(2, &(-v.column(2).clone_owned()));
            }
            let rotation = v * u.transpose();
            let translation = target_centroid - rotation * source_centroid;
            (rotation, translation)
        }
        None => {
            panic!("Unable to solve SVD for GNC");
        }
    }
}
