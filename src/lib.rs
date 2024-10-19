use feature_matching::mutual_matching;
use gnc_solver::{solve_rotation_translation, GNCSolverParams};
use graph_pruning::correspondance_graph_pruning;
use kdtree::KdTreePointCloud;
use log::info;
use nalgebra::{Matrix3, Vector3};
use normal_estimation::estimate_normals_and_get_neigbours_indexes;
use point::Point3D;
use point_cloud::PointCloud;
use point_feature_histograms::get_fastest_point_feature_histogram;

use numpy::{Ix2, PyArray, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::{Bound, Python};

pub mod constants;
pub mod feature_matching;
pub mod gnc_solver;
pub mod graph_pruning;
pub mod kdtree;
pub mod normal_estimation;
pub mod point;
pub mod point_cloud;
pub mod point_feature_histograms;

pub fn find_point_cloud_transformation(
    source: PointCloud,
    target: PointCloud,
    voxel_size: f64,
) -> (Matrix3<f64>, Vector3<f64>, Vec<bool>) {
    env_logger::init();
    let source_kdtree = KdTreePointCloud::new(&source);
    let target_kdtree = KdTreePointCloud::new(&target);

    let neibour_search_radius = 3.5 * voxel_size;
    let min_neigbours = 3;
    let min_linearity = 0.99;

    info!("calculating normals");
    let (source_normals, source_neighbours_indexes) = estimate_normals_and_get_neigbours_indexes(
        &source,
        &source_kdtree,
        neibour_search_radius,
        min_neigbours,
        min_linearity,
    );
    let (target_normals, target_neighbours_indexes) = estimate_normals_and_get_neigbours_indexes(
        &target,
        &target_kdtree,
        neibour_search_radius,
        min_neigbours,
        min_linearity,
    );

    info!("calculating histograms");
    let source_feature_histograms =
        get_fastest_point_feature_histogram(&source, &source_normals, &source_neighbours_indexes);
    let target_feature_histograms =
        get_fastest_point_feature_histogram(&target, &target_normals, &target_neighbours_indexes);

    info!("preforming mutual matching");
    let max_number_of_correspondances = 3000;

    let points_correspondances = mutual_matching(
        &source_feature_histograms,
        &target_feature_histograms,
        max_number_of_correspondances,
    );
    info!(
        "found {} mutualy matched correspondances",
        points_correspondances.len()
    );
    let distance_noise_threshold = 1.5 * voxel_size;
    info!("prunning graph");
    let filtered_correspondances = correspondance_graph_pruning(
        &points_correspondances,
        &source,
        &target,
        distance_noise_threshold,
    );

    let gnc_params = GNCSolverParams {
        gnc_factor: 1.4,
        noise_bound: 0.001,
        max_iterations: 100,
        cost_threshold: 0.005,
    };

    let mut source_filtered_points: Vec<Point3D> = Vec::new();
    let mut target_filtered_points: Vec<Point3D> = Vec::new();
    for (source_point_id, target_point_id) in filtered_correspondances.iter() {
        source_filtered_points.push(source.points[*source_point_id as usize].clone());
        target_filtered_points.push(target.points[*target_point_id as usize].clone());
    }

    info!("solving rotation/translation");
    let (rotation, translation, inliers) = solve_rotation_translation(
        &gnc_params,
        &source_filtered_points,
        &target_filtered_points,
    );

    (rotation, translation, inliers)
}

#[pyfunction]
#[pyo3(name = "find_point_cloud_transformation")]
fn find_point_cloud_transformation_py<'py>(
    py: Python<'py>,
    source_points: PyReadonlyArray2<'py, f64>,
    target_points: PyReadonlyArray2<'py, f64>,
    voxel_size: f64,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    let source_point_cloud = PointCloud::from_points(
        source_points
            .as_array()
            .outer_iter()
            .map(|x_y_z| {
                let x_y_z_ = x_y_z.as_slice().unwrap();
                Point3D::new(x_y_z_[0], x_y_z_[1], x_y_z_[2])
            })
            .collect(),
    );
    let target_point_cloud = PointCloud::from_points(
        target_points
            .as_array()
            .outer_iter()
            .map(|x_y_z| {
                let x_y_z_ = x_y_z.as_slice().unwrap();
                Point3D::new(x_y_z_[0], x_y_z_[1], x_y_z_[2])
            })
            .collect(),
    );

    let (rotation, translation, _) =
        find_point_cloud_transformation(source_point_cloud, target_point_cloud, voxel_size);
    let rotation_numpy = unsafe {
        let arr = PyArray::<f64, Ix2>::new_bound(py, [rotation.nrows(), rotation.ncols()], false);

        for r in 0..rotation.nrows() {
            for c in 0..rotation.ncols() {
                arr.uget_raw((r, c)).write(rotation[(r, c)]);
            }
        }
        arr
    };
    let translation_numpy = unsafe {
        let arr =
            PyArray::<f64, Ix2>::new_bound(py, [translation.nrows(), translation.ncols()], false);

        for r in 0..translation.nrows() {
            for c in 0..translation.ncols() {
                arr.uget_raw((r, c)).write(translation[(r, c)]);
            }
        }
        arr
    };
    (rotation_numpy, translation_numpy)
}

#[pymodule]
fn kriss_matcher(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_point_cloud_transformation_py, m)?)
}
