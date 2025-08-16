use log::info;
use nalgebra::{Matrix3, Vector3};

use crate::feature_matching::mutual_matching;
use crate::gnc_solver::{solve_rotation_translation, GNCSolverParams};
use crate::graph_pruning::correspondance_graph_pruning;
use crate::normal_estimation::estimate_normals_and_get_neighbor_indices;
use crate::point_feature_histograms::get_fastest_point_feature_histogram;
use crate::prelude::*;
use crate::query::create_point_query;

pub mod prelude {
    // there is no information on the value of bin size H in the paper.
    // I found two possible values, 5 from the original paper of FPFH
    // https://web.archive.org/web/20240906202141/
    // https://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf
    // and 11 from PCL https://web.archive.org/web/20240429124409/
    // https://pcl.readthedocs.io/projects/tutorials/en/latest/fpfh_estimation.html
    pub const HISTOGRAM_NUM_BINS: usize = 11;
    pub const HISTOGRAM_DIM: usize = HISTOGRAM_NUM_BINS * 3;

    use nalgebra::Point3;

    pub type Point = Point3<f64>;
    pub type Histogram = [f64; HISTOGRAM_DIM];
}

pub mod feature_matching;
pub mod gnc_solver;
pub mod graph_pruning;
pub mod normal_estimation;
pub mod point_feature_histograms;
pub mod query;

pub fn find_point_cloud_transformation(
    source: &[Point],
    target: &[Point],
    voxel_size: f64,
) -> (Matrix3<f64>, Vector3<f64>, Vec<bool>) {
    let source_query = create_point_query(source);
    let target_query = create_point_query(target);

    let neighbor_search_radius = 3.5 * voxel_size;
    let min_neigbours = 3;
    let min_linearity = 0.99;

    println!("calculating source normals");
    let (source_normals, source_neighbors_indexes) = estimate_normals_and_get_neighbor_indices(
        source,
        &source_query,
        neighbor_search_radius,
        min_neigbours,
        min_linearity,
    );
    println!("calculating target normals");
    let (target_normals, target_neighbors_indexes) = estimate_normals_and_get_neighbor_indices(
        target,
        &target_query,
        neighbor_search_radius,
        min_neigbours,
        min_linearity,
    );

    println!("calculating histograms");
    let source_feature_histograms =
        get_fastest_point_feature_histogram(source, &source_normals, &source_neighbors_indexes);
    let target_feature_histograms =
        get_fastest_point_feature_histogram(target, &target_normals, &target_neighbors_indexes);

    println!("preforming mutual matching");
    let max_number_of_correspondances = 3000;

    let points_correspondances = mutual_matching(
        &source_feature_histograms,
        &target_feature_histograms,
        max_number_of_correspondances,
    );
    println!(
        "found {} mutualy matched correspondances",
        points_correspondances.len()
    );
    let distance_noise_threshold = 1.5 * voxel_size;
    println!("prunning graph");
    let filtered_correspondances = correspondance_graph_pruning(
        &points_correspondances,
        source,
        target,
        distance_noise_threshold,
    );

    let gnc_params = GNCSolverParams {
        gnc_factor: 1.4,
        noise_bound: 0.001,
        max_iterations: 100,
        cost_threshold: 0.005,
    };

    let mut source_filtered_points: Vec<Point> = Vec::new();
    let mut target_filtered_points: Vec<Point> = Vec::new();
    for (source_point_id, target_point_id) in filtered_correspondances.iter() {
        source_filtered_points.push(source[*source_point_id as usize]);
        target_filtered_points.push(target[*target_point_id as usize]);
    }

    println!("solving rotation/translation");
    let (rotation, translation, inliers) = solve_rotation_translation(
        &gnc_params,
        &source_filtered_points,
        &target_filtered_points,
    );

    (rotation, translation, inliers)
}
