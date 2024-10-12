use feature_matching::mutual_matching;
use gnc_solver::{solve_rotation, GNCSolverParams};
use graph_pruning::correspondance_graph_pruning;
use kdtree::KdTreePointCloud;
use nalgebra::{Matrix3, Vector3};
use normal_estimation::estimate_normals_and_get_neigbours_indexes;
use point::Point3D;
use point_cloud::PointCloud;
use point_feature_histograms::get_fastest_point_feature_histogram;

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
    let source_kdtree = KdTreePointCloud::new(&source);
    let target_kdtree = KdTreePointCloud::new(&target);

    let neibour_search_radius = 3.5 * voxel_size;
    let min_neigbours = 3;
    let min_linearity = 0.99;

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

    let source_feature_histograms =
        get_fastest_point_feature_histogram(&source, &source_normals, &source_neighbours_indexes);
    let target_feature_histograms =
        get_fastest_point_feature_histogram(&target, &target_normals, &target_neighbours_indexes);

    let max_neigbour_distance = 5.0 * voxel_size;
    let max_number_of_correspondances = 3000;

    let points_correspondances = mutual_matching(
        &source_feature_histograms,
        &target_feature_histograms,
        max_neigbour_distance,
        max_number_of_correspondances,
    );

    let distance_noise_threshold = 1.5 * voxel_size;
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
    let (rotation, translation, inliers) = solve_rotation(
        &gnc_params,
        &source_filtered_points,
        &target_filtered_points,
    );

    (rotation, translation, inliers)
}
