use nalgebra::{Isometry3, Matrix3, UnitQuaternion, Vector3};
use std::fs::read_to_string;
use std::time::Instant;

use crate::downsample::downsample_points;
use crate::feature_matching::mutual_matching;
use crate::gnc_solver::{GNCSolverParams, solve_rotation_translation};
use crate::graph_pruning::correspondance_graph_pruning;
use crate::normal_estimation::estimate_normals;
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

pub mod downsample;
pub mod feature_matching;
pub mod gnc_solver;
pub mod graph_pruning;
pub mod normal_estimation;
pub mod point_feature_histograms;
pub mod query;

#[derive(Debug, Clone, Copy)]
pub struct KrissMatcherConfig {
    pub voxel_size: f64,
    pub use_voxel_sampling: bool,
    // pub use_quatro: bool,
    pub max_linearity: f64,
    pub num_max_corr: usize,
    // Below params just works in general cases
    pub normal_radius_gain: f64,
    pub fpfh_radius_gain: f64,
    // The smaller, more conservative
    pub robin_noise_bound_gain: f64,
    pub solver_noise_bound_gain: f64,
    pub enable_noise_bound_clamping: bool,
    // Unknown
    // pub robin_mode: RobinMode,
    pub tuple_scale: f64,
    // Always true
    // pub use_ratio_test: bool,
}
impl KrissMatcherConfig {
    pub fn build(self) -> KrissMatcher {
        assert!(self.solver_noise_bound_gain < self.robin_noise_bound_gain);
        KrissMatcher { config: self }
    }
}
impl Default for KrissMatcherConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.3,
            use_voxel_sampling: true,
            // use_quatro: false,
            max_linearity: 1.0,
            num_max_corr: 5000,
            normal_radius_gain: 3.0,
            fpfh_radius_gain: 5.0,
            robin_noise_bound_gain: 1.0, // 1.5 originally here
            solver_noise_bound_gain: 0.75,
            enable_noise_bound_clamping: true,
            // robin_mode: RobinMode::MaxCore,
            tuple_scale: 0.95,
            // use_ratio_test: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RegistrationSolution {
    pub valid: bool,
    pub translation: Vector3<f64>,
    pub rotation: Matrix3<f64>,
}
impl RegistrationSolution {
    pub fn as_isometry(&self) -> Result<Isometry3<f64>, Isometry3<f64>> {
        let iso = Isometry3::from_parts(
            self.translation.into(),
            UnitQuaternion::from_matrix(&self.rotation),
        );
        if self.valid { Ok(iso) } else { Err(iso) }
    }
}

fn read_file_for_points(path: &str) -> (Vec<Point>, Vec<Histogram>) {
    let file = read_to_string(path).unwrap();
    let mut points = vec![];
    let mut histograms = vec![];
    for line in file.lines() {
        let mut parts = line.split_whitespace().map(|s| s.parse::<f64>().unwrap());
        let point = Point::new(
            parts.next().unwrap(),
            parts.next().unwrap(),
            parts.next().unwrap(),
        );
        let histogram = std::array::from_fn(|_| parts.next().unwrap());
        points.push(point);
        histograms.push(histogram);
    }
    (points, histograms)
}

fn read_normals_from_file(path: &str) -> Vec<Option<Vector3<f64>>> {
    let file = read_to_string(path).unwrap();
    file.lines()
        .map(|line| {
            let mut parts = line.split_whitespace().map(|s| s.parse::<f64>().unwrap());
            let _index = parts.next().unwrap();
            let x = parts.next().unwrap();
            let y = parts.next().unwrap();
            let z = parts.next().unwrap();
            if x.is_nan() || y.is_nan() || z.is_nan() {
                None
            } else {
                Some(Vector3::new(x, y, z))
            }
        })
        .collect::<Vec<_>>()
}

#[derive(Debug)]
pub struct KrissMatcher {
    config: KrissMatcherConfig,
}
impl KrissMatcher {
    pub fn estimate(&mut self, source: &[Point], target: &[Point]) -> RegistrationSolution {
        let config = self.config;
        let source_vec;
        let target_vec;
        let (source, target) = if config.use_voxel_sampling {
            source_vec = Some(downsample_points(source, config.voxel_size));
            target_vec = Some(downsample_points(target, config.voxel_size));
            (
                source_vec.as_ref().unwrap().as_slice(),
                target_vec.as_ref().unwrap().as_slice(),
            )
        } else {
            (source, target)
        };

        println!("Building queries");
        let start = Instant::now();
        let source_query = create_point_query(source);
        println!("Elapsed (source): {:?}", start.elapsed());
        let start = Instant::now();
        let target_query = create_point_query(target);
        println!("Elapsed (target): {:?}", start.elapsed());

        let normal_search_radius = config.normal_radius_gain * config.voxel_size;

        println!("calculating source normals");
        let start = Instant::now();
        let source_normals = estimate_normals(
            source,
            &source_query,
            normal_search_radius,
            config.max_linearity,
        );

        println!("Elapsed: {:?}", start.elapsed());
        println!("calculating target normals");
        let start = Instant::now();
        let target_normals = estimate_normals(
            target,
            &target_query,
            normal_search_radius,
            config.max_linearity,
        );
        println!("Elapsed: {:?}", start.elapsed());

        let fpfh_search_radius = config.fpfh_radius_gain * config.voxel_size;

        println!("calculating histograms");
        let start = Instant::now();
        let (source_feature_histograms, source_point_indices) = get_fastest_point_feature_histogram(
            source,
            &source_query,
            &source_normals,
            fpfh_search_radius,
        );
        println!("Elapsed (Source): {:?}", start.elapsed());

        let start = Instant::now();
        let (target_feature_histograms, target_point_indices) = get_fastest_point_feature_histogram(
            target,
            &target_query,
            &target_normals,
            fpfh_search_radius,
        );
        println!("Elapsed (Target): {:?}", start.elapsed());

        println!("performing mutual matching");

        let start = Instant::now();
        // ROBINMatching::match
        let points_correspondances = mutual_matching(
            &source_feature_histograms,
            &target_feature_histograms,
            &source_point_indices,
            &target_point_indices,
            config.num_max_corr,
        );
        println!("Elapsed: {:?}", start.elapsed());
        println!(
            "found {} mutualy matched correspondances",
            points_correspondances.len()
        );
        let distance_noise_threshold = config.robin_noise_bound_gain * config.voxel_size;
        println!("pruning graph");
        let start = Instant::now();
        let filtered_correspondances = correspondance_graph_pruning(
            &points_correspondances,
            source,
            target,
            distance_noise_threshold,
        );
        println!("Elapsed: {:?}", start.elapsed());
        println!(
            "Pruned to {} correspondances",
            filtered_correspondances.len()
        );

        let gnc_params = GNCSolverParams {
            gnc_factor: 1.4,
            noise_bound: config.solver_noise_bound_gain * config.voxel_size,
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
        let start = Instant::now();
        // See GncSolver.cpp: RobustRegistrationSolver::solve
        let (rotation, translation, inliers) = solve_rotation_translation(
            &gnc_params,
            &source_filtered_points,
            &target_filtered_points,
        );
        println!("Elapsed: {:?}", start.elapsed());

        // TODO: Failed if there are no inliers.
        RegistrationSolution {
            valid: inliers.iter().any(|&v| v),
            rotation,
            translation,
        }
    }
}
