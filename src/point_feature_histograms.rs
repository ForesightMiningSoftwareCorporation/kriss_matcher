use nalgebra::{Vector3, distance_squared};
use rayon::prelude::*;

use crate::{prelude::*, query::PointQuery};

fn compute_features(
    point_a: Point,
    point_b: Point,
    mut normal_a: Vector3<f64>,
    mut normal_b: Vector3<f64>,
) -> Option<[f64; 3]> {
    let mut direction_vector = (point_b - point_a).try_normalize(f64::EPSILON)?;

    let mut features = [0.0; 3];

    let a1 = normal_a.dot(&direction_vector);
    let a2 = normal_b.dot(&direction_vector);
    if a1.abs().acos() > a2.abs().acos() {
        (normal_a, normal_b) = (normal_b, normal_a);
        direction_vector.neg_mut();
        features[2] = -a2;
    } else {
        features[2] = a1;
    }

    let v = direction_vector
        .cross(&normal_a)
        .try_normalize(f64::EPSILON)?;
    let w = normal_a.cross(&v);
    features[1] = v.dot(&normal_b);
    features[0] = w.dot(&normal_b).atan2(normal_a.dot(&normal_b));
    Some(features)
}

fn bin_features(features: &[f64; 3], histogram: &mut Histogram) {
    // atan2: [−π,π], cross product of normalized vectors: [-1, 1]
    let f_min = [-std::f64::consts::PI, -1.0, -1.0];
    let f_max = [std::f64::consts::PI, 1.0, 1.0];

    for (l, feature) in features.iter().enumerate() {
        let ratio = HISTOGRAM_NUM_BINS as f64 * (feature - f_min[l]) / (f_max[l] - f_min[l]);
        let ratio = ratio.max(0.0);
        let bin = (ratio.floor() as usize).min(HISTOGRAM_NUM_BINS - 1);
        histogram[l * HISTOGRAM_NUM_BINS + bin] += 1.0;
    }
}

pub fn get_fastest_point_feature_histogram(
    points: &[Point],
    query: &PointQuery,
    normals: &[Option<Vector3<f64>>],
    radius: f64,
) -> (Vec<Histogram>, Vec<usize>) {
    let sqr_radius = radius * radius;

    let start = std::time::Instant::now();

    let spf_histograms = points
        .par_iter()
        .enumerate()
        .map(|(index, &point)| {
            normals[index]?;
            let mut histogram: Histogram = [0.0_f64; HISTOGRAM_DIM];
            let mut num_neighbors = 0;
            for neighbor in query.locate_within_distance(point.into(), sqr_radius) {
                let neighbor_index = neighbor.data as usize;
                if normals[neighbor_index].is_none() || index == neighbor_index {
                    continue;
                }
                let features = compute_features(
                    point,
                    (*neighbor.geom()).into(),
                    normals[index].unwrap(),
                    normals[neighbor_index].unwrap(),
                );
                if let Some(features) = features {
                    bin_features(&features, &mut histogram);
                    num_neighbors += 1;
                }
            }
            if num_neighbors == 0 {
                return None;
            }

            let normalization_scale = 100.0 / num_neighbors as f64;
            for item in histogram.iter_mut() {
                *item *= normalization_scale
            }

            Some(histogram)
        })
        .collect::<Vec<_>>();

    println!("FPFH part 1: {:?}", start.elapsed());

    let histogram_point_indices = spf_histograms
        .iter()
        .enumerate()
        .filter_map(|(index, histogram)| histogram.is_some().then_some(index))
        .collect::<Vec<_>>();

    let start = std::time::Instant::now();

    let fpf_histograms = histogram_point_indices
        .par_iter()
        .map(|&index| {
            let point = points[index];
            let mut fpf_histogram = spf_histograms[index].unwrap();

            for neighbor in query.locate_within_distance(point.into(), sqr_radius) {
                let neighbor_index = neighbor.data as usize;
                if index == neighbor_index {
                    continue;
                }

                let Some(neigbour_spf_histogram) = spf_histograms[neighbor_index] else {
                    continue;
                };

                let neighbor_point = (*neighbor.geom()).into();
                let distance = distance_squared(&point, &neighbor_point);

                let inv_omega = 1.0 / distance;

                for (hist_index, value) in neigbour_spf_histogram.iter().enumerate() {
                    fpf_histogram[hist_index] += value * inv_omega;
                }
            }
            let histogram_sums: [f64; 3] = std::array::from_fn(|i| {
                fpf_histogram[i * HISTOGRAM_NUM_BINS..(i + 1) * HISTOGRAM_NUM_BINS]
                    .iter()
                    .sum::<f64>()
            });
            let inv_sums = histogram_sums.map(|s| if s.abs() < 1e-6 { 0.0 } else { 100.0 / s });

            for (i, item) in fpf_histogram.iter_mut().enumerate() {
                *item *= inv_sums[i / HISTOGRAM_NUM_BINS];
            }
            fpf_histogram
        })
        .collect::<Vec<_>>();

    println!("FPFH part 2: {:?}", start.elapsed());

    (fpf_histograms, histogram_point_indices)
}
