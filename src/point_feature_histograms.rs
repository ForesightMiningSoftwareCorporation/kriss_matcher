use nalgebra::{Vector3, distance};

use crate::{prelude::*, query::PointQuery};

fn compute_features(
    point_a: &Point,
    point_b: &Point,
    normal_a: &Vector3<f64>,
    normal_b: &Vector3<f64>,
) -> [f64; 3] {
    let direction_vector = Vector3::new(
        point_b.x - point_a.x,
        point_b.y - point_a.y,
        point_b.z - point_a.z,
    )
    .normalize();
    let u = normal_a;
    let v = direction_vector.cross(u).normalize();
    let w = u.cross(&v).normalize();

    let feature_1 = w.dot(normal_b).atan2(u.dot(normal_b));
    let feature_2 = v.dot(normal_b);
    let feature_3 = u.dot(&direction_vector);
    [feature_1, feature_2, feature_3]
}

fn bin_features(features: &[f64; 3], histogram: &mut Histogram) {
    let epsilon = 1e-6;

    // atan2: [−π,π], cross product of normalized vectors: [-1, 1]
    let f_min = [-std::f64::consts::PI, -1.0, -1.0];
    let f_max = [std::f64::consts::PI, 1.0, 1.0];

    for (l, feature) in features.iter().enumerate() {
        let ratio = (feature - f_min[l]) / (f_max[l] + epsilon - f_min[l]);
        let clamped_ratio = ratio.clamp(0.0, 1.0 - epsilon);
        let bin = (HISTOGRAM_NUM_BINS as f64 * clamped_ratio).floor() as usize;
        histogram[l * HISTOGRAM_NUM_BINS + bin] += 1.0;
    }
}

pub fn get_fastest_point_feature_histogram(
    points: &[Point],
    query: &PointQuery,
    normals: &[Option<Vector3<f64>>],
    radius: f64,
) -> (Vec<Histogram>, Vec<u64>) {
    let sqr_radius = radius * radius;
    let mut spf_histograms = vec![None; points.len()];
    for (index, &point) in points.iter().enumerate() {
        if normals[index].is_none() {
            continue;
        }
        let mut histogram: Histogram = [0.0_f64; HISTOGRAM_DIM];
        let mut num_neighbors = 0;
        for neighbor in query.locate_within_distance(point.into(), sqr_radius) {
            let neighbor_index = neighbor.data as usize;
            if normals[neighbor_index].is_none() || index == neighbor_index {
                continue;
            }
            let features = compute_features(
                &point,
                &(*neighbor.geom()).into(),
                &normals[index].unwrap(),
                &normals[neighbor_index].unwrap(),
            );
            bin_features(&features, &mut histogram);
            num_neighbors += 1;
        }
        if num_neighbors == 0 {
            continue;
        }

        let normalization_scale = 100.0 / num_neighbors as f64;
        for item in histogram.iter_mut() {
            *item *= normalization_scale
        }
        spf_histograms[index] = Some(histogram);
    }

    let mut fpf_histograms = vec![];
    let mut histogram_point_indices = vec![];
    for (index, &point) in points.iter().enumerate() {
        let Some(mut fpf_histogram) = spf_histograms[index] else {
            continue;
        };

        let mut num_neighbors = 0;
        for neighbor in query.locate_within_distance(point.into(), sqr_radius) {
            let neighbor_index = neighbor.data as usize;
            if normals[neighbor_index].is_none() || index == neighbor_index {
                continue;
            }
            let neighbor_point = (*neighbor.geom()).into();
            let distance = distance(&point, &neighbor_point);
            let inv_omega = 1.0 / (distance + 1e-6);
            let neigbour_spf_histogram = spf_histograms[neighbor_index].as_ref().unwrap();

            for (hist_index, value) in neigbour_spf_histogram.iter().enumerate() {
                fpf_histogram[hist_index] += value * inv_omega;
            }
            num_neighbors += 1;
        }
        let normalization_scale = 100.0 / num_neighbors as f64;
        for item in fpf_histogram.iter_mut() {
            *item *= normalization_scale
        }
        fpf_histograms.push(fpf_histogram);
        histogram_point_indices.push(index as u64);
    }

    (fpf_histograms, histogram_point_indices)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_ok() {
        let points = [
            Point::new(0.0, 0.0, 0.0),
            Point::new(1.0, 0.0, 0.0),
            Point::new(0.0, 1.0, 0.0),
        ];
        let normals = vec![
            Some(Vector3::new(0.0, 0.0, 1.0)),
            Some(Vector3::new(0.0, 0.0, 1.0)),
            Some(Vector3::new(0.0, 0.0, 1.0)),
        ];
        let neigbours_indexes = vec![Some(vec![1, 2]), Some(vec![1, 0]), Some(vec![0, 2])];
        let histograms = get_fastest_point_feature_histogram(&points, &normals, &neigbours_indexes);
        for optional_histogram in histograms.iter() {
            assert!(optional_histogram.is_some());
            let histogram = optional_histogram.as_ref().unwrap();
            assert_ne!(histogram, &[0.0_f64; HISTOGRAM_DIM]);
        }
    }
}
