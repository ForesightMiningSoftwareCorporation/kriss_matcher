use nalgebra::Vector3;

use crate::{point::Point3D, point_cloud::PointCloud, constants::HISTOGRAM_NUM_BINS};

fn compute_features(
    point_a: &Point3D,
    point_b: &Point3D,
    normal_a: &Vector3<f64>,
    normal_b: &Vector3<f64>,
) -> [f64; 3] {
    let direction_vector = Vector3::new(
        point_b.x - point_a.x,
        point_b.y - point_b.y,
        point_b.z - point_a.z,
    )
    .normalize();
    let u = normal_a;
    let v = direction_vector.cross(&u).normalize();
    let w = u.cross(&v).normalize();

    let feature_1 = w.dot(&normal_b).atan2(u.dot(&normal_b));
    let feature_2 = v.dot(&normal_b);
    let feature_3 = u.dot(&direction_vector);
    [feature_1, feature_2, feature_3]
}

fn bin_features(features: &[f64; 3], num_bins: usize, histograms: &mut Vec<f64>) {
    let epsilon = 1e-6;

    // atan2: [−π,π], cross product of normalized vectors: [-1, 1]
    let f_min = [-std::f64::consts::PI, -1.0, -1.0];
    let f_max = [std::f64::consts::PI, 1.0, 1.0];

    let h = num_bins as f64;

    for (l, feature) in features.iter().enumerate() {
        let ratio = (feature - f_min[l]) / (f_max[l] + epsilon - f_min[l]);
        let clamped_ratio = ratio.clamp(0.0, 1.0);
        let bin = (h * clamped_ratio).round() as usize;
        histograms[l * num_bins + bin] += 1.0;
    }
}

pub fn get_fastest_point_feature_histogram(
    point_cloud: &PointCloud,
    normals: &Vec<Option<Vector3<f64>>>,
    neigbours_indexes: &Vec<Option<Vec<u64>>>,
) -> Vec<Option<Vec<f64>>> {

    let mut spf_histograms = vec![None; point_cloud.len()];
    for (index, point) in point_cloud.points.iter().enumerate() {
        if normals[index].is_none() {
            continue;
        }
        let mut histograms = vec![0f64; HISTOGRAM_NUM_BINS * 3];
        // TODO: handle case when not enough neigbors with normals
        if let Some(neigbour_indexes) = &neigbours_indexes[index] {
            for &neigbour_index in neigbour_indexes.iter() {
                if normals[neigbour_index as usize].is_none() || index == neigbour_index as usize {
                    continue;
                }
                let features = compute_features(
                    &point,
                    &point_cloud.points[neigbour_index as usize],
                    &normals[index].unwrap(),
                    &normals[neigbour_index as usize].unwrap(),
                );
                bin_features(&features, HISTOGRAM_NUM_BINS, &mut histograms);
            }
            let normalization_scale = 100.0 / neigbour_indexes.len() as f64 ;
            for item in histograms.iter_mut() {
                *item *= normalization_scale
            }
            spf_histograms[index] = Some(histograms);
        }
    }

    let mut fpf_histograms = vec![None; point_cloud.len()];
    for (index, point) in point_cloud.points.iter().enumerate() {
        if normals[index].is_none() {
            continue;
        }
        if spf_histograms[index].is_none() {
            continue;
        }
        let spf_histogram = spf_histograms[index].as_ref().unwrap();
        // TODO: remove unwrap with proper Some handling
        let mut fpf_histogram = spf_histogram.clone();

        if let Some(neigbour_indexes) = &neigbours_indexes[index] {
            for &neigbour_index in neigbour_indexes.iter() {
                if normals[neigbour_index as usize].is_none() || index == neigbour_index as usize {
                    continue;
                }
                let neighbour_point = &point_cloud.points[neigbour_index as usize];
                let distance = point.distance(neighbour_point);
                let inv_omega = 1.0 / (distance + 1e-6);
                let neigbour_spf_histogram = spf_histograms[neigbour_index as usize].as_ref().unwrap();

                for (hist_index, value) in neigbour_spf_histogram.iter().enumerate() {
                    fpf_histogram[hist_index] += value * inv_omega;
                }


            }
            let normalization_scale = 100.0 / neigbour_indexes.len() as f64 ;
            for item in fpf_histogram.iter_mut() {
                *item *= normalization_scale
            }
            fpf_histograms[index] = Some(fpf_histogram);
        }
    }

   fpf_histograms
}
