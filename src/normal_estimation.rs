use nalgebra::linalg::SVD;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use rstar::primitives::GeomWithData;

use crate::prelude::*;
use crate::query::PointQuery;

const MIN_NEIGHBORS: usize = 3;

pub fn estimate_normals(
    points: &[Point],
    query: &PointQuery,
    // TODO: execute two radius searches, first with radius_fpfh and
    //       then with radius_normal
    radius: f64,
    max_linearity: f64, // = 0.99,
) -> Vec<Option<Vector3<f64>>> {
    let sqr_radius = radius * radius;
    points
        .par_iter()
        .map_with(
            Vec::<GeomWithData<[f64; 3], u64>>::new(),
            |neighbors, &point| {
                neighbors.clear();
                neighbors.extend(
                    query
                        .locate_within_distance(point.into(), sqr_radius)
                        .copied(),
                );
                let mut num_neighbors = 0;
                let mut sum = Vector3::zeros();
                for neighbor in &*neighbors {
                    sum += Vector3::from(*neighbor.geom());
                    num_neighbors += 1;
                }
                let centroid = Point::from(sum / num_neighbors as f64);
                if num_neighbors < MIN_NEIGHBORS {
                    return None;
                }

                let mut covariance = Matrix3::zeros();
                for neighbor in &*neighbors {
                    let deviation = Point::from(*neighbor.geom()) - centroid;
                    covariance += deviation * deviation.transpose();
                }
                covariance /= (num_neighbors - 1) as f64;

                let svd_solution = SVD::new(covariance, true, false);
                let singular_values = svd_solution.singular_values;
                let linearity = (singular_values[0] - singular_values[1]) / singular_values[0];
                if linearity > max_linearity {
                    None
                } else {
                    let normal = svd_solution.u.as_ref().unwrap().column(2);
                    Some(if centroid.coords.dot(&normal) > 0.0 {
                        -normal
                    } else {
                        normal.into()
                    })
                }
            },
        )
        .collect()
}
