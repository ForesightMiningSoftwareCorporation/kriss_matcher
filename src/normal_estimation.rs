use nalgebra::{MatrixXx3, Vector3};
use nalgebra_lapack::SVD;

use crate::prelude::*;
use crate::query::{PointQuery, search_radius};

const MAX_NEIGHBOR_INDICES: usize = 1000;

pub fn estimate_normals_and_get_neighbor_indices(
    point_cloud: &[Point],
    query: &PointQuery,
    // TODO: execute two radius searches, first with radius_fpfh and
    //       then with radius_normal
    radius: f64,
    min_neighbors: usize,
    min_linearity: f64, // = 0.99,
) -> Vec<Option<Vector3<f64>>> {
    let sqr_radius = radius * radius;
    let mut normals = vec![None; point_cloud.len()];
    for (i, point) in point_cloud.iter().copied().enumerate() {
        let neighbor_indices = search_radius(query, point, sqr_radius).collect::<Vec<_>>();
        if neighbor_indices.len() > MAX_NEIGHBOR_INDICES {
            println!(
                "Too many neighbors found for point {i}: {} > {MAX_NEIGHBOR_INDICES}. Perhaps your `voxel_size` is too large?",
                neighbor_indices.len()
            );
        }
        if neighbor_indices.len() < min_neighbors {
            // normals[i] = None;
            continue;
        }

        // Why not PCA? Well, the matrix shouldn't be big, so perfomance shouldn't
        // be an issue. SVD, on the other hand, should give more stable results.
        let centroid = calculate_centroid(point_cloud, &neighbor_indices);

        // TODO: Transpose this; shouldn't change the SVD.
        let mut normalized_surface = MatrixXx3::zeros(neighbor_indices.len());
        for (row, &index) in neighbor_indices.iter().enumerate() {
            let neigbour = &point_cloud[index as usize];
            normalized_surface
                .row_mut(row)
                .copy_from(&(neigbour - centroid).transpose());
        }
        let svd_solution = SVD::new(normalized_surface);
        match svd_solution {
            Some(svd) => {
                let sigma1 = svd.singular_values[0];
                let sigma2 = svd.singular_values[1];
                if sigma1.abs() < 1e-8 {
                    normals[i] = None;
                    continue;
                }
                let linearity = (sigma1 - sigma2) / sigma1;
                let tau_lin = min_linearity;
                if linearity > tau_lin {
                    println!("Linearity ({linearity})  is higher than {tau_lin}");
                    normals[i] = None;
                    continue;
                }

                let v_t = svd.vt;
                let normal = Vector3::new(v_t[(2, 0)], v_t[(2, 1)], v_t[(2, 2)]).normalize();
                normals[i] = Some(normal);
            }
            None => {
                println!("Unable to solve SVD at {i}")
            }
        }
    }
    normals
}

fn calculate_centroid(points: &[Point], indices: &[u64]) -> Point {
    let mut centroid = Point::origin();
    for index in indices.iter() {
        centroid.coords += points[*index as usize].coords;
    }
    centroid.coords /= indices.len() as f64;
    centroid
}

#[cfg(test)]
mod tests {
    use all_asserts::assert_gt;

    use super::*;

    #[test]
    fn test_normal_estimation_plane() {
        env_logger::init();

        let mut points = Vec::new();
        for x in -1..=1 {
            for y in -1..=1 {
                points.push(Point::new(x as f64, y as f64, 0.0));
            }
        }
        let point_cloud = PointCloud { points: &points };
        let kdtree = KdTreePointCloud::new(&point_cloud);
        let min_linearity = 0.99;
        let (normals, _) =
            estimate_normals_and_get_neighbor_indices(&point_cloud, &kdtree, 1.5, 3, min_linearity);

        for possible_normal in normals {
            if let Some(normal) = possible_normal {
                let dot_product = normal.dot(&Vector3::new(0.0, 0.0, 1.0));
                assert_gt!(dot_product.abs(), 0.9);
            } else {
                panic!("normal estimation failed")
            }
        }
    }

    #[test]
    fn test_not_enough_neigbours() {
        let points = vec![Point::new(1.0, 2.0, 3.0), Point::origin()];
        let point_cloud = PointCloud { points: &points };
        let kdtree = KdTreePointCloud::new(&point_cloud);

        let radius = 0.05;
        let min_neigbours = 3;
        let min_linearity = 0.3;

        let (normals, _) = estimate_normals_and_get_neighbor_indices(
            &point_cloud,
            &kdtree,
            radius,
            min_neigbours,
            min_linearity,
        );
        assert_eq!(normals.len(), 2);
        assert_eq!(normals[0], None);
        assert_eq!(normals[1], None);
    }
}
