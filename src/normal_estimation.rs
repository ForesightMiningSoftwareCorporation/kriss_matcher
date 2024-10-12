use log::warn;
use nalgebra::{DMatrix, Vector3};
use nalgebra_lapack::SVD;

use crate::{kdtree::KdTreePointCloud, point::Point3D, point_cloud::PointCloud};

pub fn estimate_normals_and_get_neigbours_indexes(
    point_cloud: &PointCloud,
    kdtree: &KdTreePointCloud,
    // TODO: execute two radius searches, first with radius_fpfh and
    //       then with radius_normal
    radius: f64,
    min_neigbours: usize,
    min_linearity: f64 // = 0.99,
) -> (Vec<Option<Vector3<f64>>>, Vec<Option<Vec<u64>>>) {
    let mut normals = vec![None; point_cloud.len()];
    let mut all_neigbours_indexes = vec![None; point_cloud.len()];
    for (i, point) in point_cloud.points.iter().enumerate() {
        let neigbours_indexes = kdtree.radius_search(point, radius);
        if neigbours_indexes.len() < min_neigbours {
            normals[i] = None;
            continue;
        }

        // Why not PCA? Well, the matrix shouldn't be big, so perfomance shouldn't
        // be an issue. SVD, on the other hand, should give more stable results.
        let centroid = calculate_centroid(point_cloud, &neigbours_indexes);

        let mut normalized_surface = DMatrix::zeros(neigbours_indexes.len(), 3);
        for (row, &index) in neigbours_indexes.iter().enumerate() {
            let neigbour = &point_cloud.points[index as usize];
            normalized_surface[(row, 0)] = neigbour.x - centroid.x;
            normalized_surface[(row, 1)] = neigbour.y - centroid.y;
            normalized_surface[(row, 2)] = neigbour.z - centroid.z;
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
                    warn!("Liniarity ({linearity})  is higher than {tau_lin}");
                    normals[i] = None;
                    continue;
                }


                let v_t = svd.vt;
                let normal = Vector3::new(v_t[(2, 0)], v_t[(2, 1)], v_t[(2, 2)]).normalize();
                normals[i] = Some(normal);
                all_neigbours_indexes[i] = Some(neigbours_indexes);
            }
            None => {
                warn!("Unable to solve SVD at {i}")
            }
        }
    }
    (normals, all_neigbours_indexes)
}

fn calculate_centroid(point_cloud: &PointCloud, indices: &Vec<u64>) -> Point3D {
    let mut centroid = Point3D::zero();
    let n = indices.len() as f64;
    for index in indices.iter() {
        let point = &point_cloud.points[*index as usize];
        centroid.x += point.x;
        centroid.y += point.y;
        centroid.z += point.z;
    }
    centroid.x /= n;
    centroid.y /= n;
    centroid.z /= n;
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
                points.push(Point3D::new(x as f64, y as f64, 0.0));
            }
        }
        let point_cloud = PointCloud::from_points(points);
        let kdtree = KdTreePointCloud::new(&point_cloud);
        let min_linearity = 0.99;
        let (normals, _) = estimate_normals_and_get_neigbours_indexes(&point_cloud, &kdtree, 1.5, 3, min_linearity);

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
        let points = vec![Point3D::new(1.0, 2.0, 3.0), Point3D::zero()];
        let point_cloud = PointCloud::from_points(points);
        let kdtree = KdTreePointCloud::new(&point_cloud);

        let radius = 0.05;
        let min_neigbours = 3;
        let min_linearity = 0.3;

        let (normals, _) = estimate_normals_and_get_neigbours_indexes(&point_cloud, &kdtree, radius, min_neigbours, min_linearity);
        assert_eq!(normals.len(), 2);
        assert_eq!(normals[0], None);
        assert_eq!(normals[1], None);

    }
}
