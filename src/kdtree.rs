use crate::{
    point::Point3D,
    point_cloud::PointCloud,
};
use kiddo::{KdTree, SquaredEuclidean};

pub struct KdTreePointCloud {
    pub kdtree: KdTree<f64, 3>,
}

impl KdTreePointCloud {
    pub fn new(point_cloud: &PointCloud) -> Self {
        let mut kdtree: KdTree<_, 3> = KdTree::new();
        point_cloud
            .points
            .iter()
            .enumerate()
            .for_each(|(i, point)| {
                kdtree.add(&point.to_vec(), i as u64);
            });
        Self { kdtree }
    }

    pub fn radius_search(&self, query_point: &Point3D, radius: f64) -> Vec<u64>{
        let neigbours = self.kdtree.within::<SquaredEuclidean>(&query_point.to_vec(), radius);
        neigbours.iter().map(|n| n.item ).collect()
    }
}
