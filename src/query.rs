use crate::prelude::*;
use rstar::{primitives::GeomWithData, RTree};

pub type PointQuery = RTree<GeomWithData<[f64; 3], u64>>;

pub fn create_point_query(points: &[Point]) -> PointQuery {
    RTree::bulk_load(
        points
            .iter()
            .copied()
            .enumerate()
            .map(|(id, p)| GeomWithData::new(p.into(), id as u64))
            .collect::<Vec<_>>(),
    )
}

pub fn search_radius(
    query: &PointQuery,
    center: Point,
    radius: f64,
) -> impl Iterator<Item = u64> + '_ {
    query
        .locate_within_distance(center.into(), radius)
        .map(|n| n.data)
}
