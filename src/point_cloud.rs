use crate::point::Point3D;

pub struct PointCloud {
    pub points: Vec<Point3D>,
}

impl PointCloud {
    pub fn new() -> Self {
        PointCloud { points: Vec::new() }
    }

    pub fn from_points(points: Vec<Point3D>) -> Self {
        PointCloud { points }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }
}
