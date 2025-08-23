use std::time::Instant;

use ahash::HashSet;
use nalgebra::Vector3;
use rayon::prelude::*;

use crate::prelude::Point;

fn get_index(point: Vector3<f64>) -> u64 {
    let coord_bit_size = 21;
    let offset = 1 << (coord_bit_size - 1);
    let voxel = point.map(|x| (x.floor() as i32 + offset) as u64);
    voxel.x + (voxel.y << coord_bit_size) + (voxel.z << (2 * coord_bit_size))
}
fn get_point(index: u64) -> Vector3<f64> {
    let coord_bit_size = 21;
    let offset = 1 << (coord_bit_size - 1);
    let mask = (1 << coord_bit_size) - 1;
    Vector3::new(
        index & mask,
        (index >> coord_bit_size) & mask,
        (index >> (2 * coord_bit_size)) & mask,
    )
    .map(|x| (x as i32 - offset) as f64 + 0.5)
}

pub fn downsample_points(points: &[Point], voxel_size: f64) -> Vec<Point> {
    println!(
        "Downsampling: {} points, voxel size: {}",
        points.len(),
        voxel_size
    );
    let start = Instant::now();
    let result = downsample_points_a(points, voxel_size);
    println!(
        "Finished downsampling in {:?}, {} points",
        start.elapsed(),
        result.len()
    );
    result
}

#[allow(dead_code)]
fn downsample_points_a(points: &[Point], voxel_size: f64) -> Vec<Point> {
    let inv_voxel_size = 1.0 / voxel_size;
    let mut points = points
        .par_iter()
        .map(|point| get_index(point.coords * inv_voxel_size))
        .collect::<Vec<_>>();
    points.par_sort_unstable();
    points.dedup();
    points
        .into_par_iter()
        .map(|index| Point::from(get_point(index) * voxel_size))
        .collect()
}
#[allow(dead_code)]
fn downsample_points_b(points: &[Point], voxel_size: f64) -> Vec<Point> {
    let inv_voxel_size = 1.0 / voxel_size;
    points
        .par_iter()
        .map(|point| get_index(point.coords * inv_voxel_size))
        .collect::<HashSet<_>>()
        .into_par_iter()
        .map(|index| Point::from(get_point(index) * voxel_size))
        .collect()
}
