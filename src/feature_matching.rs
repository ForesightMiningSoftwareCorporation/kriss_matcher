use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use ahash::HashMap;
use hora::core::ann_index::ANNIndex;
use hora::core::metrics::Metric;
use hora::core::node::Node;
use hora::index::hnsw_idx::HNSWIndex;
use hora::index::hnsw_params::HNSWParams;
use rayon::prelude::*;
use rstar::RTree;
use rstar::primitives::GeomWithData;

use crate::prelude::*;

type HistogramQuery = HNSWIndex<f64, u64>;

// type HistogramQuery = RTree<GeomWithData<[f64; HISTOGRAM_DIM], u64>>;

fn make_query_from_histograms(histograms: &[Histogram]) -> HistogramQuery {
    let mut index = HNSWIndex::new(HISTOGRAM_DIM, &HNSWParams::default());
    for (i, h) in histograms.iter().enumerate() {
        index.add_node(&Node::new_with_idx(h, i as u64)).unwrap();
    }
    index.build(Metric::Euclidean).unwrap();
    index
}

fn take2<T>(mut iter: impl Iterator<Item = T>) -> [T; 2] {
    let a = iter.next().unwrap();
    let b = iter.next().unwrap();
    [a, b]
}

// Threshold. Rename later.
const THR_RATIO_TEST: f64 = 0.9;
const THR_DIST: f64 = 60.0;
const SQR_THR_DIST: f64 = THR_DIST * THR_DIST;

pub fn mutual_matching(
    source_feature_histograms: &[Histogram],
    target_feature_histograms: &[Histogram],
    source_point_indices: &[usize],
    target_point_indices: &[usize],
    max_number_of_correspondances: usize,
) -> Vec<(u64, u64)> {
    mutual_matching_a(
        source_feature_histograms,
        target_feature_histograms,
        source_point_indices,
        target_point_indices,
        max_number_of_correspondances,
    )
}

#[allow(dead_code)]
fn mutual_matching_a(
    source_feature_histograms: &[Histogram],
    target_feature_histograms: &[Histogram],
    source_point_indices: &[usize],
    target_point_indices: &[usize],
    max_number_of_correspondances: usize,
) -> Vec<(u64, u64)> {
    println!(
        "Starting matching. Source length: {}, Target length: {}",
        source_feature_histograms.len(),
        target_feature_histograms.len()
    );
    let source_query = make_query_from_histograms(source_feature_histograms);
    let target_query = make_query_from_histograms(target_feature_histograms);

    let start = Instant::now();

    let source_to_target = std::iter::repeat_with(|| AtomicU32::new(u32::MAX))
        .take(source_feature_histograms.len())
        .collect::<Vec<_>>();
    let target_to_source = std::iter::repeat_with(|| AtomicU32::new(u32::MAX))
        .take(target_feature_histograms.len())
        .collect::<Vec<_>>();

    let ratios_target = target_feature_histograms
        .par_iter()
        .enumerate()
        .map(|(target_index, target_histogram)| {
            let source_points = source_query.search_nodes(target_histogram, 2);
            let source_points = take2(source_points.into_iter());
            let distances = source_points.each_ref().map(|(_, d)| d * d);
            let ratio = distances[0] / distances[1];
            // distances[0] > SQR_THR_DIST ||
            // results in it overpruning
            if distances[0] > SQR_THR_DIST || ratio > THR_RATIO_TEST || !ratio.is_finite() {
                return f64::INFINITY;
            }
            let nearest_source_index = source_points[0].0.idx().unwrap() as usize;
            let nearest_source_histogram = source_points[0].0.vectors();
            if source_to_target[nearest_source_index].load(Ordering::Relaxed) == u32::MAX {
                source_to_target[nearest_source_index].store(
                    target_query.search(&nearest_source_histogram, 1)[0] as u32,
                    Ordering::Relaxed,
                );
            }
            target_to_source[target_index].store(nearest_source_index as u32, Ordering::Relaxed);
            ratio
        })
        .collect::<Vec<_>>();

    println!("Elapsed (matching): {:?}", start.elapsed());

    let start = Instant::now();

    let mut matched_pairs = Vec::with_capacity(source_feature_histograms.len());
    for (target_index, source_index) in target_to_source.iter().enumerate() {
        let target_index = target_index as u32;
        let source_index = source_index.load(Ordering::Relaxed);
        if source_index != u32::MAX
            && target_index == source_to_target[source_index as usize].load(Ordering::Relaxed)
        {
            let ratio = ratios_target[target_index as usize];
            matched_pairs.push((source_index, target_index, ratio));
        }
    }

    println!("Number of matched pairs: {}", matched_pairs.len());

    println!("Elapsed (pairing): {:?}", start.elapsed());
    matched_pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    matched_pairs
        .into_iter()
        .take(max_number_of_correspondances)
        .map(|(s, t, _)| {
            (
                source_point_indices[s as usize] as u64,
                target_point_indices[t as usize] as u64,
            )
        })
        .collect()
}
