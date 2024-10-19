use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use log::debug;

use crate::constants::HISTOGRAM_NUM_BINS;

// To fix the issue mentioned in https://github.com/sdd/kiddo/issues/78
const BUCKET_SIZE: usize = 256;
type HistogramKdTree<A, const K: usize> = KdTree<A, u64, K, BUCKET_SIZE, u32>;

fn make_kdtree_from_histograms(
    histograms: &[Option<Vec<f64>>],
) -> HistogramKdTree<f64, { HISTOGRAM_NUM_BINS * 3 }> {
    let mut tree: HistogramKdTree<f64, { HISTOGRAM_NUM_BINS * 3 }> = HistogramKdTree::new();
    for (i, histogram_opt) in histograms.iter().enumerate() {
        if let Some(histogram) = histogram_opt {
            if histogram.len() != HISTOGRAM_NUM_BINS * 3 {
                panic!("Unexpected length of histogram")
            }
            let histogram_fixed: [f64; HISTOGRAM_NUM_BINS * 3] =
                histogram.as_slice().try_into().unwrap();
            debug!("Add histogram: {:?}", histogram_fixed);
            tree.add(&histogram_fixed, i as u64);
        }
    }
    tree
}

fn match_points(
    source_feature_histograms: &[Option<Vec<f64>>],
    target_kdtree: &HistogramKdTree<f64, { HISTOGRAM_NUM_BINS * 3 }>,
) -> std::collections::HashMap<u64, u64> {
    let mut source_to_target = Vec::new();
    for (source_index, source_histogram_opt) in source_feature_histograms.iter().enumerate() {
        if let Some(source_histogram) = source_histogram_opt {
            let source_histogram_fixed: [f64; HISTOGRAM_NUM_BINS * 3] =
                source_histogram.as_slice().try_into().unwrap();
            let neighbour = target_kdtree.nearest_one::<SquaredEuclidean>(&source_histogram_fixed);
            let neighbour_index = neighbour.item;
            source_to_target.push((source_index as u64, neighbour_index));
        }
    }
    let result: std::collections::HashMap<u64, u64> = source_to_target.into_iter().collect();
    result
}

fn descriptor_distance_ration(
    source_histogram: &Vec<f64>,
    target_kdtree: &HistogramKdTree<f64, { HISTOGRAM_NUM_BINS * 3 }>,
) -> Option<f64> {
    let histogram_fixed: [f64; HISTOGRAM_NUM_BINS * 3] =
        source_histogram.as_slice().try_into().unwrap();
    let nearest_neighbours = target_kdtree.nearest_n::<SquaredEuclidean>(&histogram_fixed, 2);
    if nearest_neighbours.len() != 2 {
        return None;
    }
    let ratio = nearest_neighbours[0].distance / nearest_neighbours[1].distance;
    Some(ratio)
}

pub fn mutual_matching(
    source_feature_histograms: &[Option<Vec<f64>>],
    target_feature_histograms: &[Option<Vec<f64>>],
    max_number_of_correspondances: usize, // XXX: in paper they propose 3000
) -> Vec<(u64, u64)> {
    let source_kdtree = make_kdtree_from_histograms(source_feature_histograms);
    let target_kdtree = make_kdtree_from_histograms(target_feature_histograms);
    let mut correspondance_with_ration = Vec::new();

    let source_to_target = match_points(source_feature_histograms, &target_kdtree);
    let target_to_source = match_points(target_feature_histograms, &source_kdtree);
    debug!("source to target: {:?}", source_to_target);
    debug!("target to source: {:?}", target_to_source);
    for (&source_index, &target_index) in &source_to_target {
        if let Some(&matched_index) = target_to_source.get(&target_index) {
            if matched_index == source_index {
                if let Some(source_histogram) =
                    source_feature_histograms[source_index as usize].as_ref()
                {
                    let ratio_opt = descriptor_distance_ration(source_histogram, &target_kdtree);
                    if let Some(ratio) = ratio_opt {
                        correspondance_with_ration.push((source_index, target_index, ratio));
                    }
                }
            }
        }
    }
    correspondance_with_ration.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    correspondance_with_ration
        .into_iter()
        .take(max_number_of_correspondances)
        .map(|(s, t, _)| (s, t))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ok() {
        const HISTOGRAM_NUM_BINS: usize = 11;

        let source_histograms = vec![
            Some(vec![1.0; HISTOGRAM_NUM_BINS * 3]),
            Some(vec![2.0; HISTOGRAM_NUM_BINS * 3]),
            Some(vec![3.0; HISTOGRAM_NUM_BINS * 3]),
        ];

        let target_histograms = vec![
            Some(vec![1.0; HISTOGRAM_NUM_BINS * 3]),
            Some(vec![2.0; HISTOGRAM_NUM_BINS * 3]),
            Some(vec![4.0; HISTOGRAM_NUM_BINS * 3]),
        ];

        let max_number_of_correspondances = 10;

        let correspondences = mutual_matching(
            &source_histograms,
            &target_histograms,
            max_number_of_correspondances,
        );

        let expected_correspondences = [(0u64, 0u64), (1u64, 1u64)];

        assert_eq!(
            correspondences.len(),
            expected_correspondences.len(),
            "Expected {} correspondences, found {}",
            expected_correspondences.len(),
            correspondences.len()
        );

        for &(source_idx, target_idx) in &correspondences {
            assert!(
                expected_correspondences.contains(&(source_idx, target_idx)),
                "Unexpected correspondence ({}, {})",
                source_idx,
                target_idx
            );
        }
    }
}
