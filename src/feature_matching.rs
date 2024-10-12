use kiddo::{KdTree, SquaredEuclidean};

use crate::constants::HISTOGRAM_NUM_BINS;

fn make_kdtree_from_histograms(
    histograms: &Vec<Option<Vec<f64>>>,
) -> KdTree<f64, { HISTOGRAM_NUM_BINS * 3 }> {
    let mut tree: KdTree<f64, { HISTOGRAM_NUM_BINS * 3 }> = KdTree::new();
    for (i, histogram_opt) in histograms.iter().enumerate() {
        if let Some(histogram) = histogram_opt {
            if histogram.len() != HISTOGRAM_NUM_BINS * 3 {
                panic!("Unexpected length of histogram")
            }

            let histogram_fixed: [f64; HISTOGRAM_NUM_BINS * 3] =
                histogram.as_slice().try_into().unwrap();
            tree.add(&histogram_fixed, i as u64);
        }
    }
    tree
}

fn match_points(
    source_feature_histograms: &Vec<Option<Vec<f64>>>,
    target_kdtree: &KdTree<f64, { HISTOGRAM_NUM_BINS * 3 }>,
    max_neigbour_distance: f64,
) -> std::collections::HashMap<u64, u64> {
    let mut source_to_target = Vec::new();

    for (source_index, source_histogram_opt) in source_feature_histograms.iter().enumerate() {
        if let Some(source_histogram) = source_histogram_opt {
            let source_histogram_fixed: [f64; HISTOGRAM_NUM_BINS * 3] =
                source_histogram.as_slice().try_into().unwrap();
            let neighbour = target_kdtree.nearest_one::<SquaredEuclidean>(&source_histogram_fixed);
            let neighbour_index = neighbour.item;
            let neighbour_distance = neighbour.distance;
            if neighbour_distance > max_neigbour_distance.powi(2) {
                continue;
            }
            source_to_target.push((source_index as u64, neighbour_index));
        }
    }
    let result: std::collections::HashMap<u64, u64> = source_to_target.into_iter().collect();
    result
}

fn descriptor_distance_ration(
    source_histogram: &Vec<f64>,
    target_kdtree: &KdTree<f64, { HISTOGRAM_NUM_BINS * 3 }>,
) -> Option<f64> {
    let histogram_fixed: [f64; HISTOGRAM_NUM_BINS * 3] =
        source_histogram.as_slice().try_into().unwrap();
    let nearest_neighbours = target_kdtree.nearest_n::<SquaredEuclidean>(&histogram_fixed, 2);
    if nearest_neighbours.len() != 2 {
        ()
    }
    let ratio = nearest_neighbours[0].distance / nearest_neighbours[1].distance;
    Some(ratio)
}

pub fn mutual_matching(
    source_feature_histograms: &Vec<Option<Vec<f64>>>,
    target_feature_histograms: &Vec<Option<Vec<f64>>>,
    max_neigbour_distance: f64,
    max_number_of_correspondances: usize, // XXX: in paper they propose 3000
) -> Vec<(u64, u64)> {
    let source_kdtree = make_kdtree_from_histograms(&source_feature_histograms);
    let target_kdtree = make_kdtree_from_histograms(&target_feature_histograms);
    let mut correspondance_with_ration = Vec::new();

    let source_to_target = match_points(
        source_feature_histograms,
        &target_kdtree,
        max_neigbour_distance,
    );
    let target_to_source = match_points(
        target_feature_histograms,
        &source_kdtree,
        max_neigbour_distance,
    );

    for (&source_index, &target_index) in &source_to_target {
        if let Some(&matched_index) = target_to_source.get(&target_index) {
            if matched_index == source_index {
                if let Some(source_histogram) =
                    source_feature_histograms[source_index as usize].as_ref()
                {
                    let ratio_opt = descriptor_distance_ration(&source_histogram, &target_kdtree);
                    if let Some(ratio) = ratio_opt {
                        correspondance_with_ration.push((source_index, target_index, ratio));
                    }
                }
            }
        }
    }
    correspondance_with_ration.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    let correspondance = correspondance_with_ration
        .into_iter()
        .take(max_number_of_correspondances)
        .map(|(s, t, _)| (s, t))
        .collect();
    correspondance
}
