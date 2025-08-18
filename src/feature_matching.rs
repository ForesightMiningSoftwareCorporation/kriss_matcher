use std::time::Instant;

use ahash::HashMap;
use rayon::prelude::*;
use rstar::RTree;
use rstar::primitives::GeomWithData;

use crate::prelude::*;

type HistogramQuery = RTree<GeomWithData<[f64; HISTOGRAM_DIM], u64>>;

fn make_query_from_histograms(histograms: &[Histogram]) -> HistogramQuery {
    RTree::bulk_load(
        histograms
            .iter()
            .copied()
            .enumerate()
            .map(|(i, h)| GeomWithData::new(h, i as u64))
            .collect::<Vec<_>>(),
    )
}

fn match_points(
    source_feature_histograms: &[Histogram],
    target_query: &HistogramQuery,
) -> HashMap<u64, u64> {
    source_feature_histograms
        .par_iter()
        .enumerate()
        .map(|(source_index, source_histogram)| {
            let neighbor = target_query.nearest_neighbor(source_histogram);
            let neighbor_index = neighbor.unwrap().data;
            (source_index as u64, neighbor_index)
        })
        .collect()
}

fn descriptor_distance_ratio(
    source_histogram: &Histogram,
    target_query: &HistogramQuery,
) -> Option<f64> {
    let histogram_fixed: [f64; HISTOGRAM_DIM] = source_histogram.as_slice().try_into().unwrap();
    let nearest_neighbors = target_query
        .nearest_neighbor_iter_with_distance_2(&histogram_fixed)
        .take(2)
        .collect::<Vec<_>>();
    if nearest_neighbors.len() != 2 {
        return None;
    }
    let ratio = nearest_neighbors[0].1 / nearest_neighbors[1].1;
    ratio.is_finite().then_some(ratio)
}

pub fn mutual_matching(
    source_feature_histograms: &[Histogram],
    target_feature_histograms: &[Histogram],
    source_point_indices: &[u64],
    target_point_indices: &[u64],
    max_number_of_correspondances: usize, // XXX: in paper they propose 3000
) -> Vec<(u64, u64)> {
    let source_query = make_query_from_histograms(source_feature_histograms);
    let target_query = make_query_from_histograms(target_feature_histograms);

    // let start = Instant::now();
    // let source_to_target = match_points(source_feature_histograms, &target_query);
    // println!("Elapsed (source to target): {:?}", start.elapsed());
    // let start = Instant::now();
    // let target_to_source = match_points(target_feature_histograms, &source_query);
    // println!("Elapsed (target to source): {:?}", start.elapsed());

    let start = Instant::now();

    let mut correspondance_with_ratio = source_feature_histograms
        .par_iter()
        .enumerate()
        .filter_map(|(source_index, source_histogram)| {
            let source_index = source_index as u64;
            let neighbor = target_query.nearest_neighbor(source_histogram).unwrap();
            let target_index = neighbor.data;
            let source_neighbor = source_query.nearest_neighbor(neighbor.geom()).unwrap();
            if source_neighbor.data == source_index {
                descriptor_distance_ratio(source_histogram, &target_query)
                    .map(|ratio| (source_index, target_index, ratio))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    // println!("source to target: {source_to_target:?}");
    // println!("target to source: {target_to_source:?}");
    // for (&source_index, &target_index) in &source_to_target {
    //     if let Some(&matched_index) = target_to_source.get(&target_index) {
    //         if matched_index == source_index {
    //             if let Some(source_histogram) =
    //                 source_feature_histograms[source_index as usize].as_ref()
    //             {
    //                 let ratio_opt = descriptor_distance_ratio(source_histogram, &target_query);
    //                 if let Some(ratio) = ratio_opt {
    //                     correspondance_with_ratio.push((source_index, target_index, ratio));
    //                 }
    //             }
    //         }
    //     }
    // }
    println!("Elapsed (matching): {:?}", start.elapsed());
    // TODO: Can use a binary heap here to avoid doing a full sort.
    // https://users.rust-lang.org/t/solved-best-way-to-find-largest-three-values-in-unsorted-slice/34754/6
    let start = Instant::now();
    correspondance_with_ratio.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    println!("Elapsed (sorting): {:?}", start.elapsed());

    correspondance_with_ratio
        .into_iter()
        .take(max_number_of_correspondances)
        .map(|(s, t, _)| {
            (
                source_point_indices[s as usize],
                target_point_indices[t as usize],
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ok() {
        const HISTOGRAM_NUM_BINS: usize = 11;

        let source_histograms = vec![
            Some(vec![1.0; HISTOGRAM_DIM]),
            Some(vec![2.0; HISTOGRAM_DIM]),
            Some(vec![3.0; HISTOGRAM_DIM]),
        ];

        let target_histograms = vec![
            Some(vec![1.0; HISTOGRAM_DIM]),
            Some(vec![2.0; HISTOGRAM_DIM]),
            Some(vec![4.0; HISTOGRAM_DIM]),
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
