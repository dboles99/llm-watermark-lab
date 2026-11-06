//! Multi-parameter, multi-key, and multi-hash KGW-family scoring for blind
//! and semi-blind watermark detection.

use std::collections::HashSet;

use crate::primitives::{
    binomial_z_score, score_tokens_with_green_fn, simple_1_seed, z_to_p_value,
};

/// Supported hash families for the parameter sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum HashFamily {
    Simple1,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SweepSetting {
    pub gamma: f64,
    pub context_width: usize,
    pub hash_family: HashFamily,
    pub hash_key: u64,
}

/// Result for one sweep setting.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SweepResult {
    pub setting: SweepSetting,
    pub green_count: u64,
    pub scored: u64,
    pub z_score: f64,
    pub p_value: f64,
}

/// Score a token sequence with a single setting.
pub fn score_with_setting(tokens: &[u32], vocab_size: u32, setting: &SweepSetting) -> SweepResult {
    let mut green_fn = |prev: u32| {
        let seed = match setting.hash_family {
            HashFamily::Simple1 => simple_1_seed(prev, setting.hash_key),
        };
        // Build green list using the splitmix portable RNG.
        // For context_width > 1 we would use the last `context_width` tokens as seed input;
        // here we keep the simple_1 seed and treat it as the canonical family lookup.
        crate::primitives::green_list_splitmix(vocab_size, setting.gamma, seed)
            .into_iter()
            .collect::<HashSet<u32>>()
    };
    let (green_count, scored) = score_tokens_with_green_fn(tokens, &mut green_fn, false);
    let z = binomial_z_score(green_count, scored, setting.gamma);
    SweepResult {
        setting: *setting,
        green_count,
        scored,
        z_score: z,
        p_value: z_to_p_value(z),
    }
}

/// Run a parameter sweep over a grid and return the results sorted by z-score desc.
pub fn parameter_sweep(
    tokens: &[u32],
    vocab_size: u32,
    gammas: &[f64],
    context_widths: &[usize],
    hash_keys: &[u64],
) -> Vec<SweepResult> {
    let mut results: Vec<SweepResult> = Vec::with_capacity(
        gammas.len() * context_widths.len() * hash_keys.len(),
    );
    for &gamma in gammas {
        for &cw in context_widths {
            for &hash_key in hash_keys {
                let setting = SweepSetting {
                    gamma,
                    context_width: cw,
                    hash_family: HashFamily::Simple1,
                    hash_key,
                };
                results.push(score_with_setting(tokens, vocab_size, &setting));
            }
        }
    }
    results.sort_by(|a, b| b.z_score.partial_cmp(&a.z_score).unwrap());
    results
}

/// Cross-document variance-ratio detector: score N token sequences with K random keys.
/// Returns, for each key, the z-scores across documents plus the variance ratio.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VarianceRatioResult {
    pub key_index: usize,
    pub hash_key: u64,
    pub z_scores: Vec<f64>,
    pub cross_doc_variance: f64,
    pub mean_z: f64,
    pub variance_ratio: f64,
}

pub fn variance_ratio_matrix(
    token_docs: &[Vec<u32>],
    vocab_size: u32,
    gamma: f64,
    hash_keys: &[u64],
) -> Vec<VarianceRatioResult> {
    let expected_variance = 1.0; // under H0, z-scores are N(0,1) iid
    let mut results: Vec<VarianceRatioResult> = Vec::with_capacity(hash_keys.len());
    for (idx, &hash_key) in hash_keys.iter().enumerate() {
        let mut z_scores: Vec<f64> = Vec::with_capacity(token_docs.len());
        for doc in token_docs {
            let res = score_with_setting(
                doc,
                vocab_size,
                &SweepSetting {
                    gamma,
                    context_width: 1,
                    hash_family: HashFamily::Simple1,
                    hash_key,
                },
            );
            z_scores.push(res.z_score);
        }
        let mean_z = z_scores.iter().sum::<f64>() / z_scores.len() as f64;
        let var = if z_scores.len() > 1 {
            z_scores
                .iter()
                .map(|z| (z - mean_z).powi(2))
                .sum::<f64>()
                / (z_scores.len() - 1) as f64
        } else {
            0.0
        };
        results.push(VarianceRatioResult {
            key_index: idx,
            hash_key,
            z_scores,
            cross_doc_variance: var,
            mean_z,
            variance_ratio: var / expected_variance,
        });
    }
    results
}

/// Token-frequency vectorization for two-sample tests.
/// Returns a sparse count map of token ID -> frequency for each document.
pub fn token_frequency_vectors(token_docs: &[Vec<u32>], vocab_size: u32) -> Vec<Vec<u32>> {
    token_docs
        .iter()
        .map(|tokens| {
            let mut counts = vec![0u32; vocab_size as usize];
            for &t in tokens {
                if (t as usize) < counts.len() {
                    counts[t as usize] += 1;
                }
            }
            counts
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::green_list_simple_1;

    fn build_watermarked_seq(len: usize, vocab_size: u32, gamma: f64, hash_key: u64) -> Vec<u32> {
        // Simple deterministic watermark-favoring sequence: always pick the first green token
        // from the green list seeded by the previous token.
        let mut seq = vec![0u32; 1];
        for _ in 1..len {
            let prev = seq[seq.len() - 1];
            let green = green_list_simple_1(prev, vocab_size, gamma, hash_key);
            seq.push(*green.iter().min().unwrap());
        }
        seq
    }

    #[test]
    fn parameter_sweep_finds_watermarked_settings() {
        let vocab_size = 200u32;
        let true_gamma = 0.25;
        let true_key = 15485863u64;
        let tokens = build_watermarked_seq(400, vocab_size, true_gamma, true_key);
        let results = parameter_sweep(
            &tokens,
            vocab_size,
            &[0.1, 0.25, 0.5],
            &[1usize],
            &[15485863u64, 999999u64, 123456789u64],
        );
        let best = results.first().unwrap();
        assert!(best.z_score > 10.0);
        assert!((best.setting.gamma - true_gamma).abs() < 0.01);
        assert_eq!(best.setting.hash_key, true_key);
    }

    #[test]
    fn variance_ratio_detects_fixed_key() {
        let vocab_size = 200u32;
        let gamma = 0.25;
        let true_key = 15485863u64;
        // 10 watermarked documents under the same key
        let docs: Vec<Vec<u32>> = (0..10)
            .map(|_| build_watermarked_seq(300, vocab_size, gamma, true_key))
            .collect();
        // Candidate keys: true + random
        let keys = [true_key, 111111u64, 222222u64, 333333u64, 444444u64];
        let matrix = variance_ratio_matrix(&docs, vocab_size, gamma, &keys);
        let best = matrix
            .iter()
            .min_by(|a, b| a.variance_ratio.partial_cmp(&b.variance_ratio).unwrap())
            .unwrap();
        assert_eq!(best.hash_key, true_key);
        assert!(best.mean_z > 5.0);
        assert!(best.variance_ratio < 1.0); // lower variance than H0 expected
    }

    #[test]
    fn token_frequency_vectors_sum_to_length() {
        let docs = vec![vec![1u32, 2, 2, 3], vec![3u32, 3, 1]];
        let vecs = token_frequency_vectors(&docs, 10);
        assert_eq!(vecs[0].iter().sum::<u32>(), 4);
        assert_eq!(vecs[1].iter().sum::<u32>(), 3);
    }
}
