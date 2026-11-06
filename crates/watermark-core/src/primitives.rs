//! Portable, dependency-light watermark primitives for multi-parameter,
//! multi-key, and perturbation-aware detection.

use std::collections::HashSet;

/// Deterministic splitmix64 PRNG state step.
#[inline]
pub fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// A small, fast, deterministic permutation generator using a splitmix64 stream.
/// Returns a vector of token IDs in "green-list" order.
pub fn green_list_splitmix(vocab_size: u32, gamma: f64, seed: u64) -> Vec<u32> {
    assert!(vocab_size > 0, "vocab_size must be positive");
    assert!((0.0..1.0).contains(&gamma), "gamma must be in (0,1)");
    let green_size = ((vocab_size as f64) * gamma).ceil() as usize;
    let green_size = green_size.min(vocab_size as usize).max(1);

    let mut pool: Vec<u32> = (0..vocab_size).collect();
    let mut state = seed;
    // Fisher-Yates-like shuffle using splitmix64
    for i in (1..pool.len()).rev() {
        state = splitmix64(state);
        let j = (state % (i as u64 + 1)) as usize;
        pool.swap(i, j);
    }
    pool.truncate(green_size);
    pool
}

/// Compute the `simple_1` additive PRF seed for one previous token.
#[inline]
pub fn simple_1_seed(previous_token_id: u32, hash_key: u64) -> u64 {
    (hash_key.wrapping_mul(previous_token_id as u64)) % (u64::MAX)
}

/// Compute green list for a single previous-token context.
#[inline]
#[allow(dead_code)]
pub fn green_list_simple_1(
    previous_token_id: u32,
    vocab_size: u32,
    gamma: f64,
    hash_key: u64,
) -> HashSet<u32> {
    let seed = simple_1_seed(previous_token_id, hash_key);
    green_list_splitmix(vocab_size, gamma, seed).into_iter().collect()
}

/// Score a token sequence for green-list membership, returning (green_count, scored).
pub fn score_tokens_with_green_fn(
    tokens: &[u32],
    mut green_fn: impl FnMut(u32) -> HashSet<u32>,
    ignore_repeated_ngrams: bool,
) -> (u64, u64) {
    let mut green_count: u64 = 0;
    let mut scored: u64 = 0;
    let mut seen_pairs: HashSet<(u32, u32)> = HashSet::new();

    for i in 1..tokens.len() {
        let prev = tokens[i - 1];
        let cand = tokens[i];
        let green = green_fn(prev);
        let is_green = green.contains(&cand);
        if ignore_repeated_ngrams {
            let pair = (prev, cand);
            if seen_pairs.contains(&pair) {
                continue;
            }
            seen_pairs.insert(pair);
        }
        scored += 1;
        if is_green {
            green_count += 1;
        }
    }
    (green_count, scored)
}

/// Compute z-score for a binomial green-list test.
#[inline]
pub fn binomial_z_score(green_count: u64, scored: u64, gamma: f64) -> f64 {
    if scored == 0 {
        return 0.0;
    }
    let scored_f = scored as f64;
    (green_count as f64 - gamma * scored_f) / (scored_f * gamma * (1.0 - gamma)).sqrt()
}

/// One-sided p-value from z-score via erfc.
pub fn z_to_p_value(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t * (-z * z - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398
                                + t * (1.48851587
                                    + t * (-0.82215223 + t * 0.17087277))))))))) 
        .exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Random-sampling perturbation: shuffle words in local windows with fixed seed.
pub fn word_window_shuffle(text: &str, window_size: usize, seed: u64) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }
    let mut rng_state = seed;
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        let end = (i + window_size).min(words.len());
        let slice = &words[i..end];
        let mut shuffled: Vec<String> = slice.iter().map(|w| w.to_string()).collect();
        // Fisher-Yates with splitmix64
        for j in (1..shuffled.len()).rev() {
            rng_state = splitmix64(rng_state);
            let k = (rng_state % (j as u64 + 1)) as usize;
            shuffled.swap(j, k);
        }
        out.extend(shuffled);
        i = end;
    }
    out.join(" ")
}

/// Random-sampling perturbation: drop a fraction of words deterministically.
pub fn word_drop_perturb(text: &str, drop_fraction: f64, seed: u64) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }
    let mut state = seed;
    let mut out: Vec<&str> = Vec::with_capacity(words.len());
    for w in words {
        state = splitmix64(state);
        let r = (state as f64) / (u64::MAX as f64);
        if r >= drop_fraction {
            out.push(w);
        }
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn green_list_deterministic() {
        let g1 = green_list_splitmix(100, 0.25, 12345);
        let g2 = green_list_splitmix(100, 0.25, 12345);
        assert_eq!(g1, g2);
        assert_eq!(g1.len(), 25);
    }

    #[test]
    fn green_list_size_edge() {
        let g = green_list_splitmix(10, 0.25, 1);
        assert_eq!(g.len(), 3); // ceil(2.5)
    }

    #[test]
    fn simple_1_matches_python_default() {
        let seed = simple_1_seed(42, 15_485_863);
        assert_eq!(seed, 15_485_863_u64 * 42_u64);
    }

    #[test]
    fn score_all_green() {
        let vocab_size = 100u32;
        let gamma = 0.25;
        let hash_key = 15485863u64;
        let mut green_fn = |prev: u32| green_list_simple_1(prev, vocab_size, gamma, hash_key);
        // Iteratively build a watermark-favoring sequence: t_{i+1} is a green token of t_i.
        let mut tokens: Vec<u32> = Vec::with_capacity(50);
        tokens.push(0u32);
        for _ in 1..50 {
            let prev = *tokens.last().unwrap();
            let green: Vec<u32> = green_list_simple_1(prev, vocab_size, gamma, hash_key)
                .into_iter()
                .collect();
            tokens.push(green[0]);
        }
        let (gc, sc) = score_tokens_with_green_fn(&tokens, &mut green_fn, false);
        assert_eq!(sc, (tokens.len() - 1) as u64);
        assert_eq!(gc, sc); // every token after first is green
    }

    #[test]
    fn word_shuffle_preserves_length() {
        let text = "the quick brown fox jumps over the lazy dog";
        let s = word_window_shuffle(text, 3, 42);
        assert_eq!(s.split_whitespace().count(), text.split_whitespace().count());
    }

    #[test]
    fn word_drop_reduces_length() {
        let text = "the quick brown fox jumps over the lazy dog";
        let s = word_drop_perturb(text, 0.5, 42);
        assert!(s.split_whitespace().count() < text.split_whitespace().count());
    }
}
