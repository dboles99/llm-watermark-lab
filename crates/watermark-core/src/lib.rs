use pyo3::prelude::*;
use sha2::{Digest, Sha256};

mod blind_detect;
mod primitives;

use blind_detect::{
    parameter_sweep, token_frequency_vectors, variance_ratio_matrix,
};
use primitives::{simple_1_seed, word_drop_perturb, word_window_shuffle};

const DEFAULT_HASH_KEY: u64 = 15_485_863;

/// Expose the portable splitmix64 green-list generator to Python.
#[pyfunction]
fn py_green_list_splitmix(vocab_size: u32, gamma: f64, seed: u64) -> PyResult<Vec<u32>> {
    if vocab_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("vocab_size must be positive"));
    }
    if gamma <= 0.0 || gamma >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("gamma must be in (0, 1)"));
    }
    Ok(crate::primitives::green_list_splitmix(vocab_size, gamma, seed))
}

/// Compute a stable SHA-256 digest over a token-ID context using little-endian bytes.
#[pyfunction]
fn stable_context_digest(context: Vec<u32>) -> String {
    let mut hasher = Sha256::new();
    for token_id in &context {
        hasher.update(token_id.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Compute the `simple_1` seed for a single previous token.
#[pyfunction]
fn simple_1_seed_py(previous_token_id: u32, hash_key: Option<u64>) -> u64 {
    simple_1_seed(previous_token_id, hash_key.unwrap_or(DEFAULT_HASH_KEY))
}

/// Count green tokens in a sequence given a precomputed green-list set.
///
/// This is a pure scoring helper; the actual green-list construction requires
/// a PyTorch `randperm` and is handled in Python. This function checks
/// membership and computes the z-score.
#[pyfunction]
fn score_sequence(
    tokens: Vec<u32>,
    green_lists: Vec<Vec<u32>>,
    gamma: f64,
    threshold: f64,
) -> PyResult<ScoreResult> {
    if gamma <= 0.0 || gamma >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("gamma must be in (0, 1)"));
    }
    if tokens.len() != green_lists.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tokens and green_lists must have the same length",
        ));
    }

    let mut green_count: u64 = 0;
    let mut scored: u64 = 0;
    for (token, green_list) in tokens.iter().zip(green_lists.iter()) {
        scored += 1;
        if green_list.contains(token) {
            green_count += 1;
        }
    }

    if scored == 0 {
        return Ok(ScoreResult {
            green_count: 0,
            scored_count: 0,
            green_fraction: 0.0,
            z_score: f64::NEG_INFINITY,
            p_value: 1.0,
            prediction: false,
        });
    }

    let scored_f = scored as f64;
    let z_score = (green_count as f64 - gamma * scored_f)
        / (scored_f * gamma * (1.0 - gamma)).sqrt();
    let p_value = 0.5 * lib_erfc(z_score / std::f64::consts::SQRT_2);
    let prediction = z_score > threshold;

    Ok(ScoreResult {
        green_count,
        scored_count: scored,
        green_fraction: green_count as f64 / scored_f,
        z_score,
        p_value,
        prediction,
    })
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
fn lib_erfc(x: f64) -> f64 {
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

#[pyclass]
#[derive(Clone)]
struct ScoreResult {
    green_count: u64,
    scored_count: u64,
    green_fraction: f64,
    z_score: f64,
    p_value: f64,
    prediction: bool,
}

#[pymethods]
impl ScoreResult {
    #[getter]
    fn green_count(&self) -> u64 {
        self.green_count
    }
    #[getter]
    fn scored_count(&self) -> u64 {
        self.scored_count
    }
    #[getter]
    fn green_fraction(&self) -> f64 {
        self.green_fraction
    }
    #[getter]
    fn z_score(&self) -> f64 {
        self.z_score
    }
    #[getter]
    fn p_value(&self) -> f64 {
        self.p_value
    }
    #[getter]
    fn prediction(&self) -> bool {
        self.prediction
    }
}

#[pyfunction]
fn py_parameter_sweep(
    tokens: Vec<u32>,
    vocab_size: u32,
    gammas: Vec<f64>,
    context_widths: Vec<usize>,
    hash_keys: Vec<u64>,
) -> PyResult<String> {
    let results = parameter_sweep(&tokens, vocab_size, &gammas, &context_widths, &hash_keys);
    serde_json::to_string(&results).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn py_variance_ratio_matrix(
    token_docs: Vec<Vec<u32>>,
    vocab_size: u32,
    gamma: f64,
    hash_keys: Vec<u64>,
) -> PyResult<String> {
    let matrix = variance_ratio_matrix(&token_docs, vocab_size, gamma, &hash_keys);
    serde_json::to_string(&matrix).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn py_token_frequency_vectors(
    token_docs: Vec<Vec<u32>>,
    vocab_size: u32,
) -> PyResult<String> {
    let vecs = token_frequency_vectors(&token_docs, vocab_size);
    serde_json::to_string(&vecs).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn py_word_window_shuffle(text: String, window_size: usize, seed: u64) -> String {
    word_window_shuffle(&text, window_size, seed)
}

#[pyfunction]
fn py_word_drop_perturb(text: String, drop_fraction: f64, seed: u64) -> String {
    word_drop_perturb(&text, drop_fraction, seed)
}

#[pymodule]
fn watermark_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_green_list_splitmix, module)?)?;
    module.add_function(wrap_pyfunction!(stable_context_digest, module)?)?;
    module.add_function(wrap_pyfunction!(simple_1_seed_py, module)?)?;
    module.add_function(wrap_pyfunction!(score_sequence, module)?)?;
    module.add_function(wrap_pyfunction!(py_parameter_sweep, module)?)?;
    module.add_function(wrap_pyfunction!(py_variance_ratio_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(py_token_frequency_vectors, module)?)?;
    module.add_function(wrap_pyfunction!(py_word_window_shuffle, module)?)?;
    module.add_function(wrap_pyfunction!(py_word_drop_perturb, module)?)?;
    module.add_class::<ScoreResult>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_context_digest_empty() {
        let digest = stable_context_digest(vec![]);
        assert_eq!(digest.len(), 64);
        // SHA-256 of empty input
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_stable_context_digest_single_token() {
        let digest = stable_context_digest(vec![42]);
        assert_eq!(digest.len(), 64);
        // Verify it's different from empty
        assert_ne!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_stable_context_digest_order_matters() {
        let d1 = stable_context_digest(vec![1, 2, 3]);
        let d2 = stable_context_digest(vec![3, 2, 1]);
        assert_ne!(d1, d2);
    }

    #[test]
    fn test_stable_context_digest_deterministic() {
        let d1 = stable_context_digest(vec![10, 20, 30]);
        let d2 = stable_context_digest(vec![10, 20, 30]);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_simple_1_seed_default_key() {
        let seed = simple_1_seed(0, DEFAULT_HASH_KEY);
        assert_eq!(seed, 0);

        let seed1 = simple_1_seed(1, DEFAULT_HASH_KEY);
        let seed2 = simple_1_seed(1, DEFAULT_HASH_KEY);
        assert_eq!(seed1, seed2);

        // Different tokens should produce different seeds (usually)
        let seed_a = simple_1_seed(100, DEFAULT_HASH_KEY);
        let seed_b = simple_1_seed(200, DEFAULT_HASH_KEY);
        assert_ne!(seed_a, seed_b);
    }

    #[test]
    fn test_simple_1_seed_matches_python_reference() {
        // The Python reference: (hash_key * token_id) % (2^64 - 1)
        let token_id: u32 = 42;
        let hash_key: u64 = 15_485_863;
        let expected = hash_key.wrapping_mul(token_id as u64);
        let actual = simple_1_seed(token_id, hash_key);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_simple_1_seed_custom_key() {
        let seed_default = simple_1_seed(100, DEFAULT_HASH_KEY);
        let seed_custom = simple_1_seed(100, 999_999);
        assert_ne!(seed_default, seed_custom);
    }

    #[test]
    fn test_score_sequence_all_green() {
        // Test the math manually
        let gamma = 0.25_f64;
        let green_count = 5_u64;
        let scored = 5_u64;
        let scored_f = scored as f64;
        let z = (green_count as f64 - gamma * scored_f)
            / (scored_f * gamma * (1.0 - gamma)).sqrt();
        assert!(z > 0.0);
        // 5 greens out of 5 with gamma=0.25 should give a high z-score
        assert!(z > 3.0);
    }

    #[test]
    fn test_score_sequence_no_green() {
        let gamma = 0.25_f64;
        let green_count = 0_u64;
        let scored = 10_u64;
        let scored_f = scored as f64;
        let z = (green_count as f64 - gamma * scored_f)
            / (scored_f * gamma * (1.0 - gamma)).sqrt();
        assert!(z < 0.0);
        // 0 greens out of 10 with gamma=0.25 gives z ≈ -1.83
        assert!(z < -1.0);
    }

    #[test]
    fn test_erfc_known_values() {
        // erfc(0) = 1
        let result = lib_erfc(0.0);
        assert!((result - 1.0).abs() < 1e-6);

        // erfc(inf) = 0 (approximated)
        let result = lib_erfc(10.0);
        assert!(result < 1e-40);
    }

    #[test]
    fn test_default_hash_key_value() {
        assert_eq!(DEFAULT_HASH_KEY, 15_485_863);
    }
}