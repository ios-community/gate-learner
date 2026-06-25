#![allow(clippy::type_complexity)]

//! Dataset generator for N-input logic gates (OR and XOR).

use crate::error::GateLearnerError;

/// Generates the truth table for an N-input OR gate.
///
/// The output is `1.0` if any input is `1.0`, otherwise `0.0`.
///
/// # Errors
///
/// Returns [`GateLearnerError::InvalidArchitecture`] if `inputs` is too large (>= 20)
/// to prevent excessive memory allocation or overflow.
///
/// # Examples
///
/// ```
/// # use gate_learner::dataset::generate_or;
/// let dataset = generate_or(2).unwrap();
/// assert_eq!(dataset.len(), 4);
/// ```
pub fn generate_or(inputs: usize) -> Result<Vec<(Vec<f32>, Vec<f32>)>, GateLearnerError> {
    if inputs >= 20 {
        return Err(GateLearnerError::InvalidArchitecture(format!(
            "N-input value ({inputs}) is too large, max supported is 19."
        )));
    }
    let num_samples = 1 << inputs;
    let mut dataset = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let mut input = Vec::with_capacity(inputs);
        let mut any_one = false;
        for bit in 0..inputs {
            let val = if (i >> bit) & 1 == 1 {
                any_one = true;
                1.0f32
            } else {
                0.0f32
            };
            input.push(val);
        }
        let target = vec![if any_one { 1.0f32 } else { 0.0f32 }];
        dataset.push((input, target));
    }
    Ok(dataset)
}

/// Generates the truth table for an N-input XOR gate (parity problem).
///
/// The output is `1.0` if the sum of inputs is odd, otherwise `0.0`.
///
/// # Errors
///
/// Returns [`GateLearnerError::InvalidArchitecture`] if `inputs` is too large (>= 20)
/// to prevent excessive memory allocation or overflow.
///
/// # Examples
///
/// ```
/// # use gate_learner::dataset::generate_xor;
/// let dataset = generate_xor(2).unwrap();
/// assert_eq!(dataset.len(), 4);
/// ```
pub fn generate_xor(inputs: usize) -> Result<Vec<(Vec<f32>, Vec<f32>)>, GateLearnerError> {
    if inputs >= 20 {
        return Err(GateLearnerError::InvalidArchitecture(format!(
            "N-input value ({inputs}) is too large, max supported is 19."
        )));
    }

    let num_samples = 1 << inputs;
    let mut dataset = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let mut input = Vec::with_capacity(inputs);
        let mut ones_count = 0;
        for bit in 0..inputs {
            let val = if (i >> bit) & 1 == 1 {
                ones_count += 1;
                1.0f32
            } else {
                0.0f32
            };
            input.push(val);
        }
        let target = vec![if ones_count % 2 == 1 { 1.0f32 } else { 0.0f32 }];
        dataset.push((input, target));
    }
    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::GateLearnerError;

    #[test]
    fn test_generate_or_2_inputs() {
        let dataset = generate_or(2).unwrap();
        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset[0].0, vec![0.0f32, 0.0f32]);
        assert_eq!(dataset[0].1, vec![0.0f32]);
        assert_eq!(dataset[1].0, vec![1.0f32, 0.0f32]);
        assert_eq!(dataset[1].1, vec![1.0f32]);
        assert_eq!(dataset[2].0, vec![0.0f32, 1.0f32]);
        assert_eq!(dataset[2].1, vec![1.0f32]);
        assert_eq!(dataset[3].0, vec![1.0f32, 1.0f32]);
        assert_eq!(dataset[3].1, vec![1.0f32]);
    }

    #[test]
    fn test_generate_or_1_input() {
        let dataset = generate_or(1).unwrap();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset[0].0, vec![0.0f32]);
        assert_eq!(dataset[0].1, vec![0.0f32]);
        assert_eq!(dataset[1].0, vec![1.0f32]);
        assert_eq!(dataset[1].1, vec![1.0f32]);
    }

    #[test]
    fn test_generate_or_3_inputs() {
        let dataset = generate_or(3).unwrap();
        assert_eq!(dataset.len(), 8);
        assert_eq!(dataset[0].0, vec![0.0f32, 0.0f32, 0.0f32]);
        assert_eq!(dataset[0].1, vec![0.0f32]);
        assert_eq!(dataset[1].0, vec![1.0f32, 0.0f32, 0.0f32]);
        assert_eq!(dataset[1].1, vec![1.0f32]);
        assert_eq!(dataset[2].0, vec![0.0f32, 1.0f32, 0.0f32]);
        assert_eq!(dataset[2].1, vec![1.0f32]);
        assert_eq!(dataset[7].0, vec![1.0f32, 1.0f32, 1.0f32]);
        assert_eq!(dataset[7].1, vec![1.0f32]);
    }

    #[test]
    fn test_generate_xor_2_inputs() {
        let dataset = generate_xor(2).unwrap();
        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset[0].0, vec![0.0f32, 0.0f32]);
        assert_eq!(dataset[0].1, vec![0.0f32]);
        assert_eq!(dataset[1].0, vec![1.0f32, 0.0f32]);
        assert_eq!(dataset[1].1, vec![1.0f32]);
        assert_eq!(dataset[2].0, vec![0.0f32, 1.0f32]);
        assert_eq!(dataset[2].1, vec![1.0f32]);
        assert_eq!(dataset[3].0, vec![1.0f32, 1.0f32]);
        assert_eq!(dataset[3].1, vec![0.0f32]);
    }

    #[test]
    fn test_generate_xor_1_input() {
        let dataset = generate_xor(1).unwrap();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset[0].0, vec![0.0f32]);
        assert_eq!(dataset[0].1, vec![0.0f32]);
        assert_eq!(dataset[1].0, vec![1.0f32]);
        assert_eq!(dataset[1].1, vec![1.0f32]);
    }

    #[test]
    fn test_generate_xor_3_inputs() {
        let dataset = generate_xor(3).unwrap();
        assert_eq!(dataset.len(), 8);
        assert_eq!(dataset[0].0, vec![0.0f32, 0.0f32, 0.0f32]);
        assert_eq!(dataset[0].1, vec![0.0f32]);
        assert_eq!(dataset[1].0, vec![1.0f32, 0.0f32, 0.0f32]);
        assert_eq!(dataset[1].1, vec![1.0f32]);
        assert_eq!(dataset[2].0, vec![0.0f32, 1.0f32, 0.0f32]);
        assert_eq!(dataset[2].1, vec![1.0f32]);
        assert_eq!(dataset[3].0, vec![1.0f32, 1.0f32, 0.0f32]);
        assert_eq!(dataset[3].1, vec![0.0f32]);
        assert_eq!(dataset[7].0, vec![1.0f32, 1.0f32, 1.0f32]);
        assert_eq!(dataset[7].1, vec![1.0f32]);
    }

    #[test]
    fn test_generate_or_large_inputs_error() {
        let err = generate_or(20).unwrap_err();
        assert!(matches!(err, GateLearnerError::InvalidArchitecture(_)));
        assert_eq!(
            err.to_string(),
            "invalid network architecture: N-input value (20) is too large, max supported is 19."
        );
    }

    #[test]
    fn test_generate_xor_large_inputs_error() {
        let err = generate_xor(20).unwrap_err();
        assert!(matches!(err, GateLearnerError::InvalidArchitecture(_)));
        assert_eq!(
            err.to_string(),
            "invalid network architecture: N-input value (20) is too large, max supported is 19."
        );
    }
}
