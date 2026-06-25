#![allow(clippy::needless_range_loop)]

//! Core mathematical engine for the Multilayer Perceptron (MLP) and Adam Optimizer.

use crate::error::GateLearnerError;
use rand::{Rng, RngExt};
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};

/// Type of activation function.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, clap::ValueEnum,
)]
pub enum ActivationType {
    /// Sigmoid activation function.
    Sigmoid,
    /// `ReLU` activation function.
    Relu,
}

/// Sigmoid activation function.
///
/// # Examples
///
/// ```
/// # use gate_learner::core::sigmoid;
/// let val = sigmoid(0.0);
/// assert!((val - 0.5).abs() < f32::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

/// Derivative of the Sigmoid activation function.
///
/// # Examples
///
/// ```
/// # use gate_learner::core::sigmoid_derivative;
/// let val = sigmoid_derivative(0.5);
/// assert!((val - 0.25).abs() < f32::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn sigmoid_derivative(y: f32) -> f32 {
    y * (1.0f32 - y)
}

/// `ReLU` activation function.
///
/// # Examples
///
/// ```
/// # use gate_learner::core::relu;
/// assert_eq!(relu(2.5), 2.5);
/// assert_eq!(relu(-1.5), 0.0);
/// ```
#[inline]
#[must_use]
pub fn relu(x: f32) -> f32 {
    if x > 0.0f32 { x } else { 0.0f32 }
}

/// Derivative of the `ReLU` activation function.
///
/// # Examples
///
/// ```
/// # use gate_learner::core::relu_derivative;
/// assert_eq!(relu_derivative(2.5), 1.0);
/// assert_eq!(relu_derivative(0.0), 0.0);
/// ```
#[inline]
#[must_use]
pub fn relu_derivative(y: f32) -> f32 {
    if y > 0.0f32 { 1.0f32 } else { 0.0f32 }
}

/// Configuration for the artificial neural network architecture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NetworkConfig {
    /// Number of neurons in the input layer (matches the N-input of the logic gate).
    pub input_size: usize,
    /// Number of neurons in each hidden layer.
    pub hidden_sizes: Vec<usize>,
    /// Number of neurons in the output layer (typically 1 for logic gates).
    pub output_size: usize,
    /// Activation function for the hidden layers.
    pub hidden_activation: ActivationType,
}

/// Representation of a single layer in the neural network.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Layer {
    /// Number of input neurons.
    pub input_size: usize,
    /// Number of output neurons.
    pub output_size: usize,
    /// Weights stored in a flat vector.
    pub weights: Vec<f32>,
    /// Biases for each output neuron.
    pub biases: Vec<f32>,
    /// Activation function used by this layer.
    pub activation: ActivationType,
}

impl Layer {
    /// Creates a new layer with He Uniform or Xavier Uniform initialisation.
    #[must_use]
    pub fn new_with_rng<R: Rng + ?Sized>(
        input_size: usize,
        output_size: usize,
        activation: ActivationType,
        rng: &mut R,
    ) -> Self {
        let limit = match activation {
            ActivationType::Relu => (6.0f32 / input_size as f32).sqrt(),
            ActivationType::Sigmoid => (6.0f32 / (input_size + output_size) as f32).sqrt(),
        };

        let weights: Vec<f32> = (0..(input_size * output_size))
            .map(|_| rng.random_range(-limit..limit))
            .collect();

        let biases = vec![0.0f32; output_size];

        Self {
            input_size,
            output_size,
            weights,
            biases,
            activation,
        }
    }

    /// Creates a new layer with appropriate initialisation.
    #[must_use]
    pub fn new(input_size: usize, output_size: usize, activation: ActivationType) -> Self {
        let mut rng = ChaCha8Rng::from_rng(&mut rand::rng());
        Self::new_with_rng(input_size, output_size, activation, &mut rng)
    }

    /// Performs forward propagation for a single layer.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.input_size);
        debug_assert_eq!(output.len(), self.output_size);

        for j in 0..self.output_size {
            let mut sum = self.biases[j];
            for i in 0..self.input_size {
                sum += input[i] * self.weights[j * self.input_size + i];
            }
            output[j] = match self.activation {
                ActivationType::Sigmoid => sigmoid(sum),
                ActivationType::Relu => relu(sum),
            };
        }
    }
}

/// Representation of the Multi-Layer Perceptron artificial neural network.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultilayerPerceptron {
    /// Layers constituting the neural network.
    pub layers: Vec<Layer>,
    #[serde(skip)]
    activations: Vec<Vec<f32>>,
    #[serde(skip)]
    deltas: Vec<Vec<f32>>,
    #[serde(skip)]
    pub(crate) weight_grads: Vec<Vec<f32>>,
    #[serde(skip)]
    pub(crate) bias_grads: Vec<Vec<f32>>,
}

impl MultilayerPerceptron {
    /// Creates a new MLP instance with randomised weight initialisation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use gate_learner::core::{MultilayerPerceptron, NetworkConfig, ActivationType};
    /// let config = NetworkConfig {
    ///     input_size: 2,
    ///     hidden_sizes: vec![3],
    ///     output_size: 1,
    ///     hidden_activation: ActivationType::Relu,
    /// };
    /// let mlp = MultilayerPerceptron::new(&config);
    /// assert!(mlp.is_ok());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`GateLearnerError::InvalidArchitecture`] if the layer sizes are invalid (e.g., 0).
    pub fn new(config: &NetworkConfig) -> Result<Self, GateLearnerError> {
        let mut rng = ChaCha8Rng::from_rng(&mut rand::rng());
        let seed = rng.random();
        Self::new_with_seed(config, seed)
    }

    /// Creates a new MLP instance with a specific seed for reproducibility.
    ///
    /// # Errors
    ///
    /// Returns [`GateLearnerError::InvalidArchitecture`] if the layer sizes are invalid (e.g., 0).
    pub fn new_with_seed(config: &NetworkConfig, seed: u64) -> Result<Self, GateLearnerError> {
        if config.input_size == 0 {
            return Err(GateLearnerError::InvalidArchitecture(
                "Input size cannot be zero".to_string(),
            ));
        }

        if config.output_size == 0 {
            return Err(GateLearnerError::InvalidArchitecture(
                "Output size cannot be zero".to_string(),
            ));
        }

        for (i, &size) in config.hidden_sizes.iter().enumerate() {
            if size == 0 {
                return Err(GateLearnerError::InvalidArchitecture(format!(
                    "Hidden layer {i} size cannot be zero"
                )));
            }
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut layers = Vec::new();
        let mut prev_size = config.input_size;

        for &hidden_size in &config.hidden_sizes {
            layers.push(Layer::new_with_rng(
                prev_size,
                hidden_size,
                config.hidden_activation,
                &mut rng,
            ));
            prev_size = hidden_size;
        }
        layers.push(Layer::new_with_rng(
            prev_size,
            config.output_size,
            ActivationType::Sigmoid,
            &mut rng,
        ));

        let mut mlp = Self {
            layers,
            activations: Vec::new(),
            deltas: Vec::new(),
            weight_grads: Vec::new(),
            bias_grads: Vec::new(),
        };
        mlp.ensure_buffers();
        Ok(mlp)
    }

    /// Ensures that the internal buffers for activations and deltas are allocated and correctly sized.
    pub fn ensure_buffers(&mut self) {
        let num_layers = self.layers.len();
        if self.activations.len() != num_layers + 1 {
            self.activations = Vec::with_capacity(num_layers + 1);
            self.activations
                .push(vec![0.0f32; self.layers[0].input_size]);
            for layer in &self.layers {
                self.activations.push(vec![0.0f32; layer.output_size]);
            }
        }
        if self.deltas.len() != num_layers {
            self.deltas = Vec::with_capacity(num_layers);
            for layer in &self.layers {
                self.deltas.push(vec![0.0f32; layer.output_size]);
            }
        }
        if self.weight_grads.len() != num_layers {
            self.weight_grads = Vec::with_capacity(num_layers);
            for layer in &self.layers {
                self.weight_grads.push(vec![0.0f32; layer.weights.len()]);
            }
        }
        if self.bias_grads.len() != num_layers {
            self.bias_grads = Vec::with_capacity(num_layers);
            for layer in &self.layers {
                self.bias_grads.push(vec![0.0f32; layer.biases.len()]);
            }
        }
    }

    /// Performs forward propagation for the given input.
    ///
    /// # Panics
    ///
    /// This function will panic if the length of `input` does not match the network's `input_size`
    /// or if the length of `output` does not match the network's `output_size`.
    pub fn forward(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(
            input.len(),
            self.layers[0].input_size,
            "Input size mismatch"
        );
        assert_eq!(
            output.len(),
            self.layers.last().unwrap().output_size,
            "Output size mismatch"
        );
        self.ensure_buffers();

        self.activations[0].copy_from_slice(input);
        for l in 0..self.layers.len() {
            let (left, right) = self.activations.split_at_mut(l + 1);
            let prev_act = &left[l];
            let next_act = &mut right[0];
            self.layers[l].forward(prev_act, next_act);
        }
        output.copy_from_slice(&self.activations[self.layers.len()]);
    }

    /// Zeroes out the accumulated gradients.
    pub fn zero_gradients(&mut self) {
        self.ensure_buffers();
        for grads in &mut self.weight_grads {
            grads.fill(0.0f32);
        }
        for grads in &mut self.bias_grads {
            grads.fill(0.0f32);
        }
    }

    /// Accumulates gradients for a single input-target pair.
    #[allow(clippy::too_many_lines)]
    pub fn accumulate_gradients(&mut self, input: &[f32], target: &[f32]) -> (f32, &[f32]) {
        self.ensure_buffers();

        debug_assert_eq!(
            self.activations[0].len(),
            input.len(),
            "Input activation buffer size mismatch"
        );
        self.activations[0].copy_from_slice(input);
        for l in 0..self.layers.len() {
            let (left, right) = self.activations.split_at_mut(l + 1);
            let prev_act = &left[l];
            let next_act = &mut right[0];
            self.layers[l].forward(prev_act, next_act);
        }

        let output_act = &self.activations[self.layers.len()];
        debug_assert_eq!(
            output_act.len(),
            target.len(),
            "Output activation and target size mismatch for loss calculation"
        );
        let mut loss = 0.0f32;
        for (out, tar) in output_act.iter().zip(target.iter()) {
            let out_clipped = out.clamp(1e-7f32, 1.0f32 - 1e-7f32);
            loss -= tar * out_clipped.ln() + (1.0f32 - tar) * (1.0f32 - out_clipped).ln();
        }

        let num_layers = self.layers.len();

        {
            let out_act = &self.activations[num_layers];
            let out_delta = &mut self.deltas[num_layers - 1];
            debug_assert_eq!(
                out_delta.len(),
                out_act.len(),
                "Output delta buffer size mismatch"
            );
            for j in 0..out_delta.len() {
                out_delta[j] = out_act[j] - target[j];
            }
        }

        for l in (0..num_layers - 1).rev() {
            let layer = &self.layers[l + 1];
            let act = &self.activations[l + 1];

            let (left_deltas, right_deltas) = self.deltas.split_at_mut(l + 1);
            let next_delta = &right_deltas[0];
            let current_delta = &mut left_deltas[l];
            debug_assert_eq!(
                current_delta.len(),
                act.len(),
                "Current delta buffer size mismatch"
            );

            for i in 0..current_delta.len() {
                let mut sum = 0.0f32;
                for j in 0..layer.output_size {
                    let weight_idx = j * layer.input_size + i;
                    debug_assert!(
                        weight_idx < layer.weights.len(),
                        "Weight index out of bounds in MultilayerPerceptron::accumulate_gradients (hidden delta)"
                    );
                    sum += next_delta[j] * layer.weights[weight_idx];
                }
                let a = act[i];
                current_delta[i] = sum
                    * match self.layers[l].activation {
                        ActivationType::Sigmoid => sigmoid_derivative(a),
                        ActivationType::Relu => relu_derivative(a),
                    };
            }
        }

        for l in 0..num_layers {
            let act = &self.activations[l];
            let delta = &self.deltas[l];
            let layer = &self.layers[l];

            let w_grads = &mut self.weight_grads[l];
            let b_grads = &mut self.bias_grads[l];
            debug_assert_eq!(
                w_grads.len(),
                layer.weights.len(),
                "Weight gradient buffer size mismatch"
            );
            debug_assert_eq!(
                b_grads.len(),
                layer.biases.len(),
                "Bias gradient buffer size mismatch"
            );

            for i in 0..layer.input_size {
                for j in 0..layer.output_size {
                    let weight_grad_idx = j * layer.input_size + i;
                    debug_assert!(
                        weight_grad_idx < w_grads.len(),
                        "Weight gradient index out of bounds in MultilayerPerceptron::accumulate_gradients (weight update)"
                    );
                    w_grads[weight_grad_idx] += delta[j] * act[i];
                }
            }

            for j in 0..layer.output_size {
                debug_assert!(
                    j < b_grads.len(),
                    "Bias gradient index out of bounds in MultilayerPerceptron::accumulate_gradients (bias update)"
                );
                b_grads[j] += delta[j];
            }
        }

        (loss, &self.activations[num_layers])
    }
}

/// Adam Optimizer for updating the weights and biases of a `MultilayerPerceptron`.
pub struct AdamOptimizer {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
    m_weights: Vec<Vec<f32>>,
    v_weights: Vec<Vec<f32>>,
    m_biases: Vec<Vec<f32>>,
    v_biases: Vec<Vec<f32>>,
}

impl AdamOptimizer {
    /// Creates a new Adam Optimizer instance matching the architecture of the given MLP.
    #[must_use]
    pub fn new(mlp: &MultilayerPerceptron) -> Self {
        let mut m_weights = Vec::new();
        let mut v_weights = Vec::new();
        let mut m_biases = Vec::new();
        let mut v_biases = Vec::new();

        for layer in &mlp.layers {
            m_weights.push(vec![0.0f32; layer.weights.len()]);
            v_weights.push(vec![0.0f32; layer.weights.len()]);
            m_biases.push(vec![0.0f32; layer.biases.len()]);
            v_biases.push(vec![0.0f32; layer.biases.len()]);
        }

        Self {
            beta1: 0.9f32,
            beta2: 0.999f32,
            epsilon: 1e-8f32,
            t: 0,
            m_weights,
            v_weights,
            m_biases,
            v_biases,
        }
    }

    /// Updates the weights and biases of the MLP using the accumulated gradients.
    pub fn update(
        &mut self,
        mlp: &mut MultilayerPerceptron,
        learning_rate: f32,
        l2_lambda: f32,
        batch_size: usize,
    ) {
        self.t += 1;
        let scale = 1.0f32 / batch_size as f32;

        let correction1 = 1.0f32 - self.beta1.powi(self.t as i32);
        let correction2 = 1.0f32 - self.beta2.powi(self.t as i32);

        for l in 0..mlp.layers.len() {
            let layer = &mut mlp.layers[l];
            let w_grads = &mlp.weight_grads[l];
            let b_grads = &mlp.bias_grads[l];

            for i in 0..layer.weights.len() {
                let g = w_grads[i] * scale + l2_lambda * layer.weights[i];

                self.m_weights[l][i] =
                    self.beta1 * self.m_weights[l][i] + (1.0f32 - self.beta1) * g;
                self.v_weights[l][i] =
                    self.beta2 * self.v_weights[l][i] + (1.0f32 - self.beta2) * g * g;

                let m_hat = self.m_weights[l][i] / correction1;
                let v_hat = self.v_weights[l][i] / correction2;

                layer.weights[i] -= (learning_rate / (v_hat.sqrt() + self.epsilon)) * m_hat;
            }

            for i in 0..layer.biases.len() {
                let g = b_grads[i] * scale;

                self.m_biases[l][i] = self.beta1 * self.m_biases[l][i] + (1.0f32 - self.beta1) * g;
                self.v_biases[l][i] =
                    self.beta2 * self.v_biases[l][i] + (1.0f32 - self.beta2) * g * g;

                let m_hat = self.m_biases[l][i] / correction1;
                let v_hat = self.v_biases[l][i] / correction2;

                layer.biases[i] -= (learning_rate / (v_hat.sqrt() + self.epsilon)) * m_hat;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn create_test_mlp(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        hidden_activation: ActivationType,
        seed: u64,
    ) -> MultilayerPerceptron {
        let config = NetworkConfig {
            input_size,
            hidden_sizes,
            output_size,
            hidden_activation,
        };
        MultilayerPerceptron::new_with_seed(&config, seed).unwrap()
    }

    #[test]
    fn test_layer_new() {
        let layer = Layer::new(2, 3, ActivationType::Relu);
        assert_eq!(layer.input_size, 2);
        assert_eq!(layer.output_size, 3);
        assert_eq!(layer.weights.len(), 6);
        assert_eq!(layer.biases.len(), 3);
        assert_eq!(layer.activation, ActivationType::Relu);
    }

    #[test]
    fn test_invalid_architecture_input_zero() {
        let config = NetworkConfig {
            input_size: 0,
            hidden_sizes: vec![3],
            output_size: 1,
            hidden_activation: ActivationType::Relu,
        };
        let err = MultilayerPerceptron::new(&config).unwrap_err();
        assert!(matches!(err, GateLearnerError::InvalidArchitecture(_)));
        assert_eq!(
            err.to_string(),
            "invalid network architecture: Input size cannot be zero"
        );
    }

    #[test]
    fn test_invalid_architecture_output_zero() {
        let config = NetworkConfig {
            input_size: 2,
            hidden_sizes: vec![3],
            output_size: 0,
            hidden_activation: ActivationType::Relu,
        };
        let err = MultilayerPerceptron::new(&config).unwrap_err();
        assert!(matches!(err, GateLearnerError::InvalidArchitecture(_)));
        assert_eq!(
            err.to_string(),
            "invalid network architecture: Output size cannot be zero"
        );
    }

    #[test]
    fn test_invalid_architecture_hidden_zero() {
        let config = NetworkConfig {
            input_size: 2,
            hidden_sizes: vec![3, 0],
            output_size: 1,
            hidden_activation: ActivationType::Relu,
        };
        let err = MultilayerPerceptron::new(&config).unwrap_err();
        assert!(matches!(err, GateLearnerError::InvalidArchitecture(_)));
        assert_eq!(
            err.to_string(),
            "invalid network architecture: Hidden layer 1 size cannot be zero"
        );
    }

    #[test]
    fn test_activation_functions_correctness() {
        assert!((sigmoid(0.0f32) - 0.5f32).abs() < f32::EPSILON);
        assert!((sigmoid(1.0f32) - 0.731_058_6_f32).abs() < 1e-6f32);
        assert!((sigmoid(-1.0f32) - 0.268_941_43_f32).abs() < 1e-6f32);

        assert!((sigmoid_derivative(0.5f32) - 0.25f32).abs() < f32::EPSILON);
        let s_val = sigmoid(1.0f32);
        assert!((sigmoid_derivative(s_val) - (s_val * (1.0f32 - s_val))).abs() < f32::EPSILON);

        assert_eq!(relu(2.5f32), 2.5f32);
        assert_eq!(relu(-1.5f32), 0.0f32);
        assert_eq!(relu(0.0f32), 0.0f32);

        assert_eq!(relu_derivative(2.5f32), 1.0f32);
        assert_eq!(relu_derivative(-1.5f32), 0.0f32);
        assert_eq!(relu_derivative(0.0f32), 0.0f32);
    }

    #[test]
    fn test_activation_bounds() {
        assert!((sigmoid(1000.0f32) - 1.0f32).abs() < 1e-7f32);
        assert!((relu(1000.0f32) - 1000.0f32).abs() < 1e-7f32);
        assert!((relu_derivative(1000.0f32) - 1.0f32).abs() < 1e-7f32);

        assert!((sigmoid(-1000.0f32) - 0.0f32).abs() < 1e-7f32);
        assert!((relu(-1000.0f32) - 0.0f32).abs() < 1e-7f32);
        assert!((relu_derivative(-1000.0f32) - 0.0f32).abs() < 1e-7f32);

        assert!(sigmoid(f32::NAN).is_nan());
        assert!(!relu(f32::NAN).is_nan());
        assert!(!sigmoid(1000.0f32).is_nan());
        assert!(!sigmoid(-1000.0f32).is_nan());
        assert!(!relu(1000.0f32).is_nan());
        assert!(!relu(-1000.0f32).is_nan());
    }

    #[test]
    fn test_layer_initialisation_and_forward() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let layer = Layer::new_with_rng(2, 3, ActivationType::Relu, &mut rng);

        assert_eq!(layer.input_size, 2);
        assert_eq!(layer.output_size, 3);
        assert_eq!(layer.weights.len(), 2 * 3);
        assert_eq!(layer.biases.len(), 3);
        assert_eq!(layer.activation, ActivationType::Relu);

        let mut custom_layer = Layer {
            input_size: 2,
            output_size: 2,
            weights: vec![0.1f32, 0.3f32, 0.2f32, 0.4f32],
            biases: vec![0.0f32, 0.0f32],
            activation: ActivationType::Relu,
        };

        let input = vec![1.0f32, 1.0f32];
        let mut output = vec![0.0f32; 2];
        custom_layer.forward(&input, &mut output);
        assert!((output[0] - 0.4f32).abs() < f32::EPSILON);
        assert!((output[1] - 0.6f32).abs() < f32::EPSILON);

        custom_layer.activation = ActivationType::Sigmoid;
        custom_layer.forward(&input, &mut output);
        assert!((output[0] - sigmoid(0.4f32)).abs() < 1e-7f32);
        assert!((output[1] - sigmoid(0.6f32)).abs() < 1e-7f32);
    }

    #[test]
    fn test_buffer_allocation() {
        let config = NetworkConfig {
            input_size: 2,
            hidden_sizes: vec![3, 4],
            output_size: 1,
            hidden_activation: ActivationType::Relu,
        };
        let mut mlp = MultilayerPerceptron::new(&config).unwrap();
        mlp.ensure_buffers();

        let num_layers = mlp.layers.len();
        assert_eq!(mlp.activations.len(), num_layers + 1);
        assert_eq!(mlp.activations[0].len(), 2);
        assert_eq!(mlp.activations[1].len(), 3);
        assert_eq!(mlp.activations[2].len(), 4);
        assert_eq!(mlp.activations[3].len(), 1);

        assert_eq!(mlp.deltas.len(), num_layers);
        assert_eq!(mlp.deltas[0].len(), 3);
        assert_eq!(mlp.deltas[1].len(), 4);
        assert_eq!(mlp.deltas[2].len(), 1);

        assert_eq!(mlp.weight_grads.len(), num_layers);
        assert_eq!(mlp.weight_grads[0].len(), 2 * 3);
        assert_eq!(mlp.weight_grads[1].len(), 3 * 4);
        assert_eq!(mlp.weight_grads[2].len(), 4);

        assert_eq!(mlp.bias_grads.len(), num_layers);
        assert_eq!(mlp.bias_grads[0].len(), 3);
        assert_eq!(mlp.bias_grads[1].len(), 4);
        assert_eq!(mlp.bias_grads[2].len(), 1);
    }

    #[test]
    #[should_panic(expected = "Input size mismatch")]
    fn test_forward_panic_on_input_size_mismatch() {
        let config = NetworkConfig {
            input_size: 2,
            hidden_sizes: vec![3],
            output_size: 1,
            hidden_activation: ActivationType::Relu,
        };
        let mut mlp = MultilayerPerceptron::new(&config).unwrap();
        let invalid_input = vec![1.0f32, 0.0f32, 0.5f32];
        let mut output = vec![0.0f32; 1];
        mlp.forward(&invalid_input, &mut output);
    }

    #[test]
    fn test_zero_gradients() {
        let mut mlp = create_test_mlp(2, vec![3], 1, ActivationType::Relu, 1);
        mlp.weight_grads[0].fill(0.5f32);
        mlp.bias_grads[0].fill(0.5f32);

        mlp.zero_gradients();

        for grads in &mlp.weight_grads {
            assert!(grads.iter().all(|&g| g.abs() < f32::EPSILON));
        }
        for grads in &mlp.bias_grads {
            assert!(grads.iter().all(|&g| g.abs() < f32::EPSILON));
        }
    }

    #[test]
    fn test_accumulate_gradients_and_adam_update() {
        let config = NetworkConfig {
            input_size: 2,
            hidden_sizes: vec![2],
            output_size: 1,
            hidden_activation: ActivationType::Relu,
        };
        let mut mlp = MultilayerPerceptron::new_with_seed(&config, 123).unwrap();
        let mut optimizer = AdamOptimizer::new(&mlp);

        let initial_weights_l0 = mlp.layers[0].weights.clone();
        let initial_biases_l0 = mlp.layers[0].biases.clone();

        let input = vec![0.0f32, 1.0f32];
        let target = vec![1.0f32];
        let learning_rate = 0.1f32;
        let l2_lambda = 0.001f32;
        let batch_size = 1;

        let _loss = mlp.accumulate_gradients(&input, &target);
        optimizer.update(&mut mlp, learning_rate, l2_lambda, batch_size);

        assert!(
            mlp.layers[0]
                .weights
                .iter()
                .zip(initial_weights_l0.iter())
                .any(|(w_new, w_old)| (w_new - w_old).abs() > f32::EPSILON)
        );
        assert!(
            mlp.layers[0]
                .biases
                .iter()
                .zip(initial_biases_l0.iter())
                .any(|(b_new, b_old)| (b_new - b_old).abs() > f32::EPSILON)
        );

        let mut mlp_batch = create_test_mlp(2, vec![2], 1, ActivationType::Relu, 124);
        let mut optimizer_batch = AdamOptimizer::new(&mlp_batch);
        let initial_weights_batch = mlp_batch.layers[0].weights.clone();

        let dataset_batch = [
            (vec![0.0f32, 0.0f32], vec![0.0f32]),
            (vec![0.0f32, 1.0f32], vec![1.0f32]),
            (vec![1.0f32, 0.0f32], vec![1.0f32]),
            (vec![1.0f32, 1.0f32], vec![0.0f32]),
        ];
        let batch_size_test = 2;

        mlp_batch.zero_gradients();
        for item in dataset_batch.iter().take(batch_size_test) {
            let (input_k, target_k) = item;
            mlp_batch.accumulate_gradients(input_k, target_k);
        }
        optimizer_batch.update(&mut mlp_batch, learning_rate, l2_lambda, batch_size_test);

        assert!(
            mlp_batch.layers[0]
                .weights
                .iter()
                .zip(initial_weights_batch.iter())
                .any(|(w_new, w_old)| (w_new - w_old).abs() > f32::EPSILON)
        );
    }

    #[test]
    fn test_adam_optimizer_initialisation() {
        let mlp = create_test_mlp(2, vec![3, 4], 1, ActivationType::Relu, 1);
        let optimizer = AdamOptimizer::new(&mlp);

        assert_eq!(optimizer.beta1, 0.9f32);
        assert_eq!(optimizer.beta2, 0.999f32);
        assert_eq!(optimizer.epsilon, 1e-8f32);
        assert_eq!(optimizer.t, 0);

        assert_eq!(optimizer.m_weights.len(), mlp.layers.len());
        assert_eq!(optimizer.v_weights.len(), mlp.layers.len());
        assert_eq!(optimizer.m_biases.len(), mlp.layers.len());
        assert_eq!(optimizer.v_biases.len(), mlp.layers.len());

        for l in 0..mlp.layers.len() {
            assert_eq!(optimizer.m_weights[l].len(), mlp.layers[l].weights.len());
            assert_eq!(optimizer.v_weights[l].len(), mlp.layers[l].weights.len());
            assert_eq!(optimizer.m_biases[l].len(), mlp.layers[l].biases.len());
            assert_eq!(optimizer.v_biases[l].len(), mlp.layers[l].biases.len());

            assert!(optimizer.m_weights[l].iter().all(|&v| v == 0.0f32));
            assert!(optimizer.v_weights[l].iter().all(|&v| v == 0.0f32));
            assert!(optimizer.m_biases[l].iter().all(|&v| v == 0.0f32));
            assert!(optimizer.v_biases[l].iter().all(|&v| v == 0.0f32));
        }
    }

    #[test]
    fn test_mlp_forward_success() {
        let config = NetworkConfig {
            input_size: 2,
            hidden_sizes: vec![3],
            output_size: 1,
            hidden_activation: ActivationType::Relu,
        };
        let mut mlp = MultilayerPerceptron::new_with_seed(&config, 42).unwrap();
        let input = vec![1.0f32, 0.0f32];
        let mut output = vec![0.0f32; 1];
        mlp.forward(&input, &mut output);
        assert_eq!(output.len(), 1);
        assert!(output[0] >= 0.0f32 && output[0] <= 1.0f32);
    }
}
