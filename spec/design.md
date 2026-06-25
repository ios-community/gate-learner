# Architecture & Design Specification: Gate Learner

**Role:** Architect Directive → Senior Engineer Blueprint

**Revision:** 0.2.0 | **Toolchain:** Rust 1.80.0 (Edition 2024)

## Architect Directives

- **Decoupling Rule:** The neural network mathematical logic (`core`) must be completely decoupled from I/O logic (`storage`), visualisation (`plot`), and the user interface (`cli`). Interaction between modules is bridged by pure data transfer objects (DTOs) without circular dependencies.
- **Memory Ownership Constraint:** Avoid repetitive memory allocations inside the training loop. Matrix/vector structures for activations, deltas, and gradients are allocated once at initialization (`ensure_buffers`) and mutated in-place where possible.
- **Thread Model Requirement:** The training process runs synchronously on the main thread. However, the model structure (`MultilayerPerceptron`) must implement `Send + Sync` so that it can be run within a thread pool in the future (e.g., for parallel hyperparameter grid search).
- **Feature Gating:** Provide an optional `visualise` feature (using the `plotters` dependency) so that the core engine can still compile without graphical dependencies in minimal server environments.

## Senior Engineer Specification

### 1. System Architecture

```text
+-----------------------------------------------------------------------+
|                              CLI Module                               |
|       (Accepts arguments: Gate Type, N-Input, Epochs, LR, Batch, L2)  |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------+-----------------------------------+
|                           Dataset Generator                           |
|             (Generates input combinations & targets for OR/XOR)       |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------+-----------------------------------+
|                           Core ML Engine                              |
|   +---------------------------------------------------------------+   |
|   |                     MultilayerPerceptron                      |   |
|   |  - Forward Pass (Sigmoid / ReLU Activation)                   |   |
|   |  - Gradient Accumulation (BCE Loss)                           |   |
|   +-------------------------------+-------------------------------+   |
|                                   |                                   |
|                                   v                                   |
|   +---------------------------------------------------------------+   |
|   |                        AdamOptimizer                          |   |
|   |  - Parameter Update (L2 Regularisation & Bias Correction)     |   |
|   +---------------------------------------------------------------+   |
+-----------------------------------+-----------------------------------+
                                    |
                                    +-----------------------+
                                    |                       |
                                    v                       v
+-----------------------------------+---+               +---+--------------------+
|            Storage Module             |               |  Plot Module (Feature) |
|  - Model Serialisation to JSON        |               |  - Render Loss Curve   |
|  - History Serialisation to JSON      |               |  - Render Accuracy     |
+---------------------------------------+               +------------------------+
```

- **Pipeline:** CLI Input → Parameter Validation → MLP & Dataset Initialisation → Training Loop (Zero Gradients → Accumulate Gradients per Batch → Adam Update → Early Stopping Evaluation) → Result Recording (JSON) → Plot Rendering (PNG).

### 2. Module Structure & Responsibilities

| Module | Path | Responsibility | Architect Constraint |
| --- | --- | --- | --- |
| `lib` | `src/lib.rs` | Crate root, module declarations, public re-exports, and enforcement of documentation lints. | `#![deny(unsafe_code)]` |
| `core` | `src/core.rs` | Implementation of `Layer`, `MultilayerPerceptron`, activation functions (Sigmoid, ReLU), and `AdamOptimizer`. | Free of external dependencies, efficient memory allocation. |
| `dataset` | `src/dataset.rs` | Dynamic generation of OR and XOR truth tables for $N$-inputs. | Produces clean data vectors with safety bounds ($N < 20$). |
| `storage` | `src/storage.rs` | Saving and loading model weights and training history to/from JSON files. | Uses `serde` safely without panicking. |
| `plot` | `src/plot.rs` | Generating PNG plots to visualise loss and accuracy per epoch. | Gated under the `#[cfg(feature = "visualise")]` flag. |
| `error` | `src/error.rs` | Definition of the `GateLearnerError` enum for centralised error handling. | Uses `thiserror` implementation for easy conversion. |

### 3. Concurrency & Memory Model

- **Read/Write Path:** During training, weight mutation is performed directly on the internal `Layer` structure using exclusive references (`&mut self`). No interior mutability (`RefCell` or `Mutex`) is used within the training hot loop to guarantee maximum performance.
- **Ownership:** The training function consumes the configuration and returns ownership of the `TrainingHistory` and the updated `MultilayerPerceptron` to the caller.

### 4. API Surface Contract

```rust
/// Supported activation function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ActivationType {
    /// Sigmoid activation function.
    Sigmoid,
    /// ReLU activation function.
    Relu,
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

/// Representation of the Multi-Layer Perceptron artificial neural network.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultilayerPerceptron {
    /// Layers constituting the neural network.
    pub layers: Vec<Layer>,
}

impl MultilayerPerceptron {
    /// Creates a new MLP instance with randomised weight initialisation (He/Xavier Uniform).
    pub fn new(config: NetworkConfig) -> Result<Self, GateLearnerError>;

    /// Creates a new MLP instance with a specific seed for reproducibility.
    pub fn new_with_seed(config: NetworkConfig, seed: u64) -> Result<Self, GateLearnerError>;

    /// Performs forward propagation for the given input.
    pub fn forward(&mut self, input: &[f32], output: &mut [f32]);

    /// Zeroes out the accumulated gradients before processing a new batch.
    pub fn zero_gradients(&mut self);

    /// Accumulates gradients for a single input-target pair using BCE Loss.
    /// Returns the loss value and a reference to the output activations.
    pub fn accumulate_gradients(&mut self, input: &[f32], target: &[f32]) -> (f32, &[f32]);
}

/// Adam Optimizer for updating the weights and biases of a MultilayerPerceptron.
pub struct AdamOptimizer { ... }

impl AdamOptimizer {
    /// Creates a new Adam Optimizer instance matching the architecture of the given MLP.
    pub fn new(mlp: &MultilayerPerceptron) -> Self;

    /// Updates the weights and biases of the MLP using the accumulated gradients.
    pub fn update(
        &mut self,
        mlp: &mut MultilayerPerceptron,
        learning_rate: f32,
        l2_lambda: f32,
        batch_size: usize,
    );
}
```

### 5. Math Implementation Details

- **Activation Functions:**
  - **Sigmoid:** $f(x) = \frac{1}{1 + e^{-x}}$ | Derivative: $f'(y) = y \cdot (1 - y)$ where $y = f(x)$.
  - **ReLU:** $f(x) = \max(0, x)$ | Derivative: $f'(y) = 1$ if $y > 0$, else $0$.
- **Weight Initialisation:**
  - **Xavier/Glorot Uniform** (for Sigmoid): Limits of $\pm \sqrt{\frac{6}{\text{input\_size} + \text{output\_size}}}$.
  - **He Uniform** (for ReLU): Limits of $\pm \sqrt{\frac{6}{\text{input\_size}}}$.
- **Loss Function:**
  - **Binary Cross Entropy (BCE):** $L = - [t \ln(o) + (1 - t) \ln(1 - o)]$ where $t$ is the target and $o$ is the output clipped to avoid undefined values ($\ln(0)$).
- **Optimizer:**
  - **Adam Optimizer** with bias correction for the first moment ($m$) and second moment ($v$), supporting L2 regularisation penalties applied directly to the gradients.
