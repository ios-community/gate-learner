# Architecture & Design

This document describes the high-level architecture of the `gate_learner` project.

## Module Overview

The project is structured as a decoupled library with a thin CLI binary wrapper. This separation ensures that the core mathematical engine remains independent of I/O, CLI parsing, and plotting logic.

```text
+-----------------------------------------------------------------------+
|                              CLI Module                               |
|                           (src/cli.rs, main)                          |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------+-----------------------------------+
|                           Dataset Generator                           |
|                           (src/dataset.rs)                            |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------+-----------------------------------+
|                           Core ML Engine                              |
|                           (src/core.rs)                               |
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
|          (src/storage.rs)             |               |    (src/plot.rs)       |
+---------------------------------------+               +------------------------+
```

### 1. Core ML Engine (`src/core.rs`)
Contains the mathematical representation of the neural network.
- **`Layer`**: Represents a single layer containing weights, biases, and an activation function.
- **`MultilayerPerceptron`**: Manages a sequence of layers. It holds pre-allocated internal buffers for activations, deltas, and gradients to avoid heap allocations during the training loop.
- **`AdamOptimizer`**: Implements the Adam optimization algorithm. It maintains its own state buffers (first and second moments) matching the network's architecture.

### 2. Dataset Generator (`src/dataset.rs`)
Generates truth tables for $N$-input OR and XOR gates. It includes safety checks to prevent memory exhaustion by limiting $N < 20$ (since dataset size scales exponentially as $2^N$).

### 3. Storage Module (`src/storage.rs`)
Handles JSON serialisation and deserialisation of the trained model weights and training history using `serde_json`.

### 4. Plot Module (`src/plot.rs`)
Gated behind the `visualise` feature. It uses the `plotters` crate to render a dual-axis PNG chart showing the training loss and accuracy curves over epochs.

### 5. CLI Module (`src/cli.rs` & `src/main.rs`)
Parses command-line arguments using `clap`, coordinates the training loop, manages early stopping, and handles output generation.

## Key Architectural Decisions

### Memory Management & Allocation
To achieve high performance and meet the strict latency targets, heap allocations are avoided inside the training loop. 
- The `MultilayerPerceptron` allocates its activation, delta, and gradient buffers once during initialization via `ensure_buffers()`.
- During the forward and backward passes, these buffers are mutated in-place.

### Thread Safety
The training process runs synchronously on a single thread to avoid multi-threading overhead on small logic gate datasets. However, `MultilayerPerceptron` implements `Send + Sync` because all its internal fields are thread-safe primitive types (`Vec<f32>`). This allows the model to be safely transferred across threads if needed in the future.

### Error Handling
All fallible operations return a `Result<T, GateLearnerError>`. Performance-critical internal functions use `debug_assert!` or standard assertions to verify array boundaries, avoiding runtime overhead in release builds while maintaining safety.
