# Gate Learner

A lightweight, high-performance Multilayer Perceptron (MLP) engine written from scratch in Rust to train and evaluate $N$-input logic gates (such as OR and XOR).

[![Crates.io](https://img.shields.io/badge/rust-1.80.0%2B-blue.svg)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

## Overview

`gate_learner` is a zero-unsafe, single-threaded neural network engine designed to solve non-linear classification problems (like the XOR parity problem) for arbitrary $N$-input dimensions. It features a manually implemented forward pass, backpropagation, Binary Cross Entropy (BCE) loss, and an Adam Optimizer with L2 regularisation.

## Features

- **Custom MLP Engine** \
Built entirely from scratch without external machine learning or tensor libraries.
- **Adam Optimizer** \
Includes momentum, RMSProp-like scaling, bias correction, and L2 regularisation.
- **Flexible Activations** \
Supports both Sigmoid and ReLU activation functions with appropriate weight initialisation (Xavier/Glorot and He Uniform).
- **Dynamic Dataset Generator** \
Generates truth tables for $N$-input OR and XOR gates dynamically.
- **Early Stopping** \
Halts training automatically when the model converges (Loss < 0.01) or stalls (Loss change < 1e-6 over 50 epochs).
- **Visualisation** \
Optional dual-axis plotting of training loss and accuracy curves using the `plotters` library.
- **Serialisation** \
Save and load trained models and training histories to/from JSON.

## Installation

Ensure you have Rust 1.80.0 or newer installed. Clone the repository and build the project:

```bash
git clone https://github.com/ios-community/gate-learner.git
cd gate-learner
cargo build --release
```

## Usage

### Command Line Interface (CLI)

Run the binary to train an MLP on a specific logic gate. For example, to train a 2-input XOR gate:

```bash
cargo run --release -- \
  --gate xor \
  --inputs 2 \
  --epochs 10000 \
  --lr 0.01 \
  --batch-size 4 \
  --hidden-sizes 4 \
  --hidden-activation sigmoid \
  --seed 42
```

#### CLI Arguments

- `-g, --gate <GATE>`: Logic gate type (`or`, `xor`) [default: `xor`].
- `-i, --inputs <INPUTS>`: Number of inputs (1 to 19) [default: 2].
- `-e, --epochs <EPOCHS>`: Maximum training epochs [default: 10000].
- `-l, --lr <LR>`: Learning rate for the Adam Optimizer [default: 0.01].
- `-b, --batch-size <BATCH_SIZE>`: Batch size for training [default: 4].
- `--l2 <L2>`: L2 regularisation penalty coefficient [default: 0.0001].
- `--hidden-sizes <HIDDEN_SIZES>`: Comma-separated hidden layer sizes [default: `4`].
- `--hidden-activation <ACTIVATION>`: Hidden layer activation function (`sigmoid`, `relu`) [default: `relu`].
- `--seed <SEED>`: Optional seed for reproducible weight initialisation and dataset shuffling.
- `--model-out <PATH>`: Path to save the trained model JSON [default: `model.json`].
- `--history-out <PATH>`: Path to save the training history JSON [default: `history.json`].
- `--plot-out <PATH>`: Path to save the training plot PNG [default: `plot.png`].

### Library Usage

You can also use `gate_learner` as a library in your own Rust projects. Add it to your `Cargo.toml` dependencies, then use the API:

```rust
use gate_learner::core::{ActivationType, MultilayerPerceptron, NetworkConfig};

fn main() -> Result<(), gate_learner::error::GateLearnerError> {
    let config = NetworkConfig {
        input_size: 2,
        hidden_sizes: vec![4],
        output_size: 1,
        hidden_activation: ActivationType::Relu,
    };

    // Initialise the network
    let mut mlp = MultilayerPerceptron::new(config)?;

    // Perform a forward pass
    let input = vec![1.0f32, 0.0f32];
    let mut output = vec![0.0f32; 1];
    mlp.forward(&input, &mut output);

    println!("Output: {:?}", output);
    Ok(())
}
```

## Feature Gating

The plotting functionality is optional and can be disabled to compile the engine in minimal or headless environments without graphical dependencies.

- **`visualise` (Enabled by default):** Pulls in the `plotters` dependency to render PNG plots.

To compile without plotting support:

```bash
cargo build --release --no-default-features
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
