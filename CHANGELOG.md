# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-25

### Added
- Initial release of the `gate_learner` neural network engine.
- Custom Multilayer Perceptron (MLP) implementation with support for arbitrary hidden layer architectures.
- Sigmoid and ReLU activation functions with Xavier/Glorot and He Uniform weight initialisation.
- Adam Optimizer with L2 regularisation, momentum, and bias correction.
- Dynamic dataset generator for $N$-input OR and XOR logic gates.
- Early stopping mechanisms based on convergence (Loss < 0.01) and stalling (Loss change < 1e-6 over 50 epochs).
- JSON serialisation and deserialisation for models and training histories.
- Dual-axis plotting of training loss and accuracy curves (gated under the `visualise` feature).
- Comprehensive unit tests, integration tests, and performance benchmarks using `criterion`.
