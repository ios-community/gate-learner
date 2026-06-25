# Requirements Specification: Gate Learner Project

**Role:** Architect Engineer → Senior Software Engineer

**Status:** Approved
**Target Registry:** GitHub / Internal
**MSRV:** Rust 1.80.0 | Edition: 2024

## Architect Directives

- **Primary Objective:** Build a Rust-based CLI application to train a multilayer perceptron (MLP) from scratch to learn $N$-input logic gates (OR and XOR), record training results to JSON, and visualise them into static image files (PNG).
- **Crate Type:** `bin` (CLI application) with an internal `lib` module (`gate_learner`) that can be tested independently.
- **Concurrency Model:** Single-threaded for the core training process to avoid thread coordination overhead on small workloads, but data structures must be thread-safe (`Send + Sync`).
- **Memory Safety:** Zero unsafe (`#![deny(unsafe_code)]`). Minimal and efficient memory allocation using standard vectors (`Vec<f32>`) allocated once at initialization and mutated in-place during training.
- **Portability:** `std` (required for file I/O, JSON writing, and PNG image rendering).
- **Documentation Contract:** Fully comply with the "Rust Documentation Standard" with doctest coverage on key public functions.

## Functional Requirements (FR)

| ID | Requirement | Owner | Description |
| --- | --- | --- | --- |
| FR-01 | **N-Input MLP Engine** | Senior SE | Implement a multilayer perceptron (MLP) from scratch with a dynamic number of inputs $N$, supporting one or more hidden layers to solve non-linear problems (XOR). |
| FR-02 | **Dataset Generator** | Senior SE | Automatic generator for OR and XOR logic gate truth tables with $N$-inputs (producing $2^N$ input combinations and their target outputs). |
| FR-03 | **Training & Backpropagation** | Senior SE | Training algorithm using the Adam Optimizer with Sigmoid/ReLU activation and Binary Cross Entropy (BCE) loss, written manually without external ML crates. |
| FR-04 | **Early Stopping Mechanism** | Senior SE | Automatically halt training if the average BCE loss falls below `0.01` ("Already Smart") or if the average loss change is less than `1e-6` over 50 consecutive epochs ("Stuck/Stalled"). |
| FR-05 | **JSON Serialisation** | Senior SE | Save trained model weights and biases, as well as training history (loss and accuracy per epoch), to `.json` files. |
| FR-06 | **Static Plot Generation** | Senior SE | Generate static image files (`.png`) visualising the loss reduction and accuracy improvement curves during training using the `plotters` library. |
| FR-07 | **CLI Interface** | Senior SE | Command-line interface to select the gate type (OR/XOR), number of inputs ($N$), number of epochs, learning rate, batch size, L2 regularisation coefficient, hidden layer sizes, activation type, random seed, and output storage paths. |

## Non-Functional Requirements (NFR)

| ID | Category | Constraint | Validation Method |
| --- | --- | --- | --- |
| NFR-01 | Performance | Training 2-input OR/XOR gates to convergence (Loss < 0.01) must complete in < 50 ms. | Benchmark testing using `criterion`. |
| NFR-02 | Safety | No unsafe code allowed in the codebase. | Static analysis via `#![deny(unsafe_code)]` and `cargo clippy` verification. |
| NFR-03 | Documentation | Full compliance with strict documentation lints and all doctests must pass. | Execution of `RUSTDOCFLAGS="-D warnings" cargo doc` and `cargo test --doc`. |
| NFR-04 | Robustness | Graceful error handling for I/O failures (e.g., failed to write JSON or PNG files) without causing unexpected panics. | Integration testing with file write permission failure scenarios. |

## Performance Targets

| Component | Metric | Bound | Measurement |
| --- | --- | --- | --- |
| MLP Forward Pass | Latency | ≤ 500 ns per evaluation (2-input) | `criterion` benchmark |
| MLP Backward Pass | Latency | ≤ 2 µs per update step | `criterion` benchmark |

## Out of Scope

- GPU acceleration (CUDA/Vulkan).
- Use of external tensor computation libraries (such as `ndarray`, `tch`, or `burn`).
- Interactive graphical user interfaces (GUI).
