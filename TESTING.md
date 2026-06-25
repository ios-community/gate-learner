# Testing & CI/CD Guide

This document explains how to run the test suite, generate coverage reports, execute performance benchmarks, and understand the automated CI/CD pipeline for the `gate_learner` project.

## Prerequisites

Ensure you have the Rust toolchain installed. To run coverage reports locally, you will also need `cargo-tarpaulin`:

```bash
cargo install cargo-tarpaulin
```

---

## Running Tests Locally

### 1. Unit and Integration Tests
To run all unit tests (located at the bottom of each module file) and integration tests (located in `tests/`):

```bash
cargo test
```

### 2. Documentation Tests
To verify that all code examples in the documentation and doc comments compile and run correctly:

```bash
cargo test --doc
```

### 3. Testing Specific Features
To run tests without the optional `visualise` feature:

```bash
cargo test --no-default-features
```

---

## Code Coverage

We use `cargo-tarpaulin` to measure test coverage. The configuration is defined in `tarpaulin.toml`.

To run coverage and generate an HTML report locally:

```bash
cargo tarpaulin
```

This will generate a `tarpaulin-report.html` file in the root directory, which you can open in any web browser to inspect line-by-line coverage.

---

## Benchmarking

We use `criterion` to measure the latency of critical operations (such as the forward pass and Adam update steps).

To run the benchmarks:

```bash
cargo bench
```

This will execute the benchmarks defined in `benches/mlp_benchmark.rs` and generate detailed HTML reports under `target/criterion/report/index.html`.

---

## Continuous Integration & Deployment (CI/CD)

The project uses GitHub Actions to automate code quality checks, testing, coverage verification, and documentation deployment. The workflow is defined in `.github/workflows/ci.yml`.

### Workflow Jobs

1. **Code Quality & Formatting (`check`):**
   - Verifies code formatting using `cargo fmt`.
   - Runs static analysis using `cargo clippy` with strict warning denials (`-D warnings`) across all features.

2. **Test Suite (`test`):**
   - Runs the test suite with `--all-features` to ensure full coverage.
   - Runs the test suite with `--no-default-features` to verify compilation in minimal/headless environments.

3. **Code Coverage (`coverage`):**
   - Installs `cargo-tarpaulin` and generates an XML coverage report to verify that test coverage remains stable.

4. **Deploy Documentation (`deploy-docs`):**
   - **Trigger:** Runs only on successful pushes to the `main` branch.
   - **Action:** Builds the API documentation using `cargo doc --no-deps --all-features`.
   - **Redirect:** Automatically generates a root `index.html` file that redirects visitors from the root URL directly to the `gate_learner` documentation.
   - **Deployment:** Deploys the generated documentation directly to GitHub Pages using GitHub's modern Actions-based deployment.

### Enabling GitHub Pages in Your Repository

To allow the CI/CD pipeline to deploy the documentation, you must configure your GitHub repository settings:

1. Go to your repository on GitHub.
2. Navigate to **Settings** > **Pages**.
3. Under **Build and deployment** > **Source**, select **GitHub Actions** from the dropdown menu.
4. The workflow will now automatically deploy the documentation to `https://github.com/ios-community/gate-learner` on every push to `main`.
```
