#![deny(unsafe_code)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_precision_loss)]

//! Core CLI application logic for the Boole Project.

use crate::core::{ActivationType, AdamOptimizer, MultilayerPerceptron, NetworkConfig};
use crate::dataset::{generate_or, generate_xor};
use crate::error::GateLearnerError;
use crate::plot::plot_history;
use crate::storage::{TrainingHistory, save_history, save_model};
use clap::Parser;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use std::path::PathBuf;

fn validate_lr(s: &str) -> Result<f32, String> {
    let val: f32 = s.parse().map_err(|_| "Must be a valid float".to_string())?;
    if val <= 0.0f32 {
        Err("Learning rate must be greater than 0.0".to_string())
    } else {
        Ok(val)
    }
}

/// CLI arguments for the Boole Project.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Train an MLP to learn N-input logic gates (OR/XOR)"
)]
pub struct Args {
    /// Type of logic gate to learn (or, xor).
    #[arg(short, long, default_value = "xor")]
    pub gate: GateType,

    /// Number of inputs for the logic gate.
    #[arg(short, long, default_value_t = 2, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1u64..=19u64))]
    pub inputs: usize,

    /// Number of training epochs.
    #[arg(short, long, default_value_t = 10000, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1u64..))]
    pub epochs: usize,

    /// Learning rate for the Adam Optimizer.
    #[arg(short, long, default_value_t = 0.01, value_parser = validate_lr)]
    pub lr: f32,

    /// Batch size for training.
    #[arg(short, long, default_value_t = 4, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1u64..))]
    pub batch_size: usize,

    /// L2 regularisation penalty coefficient.
    #[arg(long, default_value_t = 0.0001)]
    pub l2: f32,

    /// Path to save the trained model weights (JSON).
    #[arg(long, default_value = "model.json")]
    pub model_out: PathBuf,

    /// Path to save the training history (JSON).
    #[arg(long, default_value = "history.json")]
    pub history_out: PathBuf,

    /// Path to save the training plot (PNG).
    #[arg(long, default_value = "plot.png")]
    pub plot_out: PathBuf,

    /// Hidden layer sizes.
    #[arg(long, value_delimiter = ',', default_value = "4")]
    pub hidden_sizes: Vec<usize>,

    /// Hidden layer activation function.
    #[arg(long, default_value = "relu")]
    pub hidden_activation: ActivationType,

    /// Seed for reproducibility.
    #[arg(long)]
    pub seed: Option<u64>,
}

/// Type of logic gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum GateType {
    /// OR logic gate.
    Or,
    /// XOR logic gate.
    Xor,
}

/// Runs the main application logic.
///
/// # Errors
///
/// Returns a [`GateLearnerError`] if any part of the pipeline fails.
#[allow(clippy::too_many_lines)]
pub fn run_app(args: &Args) -> Result<(), GateLearnerError> {
    println!("--- Boole Project: MLP Logic Gate Learner ---");
    println!("Gate Type:         {gate:?}", gate = args.gate);
    println!("Inputs (N):        {inputs}", inputs = args.inputs);
    println!("Epochs:            {epochs}", epochs = args.epochs);
    println!("Learning Rate:     {lr}", lr = args.lr);
    println!(
        "Batch Size:        {batch_size}",
        batch_size = args.batch_size
    );
    println!("L2 Penalty:        {l2}", l2 = args.l2);
    println!(
        "Hidden Sizes:      {hidden_sizes:?}",
        hidden_sizes = args.hidden_sizes
    );
    println!(
        "Hidden Activation: {hidden_activation:?}",
        hidden_activation = args.hidden_activation
    );
    println!("Seed:              {seed:?}", seed = args.seed);

    let mut dataset = match args.gate {
        GateType::Or => generate_or(args.inputs)?,
        GateType::Xor => generate_xor(args.inputs)?,
    };

    let config = NetworkConfig {
        input_size: args.inputs,
        hidden_sizes: args.hidden_sizes.clone(),
        output_size: 1,
        hidden_activation: args.hidden_activation,
    };

    let mut mlp = if let Some(seed) = args.seed {
        MultilayerPerceptron::new_with_seed(&config, seed)?
    } else {
        MultilayerPerceptron::new(&config)?
    };
    let mut optimizer = AdamOptimizer::new(&mlp);

    let mut shuffle_rng = if let Some(seed) = args.seed {
        ChaCha8Rng::seed_from_u64(seed)
    } else {
        ChaCha8Rng::from_rng(&mut rand::rng())
    };

    let mut loss_history = Vec::with_capacity(args.epochs);
    let mut acc_history = Vec::with_capacity(args.epochs);

    for epoch in 0..args.epochs {
        dataset.shuffle(&mut shuffle_rng);

        let mut epoch_loss = 0.0f32;
        let mut correct_predictions = 0_usize;

        for batch in dataset.chunks(args.batch_size) {
            let current_batch_size = batch.len();

            mlp.zero_gradients();

            for (input, target) in batch {
                let (loss, output) = mlp.accumulate_gradients(input, target);
                epoch_loss += loss;

                let predicted = if output[0] >= 0.5f32 { 1.0f32 } else { 0.0f32 };
                if (predicted - target[0]).abs() < f32::EPSILON {
                    correct_predictions += 1;
                }
            }

            optimizer.update(&mut mlp, args.lr, args.l2, current_batch_size);
        }

        let (avg_loss, accuracy) = if dataset.is_empty() {
            (0.0f32, 0.0f32)
        } else {
            let avg = epoch_loss / dataset.len() as f32;
            let acc = correct_predictions as f32 / dataset.len() as f32;
            (avg, acc)
        };

        loss_history.push(avg_loss);
        acc_history.push(accuracy);

        if epoch % (args.epochs / 10).max(1) == 0 || epoch == args.epochs - 1 {
            let acc_pct = accuracy * 100.0;
            println!("Epoch {epoch:5} | Loss (BCE): {avg_loss:.6} | Accuracy: {acc_pct:.2}%");
        }

        if avg_loss < 0.01f32 {
            println!(
                "Early stopping triggered: Model is 'Already Smart' (Loss {avg_loss:.6} < 0.01) at epoch {epoch}."
            );
            break;
        }

        if loss_history.len() >= 50 {
            let window = &loss_history[loss_history.len() - 50..];
            let max_loss = window.iter().copied().fold(f32::MIN, f32::max);
            let min_loss = window.iter().copied().fold(f32::MAX, f32::min);
            if (max_loss - min_loss) < 1e-6f32 {
                let diff = max_loss - min_loss;
                println!(
                    "Early stopping triggered: Model is 'Stuck/Stalled' (Loss change {diff:.8} < 1e-6 over last 50 epochs) at epoch {epoch}."
                );
                break;
            }
        }
    }

    loss_history.shrink_to_fit();
    acc_history.shrink_to_fit();

    let model_out_path = args.model_out.display();
    println!("Saving model to {model_out_path}");
    save_model(&mlp, &args.model_out)?;

    let history = TrainingHistory {
        loss: loss_history,
        accuracy: acc_history,
    };

    let history_out_path = args.history_out.display();
    println!("Saving history to {history_out_path}");
    save_history(&history, &args.history_out)?;

    let plot_out_path = args.plot_out.display();
    println!("Generating plot to {plot_out_path}");
    match plot_history(&history, &args.plot_out) {
        Ok(()) => println!("Plot successfully generated."),
        Err(GateLearnerError::FeatureNotEnabled(msg)) => {
            eprintln!("==================================================");
            eprintln!(" PERINGATAN: Pembuatan gambar grafik dilewati!");
            eprintln!(" Alasan: {msg}");
            eprintln!("==================================================");
        }
        Err(e) => return Err(e),
    }

    println!("Training complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_run_app_or() {
        let temp_dir = tempdir().unwrap();
        let model_file = temp_dir.path().join("model.json");
        let history_file = temp_dir.path().join("history.json");
        let plot_file = temp_dir.path().join("plot.png");
        let args = Args {
            gate: GateType::Or,
            inputs: 2,
            epochs: 5,
            lr: 0.1f32,
            batch_size: 2,
            l2: 0.0001f32,
            model_out: model_file,
            history_out: history_file,
            plot_out: plot_file,
            hidden_sizes: vec![2],
            hidden_activation: ActivationType::Relu,
            seed: Some(42),
        };
        let result = run_app(&args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_app_xor() {
        let temp_dir = tempdir().unwrap();
        let model_file = temp_dir.path().join("model.json");
        let history_file = temp_dir.path().join("history.json");
        let plot_file = temp_dir.path().join("plot.png");
        let args = Args {
            gate: GateType::Xor,
            inputs: 2,
            epochs: 5,
            lr: 0.1f32,
            batch_size: 2,
            l2: 0.0001f32,
            model_out: model_file,
            history_out: history_file,
            plot_out: plot_file,
            hidden_sizes: vec![4],
            hidden_activation: ActivationType::Sigmoid,
            seed: Some(42),
        };
        let result = run_app(&args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_app_no_seed() {
        let temp_dir = tempdir().unwrap();
        let model_file = temp_dir.path().join("model.json");
        let history_file = temp_dir.path().join("history.json");
        let plot_file = temp_dir.path().join("plot.png");
        let args = Args {
            gate: GateType::Or,
            inputs: 2,
            epochs: 2,
            lr: 0.1f32,
            batch_size: 2,
            l2: 0.0001f32,
            model_out: model_file,
            history_out: history_file,
            plot_out: plot_file,
            hidden_sizes: vec![2],
            hidden_activation: ActivationType::Relu,
            seed: None,
        };
        let result = run_app(&args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_app_already_smart() {
        let temp_dir = tempdir().unwrap();
        let model_file = temp_dir.path().join("model.json");
        let history_file = temp_dir.path().join("history.json");
        let plot_file = temp_dir.path().join("plot.png");
        let args = Args {
            gate: GateType::Or,
            inputs: 2,
            epochs: 1000,
            lr: 0.1f32,
            batch_size: 2,
            l2: 0.0001f32,
            model_out: model_file,
            history_out: history_file,
            plot_out: plot_file,
            hidden_sizes: vec![2],
            hidden_activation: ActivationType::Relu,
            seed: Some(42),
        };
        let result = run_app(&args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_app_stuck_stalled() {
        let temp_dir = tempdir().unwrap();
        let model_file = temp_dir.path().join("model.json");
        let history_file = temp_dir.path().join("history.json");
        let plot_file = temp_dir.path().join("plot.png");
        let args = Args {
            gate: GateType::Or,
            inputs: 2,
            epochs: 55,
            lr: 0.0f32,
            batch_size: 2,
            l2: 0.0001f32,
            model_out: model_file,
            history_out: history_file,
            plot_out: plot_file,
            hidden_sizes: vec![2],
            hidden_activation: ActivationType::Relu,
            seed: Some(42),
        };
        let result = run_app(&args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_app_plot_error() {
        let temp_dir = tempdir().unwrap();
        let model_file = temp_dir.path().join("model.json");
        let history_file = temp_dir.path().join("history.json");
        let plot_file = PathBuf::from("/nonexistent_dir/plot.png");
        let args = Args {
            gate: GateType::Or,
            inputs: 2,
            epochs: 2,
            lr: 0.1f32,
            batch_size: 2,
            l2: 0.0001f32,
            model_out: model_file,
            history_out: history_file,
            plot_out: plot_file,
            hidden_sizes: vec![2],
            hidden_activation: ActivationType::Relu,
            seed: Some(42),
        };
        let result = run_app(&args);
        #[cfg(feature = "visualise")]
        assert!(result.is_err());
        #[cfg(not(feature = "visualise"))]
        assert!(result.is_ok());
    }
}
