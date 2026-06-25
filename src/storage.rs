//! Serialisation and deserialisation of models and training history.

use crate::core::MultilayerPerceptron;
use crate::error::GateLearnerError;
use std::fs::File;
use std::path::Path;

/// Represents the training history of a model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingHistory {
    /// Loss value per epoch.
    pub loss: Vec<f32>,
    /// Accuracy value per epoch.
    pub accuracy: Vec<f32>,
}

/// Saves the trained model to a JSON file.
///
/// # Errors
///
/// Returns [`GateLearnerError::Io`] if writing to the file fails,
/// or [`GateLearnerError::Json`] if serialisation fails.
pub fn save_model<P: AsRef<Path>>(
    model: &MultilayerPerceptron,
    path: P,
) -> Result<(), GateLearnerError> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, model)?;
    Ok(())
}

/// Loads a model from a JSON file.
///
/// # Errors
///
/// Returns [`GateLearnerError::Io`] if reading from the file fails,
/// or [`GateLearnerError::Json`] if deserialisation fails.
pub fn load_model<P: AsRef<Path>>(path: P) -> Result<MultilayerPerceptron, GateLearnerError> {
    let file = File::open(path)?;
    let mut model: MultilayerPerceptron = serde_json::from_reader(file)?;
    model.ensure_buffers();
    Ok(model)
}

/// Saves the training history to a JSON file.
///
/// # Errors
///
/// Returns [`GateLearnerError::Io`] if writing to the file fails,
/// or [`GateLearnerError::Json`] if serialisation fails.
pub fn save_history<P: AsRef<Path>>(
    history: &TrainingHistory,
    path: P,
) -> Result<(), GateLearnerError> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, history)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ActivationType, MultilayerPerceptron, NetworkConfig};
    use std::{
        io::{self, Write},
        path::PathBuf,
    };
    use tempfile::NamedTempFile;

    fn create_dummy_mlp() -> MultilayerPerceptron {
        let config = NetworkConfig {
            input_size: 2,
            hidden_sizes: vec![3],
            output_size: 1,
            hidden_activation: ActivationType::Relu,
        };
        MultilayerPerceptron::new(&config).unwrap()
    }

    fn create_dummy_history() -> TrainingHistory {
        TrainingHistory {
            loss: vec![0.5f32, 0.2f32, 0.1f32],
            accuracy: vec![0.5f32, 0.8f32, 0.9f32],
        }
    }

    #[test]
    fn test_save_and_load_model_success() {
        let original_mlp = create_dummy_mlp();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        save_model(&original_mlp, path).unwrap();

        let loaded_mlp = load_model(path).unwrap();

        assert_eq!(original_mlp.layers.len(), loaded_mlp.layers.len());
        assert_eq!(
            original_mlp.layers[0].input_size,
            loaded_mlp.layers[0].input_size
        );
        assert_eq!(
            original_mlp.layers[0].output_size,
            loaded_mlp.layers[0].output_size
        );
        assert!(!loaded_mlp.layers[0].weights.is_empty());
        assert_eq!(
            original_mlp.layers[0].weights.len(),
            loaded_mlp.layers[0].weights.len()
        );
        assert_eq!(
            original_mlp.layers[0].biases.len(),
            loaded_mlp.layers[0].biases.len()
        );
    }

    #[test]
    fn test_save_history_success() {
        let original_history = create_dummy_history();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        save_history(&original_history, path).unwrap();

        let file_content = std::fs::read_to_string(path).unwrap();
        let loaded_history: TrainingHistory = serde_json::from_str(&file_content).unwrap();

        assert_eq!(original_history.loss, loaded_history.loss);
        assert_eq!(original_history.accuracy, loaded_history.accuracy);
    }

    #[test]
    fn test_load_model_file_not_found() {
        let non_existent_path = Path::new("non_existent_model.json");
        let err = load_model(non_existent_path).unwrap_err();
        match err {
            GateLearnerError::Io(io_err) => {
                assert_eq!(io_err.kind(), io::ErrorKind::NotFound);
            }
            _ => panic!("Expected GateLearnerError::Io, got {err:?}"),
        }
    }

    #[test]
    fn test_save_model_io_error() {
        let invalid_path = PathBuf::from("/nonexistent_dir/model.json");
        let mlp = create_dummy_mlp();
        let err = save_model(&mlp, &invalid_path).unwrap_err();
        assert!(matches!(err, GateLearnerError::Io(_)));
        assert!(err.to_string().contains("I/O error: "));
    }

    #[test]
    fn test_load_model_json_error() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();
        writeln!(temp_file, "{{ \"invalid_json\": }}").unwrap();

        let err = load_model(path).unwrap_err();
        assert!(matches!(err, GateLearnerError::Json(_)));
        assert!(err.to_string().contains("JSON error: "));
    }

    #[test]
    fn test_save_history_io_error() {
        let invalid_path = PathBuf::from("/nonexistent_dir/history.json");
        let history = create_dummy_history();
        let err = save_history(&history, &invalid_path).unwrap_err();
        assert!(matches!(err, GateLearnerError::Io(_)));
        assert!(err.to_string().contains("I/O error: "));
    }
}
