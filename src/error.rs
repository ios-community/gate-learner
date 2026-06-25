//! Centralised error handling for the Boole Project.

use thiserror::Error;

/// Represents all errors that can occur in the `boole_project` crate.
#[derive(Debug, Error)]
pub enum GateLearnerError {
    /// Returned when the network architecture configuration is invalid.
    #[error("invalid network architecture: {0}")]
    InvalidArchitecture(String),

    /// Returned when an I/O operation fails.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Returned when JSON serialisation or deserialisation fails.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Returned when plotting fails.
    #[error("plotting error: {0}")]
    Plot(String),

    /// Returned when the requested feature is not compiled.
    #[error("feature not enabled: {0}")]
    FeatureNotEnabled(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;
    use std::io;

    #[test]
    fn test_error_display_invalid_architecture() {
        let err = GateLearnerError::InvalidArchitecture("Layer size mismatch".to_string());
        assert_eq!(
            err.to_string(),
            "invalid network architecture: Layer size mismatch"
        );
    }

    #[test]
    fn test_error_display_io() {
        let io_error = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err = GateLearnerError::Io(io_error);
        assert_eq!(err.to_string(), "I/O error: file not found");
    }

    #[test]
    fn test_error_display_json() {
        let json_error = serde_json::from_str::<serde_json::Value>("{invalid json").unwrap_err();
        let err = GateLearnerError::Json(json_error);
        assert!(err.to_string().starts_with("JSON error: "));
    }

    #[test]
    fn test_error_display_plot() {
        let err = GateLearnerError::Plot("Failed to draw line".to_string());
        assert_eq!(err.to_string(), "plotting error: Failed to draw line");
    }

    #[test]
    fn test_error_display_feature_not_enabled() {
        let err = GateLearnerError::FeatureNotEnabled("Visualisation is off".to_string());
        assert_eq!(err.to_string(), "feature not enabled: Visualisation is off");
    }

    #[test]
    fn test_error_conversion_from_io() {
        let io_error = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
        let gate_learner_error: GateLearnerError = io_error.into();
        assert!(matches!(gate_learner_error, GateLearnerError::Io(_)));
        assert_eq!(gate_learner_error.to_string(), "I/O error: access denied");
    }

    #[test]
    fn test_error_conversion_from_json() {
        let json_str = "{ \"key\": \"value\" ";
        let json_error = serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let gate_learner_error: GateLearnerError = json_error.into();
        assert!(matches!(gate_learner_error, GateLearnerError::Json(_)));
        assert!(gate_learner_error.to_string().contains("JSON error: "));
    }
}
