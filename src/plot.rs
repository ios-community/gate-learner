//! Plotting engine for visualising training history.

use crate::error::GateLearnerError;
use crate::storage::TrainingHistory;
use std::path::Path;

/// Renders a dual-axis PNG plot showing Loss and Accuracy over epochs.
///
/// # Errors
///
/// Returns [`GateLearnerError::FeatureNotEnabled`] if the `visualise` feature is not enabled.
/// Returns [`GateLearnerError::Plot`] if rendering fails.
#[allow(unused_variables)]
pub fn plot_history<P: AsRef<Path>>(
    history: &TrainingHistory,
    path: P,
) -> Result<(), GateLearnerError> {
    #[cfg(not(feature = "visualise"))]
    {
        Err(GateLearnerError::FeatureNotEnabled(
            "The 'visualise' feature is required to generate plots. Recompile with --features visualise.".to_string()
        ))
    }
    #[cfg(feature = "visualise")]
    {
        use plotters::prelude::*;

        let root = BitMapBackend::new(path.as_ref(), (1024, 768)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| GateLearnerError::Plot(e.to_string()))?;

        let epochs = history.loss.len();
        if epochs == 0 {
            return Err(GateLearnerError::Plot(
                "Cannot plot empty history".to_string(),
            ));
        }

        let max_loss = history.loss.iter().copied().fold(0.0f32, f32::max);

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption("Training Loss & Accuracy", ("sans-serif", 40).into_font())
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Right, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 60)
            .build_cartesian_2d(0..epochs, 0.0f32..max_loss)
            .map_err(|e| GateLearnerError::Plot(e.to_string()))?
            .set_secondary_coord(0..epochs, 0.0f32..1.05f32);

        chart
            .configure_mesh()
            .x_desc("Epoch")
            .y_desc("Loss (BCE)")
            .draw()
            .map_err(|e| GateLearnerError::Plot(e.to_string()))?;

        chart
            .configure_secondary_axes()
            .y_desc("Accuracy")
            .draw()
            .map_err(|e| GateLearnerError::Plot(e.to_string()))?;

        chart
            .draw_series(LineSeries::new(
                history.loss.iter().enumerate().map(|(x, &y)| (x, y)),
                &RED,
            ))
            .map_err(|e| GateLearnerError::Plot(e.to_string()))?
            .label("Loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        chart
            .draw_secondary_series(LineSeries::new(
                history.accuracy.iter().enumerate().map(|(x, &y)| (x, y)),
                &BLUE,
            ))
            .map_err(|e| GateLearnerError::Plot(e.to_string()))?
            .label("Accuracy")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()
            .map_err(|e| GateLearnerError::Plot(e.to_string()))?;

        root.present()
            .map_err(|e| GateLearnerError::Plot(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tempfile::NamedTempFile;

    fn create_dummy_history() -> TrainingHistory {
        TrainingHistory {
            loss: vec![0.5f32, 0.2f32, 0.1f32],
            accuracy: vec![0.5f32, 0.8f32, 0.9f32],
        }
    }

    #[test]
    #[cfg(feature = "visualise")]
    fn test_plot_history_success() {
        let history = create_dummy_history();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().with_extension("png");

        plot_history(&history, &path).unwrap();

        assert!(path.exists());
        assert!(!std::fs::read(&path).unwrap().is_empty());
    }

    #[test]
    #[cfg(feature = "visualise")]
    fn test_plot_history_empty_history_error() {
        let history = TrainingHistory {
            loss: vec![],
            accuracy: vec![],
        };
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().with_extension("png");

        let err = plot_history(&history, &path).unwrap_err();
        assert!(matches!(err, GateLearnerError::Plot(_)));
        assert_eq!(err.to_string(), "plotting error: Cannot plot empty history");
    }

    #[test]
    #[cfg(not(feature = "visualise"))]
    fn test_plot_history_feature_not_enabled_error() {
        let history = create_dummy_history();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().with_extension("png");

        let err = plot_history(&history, &path).unwrap_err();
        assert!(matches!(err, GateLearnerError::FeatureNotEnabled(_)));
        assert_eq!(
            err.to_string(),
            "feature not enabled: The 'visualise' feature is required to generate plots. Recompile with --features visualise."
        );
    }

    #[test]
    #[cfg(feature = "visualise")]
    fn test_plot_history_io_error() {
        let history = create_dummy_history();
        let invalid_path = Path::new("/nonexistent_dir/plot.png");

        let err = plot_history(&history, invalid_path).unwrap_err();
        assert!(matches!(err, GateLearnerError::Plot(_)));
        assert!(err.to_string().contains("plotting error: "));
    }
}
