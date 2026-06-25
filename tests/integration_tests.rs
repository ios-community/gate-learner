use gate_learner::core::{ActivationType, AdamOptimizer, MultilayerPerceptron, NetworkConfig};
use gate_learner::dataset::{generate_or, generate_xor};

#[test]
fn test_or_gate_convergence() {
    let dataset = generate_or(2).unwrap();
    let config = NetworkConfig {
        input_size: 2,
        hidden_sizes: vec![4],
        output_size: 1,
        hidden_activation: ActivationType::Relu,
    };
    let mut mlp = MultilayerPerceptron::new_with_seed(&config, 42).unwrap();
    let mut optimizer = AdamOptimizer::new(&mlp);

    let mut converged = false;
    for _ in 0..1000 {
        let mut epoch_loss = 0.0f32;
        mlp.zero_gradients();
        for (input, target) in &dataset {
            let (loss, _) = mlp.accumulate_gradients(input, target);
            epoch_loss += loss;
        }
        optimizer.update(&mut mlp, 0.1f32, 0.0001f32, dataset.len());
        let avg_loss = epoch_loss / dataset.len() as f32;
        if avg_loss < 0.01f32 {
            converged = true;
            break;
        }
    }
    assert!(converged);
}

#[test]
fn test_xor_gate_convergence() {
    let dataset = generate_xor(2).unwrap();
    let config = NetworkConfig {
        input_size: 2,
        hidden_sizes: vec![4],
        output_size: 1,
        hidden_activation: ActivationType::Sigmoid,
    };
    let mut mlp = MultilayerPerceptron::new_with_seed(&config, 12345).unwrap();
    let mut optimizer = AdamOptimizer::new(&mlp);

    let mut converged = false;
    for _ in 0..5000 {
        let mut epoch_loss = 0.0f32;
        mlp.zero_gradients();
        for (input, target) in &dataset {
            let (loss, _) = mlp.accumulate_gradients(input, target);
            epoch_loss += loss;
        }
        optimizer.update(&mut mlp, 0.1f32, 0.0001f32, dataset.len());
        let avg_loss = epoch_loss / dataset.len() as f32;
        if avg_loss < 0.01f32 {
            converged = true;
            break;
        }
    }
    assert!(converged);
}
