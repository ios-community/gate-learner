use criterion::{Criterion, criterion_group, criterion_main};
use gate_learner::core::{
    ActivationType, AdamOptimizer, MultilayerPerceptron, NetworkConfig, relu, sigmoid,
};
use std::hint::black_box;

fn bench_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Activation Functions");
    let x = 0.5f32;
    group.bench_function("sigmoid", |b| b.iter(|| sigmoid(black_box(x))));
    group.bench_function("relu", |b| b.iter(|| relu(black_box(x))));
    group.finish();
}

fn bench_architectures(c: &mut Criterion) {
    let mut group = c.benchmark_group("Architectures");

    let config_wide = NetworkConfig {
        input_size: 2,
        hidden_sizes: vec![128],
        output_size: 1,
        hidden_activation: ActivationType::Relu,
    };
    let mut mlp_wide = MultilayerPerceptron::new(&config_wide).unwrap();
    let input_wide = vec![1.0f32, 0.0f32];
    let mut output_wide = vec![0.0f32; 1];
    group.bench_function("forward_wide_128", |b| {
        b.iter(|| mlp_wide.forward(black_box(&input_wide), black_box(&mut output_wide)))
    });

    let config_deep = NetworkConfig {
        input_size: 2,
        hidden_sizes: vec![4, 4, 4, 4, 4],
        output_size: 1,
        hidden_activation: ActivationType::Sigmoid,
    };
    let mut mlp_deep = MultilayerPerceptron::new(&config_deep).unwrap();
    let input_deep = vec![1.0f32, 0.0f32];
    let mut output_deep = vec![0.0f32; 1];
    group.bench_function("forward_deep_5_layers", |b| {
        b.iter(|| mlp_deep.forward(black_box(&input_deep), black_box(&mut output_deep)))
    });

    group.finish();
}

fn bench_input_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Input Scaling");

    let config_2 = NetworkConfig {
        input_size: 2,
        hidden_sizes: vec![4],
        output_size: 1,
        hidden_activation: ActivationType::Relu,
    };
    let mut mlp_2 = MultilayerPerceptron::new(&config_2).unwrap();
    let input_2 = vec![1.0f32, 0.0f32];
    let mut output_2 = vec![0.0f32; 1];
    group.bench_function("forward_input_2", |b| {
        b.iter(|| mlp_2.forward(black_box(&input_2), black_box(&mut output_2)))
    });

    let config_32 = NetworkConfig {
        input_size: 32,
        hidden_sizes: vec![64],
        output_size: 1,
        hidden_activation: ActivationType::Relu,
    };
    let mut mlp_32 = MultilayerPerceptron::new(&config_32).unwrap();
    let input_32 = vec![1.0f32; 32];
    let mut output_32 = vec![0.0f32; 1];
    group.bench_function("forward_input_32", |b| {
        b.iter(|| mlp_32.forward(black_box(&input_32), black_box(&mut output_32)))
    });

    group.finish();
}

fn bench_batch_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batch Scaling");

    let config = NetworkConfig {
        input_size: 2,
        hidden_sizes: vec![4],
        output_size: 1,
        hidden_activation: ActivationType::Relu,
    };
    let mut mlp = MultilayerPerceptron::new(&config).unwrap();

    for &batch_size in &[1, 16, 64] {
        let mut optimizer = AdamOptimizer::new(&mlp);
        group.bench_function(format!("adam_update_batch_{batch_size}"), |b| {
            b.iter(|| {
                optimizer.update(
                    black_box(&mut mlp),
                    black_box(0.01f32),
                    black_box(0.0001f32),
                    black_box(batch_size),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_activation_functions,
    bench_architectures,
    bench_input_scaling,
    bench_batch_scaling
);
criterion_main!(benches);
