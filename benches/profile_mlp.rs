use gate_learner::core::{
    ActivationType, AdamOptimizer, MultilayerPerceptron, NetworkConfig, relu, sigmoid,
};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let x = 0.5f32;
    for _ in 0..10000 {
        std::hint::black_box(sigmoid(std::hint::black_box(x)));
        std::hint::black_box(relu(std::hint::black_box(x)));
    }

    let config_wide = NetworkConfig {
        input_size: 2,
        hidden_sizes: vec![128],
        output_size: 1,
        hidden_activation: ActivationType::Relu,
    };
    let mut mlp_wide = MultilayerPerceptron::new(&config_wide).unwrap();
    let input_wide = vec![1.0f32, 0.0f32];
    let mut output_wide = vec![0.0f32; 1];
    for _ in 0..10000 {
        mlp_wide.forward(
            std::hint::black_box(&input_wide),
            std::hint::black_box(&mut output_wide),
        );
    }

    let config_deep = NetworkConfig {
        input_size: 2,
        hidden_sizes: vec![4, 4, 4, 4, 4],
        output_size: 1,
        hidden_activation: ActivationType::Sigmoid,
    };
    let mut mlp_deep = MultilayerPerceptron::new(&config_deep).unwrap();
    let input_deep = vec![1.0f32, 0.0f32];
    let mut output_deep = vec![0.0f32; 1];
    for _ in 0..10000 {
        mlp_deep.forward(
            std::hint::black_box(&input_deep),
            std::hint::black_box(&mut output_deep),
        );
    }

    let config_2 = NetworkConfig {
        input_size: 2,
        hidden_sizes: vec![4],
        output_size: 1,
        hidden_activation: ActivationType::Relu,
    };
    let mut mlp_2 = MultilayerPerceptron::new(&config_2).unwrap();
    let input_2 = vec![1.0f32, 0.0f32];
    let mut output_2 = vec![0.0f32; 1];
    for _ in 0..10000 {
        mlp_2.forward(
            std::hint::black_box(&input_2),
            std::hint::black_box(&mut output_2),
        );
    }

    let config_32 = NetworkConfig {
        input_size: 32,
        hidden_sizes: vec![64],
        output_size: 1,
        hidden_activation: ActivationType::Relu,
    };
    let mut mlp_32 = MultilayerPerceptron::new(&config_32).unwrap();
    let input_32 = vec![1.0f32; 32];
    let mut output_32 = vec![0.0f32; 1];
    for _ in 0..10000 {
        mlp_32.forward(
            std::hint::black_box(&input_32),
            std::hint::black_box(&mut output_32),
        );
    }

    let config = NetworkConfig {
        input_size: 2,
        hidden_sizes: vec![4],
        output_size: 1,
        hidden_activation: ActivationType::Relu,
    };
    let mut mlp = MultilayerPerceptron::new(&config).unwrap();

    for &batch_size in &[1, 16, 64] {
        let mut optimizer = AdamOptimizer::new(&mlp);
        for _ in 0..10000 {
            optimizer.update(
                std::hint::black_box(&mut mlp),
                std::hint::black_box(0.01f32),
                std::hint::black_box(0.0001f32),
                std::hint::black_box(batch_size),
            );
        }
    }
}
