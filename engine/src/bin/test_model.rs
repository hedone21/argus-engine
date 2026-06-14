use argus_engine::backend::cpu::CpuBackend;
use argus_engine::memory::galloc::Galloc;
use argus_engine::models::transformer::TransformerModel;
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    println!("=== LLM RS2 Model Loading Test ===");

    let backend = Arc::new(CpuBackend::new());
    let memory = Galloc::new();

    let model_path = "/data/local/tmp/argus_engine/models/llama3.2-1b";
    println!("Loading model from: {}", model_path);

    let start = Instant::now();
    let model = TransformerModel::load(model_path, backend.clone(), &memory)?;
    let duration = start.elapsed();

    println!("Model loaded successfully in {:?}", duration);
    println!("Config: {:?}", model.config);
    println!("Number of layers: {}", model.layers.len());
    println!("Vocab size: {}", model.config.vocab_size);
    println!(
        "Embed Tokens Shape: {:?}",
        model.embed_tokens.shape().dims()
    );

    Ok(())
}
