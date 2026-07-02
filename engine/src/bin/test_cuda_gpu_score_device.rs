//! CUDA device test — GPU-side attention score accumulator + `attn-score` CUDA reducer.
//!
//! The discrete-GPU twin of the OpenCL `test_faithful_h2o_gpu_seed_device`. Drives the
//! plugin's fused reduce kernel on-device over a controlled `score_buf` and asserts the
//! resulting cumulative `importance` / `head_importance` match the hand-computed
//! per-layer-MAX + GQA-average policy (`score_reduce.cu`). Also exercises the A2SF decay
//! (a second `end_step` with `decay=0.5`) and the prefill `seed_cumulative` round-trip.
//!
//! Proves end-to-end on the device: `find_cuda_score_reducer` → `from_raw_context`
//! (ManuallyDrop borrowed ctx) → nvcc→PTX compile → module load → `launch_kernel` on the
//! engine's lent `CUstream` → `sync_to_cpu`. Requires a CUDA GPU.
//!
//! Run: `cargo run --no-default-features --features cuda --bin test_cuda_gpu_score_device`

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("test_cuda_gpu_score_device requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use argus_engine::backend::Backend;
    use argus_engine::backend::cuda_pc::CudaBackend;

    let be = CudaBackend::new()?;

    // Geometry: 2 layers, 2 query heads, 1 KV head (n_rep=2), max_seq=4, cache_seq_len=3.
    let (n_layers, n_heads_q, n_kv_heads, max_seq) = (2usize, 2usize, 1usize, 4usize);
    let stride = max_seq;
    let cache_seq_len = 3usize;

    // --- Case 1: single reduce, no decay (decay=0.0 → decay_factor=1.0). ---
    be.init_gpu_score_acc(n_layers, n_heads_q, n_kv_heads, max_seq, 0.0)?;
    be.gpu_score_acc_mut().expect("acc").set_active(true);

    // score_buf layout [layer][head][stride]; write per (layer, head) the 3 post-softmax weights.
    let mut scores = vec![0.0f32; n_layers * n_heads_q * stride];
    let mut set = |l: usize, h: usize, v: [f32; 3]| {
        let base = l * n_heads_q * stride + h * stride;
        scores[base] = v[0];
        scores[base + 1] = v[1];
        scores[base + 2] = v[2];
    };
    set(0, 0, [0.1, 0.2, 0.7]);
    set(0, 1, [0.3, 0.3, 0.4]);
    set(1, 0, [0.5, 0.4, 0.1]);
    set(1, 1, [0.2, 0.2, 0.6]);
    be.gpu_score_acc().unwrap().debug_fill_score_buf(&scores)?;

    let cu_stream = be.context().default_stream().cu_stream() as *mut std::ffi::c_void;
    be.gpu_score_acc_mut()
        .unwrap()
        .end_step(cu_stream, cache_seq_len)?;
    be.synchronize()?;
    let (flat, head) = be.gpu_score_acc().unwrap().sync_to_cpu()?;

    // Expected (per-layer flat sum, MAX across layers):
    //   L0 flat = [0.4, 0.5, 1.1], L1 flat = [0.7, 0.6, 0.7] → MAX = [0.7, 0.6, 1.1].
    //   head (GQA avg, n_rep=2 → *0.5) = MAX*0.5 = [0.35, 0.30, 0.55].
    let exp_flat = [0.7f32, 0.6, 1.1];
    let exp_head = [0.35f32, 0.30, 0.55];
    for t in 0..cache_seq_len {
        assert!(
            (flat[t] - exp_flat[t]).abs() < 1e-5,
            "case1 flat[{t}]={} != {}",
            flat[t],
            exp_flat[t]
        );
        assert!(
            (head[t] - exp_head[t]).abs() < 1e-5,
            "case1 head[{t}]={} != {}",
            head[t],
            exp_head[t]
        );
    }
    println!(
        "[case1] no-decay reduce OK: flat={:?} head={:?}",
        &flat[..3],
        &head[..3]
    );

    // --- Case 2: A2SF decay accumulation (decay=0.5 → decay_factor=0.5). ---
    // importance starts fresh; after two identical end_steps with the same score_buf:
    //   step1: imp = 0*0.5 + step_flat = step_flat
    //   step2: imp = step_flat*0.5 + step_flat = 1.5 * step_flat
    be.init_gpu_score_acc(n_layers, n_heads_q, n_kv_heads, max_seq, 0.5)?;
    be.gpu_score_acc_mut().unwrap().set_active(true);
    be.gpu_score_acc().unwrap().debug_fill_score_buf(&scores)?;
    let cu_stream = be.context().default_stream().cu_stream() as *mut std::ffi::c_void;
    be.gpu_score_acc_mut()
        .unwrap()
        .end_step(cu_stream, cache_seq_len)?;
    be.gpu_score_acc_mut()
        .unwrap()
        .end_step(cu_stream, cache_seq_len)?;
    be.synchronize()?;
    let (flat2, _head2) = be.gpu_score_acc().unwrap().sync_to_cpu()?;
    for t in 0..cache_seq_len {
        let expect = 1.5 * exp_flat[t];
        assert!(
            (flat2[t] - expect).abs() < 1e-5,
            "case2 flat[{t}]={} != {} (decay accumulate)",
            flat2[t],
            expect
        );
    }
    println!("[case2] decay=0.5 accumulate OK: flat={:?}", &flat2[..3]);

    // --- Case 3: prefill seed round-trip (seed_cumulative → sync_to_cpu). ---
    be.init_gpu_score_acc(n_layers, n_heads_q, n_kv_heads, max_seq, 0.0)?;
    be.gpu_score_acc_mut().unwrap().set_active(true);
    let seed_flat: Vec<f32> = (0..max_seq).map(|i| i as f32 + 1.0).collect();
    let seed_head: Vec<f32> = (0..n_kv_heads * max_seq)
        .map(|i| (i as f32) * 0.25)
        .collect();
    be.gpu_score_acc()
        .unwrap()
        .seed_cumulative(&seed_flat, &seed_head)?;
    be.synchronize()?;
    let (rf, rh) = be.gpu_score_acc().unwrap().sync_to_cpu()?;
    assert_eq!(rf, seed_flat, "case3 seed flat round-trip");
    assert_eq!(rh, seed_head, "case3 seed head round-trip");
    println!("[case3] prefill seed round-trip OK");

    println!("ALL PASS: CUDA GPU score accumulator + attn-score CUDA reducer verified on-device");
    Ok(())
}
