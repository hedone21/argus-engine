//! Phase 4-4: standard happy path (`session::assembly::is_standard_happy_path` 진입)
//! 분기 추출.
//!
//! `bin/generate.rs::main()` L1764~1844 분기를 외과적으로 이동.
//! DecodeLoop + ModelForward 위임 경로.

use std::io::Write;
use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::auf::source_hash::compute_source_hash;
use crate::backend::Backend;
use crate::format::SnapshotRestore;
use crate::hardware::{DeviceTarget, Hardware};
use crate::inference::sampling::{self, SamplingConfig};
use crate::kv::kv_cache::KVCache;
use crate::kv::standard_format::StandardFormat;
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::session::assembly::build_standard_loop;
use crate::session::chat::stream_stage::{ChatStreamSlot, IncDetok};
use crate::session::cli::Args;
use crate::session::prefix_cache::{RestoredPrefix, try_restore_prefix};
use crate::session::resilience_adapter::ResilienceAdapter;

pub struct StandardHappyCtx {
    pub args: Args,
    pub backend: Arc<dyn Backend>,
    pub memory: Arc<dyn Memory>,
    pub hardware: Arc<Hardware>,
    pub model: TransformerModel,
    pub tokenizer: Tokenizer,
    pub kv_caches: Vec<KVCache>,
    pub tokens: Vec<u32>,
    pub max_seq_len: usize,
    pub sampling_config: SamplingConfig,
    pub vocab_size: usize,
    /// P4: ResilienceAdapter 주입 (None 이면 NoOp default).
    pub resilience: Option<ResilienceAdapter>,
}

pub fn run_standard_happy_path(ctx: StandardHappyCtx) -> anyhow::Result<()> {
    let StandardHappyCtx {
        args,
        backend,
        memory,
        hardware,
        model,
        tokenizer,
        mut kv_caches,
        tokens,
        max_seq_len,
        sampling_config,
        vocab_size,
        resilience,
    } = ctx;

    // Phase α-W-2: hardware resolver 에서 cpu secondary Arc 를 재바인딩.
    // 로컬이 정확히 같은 Arc 를 보유하므로 본문 사용처는 무변경.
    let cpu_backend_arc = hardware
        .resolve(DeviceTarget::Cpu)
        .expect("Cpu always resolves")
        .0
        .clone();

    eprintln!(
        "[Phase4-4.5] standard happy path → DecodeLoop+ModelForward (tokens={}, budget={})",
        tokens.len(),
        args.num_tokens
    );

    // ── prefix cache restore 배선 (ENG-082, INV-190) ─────────────────────────
    // --prefix-cache 가 있고 opaque KV가 아닌 경우, kv_caches를 임시 StandardFormat으로
    // wrap → try_restore_prefix → KVCache 재조립 → build_standard_loop에 넘긴다.
    // restore 성공 시 KVCache.current_pos 가 이미 token_count 로 갱신돼 있음.
    // 두 flag 모두 None 이면 이 블록에 완전히 미진입 (성능 무영향, INV-190).
    let has_prefix_flags = args.prefix_cache.is_some() || args.save_prefix_cache.is_some();

    // hash는 두 flag 중 하나라도 있을 때만 계산
    let prefix_hashes: Option<([u8; 32], [u8; 32])> = if has_prefix_flags {
        let model_hash_res = compute_source_hash(std::path::Path::new(&args.model_path));
        let is_gguf_or_auf =
            args.model_path.ends_with(".gguf") || args.model_path.ends_with(".auf");
        let tokenizer_path = crate::session::bin_setup::resolve_tokenizer_path(
            &args,
            &args.model_path,
            is_gguf_or_auf,
        );
        let tok_hash_res = compute_source_hash(std::path::Path::new(&tokenizer_path));
        match (model_hash_res, tok_hash_res) {
            (Ok((mh, _, _)), Ok((th, _, _))) => Some((mh, th)),
            _ => {
                eprintln!("[PrefixCache] hash computation failed — fresh prefill");
                None
            }
        }
    } else {
        None
    };

    // restore: --prefix-cache 지정 시 시도. kv_caches drain → wrap → restore → 회수.
    // full restore 시 RestoredPrefix.last_logits 를 decode 루프에 직접 주입(re-forward 불필요).
    let restore_result: Option<RestoredPrefix> = if let (
        Some(path_str),
        Some((model_hash, tok_hash)),
    ) = (&args.prefix_cache, &prefix_hashes)
    {
        let is_opaque = kv_caches.first().is_some_and(|c| c.is_opaque());
        // W-ALLOC: a per-layer mixed-precision set has heterogeneous byte layouts across layers, but
        // the snapshot stamps ONE format_id (fmts.first()) for the whole save — restoring it would
        // mis-describe the non-first layers. Reject the snapshot (fresh prefill) for a mixed set
        // rather than silently round-trip a per-layer layout mismatch.
        let is_mixed = kv_caches.first().is_some_and(|first| {
            let d0 = first.kv_dtype();
            kv_caches.iter().any(|c| c.kv_dtype() != d0)
        });
        if is_opaque || is_mixed {
            eprintln!(
                "[PrefixCache] {} KV format — snapshot unsupported, fresh prefill",
                if is_mixed {
                    "per-layer mixed"
                } else {
                    "opaque"
                }
            );
            None
        } else {
            let n_layers = kv_caches.len();
            let mut fmts: Vec<StandardFormat> = kv_caches
                .drain(..)
                .enumerate()
                .map(|(i, c)| StandardFormat::new(i, c))
                .collect();

            let snapshot_refs: Vec<&dyn SnapshotRestore> =
                fmts.iter().map(|f| f as &dyn SnapshotRestore).collect();

            let kv_heads = model.config.num_key_value_heads as u32;
            let head_dim = model.config.head_dim as u32;
            let format_id = fmts.first().map(|f| f.snapshot_format_id()).unwrap_or(0);

            let restored = match try_restore_prefix(
                std::path::Path::new(path_str),
                model_hash,
                tok_hash,
                format_id,
                &tokens,
                &snapshot_refs,
                kv_heads,
                head_dim,
                backend.as_ref(),
            ) {
                Ok(Some(r)) => {
                    if r.token_count == tokens.len() {
                        // vocab 크기 검증: logits_len != vocab_size 이면 무효화
                        if !r.last_logits.is_empty() && r.last_logits.len() != vocab_size {
                            eprintln!(
                                "[PrefixCache] logits_len mismatch ({} != {}) — fresh prefill",
                                r.last_logits.len(),
                                vocab_size
                            );
                            None
                        } else {
                            eprintln!(
                                "[PrefixCache] restored {} tokens (skipped prefill)",
                                r.token_count
                            );
                            Some(r)
                        }
                    } else {
                        eprintln!(
                            "[PrefixCache] partial restore {} / {} tokens",
                            r.token_count,
                            tokens.len()
                        );
                        Some(r)
                    }
                }
                Ok(None) => {
                    eprintln!("[PrefixCache] miss — fresh prefill");
                    None
                }
                Err(e) => {
                    eprintln!("[PrefixCache] restore error: {e} — fresh prefill");
                    None
                }
            };

            // kv_caches 재조립 (drain 했으므로 복원)
            kv_caches.extend(fmts.drain(..).map(|f| f.into_kv_cache()));
            debug_assert_eq!(kv_caches.len(), n_layers);
            restored
        }
    } else {
        None
    };

    // argus-cli eviction(공용 path): score-free `CacheManager`(none/sliding/streaming/
    // `--load-plugin` stage)를 CLI `eviction <policy>` 로 구성한다. `eviction none` 이고 swap-dir
    // 도 없으면 None → build_standard_loop 의 eviction 배선 미진입(기존 happy-path 와 byte-identical,
    // 회귀 0). score-based(h2o/d2o)·offload(--swap-dir)·기타 미지원 모드는 argus_cli 진입부에서
    // 이미 reject 되므로 여기 도달하는 정책은 항상 score-free.
    let cache_manager = crate::session::assembly::build_resilience_cache_manager(&args, &backend)?;

    // Per-token streaming slot: the decode loop's ChatStreamStage emits each kept token into it.
    // Armed below for the synchronous run() so tokens print as they land (instead of one batch
    // decode+println after the loop). build_standard_loop forces a registry when this is Some.
    let stream_slot = ChatStreamSlot::new();

    // bin_setup이 --kv-format/--kv-type dispatch로 할당한 kv_caches를
    // 그대로 소비한다(과거엔 drop 후 build_standard_loop이 typed로 재할당 →
    // --kv-format opaque 선택이 decode 경로에 도달 못 했다).
    let mut decode_loop = build_standard_loop(
        backend.clone(),
        memory.clone(),
        cpu_backend_arc.clone(),
        model,
        kv_caches,
        max_seq_len,
        sampling_config.clone(),
        !args.no_gpu_plan,
        resilience,
        args.effective_read_stage(),
        cache_manager,
        args.eviction_target_ratio(),
        Some(Arc::clone(&stream_slot)),
    )?;

    // ── prefill / restore 분기 ────────────────────────────────────────────────
    let t_prefill = std::time::Instant::now();
    let did_full_restore = matches!(&restore_result, Some(r) if r.token_count == tokens.len());
    let mut last_logits = match restore_result {
        Some(r) if r.token_count == tokens.len() => {
            // full restore: KV 전체 복원됨. re-forward 불필요.
            // set_pos(tc) 로 decode 루프 pos 를 정확히 설정하고,
            // 저장된 logits를 직접 사용한다.
            // last_logits가 비어 있으면 re-forward로 폴백 (구버전 캐시 호환).
            let tc = r.token_count;
            decode_loop.set_pos(tc);
            if r.last_logits.is_empty() {
                // logits 없는 캐시 파일 (하위 호환 폴백): re-forward 1 token.
                // tc >= 1 은 try_restore_prefix contract.
                // prefill(tokens[tc-1..tc]) 이 tokens[tc-1] 만 observe_token → 앞 tokens[..tc-1] 주입.
                decode_loop.seed_sampler_history(&tokens[..tc - 1]);
                decode_loop.set_pos(tc - 1);
                decode_loop.prefill(&tokens[tc - 1..tc])?
            } else {
                // sampler history: fresh prefill 의 observe_token(each prompt token) 와 동치로 주입.
                decode_loop.seed_sampler_history(&tokens);
                r.last_logits
            }
        }
        Some(r) if r.token_count < tokens.len() => {
            // 부분 복원: [tc..] 잔여 토큰 prefill (start_pos = tc).
            // prefill(tokens[tc..]) 이 tokens[tc..] 를 observe_token → tokens[..tc] 는 여기서 주입.
            let tc = r.token_count;
            decode_loop.seed_sampler_history(&tokens[..tc]);
            decode_loop.set_pos(tc);
            decode_loop.prefill(&tokens[tc..])?
        }
        _ => {
            // fresh prefill
            decode_loop.prefill(&tokens)?
        }
    };
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    // ── save prefix cache (INV-189: prefill 완료 직후·decode 진입 전·eviction 전) ──
    // fresh prefill 또는 partial restore 후에만 save. full restore 경우 덮어쓰기 불필요.
    if let (Some(save_path_str), Some((model_hash, tok_hash))) =
        (&args.save_prefix_cache, &prefix_hashes)
        && !did_full_restore
    {
        let save_path = std::path::Path::new(save_path_str);
        match decode_loop.save_kv_prefix(
            save_path,
            model_hash,
            tok_hash,
            &tokens,
            &last_logits,
            backend.as_ref(),
        ) {
            Ok(()) => eprintln!(
                "[PrefixCache] saved {} tokens to {:?}",
                tokens.len(),
                save_path
            ),
            Err(e) => eprintln!("[PrefixCache] save warning: {e} (run continues)"),
        }
    }

    // Phase 4-4.7: first_token을 raw argmax가 아니라 production fallback과
    // 동일한 `sampling::sample(&mut logits, &tokens, ...)` 호출로 산출.
    // `tokens` 전체가 rep history로 들어가 prompt suffix에 rep penalty가
    // 적용된다.
    let first_token = sampling::sample(
        &mut last_logits,
        &tokens,
        vocab_size,
        &sampling_config,
        None,
    );

    // ── per-token streaming output ────────────────────────────────────────────
    // Echo the prompt, print the first sampled token, then arm the stream slot so every token the
    // decode loop produces is detokenized and flushed as it lands (the chat repl/server pattern).
    // `IncDetok` decodes the growing id vec and emits only newly-completed UTF-8, so the bytes
    // written equal the previous `decode(prompt + first + generated)` + '\n' exactly — same final
    // output, just delivered live. (detok+flush runs inside the decode-timed region, but its cost is
    // ~µs/token, negligible against per-token forward.)
    let mut detok = IncDetok::new();
    let mut out = std::io::stdout().lock();
    for &t in &tokens {
        let piece = detok.push(t, &tokenizer);
        if !piece.is_empty() {
            let _ = write!(out, "{piece}");
        }
    }
    let piece = detok.push(first_token, &tokenizer);
    if !piece.is_empty() {
        let _ = write!(out, "{piece}");
    }
    let _ = out.flush();

    let t_decode = std::time::Instant::now();
    let result = {
        let detok_ref = &mut detok;
        let out_ref = &mut out;
        let mut cb = |tok: u32| {
            let piece = detok_ref.push(tok, &tokenizer);
            if !piece.is_empty() {
                let _ = write!(out_ref, "{piece}");
                let _ = out_ref.flush();
            }
        };
        let _guard = stream_slot.arm(&mut cb);
        decode_loop.run(args.num_tokens - 1, first_token)?
    };
    let decode_total_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

    // Flush bytes held back at a multi-byte boundary, then end the line (matches the old println).
    let tail = detok.flush(&tokenizer);
    if !tail.is_empty() {
        let _ = write!(out, "{tail}");
    }
    let _ = writeln!(out);
    let _ = out.flush();
    drop(out); // release the stdout lock before the summary println!s below

    let decode_tokens = result.tokens_generated.len();
    let total_gen = 1 + decode_tokens;
    let decode_per_tok = if decode_tokens > 0 {
        decode_total_ms / decode_tokens as f64
    } else {
        0.0
    };
    let avg_tbt = (prefill_ms + decode_total_ms) / total_gen as f64;
    println!("TTFT: {:.2} ms", prefill_ms);
    if decode_tokens > 0 {
        println!(
            "Decode: {:.2} ms/tok ({:.1} tok/s) [{} tokens]",
            decode_per_tok,
            1000.0 / decode_per_tok.max(0.001),
            decode_tokens,
        );
    }
    println!(
        "Avg TBT: {:.2} ms ({:.1} tokens/sec)",
        avg_tbt,
        1000.0 / avg_tbt.max(0.001),
    );
    eprintln!(
        "[Phase4-4.5] generated={} (first={} + run={}) stopped_by={:?} final_pos={}",
        total_gen, first_token, decode_tokens, result.stopped_by, result.final_pos
    );
    Ok(())
}
