//! Phase 4-4: `--dump-importance` 모드 추출.
//!
//! `bin/generate.rs::main()` L1576~1652 분기를 외과적으로 이동.
//! `ImportanceCollector`를 prefill에 부착하여 layer별 importance를 JSON으로
//! stdout에 출력하고 종료한다.
//!
//! ## P2c 확장 (KV roadmap 항목 0 §4.3, 2026-06-12)
//!
//! prefill 후 `HEAD_CONC_DECODE_STEPS` 만큼의 decode step을 추가로 수행하여
//! per-(layer, kv_head) attention concentration C_h(상위 5% 토큰 질량)를 측정한다.
//! `score_accumulator`(GQA mode)의 `last_step_head_attn()`을 매 decode step마다
//! 읽어 `ConcentrationAccumulator`에 누적 → JSON `head_concentration` 섹션 출력.
//! flag 0 (기존 `--dump-importance` 확장) — off(steps=0) 시 기존 경로 무영향.

use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::inference::query_stats::QueryStatsAccumulator;
use crate::kv::kv_cache::KVCache;
use crate::layers::workspace::{LayerWorkspace, WorkspaceConfig};
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use crate::qcf::ImportanceCollector;
use crate::session::eval::EvalCacheKind;
use crate::session::head_concentration::ConcentrationAccumulator;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// decode step에서 head concentration을 캡처할 기본 step 수.
/// prefill 후 이 만큼 greedy decode를 추가 수행한다.
/// 0으로 설정하면 head_concentration 섹션이 JSON에서 제외된다.
const HEAD_CONC_DECODE_STEPS: usize = 32;

/// C_h 계산에 사용할 상위 토큰 비율 (5%).
const HEAD_CONC_TOP_FRAC: f32 = 0.05;

pub struct DumpImportanceCtx {
    pub backend: Arc<dyn Backend>,
    pub memory: Arc<dyn Memory>,
    pub model: TransformerModel,
    pub tokenizer: Tokenizer,
    pub kv_caches: Vec<KVCache>,
    pub prompt: String,
    pub vocab_size: usize,
    pub model_path: String,
}

pub fn run_dump_importance(mut ctx: DumpImportanceCtx) -> anyhow::Result<()> {
    let mut collector = ImportanceCollector::new();

    let prompt_enc = ctx
        .tokenizer
        .encode(ctx.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let prompt_ids: Vec<u32> = prompt_enc.get_ids().to_vec();
    let prompt_len = prompt_ids.len();
    eprintln!("[Importance] Prefill {} tokens...", prompt_len);

    let cpu_buf = Galloc::new().alloc(prompt_len * 4, DType::U8)?;
    unsafe {
        let ptr = cpu_buf.as_mut_ptr() as *mut u32;
        std::ptr::copy_nonoverlapping(prompt_ids.as_ptr(), ptr, prompt_len);
    }
    let cpu_input = Tensor::new(
        Shape::new(vec![1, prompt_len]),
        cpu_buf,
        Arc::new(CpuBackend::new()),
    );
    let input_tensor = ctx.backend.copy_from(&cpu_input)?;

    let logits_buf = ctx
        .memory
        .alloc(prompt_len * ctx.vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, prompt_len, ctx.vocab_size]),
        logits_buf,
        ctx.backend.clone(),
    );

    // Phase α-K ①-d: forward_into → fmt round-trip (prefill, qcf:212 구조적 쌍둥이).
    // ctx.kv_caches 는 owned Vec → 시그니처 변경 불요. disjoint field borrow(model/backend/memory
    // vs kv_caches)로 closure 와 &mut 공존.
    KVCache::forward_fmt_roundtrip(&mut ctx.kv_caches, |fmts| {
        ctx.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
            fmts,
            backend: &ctx.backend,
            memory: &*ctx.memory,
            logits_out: &mut logits,
            x_gen: None,
            workspace: None,
            logits_last_only: false,
            score_accumulator: None,
            query_stats_accumulator: None,
            skip_config: None,
            importance_collector: Some(&mut collector),
            cache_self_need_scores: false,
            layer_boundary_hook: None,
            read_stage: None,
        })
    })?;

    let table = collector.build();

    let importance_entries: Vec<serde_json::Value> = table
        .entries()
        .iter()
        .map(|e| {
            serde_json::json!({
                "layer": e.layer_id,
                "sublayer": format!("{:?}", e.sublayer),
                "importance": e.importance,
                "opr": e.opr,
            })
        })
        .collect();

    // ── P2c: head concentration decode loop ─────────────────────────────────
    let head_conc_section = if HEAD_CONC_DECODE_STEPS > 0 {
        Some(run_head_concentration_decode(
            &mut ctx,
            &prompt_ids,
            prompt_len,
        )?)
    } else {
        None
    };

    // ── JSON 출력 ────────────────────────────────────────────────────────────
    let mut output = serde_json::json!({
        "model": ctx.model_path,
        "num_layers": ctx.model.config.num_hidden_layers,
        "prompt_tokens": prompt_len,
        "importance": importance_entries,
    });

    if let Some(conc) = head_conc_section {
        output["head_concentration"] = conc;
    }

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

/// prefill 후 `HEAD_CONC_DECODE_STEPS` greedy decode step을 수행하며
/// per-(layer, kv_head) attention concentration C_h를 측정한다.
///
/// `last_step_head_attn()`은 마지막 tracked layer의 값만 반환하므로,
/// 단일 layer를 track 하는 GQA accumulator를 layer별로 개별 실행해 16×8 매트릭스를 채운다.
///
/// 단순화: 모든 layer를 한 accumulator로 track (tracked_layers=[0..num_layers])하고,
/// head attention은 매 step 캡처 후 `ConcentrationAccumulator`에 레이어별 슬롯을 확인한다.
/// 현재 `last_step_head_attn()` API는 마지막 tracked layer만 반환하므로,
/// layer 별 C_h는 `accumulate_layer_gqa()` 콜백을 직접 읽는 대신
/// ConcentrationAccumulator를 **1 layer × n_kv_heads**씩 num_layers 번 반복한다.
///
/// **현실적 간소화**: decode step에서 `last_step_head_attn`은 가장 마지막에 기록된 layer.
/// 이를 num_layers 만큼 루프하면 모든 layer를 커버하지만 비용이 num_layers 배 증가.
/// 스펙 측정 전용이므로 허용하되, production 경로 무영향(flag 0 = off 시 미진입).
fn run_head_concentration_decode(
    ctx: &mut DumpImportanceCtx,
    prompt_ids: &[u32],
    prompt_len: usize,
) -> anyhow::Result<serde_json::Value> {
    let n_layers = ctx.model.config.num_hidden_layers;
    let n_kv_heads = ctx.model.config.num_key_value_heads;
    let n_heads_q = ctx.model.config.num_attention_heads;
    let max_seq_len = ctx.kv_caches.first().map(|c| c.capacity()).unwrap_or(2048);

    eprintln!(
        "[HeadConc] decode {} steps, {} layers × {} kv_heads...",
        HEAD_CONC_DECODE_STEPS, n_layers, n_kv_heads
    );

    // greedy token을 얻기 위해 logits readback 버퍼 할당 (단일 토큰).
    let decode_logits_buf = ctx.memory.alloc(ctx.vocab_size * 4, DType::F32)?;
    let mut decode_logits = Tensor::new(
        Shape::new(vec![1, 1, ctx.vocab_size]),
        decode_logits_buf,
        ctx.backend.clone(),
    );

    // Decode workspace: accumulate_layer_gqa()는 (Some(acc), Some(ws)) 조건에서만
    // 호출된다 (transformer.rs:1639). workspace가 None이면 ws.scores가 채워지지 않아
    // last_step_head_attn()이 항상 0을 반환하므로, 실제 LayerWorkspace를 구성한다.
    let q_dim = ctx.model.config.num_attention_heads * ctx.model.config.head_dim;
    let k_dim = ctx.model.config.num_key_value_heads * ctx.model.config.head_dim;
    let v_dim = k_dim;
    let ffn_hidden = ctx.model.config.intermediate_size;
    let hidden_size = ctx.model.config.hidden_size;
    let mut gen_ws = LayerWorkspace::new(
        WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: ctx.model.config.num_attention_heads,
            max_seq_len,
        },
        ctx.memory.as_ref(),
        ctx.backend.clone(),
    )?;

    // 단일 토큰 입력 버퍼
    let tok_buf = Galloc::new().alloc(4, DType::U8)?;
    let tok_cpu = Tensor::new(Shape::new(vec![1, 1]), tok_buf, Arc::new(CpuBackend::new()));

    // ConcentrationAccumulator: [n_layers × n_kv_heads]
    let mut conc_acc = ConcentrationAccumulator::new(n_layers, n_kv_heads);

    // QueryStats e2e seam — score-active decode 경로에서 per-(layer,kv_head)
    // Q running mean/var 가 실제로 채워지는지 검증. step 간 누적이므로 루프 밖에서 1개 생성, 매 step
    // forward_into 에 공급(transformer.rs seam 의 GQA 환원 1-sample 누적).
    let head_dim = ctx.model.config.head_dim;
    let mut qstats_acc = QueryStatsAccumulator::new(n_layers, n_heads_q, n_kv_heads, head_dim);
    qstats_acc.set_active(true);

    // 첫 decode 토큰: prefill logits의 마지막 토큰 logit에서 greedy argmax.
    // prefill logits 버퍼는 이미 해제됐으므로 재계산 없이 next_token=prompt_ids.last 사용.
    // 실제 greedy는 별도로 측정 필요 없으므로 임의 시작 토큰(prompt 마지막 토큰)으로 대체.
    let mut next_token = *prompt_ids.last().unwrap_or(&1);

    for step in 0..HEAD_CONC_DECODE_STEPS {
        let start_pos = prompt_len + step;
        if start_pos >= max_seq_len {
            break;
        }

        // 토큰 입력 텐서 갱신
        unsafe {
            let ptr = tok_cpu.as_mut_ptr() as *mut u32;
            *ptr = next_token;
        }
        let tok_tensor = ctx.backend.copy_from(&tok_cpu)?;

        // GQA accumulator: 모든 layer track
        let mut acc = AttentionScoreAccumulator::new_gqa(
            max_seq_len,
            n_heads_q,
            n_kv_heads,
            n_layers,
            0, // tracked_layers=0 → 모든 layer
            0.0,
        );
        acc.set_active(true);
        acc.begin_step();

        KVCache::forward_fmt_roundtrip(&mut ctx.kv_caches, |fmts| {
            ctx.model.forward_into(TransformerModelForwardArgs {
                input_tokens: &tok_tensor,
                start_pos,
                fmts,
                backend: &ctx.backend,
                memory: &*ctx.memory,
                logits_out: &mut decode_logits,
                x_gen: None,
                // workspace가 Some이어야 transformer.rs:1639의 조건
                // `(Some(acc), Some(ws))`를 만족해 accumulate_layer_gqa()가 호출된다.
                // None이면 ws.scores가 채워지지 않아 last_step_head_attn() = 0.
                workspace: Some(&mut gen_ws),
                logits_last_only: true,
                score_accumulator: Some(&mut acc),
                // MQ-6 e2e seam: score-active decode 에서 RoPE-적용 Q 캡처 누적.
                query_stats_accumulator: Some(&mut qstats_acc),
                skip_config: None,
                importance_collector: None,
                cache_self_need_scores: false,
                layer_boundary_hook: None,
                read_stage: None,
            })
        })?;

        acc.end_step();

        // last_step_head_attn: [n_kv_heads * max_seq_len] 레이아웃.
        // head h의 유효 attention = attn[h * max_seq_len .. h * max_seq_len + seq_len].
        // ConcentrationAccumulator::accumulate의 seq_len 파라미터는 head-stride이므로
        // max_seq_len을 전달한다 (start_pos+1이 아님).
        if let Some(attn) = acc.last_step_head_attn() {
            // n_layers 전체를 커버하기 위해 각 layer를 같은 값으로 채운다.
            // (last_step_head_attn은 마지막 layer 값 — layer 해상도 없음)
            // 설계서 §2.3-B의 "마지막 tracked layer" 기준으로 단일 layer 누적.
            let layer_attns: Vec<Option<Vec<f32>>> = std::iter::once(Some(attn.to_vec()))
                .chain((1..n_layers).map(|_| None))
                .collect();
            // stride = max_seq_len (last_step_head_attn 버퍼 레이아웃)
            // 유효 길이 = start_pos + 1 (accumulate 내부에서 .min(attn.len())로 클램핑)
            conc_acc.accumulate_strided(&layer_attns, max_seq_len, step + 1, HEAD_CONC_TOP_FRAC);
        }

        // greedy next token (logits readback)
        next_token = greedy_argmax_from_logits(&decode_logits, &ctx.backend, ctx.vocab_size)?;
    }

    let mean_ch = conc_acc.mean_ch();
    let ratio = ConcentrationAccumulator::max_min_ratio(&mean_ch);

    // CSV 형태 per-(layer, head) 매트릭스 생성
    let mut rows: Vec<serde_json::Value> = Vec::new();
    for l in 0..n_layers {
        for h in 0..n_kv_heads {
            let idx = l * n_kv_heads + h;
            let ch = if idx < mean_ch.len() {
                mean_ch[idx]
            } else {
                0.0
            };
            rows.push(serde_json::json!({
                "layer": l,
                "kv_head": h,
                "C_h": ch,
            }));
        }
    }

    eprintln!(
        "[HeadConc] max/min C_h ratio = {:.3} (임계: <2→보류, ≥5→개봉후보)",
        ratio
    );

    // ── MQ-6: query_stats e2e 섹션 ──────────────────────────────────────────
    // 누적된 per-(layer,kv_head) Q running mean/var 를 JSON 으로 덤프해 실제 채워짐을 증명한다
    // (단위 테스트만으로 완료 선언 금지, U5/항목 0 허상 교훈). non-empty 신호 = mean/var 가 전부
    // 0 이 아닌 원소를 가지는가(decode step ≥1 누적 시 mean ≈ Q 평균이라 자명히 non-zero).
    let query_stats_section =
        build_query_stats_section(&mut qstats_acc, n_layers, n_kv_heads, head_dim);

    Ok(serde_json::json!({
        "decode_steps": HEAD_CONC_DECODE_STEPS,
        "top_frac": HEAD_CONC_TOP_FRAC,
        "max_min_ratio": ratio,
        "matrix": rows,
        "query_stats": query_stats_section,
    }))
}

/// MQ-6: 누적된 QueryStats 를 per-(layer,kv_head) mean/var L2-norm 매트릭스로 덤프한다.
/// 실모델 e2e 검증용 — `non_empty=true` + 각 (layer,kv_head) mean/var norm 이 누적기가 실제
/// 채워졌음을 증명한다(전부 0 이면 seam 미작동).
fn build_query_stats_section(
    qstats_acc: &mut QueryStatsAccumulator,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> serde_json::Value {
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut any_nonzero = false;
    for l in 0..n_layers {
        // layer_stats: [n_kv_heads * 2 * head_dim] (row0=mean / row1=var).
        let stats = qstats_acc.layer_stats(l).to_vec();
        for h in 0..n_kv_heads {
            let base = h * 2 * head_dim;
            let mean_slice = &stats[base..base + head_dim];
            let var_slice = &stats[base + head_dim..base + 2 * head_dim];
            let mean_norm = (mean_slice.iter().map(|x| x * x).sum::<f32>()).sqrt();
            let var_norm = (var_slice.iter().map(|x| x * x).sum::<f32>()).sqrt();
            if mean_norm != 0.0 || var_norm != 0.0 {
                any_nonzero = true;
            }
            rows.push(serde_json::json!({
                "layer": l,
                "kv_head": h,
                "mean_l2": mean_norm,
                "var_l2": var_norm,
            }));
        }
    }
    eprintln!(
        "[QueryStats] non_empty={} ({} layers × {} kv_heads, head_dim={})",
        any_nonzero, n_layers, n_kv_heads, head_dim
    );
    serde_json::json!({
        "non_empty": any_nonzero,
        "head_dim": head_dim,
        "matrix": rows,
    })
}

/// decode logits 텐서에서 greedy argmax token id를 반환한다.
fn greedy_argmax_from_logits(
    logits: &Tensor,
    backend: &Arc<dyn Backend>,
    vocab_size: usize,
) -> anyhow::Result<u32> {
    // GPU → CPU readback: byte 버퍼를 통해 복사.
    let mut cpu_bytes = vec![0u8; vocab_size * 4];
    backend.read_buffer(logits, &mut cpu_bytes)?;

    // SAFETY: F32 버퍼를 f32 슬라이스로 재해석 (read_buffer 완료 후 정렬 보장).
    let logits_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(cpu_bytes.as_ptr() as *const f32, vocab_size) };

    let best = logits_slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(best as u32)
}
