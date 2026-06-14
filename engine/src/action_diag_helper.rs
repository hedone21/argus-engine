//! Cache pressure action diagnostic helper.
//!
//! KV cache eviction (sliding/h2o/d2o) 등 cache pressure action 시점에 호출되는
//! score 분포 디버그 출력 helper. `log::info!` 로 stderr 통계 dump + 옵션 CSV
//! 파일 생성을 한 함수로 통합. 이전 `observability::events::{build_score_snapshot,
//! dump_scores_csv}` + `StderrDiagnosticSink` 의 `ScoreDiagnostic` 분기를 응축.

use std::io::Write;
use std::path::Path;

/// Action 시점 score 분포 진단을 stderr (log::info!) 와 옵션 CSV 로 dump.
///
/// `scores[..cache_pos]` 의 통계(min/max/mean/std/cv), top-K/bottom-K 위치,
/// prefix vs 나머지 평균, σ-distribution 을 한 번에 출력. `csv_path` 가 주어지면
/// raw `(position,score)` CSV 도 함께 기록.
///
/// `cache_pos == 0` 또는 `scores.len() < cache_pos` 인 경우 silently skip.
pub fn log_score_diag(
    scores: &[f32],
    cache_pos: usize,
    protected_prefix: usize,
    decode_steps: usize,
    k: usize,
    csv_path: Option<&Path>,
) {
    if cache_pos == 0 || scores.len() < cache_pos {
        return;
    }

    let active = &scores[..cache_pos];

    // Overall stats
    let min = active.iter().copied().fold(f32::INFINITY, f32::min);
    let max = active.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = active.iter().sum::<f32>() / active.len() as f32;
    let var = active.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / active.len() as f32;
    let std_dev = var.sqrt();
    let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

    // Top-K / Bottom-K
    let mut indexed: Vec<(usize, f32)> = active.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k: Vec<(usize, f32)> = indexed.iter().take(k).copied().collect();
    let bottom_k: Vec<(usize, f32)> = indexed.iter().rev().take(k).copied().collect();

    // Prefix vs rest averages
    let prefix_end = protected_prefix.min(cache_pos);
    let prefix_avg = if prefix_end > 0 {
        scores[..prefix_end].iter().sum::<f32>() / prefix_end as f32
    } else {
        0.0
    };
    let rest_avg = if cache_pos > protected_prefix {
        scores[protected_prefix..cache_pos].iter().sum::<f32>()
            / (cache_pos - protected_prefix) as f32
    } else {
        0.0
    };

    // σ-distribution for non-prefix tokens
    let (above_1sigma_frac, above_2sigma_frac) = if cache_pos > protected_prefix {
        let non_prefix = &scores[protected_prefix..cache_pos];
        let np_mean = non_prefix.iter().sum::<f32>() / non_prefix.len() as f32;
        let np_var = non_prefix
            .iter()
            .map(|s| (s - np_mean).powi(2))
            .sum::<f32>()
            / non_prefix.len() as f32;
        let np_std = np_var.sqrt();
        let above_1 = non_prefix.iter().filter(|&&s| s > np_mean + np_std).count();
        let above_2 = non_prefix
            .iter()
            .filter(|&&s| s > np_mean + 2.0 * np_std)
            .count();
        (
            above_1 as f32 / non_prefix.len() as f32,
            above_2 as f32 / non_prefix.len() as f32,
        )
    } else {
        (0.0, 0.0)
    };

    // stderr dump (log::info!)
    log::info!(
        "[ScoreDiag] cache_pos={}, prefix={}, decode_steps={}",
        cache_pos,
        protected_prefix,
        decode_steps
    );
    log::info!(
        "[ScoreDiag] Score stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}, cv={:.4}",
        min,
        max,
        mean,
        std_dev,
        cv
    );
    let top_str: Vec<String> = top_k
        .iter()
        .map(|(pos, score)| format!("{}:{:.3}", pos, score))
        .collect();
    let bot_str: Vec<String> = bottom_k
        .iter()
        .map(|(pos, score)| format!("{}:{:.3}", pos, score))
        .collect();
    log::info!("[ScoreDiag] Top-{}: [{}]", k, top_str.join(", "));
    log::info!("[ScoreDiag] Bot-{}: [{}]", k, bot_str.join(", "));
    log::info!(
        "[ScoreDiag] Prefix avg={:.4}, Rest avg={:.4}, ratio={:.2}x",
        prefix_avg,
        rest_avg,
        if rest_avg > 0.0 {
            prefix_avg / rest_avg
        } else {
            0.0
        }
    );
    let n_non_prefix = cache_pos.saturating_sub(protected_prefix);
    if n_non_prefix > 0 {
        let above_1s = (above_1sigma_frac * n_non_prefix as f32) as usize;
        let above_2s = (above_2sigma_frac * n_non_prefix as f32) as usize;
        log::info!(
            "[ScoreDiag] Non-prefix: n={}, >mean+1σ={}({:.1}%), >mean+2σ={}({:.1}%)",
            n_non_prefix,
            above_1s,
            above_1sigma_frac * 100.0,
            above_2s,
            above_2sigma_frac * 100.0,
        );
    }

    // Optional CSV dump
    if let Some(path) = csv_path
        && let Err(e) = dump_scores_csv(scores, cache_pos, path)
    {
        log::warn!("[ScoreDiag] CSV dump failed: {}", e);
    }
}

// ── Weight (precision) swap event log helpers ────────────────
//
// 이전 `StderrDiagnosticSink::emit` 의 `WeightSwap*` 분기를 응축. 각 helper 는
// 이전 enum variant 1개에 대응하며, `kind` 는 dispatcher 식별 문자열
// ("IntraForward" / "PhaseAware" / "Incremental" / "Subsystem") 을 그대로
// 받는다. helper 함수가 도메인 어휘를 응축하므로 emit 사이트는 enum 의존성을
//갖지 않는다.

pub fn log_swap_plan_committed(
    kind: &str,
    algorithm: &str,
    ratio: f32,
    k_chunk: usize,
    n_layers: usize,
) {
    log::info!(
        "[WeightSwap] PlanCommitted: kind={}, algo={}, ratio={:.2}, k={}, n_layers={}",
        kind,
        algorithm,
        ratio,
        k_chunk,
        n_layers
    );
}

pub fn log_swap_chunk_drained(
    kind: &str,
    chunk_idx: usize,
    layers_done: usize,
    latency_ms: f32,
    stages: Option<&str>,
) {
    if let Some(s) = stages {
        log::info!(
            "[WeightSwap] ChunkDrained: kind={}, idx={}, layers={}, latency={:.1}ms, stages={}",
            kind,
            chunk_idx,
            layers_done,
            latency_ms,
            s
        );
    } else {
        log::info!(
            "[WeightSwap] ChunkDrained: kind={}, idx={}, layers={}, latency={:.1}ms",
            kind,
            chunk_idx,
            layers_done,
            latency_ms
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn log_swap_plan_retired(
    kind: &str,
    qcf_actual: Option<f32>,
    token: usize,
    elapsed_ms: f32,
    ratio: Option<f32>,
    n_planned: Option<usize>,
    actually_q4: Option<usize>,
) {
    let qcf_str = qcf_actual
        .map(|v| format!("{:.4}", v))
        .unwrap_or_else(|| "n/a".to_string());
    let ratio_str = ratio
        .map(|v| format!(", ratio={:.2}", v))
        .unwrap_or_default();
    let planned_str = n_planned
        .map(|v| format!(", planned={}", v))
        .unwrap_or_default();
    let actual_str = actually_q4
        .map(|v| format!(", actually_q4={}", v))
        .unwrap_or_default();
    log::info!(
        "[WeightSwap] PlanRetired: kind={}, qcf={}, token={}, elapsed={:.1}ms{}{}{}",
        kind,
        qcf_str,
        token,
        elapsed_ms,
        ratio_str,
        planned_str,
        actual_str
    );
}

pub fn log_swap_failed(kind: &str, reason: &str, layer: Option<usize>, token: Option<usize>) {
    let layer_str = layer.map(|v| format!(", layer={}", v)).unwrap_or_default();
    let token_str = token.map(|v| format!(", token={}", v)).unwrap_or_default();
    log::warn!(
        "[WeightSwap] SwapFailed: kind={}, reason=\"{}\"{}{}",
        kind,
        reason,
        layer_str,
        token_str
    );
}

pub fn log_swap_batch_summary(
    kind: &str,
    mode: &str,
    target_layers: usize,
    max_release_pending: usize,
    max_dispatcher_pending: usize,
) {
    log::info!(
        "[WeightSwap] BatchSummary: kind={}, mode={}, target_layers={}, max_release_pending={}, max_dispatcher_pending={}",
        kind,
        mode,
        target_layers,
        max_release_pending,
        max_dispatcher_pending
    );
}

pub fn log_swap_config_warning(source: &str, message: &str) {
    log::warn!("[WeightSwap] ConfigWarning: source={}, {}", source, message);
}

pub fn log_swap_sub_batch_wait(layer_idx: usize, wait_ms: f32) {
    log::info!(
        "[WeightSwap] SubBatchWait: layer_idx={} wait_ms={:.2}",
        layer_idx,
        wait_ms
    );
}

#[allow(clippy::too_many_arguments)]
pub fn log_swap_prof_breakdown(
    layer_idx: usize,
    subname: &str,
    is_weight: bool,
    tensor_size: usize,
    lookup_us: f32,
    dim_us: f32,
    bytes_us: f32,
    permute_us: f32,
    wrap_us: f32,
    cpu_us: f32,
    upload_us: f32,
    total_us: f32,
    source: &str,
) {
    let source_str = if source.is_empty() {
        String::new()
    } else {
        format!(" source={}", source)
    };
    log::info!(
        "[swap-prof] layer={} sub={} is_weight={} size={} \
         lookup={:.1} dim={:.1} bytes={:.1} permute={:.1} \
         wrap={:.1} cpu={:.1} upload={:.1} total={:.1}{}",
        layer_idx,
        subname,
        is_weight as u8,
        tensor_size,
        lookup_us,
        dim_us,
        bytes_us,
        permute_us,
        wrap_us,
        cpu_us,
        upload_us,
        total_us,
        source_str,
    );
}

fn dump_scores_csv(scores: &[f32], cache_pos: usize, path: &Path) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "position,score")?;
    for (i, &s) in scores[..cache_pos.min(scores.len())].iter().enumerate() {
        writeln!(f, "{},{:.6}", i, s)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_score_diag_handles_empty_cache_silently() {
        // cache_pos == 0 → silent skip, no panic
        log_score_diag(&[], 0, 0, 0, 10, None);
    }

    #[test]
    fn log_score_diag_handles_short_scores_silently() {
        // scores.len() < cache_pos → silent skip
        log_score_diag(&[1.0, 2.0], 10, 0, 0, 5, None);
    }

    #[test]
    fn log_score_diag_computes_without_panic() {
        let scores: Vec<f32> = (0..32).map(|i| i as f32).collect();
        log_score_diag(&scores, 32, 4, 8, 5, None);
    }

    #[test]
    fn log_score_diag_writes_csv_when_path_given() {
        use std::io::Read;
        let scores: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tmp = std::env::temp_dir().join("llmrs_score_diag_test.csv");
        log_score_diag(&scores, 4, 0, 0, 2, Some(&tmp));
        let mut content = String::new();
        std::fs::File::open(&tmp)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();
        assert!(content.starts_with("position,score"));
        assert!(content.contains("0,1.000000"));
        let _ = std::fs::remove_file(&tmp);
    }
}
