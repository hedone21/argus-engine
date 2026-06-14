//! A2SF forgetting factor 게이트 지표 덤프 — 측정 전용 읽기 표면 (P2b 잔여).
//!
//! arXiv 2407.20485. KV roadmap 항목 0 측정 스프린트:
//! `arch/kv_roadmap_item0_measurement.md` §2.2 / §4.2.
//!
//! score accumulator 의 token importance(`importance_scores()`, 읽기 전용)에서 두 게이트 지표를
//! 산출한다 — ① **BOS ratio** = `importance[0] / mean(importance[1..])`(BOS 지배 완화의 직접 지표),
//! ② **HH 집합** = importance 상위 top-k(budget 크기) 토큰 인덱스(decay 0.0 vs 0.8 Jaccard 입력).
//!
//! **절대 제약**: 누적 로직(`begin_step`/`accumulate_*`) 무수정 — 본 모듈은 importance 슬라이스를
//! 읽기만 한다. `--dump-a2sf` 미지정 시 호출되지 않음(production 무영향).

use serde::Serialize;

/// A2SF 게이트 지표 1회 스냅샷(eviction 직전 또는 run 종료 시점).
#[derive(Debug, Clone, Serialize)]
pub struct A2sfSnapshot {
    /// 스냅샷 시점 식별(예: decode step index, run-end 은 -1).
    pub step: i64,
    /// 유효 토큰 수(현재 cache_pos, importance 의 활성 prefix 길이).
    pub current_pos: usize,
    /// BOS(pos 0) importance.
    pub bos_score: f32,
    /// non-BOS(pos 1..current_pos) importance 평균.
    pub non_bos_mean: f32,
    /// BOS/non-BOS ratio = `bos_score / non_bos_mean`(non-BOS 평균 0 이면 무한대 회피로 0).
    pub bos_ratio: f32,
    /// HH(heavy hitter) 집합 = importance 상위 top-k 토큰 인덱스(ascending). budget 크기 = `top_k`.
    pub hh_topk: Vec<usize>,
}

/// importance 슬라이스에서 A2SF 게이트 지표를 산출한다(읽기 전용).
///
/// `importance`: accumulator `importance_scores()`(`[max_seq_len]`, time-normalized 가능).
/// `current_pos`: 유효 토큰 수(importance 의 활성 prefix — 나머지는 0). `top_k`: HH 집합 크기(budget).
/// `step`: 스냅샷 식별자(run-end 은 -1).
pub fn compute_a2sf_snapshot(
    importance: &[f32],
    current_pos: usize,
    top_k: usize,
    step: i64,
) -> A2sfSnapshot {
    let n = current_pos.min(importance.len());
    let imp = &importance[..n];

    let bos_score = imp.first().copied().unwrap_or(0.0);
    let (non_bos_sum, non_bos_cnt) = if n > 1 {
        (imp[1..].iter().copied().sum::<f32>(), (n - 1) as f32)
    } else {
        (0.0, 0.0)
    };
    let non_bos_mean = if non_bos_cnt > 0.0 {
        non_bos_sum / non_bos_cnt
    } else {
        0.0
    };
    let bos_ratio = if non_bos_mean.abs() > f32::EPSILON {
        bos_score / non_bos_mean
    } else {
        0.0
    };

    // HH 집합: importance 상위 top_k 토큰 인덱스(동점은 위치 오름차순으로 결정성).
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| {
        imp[b]
            .partial_cmp(&imp[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut hh_topk: Vec<usize> = idx.into_iter().take(top_k.min(n)).collect();
    hh_topk.sort_unstable(); // ascending — Jaccard 오프라인 비교 편의.

    A2sfSnapshot {
        step,
        current_pos: n,
        bos_score,
        non_bos_mean,
        bos_ratio,
        hh_topk,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BOS 지배(BOS=3000, 나머지 ~3) → bos_ratio ≈ 1000. HH 에 BOS 포함.
    #[test]
    fn bos_dominance_ratio_and_hh() {
        let mut imp = vec![3.0f32; 10];
        imp[0] = 3000.0;
        let snap = compute_a2sf_snapshot(&imp, 10, 3, 0);
        // non-BOS 평균 = 3.0, ratio = 1000.
        assert!((snap.non_bos_mean - 3.0).abs() < 1e-4);
        assert!(
            (snap.bos_ratio - 1000.0).abs() < 1e-2,
            "ratio {}",
            snap.bos_ratio
        );
        assert!(snap.hh_topk.contains(&0), "BOS 가 HH top-3 에 포함");
        assert_eq!(snap.hh_topk.len(), 3);
        assert!(snap.hh_topk.windows(2).all(|w| w[0] < w[1]), "ascending");
    }

    /// 완화 시나리오: BOS 와 non-BOS 가 비슷 → ratio ≈ 1. HH 가 BOS 외 토큰 포함.
    #[test]
    fn flattened_distribution_lower_ratio() {
        let imp = vec![5.0, 4.0, 3.0, 9.0, 8.0, 1.0, 2.0, 0.5];
        let snap = compute_a2sf_snapshot(&imp, 8, 3, 7);
        // non-BOS 평균 = (4+3+9+8+1+2+0.5)/7 ≈ 3.93, ratio = 5/3.93 ≈ 1.27.
        assert!(
            snap.bos_ratio < 2.0,
            "완화된 분포 ratio < 2: {}",
            snap.bos_ratio
        );
        // HH top-3 = pos {3(9.0), 4(8.0), 0(5.0)} → ascending [0,3,4].
        assert_eq!(snap.hh_topk, vec![0, 3, 4]);
    }

    /// current_pos 가 importance 길이보다 작으면 활성 prefix 만 본다(나머지 0 무시).
    #[test]
    fn respects_current_pos_prefix() {
        let mut imp = vec![0.0f32; 100];
        imp[0] = 50.0;
        imp[1] = 10.0;
        imp[2] = 20.0;
        let snap = compute_a2sf_snapshot(&imp, 3, 2, 0);
        assert_eq!(snap.current_pos, 3);
        assert!((snap.non_bos_mean - 15.0).abs() < 1e-4, "(10+20)/2");
        assert_eq!(snap.hh_topk, vec![0, 2], "top-2 = pos 0(50),2(20)");
    }

    /// non-BOS 평균이 0(전부 0 또는 단일 토큰) → ratio 0(무한대 회피).
    #[test]
    fn zero_non_bos_mean_yields_zero_ratio() {
        let imp = vec![5.0f32, 0.0, 0.0];
        let snap = compute_a2sf_snapshot(&imp, 3, 1, 0);
        assert_eq!(snap.bos_ratio, 0.0);
        // 단일 토큰.
        let single = compute_a2sf_snapshot(&[7.0], 1, 1, 0);
        assert_eq!(single.bos_ratio, 0.0);
        assert_eq!(single.hh_topk, vec![0]);
    }
}
