//! Adaptive multi-layer prefetch controller.
//!
//! Dynamically adjusts prefetch depth based on observed preload vs forward timing.
//! When preload takes longer than forward (pipeline stall), depth increases.
//! When there is consistent slack, depth decreases (saves memory).
//!
//! Both increases and decreases move by `adjust_step` (default 16, so one
//! full layer-group per tick). The step also acts as the decrease floor —
//! depth never drops below `adjust_step`, guaranteeing at least one primed
//! prefetch window at all times.

use std::time::Duration;

/// Adaptive prefetch depth controller for offload KV cache pipelines.
///
/// Tracks EMA of preload and forward times, adjusting depth at token boundaries.
/// - Increase: immediate on stall (preload > forward)
/// - Decrease: after `decrease_patience` consecutive slack observations (anti-oscillation)
///
/// S6 보강: `priority_hint` 슬롯 — read-plan이 layer i 의 산출한 `select` 목록을
/// layer i+1 의 prefetch 우선 힌트로 저장한다. 힌트는 1회성(take)으로 소비되며, preload_locked 는
/// 지금은 선입선출 순서라 hint 가 실제 I/O 순서를 변경하지는 않는다. 향후 preload_locked 에 hint
/// 우선 순서 지원을 추가할 때 소비 표면(take_priority_hint)을 확장하면 된다.
/// read_stage=None 이면 hint 슬롯은 항상 None 을 유지(기존 경로 byte-identical).
pub struct PrefetchController {
    /// Current prefetch depth (≥ 1).
    depth: usize,
    /// Maximum allowed depth (memory cap).
    max_depth: usize,
    /// EMA of preload duration (μs).
    preload_ema_us: f64,
    /// EMA of forward duration (μs).
    forward_ema_us: f64,
    /// EMA smoothing coefficient.
    alpha: f64,
    /// Number of samples recorded so far.
    samples: usize,
    /// Warmup period: no adjustments until this many samples.
    warmup_samples: usize,
    /// Consecutive observations of sufficient slack.
    slack_streak: usize,
    /// Number of consecutive slack observations required before decreasing depth.
    decrease_patience: usize,
    /// Step size for each increase / decrease tick (≥ 1).
    adjust_step: usize,
    /// S6 priority hint: layer index + select list from read-plan.
    /// `Some((layer_idx, select))` = layer `layer_idx` 의 preload 시 이 select 목록을 우선 힌트로 삼는다.
    /// `None` = 힌트 없음(기본, read_stage=None 경로).
    priority_hint: Option<(usize, Vec<usize>)>,
}

/// Default starting prefetch depth used by [`PrefetchController::new`].
///
/// Chosen to match typical on-device transformer layer counts (Llama 3.2 1B
/// = 16 layers), so the pipeline starts fully primed and the adaptive loop
/// only has to shrink if memory pressure or slack is observed.
pub const DEFAULT_INITIAL_DEPTH: usize = 16;

/// Default step size for each adaptive increase / decrease tick.
///
/// One "layer group" (matches `DEFAULT_INITIAL_DEPTH`) so the controller
/// reacts in chunks that are meaningful for memory accounting, and so the
/// step also serves as the decrease floor (depth never drops below one
/// full window).
pub const DEFAULT_ADJUST_STEP: usize = 16;

impl PrefetchController {
    /// Create a new controller with [`DEFAULT_INITIAL_DEPTH`] as the starting
    /// depth and [`DEFAULT_ADJUST_STEP`] as the per-tick change.
    ///
    /// - `max_depth`: upper bound on prefetch depth (memory limit)
    /// - `num_layers`: total transformer layers (used as warmup period)
    pub fn new(max_depth: usize, num_layers: usize) -> Self {
        Self::with_tuning(
            max_depth,
            DEFAULT_INITIAL_DEPTH,
            DEFAULT_ADJUST_STEP,
            num_layers,
        )
    }

    /// Create a new controller with an explicit starting depth.
    ///
    /// Uses [`DEFAULT_ADJUST_STEP`] for the per-tick change. `initial_depth`
    /// is clamped to `[1, max_depth]`.
    pub fn with_initial_depth(max_depth: usize, initial_depth: usize, num_layers: usize) -> Self {
        Self::with_tuning(max_depth, initial_depth, DEFAULT_ADJUST_STEP, num_layers)
    }

    /// Create a new controller with an explicit starting depth and step size.
    ///
    /// `adjust_step` doubles as the decrease floor: `depth` never drops
    /// below `adjust_step` once `adjust()` starts running, so a step of
    /// 16 keeps at least one full 16-layer window prefetched. Tests that
    /// want ±1 behavior pass `adjust_step = 1`.
    pub fn with_tuning(
        max_depth: usize,
        initial_depth: usize,
        adjust_step: usize,
        num_layers: usize,
    ) -> Self {
        let max_depth = max_depth.max(1);
        Self {
            depth: initial_depth.clamp(1, max_depth),
            max_depth,
            preload_ema_us: 0.0,
            forward_ema_us: 0.0,
            alpha: 0.3,
            samples: 0,
            warmup_samples: num_layers,
            slack_streak: 0,
            decrease_patience: 3,
            adjust_step: adjust_step.max(1),
            priority_hint: None,
        }
    }

    /// Current prefetch depth.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Maximum allowed prefetch depth.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// EMA of preload time in microseconds.
    pub fn preload_ema_us(&self) -> f64 {
        self.preload_ema_us
    }

    /// EMA of forward time in microseconds.
    pub fn forward_ema_us(&self) -> f64 {
        self.forward_ema_us
    }

    /// Record a (preload_duration, forward_duration) observation.
    pub fn record(&mut self, preload_dur: Duration, forward_dur: Duration) {
        let preload_us = preload_dur.as_secs_f64() * 1_000_000.0;
        let forward_us = forward_dur.as_secs_f64() * 1_000_000.0;

        if self.samples == 0 {
            self.preload_ema_us = preload_us;
            self.forward_ema_us = forward_us;
        } else {
            self.preload_ema_us =
                self.alpha * preload_us + (1.0 - self.alpha) * self.preload_ema_us;
            self.forward_ema_us =
                self.alpha * forward_us + (1.0 - self.alpha) * self.forward_ema_us;
        }
        self.samples += 1;
    }

    /// Record only a preload duration observation.
    pub fn record_preload(&mut self, preload_dur: Duration) {
        let preload_us = preload_dur.as_secs_f64() * 1_000_000.0;
        if self.samples == 0 {
            self.preload_ema_us = preload_us;
        } else {
            self.preload_ema_us =
                self.alpha * preload_us + (1.0 - self.alpha) * self.preload_ema_us;
        }
    }

    /// Record only a forward duration observation.
    pub fn record_forward(&mut self, forward_dur: Duration) {
        let forward_us = forward_dur.as_secs_f64() * 1_000_000.0;
        if self.samples == 0 {
            self.forward_ema_us = forward_us;
        } else {
            self.forward_ema_us =
                self.alpha * forward_us + (1.0 - self.alpha) * self.forward_ema_us;
        }
        self.samples += 1; // Count forward calls as samples (one per layer)
    }

    /// S6: read-plan 보강 채널 — layer `layer_idx` 의 prefetch 우선 힌트를 저장한다.
    ///
    /// layer i−1 의 read_plan 이 산출한 `select` 를 layer i 의 preload 우선 목록으로 공급.
    /// 현재 `preload_locked` 는 선입선출이라 hint 가 I/O 순서를 바꾸지는 않는다. 힌트는 "어느 KV 를
    /// layer i 가 읽을지"를 알고 있으므로, 향후 `preload_locked` 에 page-id 우선 로드를 추가할 때
    /// 이 표면을 확장한다. 현재는 hint 저장 + 관찰(테스트)만 제공한다.
    ///
    /// 기존 경로(`read_stage=None`)는 이 메서드를 호출하지 않아 `priority_hint` 가 항상 `None` 유지
    /// → byte-identical(INV-147).
    pub fn set_priority_hint(&mut self, layer_idx: usize, select: Vec<usize>) {
        self.priority_hint = Some((layer_idx, select));
    }

    /// S6: 저장된 우선 힌트를 소비(take)한다.
    ///
    /// 한 번 take 하면 `None` 으로 리셋. 호출자는 반환된 `(layer_idx, select)` 를 preload 우선
    /// 순서로 활용한다. 힌트가 없으면 `None` 반환 — 호출자는 기존 순차 preload 를 그대로 수행한다.
    pub fn take_priority_hint(&mut self) -> Option<(usize, Vec<usize>)> {
        self.priority_hint.take()
    }

    /// Adjust depth based on accumulated timing data.
    /// Call once per token boundary (after all layers complete).
    pub fn adjust(&mut self) {
        if self.samples < self.warmup_samples {
            return; // Not enough data yet
        }
        if self.forward_ema_us < 1.0 {
            return; // Avoid division by near-zero
        }

        let slack_ratio = 1.0 - (self.preload_ema_us / self.forward_ema_us);

        if slack_ratio < 0.0 {
            // Stall: preload takes longer than forward → increase immediately
            self.depth = (self.depth + self.adjust_step).min(self.max_depth);
            self.slack_streak = 0;
        } else if slack_ratio > 0.3 {
            // Ample slack → count toward decrease. `adjust_step` is also the
            // floor: never drop below one full window (e.g. 16), so a slack
            // streak that would push depth under `adjust_step` is clamped.
            self.slack_streak += 1;
            if self.slack_streak >= self.decrease_patience && self.depth > self.adjust_step {
                self.depth = self
                    .depth
                    .saturating_sub(self.adjust_step)
                    .max(self.adjust_step);
                self.slack_streak = 0;
            }
        } else {
            // Balanced — hold steady
            self.slack_streak = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dur_us(us: u64) -> Duration {
        Duration::from_micros(us)
    }

    #[test]
    fn test_warmup_no_adjust() {
        let num_layers = 4;
        let mut ctrl = PrefetchController::with_tuning(4, 1, 1, num_layers);

        // Record fewer samples than warmup → depth stays at 1
        for _ in 0..3 {
            ctrl.record(dur_us(5000), dur_us(2000)); // stall condition
        }
        ctrl.adjust();
        assert_eq!(ctrl.depth(), 1, "depth must not change during warmup");
    }

    #[test]
    fn test_increase_on_stall() {
        let num_layers = 2;
        let mut ctrl = PrefetchController::with_tuning(4, 1, 1, num_layers);
        assert_eq!(ctrl.depth(), 1);

        // Record enough samples to exit warmup (preload > forward = stall)
        for _ in 0..3 {
            ctrl.record(dur_us(5000), dur_us(2000));
        }
        ctrl.adjust();
        assert_eq!(ctrl.depth(), 2, "depth should increase on stall");

        // Another stall → increase again
        for _ in 0..2 {
            ctrl.record(dur_us(5000), dur_us(2000));
        }
        ctrl.adjust();
        assert_eq!(ctrl.depth(), 3);
    }

    #[test]
    fn test_decrease_with_patience() {
        let num_layers = 2;
        let mut ctrl = PrefetchController::with_tuning(4, 1, 1, num_layers);

        // Drive depth up first
        for _ in 0..3 {
            ctrl.record(dur_us(5000), dur_us(2000));
        }
        ctrl.adjust();
        assert_eq!(ctrl.depth(), 2);

        // First: stabilize EMA into slack territory (need enough samples to
        // overcome the stall EMA from warmup phase)
        for _ in 0..20 {
            ctrl.record(dur_us(500), dur_us(3000)); // slack_ratio ≈ 0.83
        }
        // Reset streak by calling adjust during a stall-ema → may increase depth
        // Let's just settle the EMA first, then count patience from scratch
        ctrl.adjust(); // first slack adjust → streak=1

        let depth_after_settle = ctrl.depth();

        // Now count patience: need 2 more consecutive slack adjusts
        for _ in 0..num_layers {
            ctrl.record(dur_us(500), dur_us(3000));
        }
        ctrl.adjust(); // streak=2
        assert_eq!(
            ctrl.depth(),
            depth_after_settle,
            "should not decrease before patience met"
        );

        for _ in 0..num_layers {
            ctrl.record(dur_us(500), dur_us(3000));
        }
        ctrl.adjust(); // streak=3 → decrease
        assert_eq!(
            ctrl.depth(),
            depth_after_settle - 1,
            "depth should decrease after patience met"
        );
    }

    #[test]
    fn test_max_depth_cap() {
        let num_layers = 2;
        let mut ctrl = PrefetchController::with_tuning(3, 1, 1, num_layers);

        // Stall many times
        for _ in 0..20 {
            for _ in 0..num_layers {
                ctrl.record(dur_us(5000), dur_us(2000));
            }
            ctrl.adjust();
        }
        assert_eq!(ctrl.depth(), 3, "depth must not exceed max_depth");
    }

    #[test]
    fn test_no_oscillation() {
        let num_layers = 2;
        let mut ctrl = PrefetchController::with_tuning(4, 1, 1, num_layers);

        // Warmup with stall → depth=2
        for _ in 0..3 {
            ctrl.record(dur_us(5000), dur_us(2000));
        }
        ctrl.adjust();
        assert_eq!(ctrl.depth(), 2);

        // Alternate: one slack round, one stall round → should not oscillate
        for _ in 0..5 {
            // Slack round
            for _ in 0..num_layers {
                ctrl.record(dur_us(500), dur_us(3000));
            }
            ctrl.adjust();

            // Stall round
            for _ in 0..num_layers {
                ctrl.record(dur_us(5000), dur_us(2000));
            }
            ctrl.adjust();
        }
        // Due to patience=3 on decrease, alternating prevents decrease,
        // but stalls keep pushing up → depth should be >= 2
        assert!(
            ctrl.depth() >= 2,
            "alternating should not cause depth to drop below 2"
        );
    }

    #[test]
    fn test_default_initial_depth_is_clamped() {
        // `new` starts at DEFAULT_INITIAL_DEPTH but is clamped down to
        // `max_depth` when the latter is smaller.
        let small = PrefetchController::new(4, 2);
        assert_eq!(small.depth(), 4, "small max_depth clamps initial depth");

        let large = PrefetchController::new(128, 2);
        assert_eq!(
            large.depth(),
            DEFAULT_INITIAL_DEPTH,
            "initial depth equals DEFAULT_INITIAL_DEPTH when max_depth allows it"
        );
    }

    #[test]
    fn test_default_step_moves_by_16_and_respects_floor() {
        // max=128 gives the step room to grow; num_layers=2 keeps warmup short.
        let num_layers = 2;
        let mut ctrl = PrefetchController::new(128, num_layers);
        assert_eq!(ctrl.depth(), DEFAULT_INITIAL_DEPTH);

        // Stall → +16 to 32.
        for _ in 0..3 {
            ctrl.record(dur_us(5000), dur_us(2000));
        }
        ctrl.adjust();
        assert_eq!(ctrl.depth(), 32, "stall bumps depth by one adjust_step");

        // 3 consecutive slack ticks → -16 back to 16 (the floor).
        for _ in 0..30 {
            ctrl.record(dur_us(500), dur_us(3000));
        }
        ctrl.adjust();
        ctrl.adjust();
        let before_decrease = ctrl.depth();
        ctrl.adjust();
        assert!(
            ctrl.depth() < before_decrease,
            "three slack adjusts should decrease depth"
        );
        assert_eq!(
            ctrl.depth(),
            DEFAULT_ADJUST_STEP,
            "decrease clamps to adjust_step floor"
        );

        // Further slack must not push below the floor.
        for _ in 0..30 {
            ctrl.record(dur_us(500), dur_us(3000));
        }
        for _ in 0..10 {
            ctrl.adjust();
        }
        assert_eq!(
            ctrl.depth(),
            DEFAULT_ADJUST_STEP,
            "depth is pinned at the adjust_step floor"
        );
    }

    #[test]
    fn test_with_initial_depth_clamps_to_range() {
        // `initial_depth=0` is promoted to 1, and an oversized initial_depth
        // is clamped to max_depth.
        let floor = PrefetchController::with_initial_depth(8, 0, 2);
        assert_eq!(floor.depth(), 1);

        let ceil = PrefetchController::with_initial_depth(8, 999, 2);
        assert_eq!(ceil.depth(), 8);
    }

    // S6: priority hint API 테스트 — set_priority_hint / take_priority_hint 정합성.

    /// hint 없는 기본 상태에서 take_priority_hint 는 None 을 반환한다.
    #[test]
    fn test_priority_hint_default_none() {
        let mut ctrl = PrefetchController::new(4, 2);
        assert!(
            ctrl.take_priority_hint().is_none(),
            "기본 상태에서 priority_hint 는 None"
        );
    }

    /// set_priority_hint 로 저장한 hint 를 take_priority_hint 로 정확히 소비한다.
    #[test]
    fn test_priority_hint_set_and_take() {
        let mut ctrl = PrefetchController::new(4, 2);
        let select = vec![0usize, 2, 5];
        ctrl.set_priority_hint(3, select.clone());

        let hint = ctrl.take_priority_hint();
        assert!(hint.is_some(), "set 후 take 는 Some 을 반환해야 한다");
        let (layer_idx, taken_select) = hint.unwrap();
        assert_eq!(layer_idx, 3, "layer_idx 일치");
        assert_eq!(taken_select, select, "select 목록 일치");
    }

    /// take 후 두 번째 take 는 None 을 반환한다(1회성 소비).
    #[test]
    fn test_priority_hint_take_consumes() {
        let mut ctrl = PrefetchController::new(4, 2);
        ctrl.set_priority_hint(1, vec![0, 1]);
        let _ = ctrl.take_priority_hint(); // 첫 번째 소비
        assert!(
            ctrl.take_priority_hint().is_none(),
            "두 번째 take 는 None(1회성 소비)"
        );
    }

    /// read_stage=None 경로 시뮬레이션: set_priority_hint 를 호출하지 않으면 hint 는 None 유지.
    /// INV-147 byte-identical 보존 — 기존 depth/EMA 에 영향 없음.
    #[test]
    fn test_priority_hint_no_interference_with_depth() {
        let num_layers = 2;
        let mut ctrl = PrefetchController::with_tuning(4, 1, 1, num_layers);

        // hint 없이 stall → depth 증가
        for _ in 0..3 {
            ctrl.record(dur_us(5000), dur_us(2000));
        }
        ctrl.adjust();
        let depth_after = ctrl.depth();
        assert_eq!(depth_after, 2, "hint 미설정 시 depth 동작 불변");

        // hint 설정 후에도 depth 동작 불변
        ctrl.set_priority_hint(2, vec![1, 3]);
        for _ in 0..2 {
            ctrl.record(dur_us(5000), dur_us(2000));
        }
        ctrl.adjust();
        assert_eq!(ctrl.depth(), 3, "hint 설정 후에도 depth 증가 불변");

        // hint 는 아직 소비되지 않음
        let hint = ctrl.take_priority_hint();
        assert!(hint.is_some(), "hint 는 depth 조정과 독립적으로 보존");
    }
}
