//! Adaptive CPU–GPU split controller for the tensor-partition decode path
//! (ticket 021, a Doppeladler-style control arm).
//!
//! One [`SegState`] per `(layer, segment)` pair — 56 on a 28-layer model (ATTN and FFN per
//! layer). Each decode token feeds it one [`Obs`]; it answers with the GPU share to run next
//! token. The three phases follow the Doppeladler heuristic (Startup → Descent → Lookup) with
//! the observation model of ticket 021 §D3 (revision 1):
//!
//! - The CPU looks at the GPU's done-flag only after its own share is finished. If the flag is
//!   still down it waits, and the wait yields the exact GPU time (`waited = true`). If the flag
//!   is already up we only know `t_gpu <= t_cpu` (`waited = false`, no value).
//! - Startup runs one token serially (both times exact) and jumps to the balance point.
//! - Descent follows the exact gradient when it has a GPU time, and probes upward
//!   (`+quantum · 2^k`) when the GPU had slack of unknown size. It keeps a bracket — the highest
//!   quantum where the CPU did not wait, the lowest where it did — and never steps outside it;
//!   adjacent bracket ends mean converged (a boundary that flips with noise still settles).
//! - Lookup holds the converged share. It leaves on the Doppeladler 1.2× slowdown rule, and on a
//!   periodic one-quantum probe that finds GPU slack after contention clears. A probe hit needs
//!   two probe tokens in a row without a wait, so one noisy token does not restart Descent.
//!
//! Pure state + arithmetic: time is an argument, so the whole controller runs on the host.

use crate::layers::tensor_partition::GPU_ONLY_THRESHOLD;

/// Descent step size for the exact-gradient update.
pub const DEFAULT_ETA: f32 = 0.25;
/// Doppeladler's contention threshold: a segment slower than this × its best time is contended.
pub const DEFAULT_CONTENTION_RATIO: f32 = 1.2;
/// FFN row quantum (GPU work-group and CPU chunk alignment).
pub const FFN_ROW_QUANTUM: usize = 128;
/// Consecutive tokens a condition must hold (convergence, contention).
const STREAK: usize = 3;
/// Descent gives up and adopts the current share after this many tokens.
const MAX_DESCENT_TOKENS: u32 = 16;
/// Upward probe step cap, in quanta.
const MAX_PROBE_QUANTA: usize = 8;
/// Every this many Lookup tokens, one runs one quantum above the converged share.
const PROBE_PERIOD: u32 = 8;
/// Lowest CPU thread count the contention rule may reduce to.
pub const MIN_THREADS: usize = 4;
/// Threads removed per CPU-contention event.
const THREAD_STEP: usize = 2;

/// Which segment of a layer a [`SegState`] controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegKind {
    /// Q heads `[0, h_g)` and Wo columns `[0, h_g·head_dim)` on the GPU.
    Attn,
    /// gate/up rows and down columns `[0, s)` on the GPU.
    Ffn,
}

/// Quantization of one segment's GPU share.
///
/// The share is held as a quantum index `q ∈ [1, n_units − 1]`: Q heads for ATTN, 128-row
/// blocks for FFN. Both ends stay split — a segment is never fully on one device while
/// partition is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegGeom {
    pub kind: SegKind,
    /// Size of the split axis: `n_heads_q` (ATTN) or `ffn_hidden` (FFN).
    pub total: usize,
}

impl SegGeom {
    pub fn attn(n_heads_q: usize) -> Self {
        Self {
            kind: SegKind::Attn,
            total: n_heads_q,
        }
    }

    pub fn ffn(ffn_hidden: usize) -> Self {
        Self {
            kind: SegKind::Ffn,
            total: ffn_hidden,
        }
    }

    /// Elements of the split axis in one quantum.
    pub fn unit(&self) -> usize {
        match self.kind {
            SegKind::Attn => 1,
            SegKind::Ffn => FFN_ROW_QUANTUM,
        }
    }

    /// Number of whole quanta along the split axis.
    pub fn n_units(&self) -> usize {
        self.total / self.unit()
    }

    /// Whether this axis can be split at all (needs two quanta).
    pub fn splittable(&self) -> bool {
        self.n_units() >= 2
    }

    /// Largest quantum index (smallest CPU share).
    pub fn q_max(&self) -> usize {
        self.n_units() - 1
    }

    /// GPU share of quantum index `q`.
    pub fn share(&self, q: usize) -> f32 {
        (q * self.unit()) as f32 / self.total as f32
    }

    /// Split point on the axis for quantum index `q` (heads for ATTN, rows for FFN).
    pub fn split(&self, q: usize) -> usize {
        q * self.unit()
    }

    /// Quantize a GPU share. `None` = partition off (`r >= GPU_ONLY_THRESHOLD`, or an axis too
    /// short to leave both devices a quantum).
    ///
    /// ATTN rounds to the nearest head; FFN aligns down to 128 rows (ticket 021 §D2).
    pub fn quantize(&self, r: f32) -> Option<usize> {
        if r >= GPU_ONLY_THRESHOLD || !self.splittable() {
            return None;
        }
        let r = r.max(0.0);
        let q = match self.kind {
            SegKind::Attn => (r * self.total as f32).round() as usize,
            SegKind::Ffn => (r * self.total as f32) as usize / self.unit(),
        };
        Some(q.clamp(1, self.q_max()))
    }
}

/// Quantized ATTN split: Q heads on the GPU, `None` = partition off.
pub fn quantize_attn(r: f32, n_heads_q: usize) -> Option<usize> {
    SegGeom::attn(n_heads_q).quantize(r)
}

/// Quantized FFN split: gate/up rows on the GPU, `None` = partition off.
pub fn quantize_ffn(r: f32, ffn_hidden: usize) -> Option<usize> {
    let g = SegGeom::ffn(ffn_hidden);
    g.quantize(r).map(|q| g.split(q))
}

/// Controller knobs (`--tp-eta`, `--tp-contention-ratio`).
#[derive(Clone, Copy, Debug)]
pub struct TpConfig {
    pub eta: f32,
    pub contention_ratio: f32,
    /// Periodic Lookup probe. Always on in the engine; tests turn it off to show the probe is
    /// the only way back after contention clears.
    pub probe: bool,
}

impl Default for TpConfig {
    fn default() -> Self {
        Self {
            eta: DEFAULT_ETA,
            contention_ratio: DEFAULT_CONTENTION_RATIO,
            probe: true,
        }
    }
}

/// One token's measurement of one segment, taken at the share [`SegState::applied`] returned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obs {
    /// The GPU was still busy when the CPU finished its share.
    pub waited: bool,
    /// CPU share start → end.
    pub t_cpu: f32,
    /// GPU share start → done-flag. Known when `waited`, and always on a serial token.
    pub t_gpu: Option<f32>,
}

impl Obs {
    /// Segment time: the slower device when known, else the CPU (the GPU finished first).
    fn t_seg(&self) -> f32 {
        match self.t_gpu {
            Some(g) if self.waited => g.max(self.t_cpu),
            _ => self.t_cpu,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Startup,
    Descent,
    Lookup,
}

/// What an observation did, for the controller's bookkeeping and the logs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegEvent {
    None,
    /// Descent settled (`forced` = hit the 16-token cap).
    Converged {
        forced: bool,
    },
    /// Lookup: slower than 1.2× best and the CPU is the slow side → reduce threads.
    ContentionCpu,
    /// Lookup: slower than 1.2× best, GPU side → back to Descent.
    ContentionGpu,
    /// Lookup probe found GPU slack → back to Descent.
    ProbeHit,
}

/// Controller state of one `(layer, segment)`.
#[derive(Clone, Debug)]
pub struct SegState {
    pub geom: SegGeom,
    pub phase: Phase,
    /// Continuous GPU share; `q` is its quantization.
    r: f32,
    q: usize,
    k_probe: u32,
    /// Last observations in Descent: `(q, waited, t_seg, t_cpu)`, oldest first.
    hist: Vec<(usize, bool, f32, f32)>,
    /// Descent bracket: highest quantum seen without a wait, lowest seen with one.
    lo: Option<usize>,
    hi: Option<usize>,
    descent_tokens: u32,
    /// Converged quantum index held in Lookup.
    pub q_opt: usize,
    pub t_best: f32,
    pub t_cpu_best: f32,
    slow_streak: usize,
    since_probe: u32,
    probing: bool,
    /// The first probe token had no wait; the next one confirms (or not) the hit.
    probe_confirm: bool,
}

impl SegState {
    pub fn new(geom: SegGeom, r0: f32) -> Self {
        let q = geom.quantize(r0).unwrap_or(geom.q_max());
        Self {
            geom,
            phase: Phase::Startup,
            r: geom.share(q),
            q,
            k_probe: 0,
            hist: Vec::with_capacity(STREAK),
            lo: None,
            hi: None,
            descent_tokens: 0,
            q_opt: q,
            t_best: 0.0,
            t_cpu_best: 0.0,
            slow_streak: 0,
            since_probe: 0,
            probing: false,
            probe_confirm: false,
        }
    }

    /// Quantum index to run this token (the probe quantum on a Lookup probe token).
    pub fn applied(&self) -> usize {
        self.q
    }

    /// GPU share this token runs at.
    pub fn applied_share(&self) -> f32 {
        self.geom.share(self.q)
    }

    /// Run this token serially (GPU share alone, then CPU share) so both times are exact.
    pub fn serial(&self) -> bool {
        self.phase == Phase::Startup
    }

    /// Re-measure from scratch next token (after a thread-count change).
    pub fn restart(&mut self) {
        self.phase = Phase::Startup;
        self.probing = false;
        self.probe_confirm = false;
        self.slow_streak = 0;
        self.hist.clear();
    }

    fn set_r(&mut self, r: f32) {
        let lo = self.geom.share(1);
        let hi = self.geom.share(self.geom.q_max());
        self.r = r.clamp(lo, hi);
        self.q = self
            .geom
            .quantize(self.r)
            .expect("r < 1 by the clamp above");
    }

    fn set_q(&mut self, q: usize) {
        self.q = q.clamp(1, self.geom.q_max());
        self.r = self.geom.share(self.q);
    }

    fn enter_descent(&mut self, k_probe: u32) {
        self.phase = Phase::Descent;
        self.k_probe = k_probe;
        self.hist.clear();
        self.lo = None;
        self.hi = None;
        self.descent_tokens = 0;
        self.probing = false;
        self.probe_confirm = false;
        self.slow_streak = 0;
    }

    fn enter_lookup(&mut self, q_opt: usize, t_best: f32, t_cpu_best: f32) {
        self.phase = Phase::Lookup;
        self.q_opt = q_opt;
        self.t_best = t_best;
        self.t_cpu_best = t_cpu_best;
        self.slow_streak = 0;
        self.since_probe = 0;
        self.probing = false;
        self.probe_confirm = false;
        self.set_q(q_opt);
    }

    /// Feed the observation of the token that just ran at [`Self::applied`].
    pub fn observe(&mut self, obs: Obs, cfg: &TpConfig) -> SegEvent {
        match self.phase {
            Phase::Startup => {
                // Serial token: both times exact. Balance point from the two measured speeds.
                let r0 = self.applied_share();
                let t_gpu = obs.t_gpu.unwrap_or(obs.t_cpu).max(f32::EPSILON);
                let t_cpu = obs.t_cpu.max(f32::EPSILON);
                let v_g = r0 / t_gpu;
                let v_c = (1.0 - r0) / t_cpu;
                self.set_r(v_g / (v_g + v_c));
                self.enter_descent(0);
                SegEvent::None
            }
            Phase::Descent => self.observe_descent(obs, cfg),
            Phase::Lookup => self.observe_lookup(obs, cfg),
        }
    }

    fn observe_descent(&mut self, obs: Obs, cfg: &TpConfig) -> SegEvent {
        let q_ran = self.q;
        self.descent_tokens += 1;
        if self.hist.len() == STREAK {
            self.hist.remove(0);
        }
        self.hist.push((q_ran, obs.waited, obs.t_seg(), obs.t_cpu));
        // The newest observation wins when noise contradicts the bracket.
        if obs.waited {
            self.hi = Some(q_ran);
            self.lo = self.lo.filter(|&lo| lo < q_ran);
        } else {
            self.lo = Some(q_ran);
            self.hi = self.hi.filter(|&hi| hi > q_ran);
        }

        // Convergence is judged on what was just observed, before moving again.
        if let Some(q_opt) = self.converged_at() {
            let t_best = self
                .hist
                .iter()
                .filter(|h| h.0 == q_opt)
                .map(|h| h.2)
                .fold(f32::INFINITY, f32::min);
            let t_best = if t_best.is_finite() {
                t_best
            } else {
                obs.t_seg()
            };
            self.enter_lookup(q_opt, t_best, self.t_cpu_median());
            return SegEvent::Converged { forced: false };
        }
        if self.descent_tokens >= MAX_DESCENT_TOKENS {
            let t_cpu_best = self.t_cpu_median();
            self.enter_lookup(q_ran, obs.t_seg(), t_cpu_best);
            return SegEvent::Converged { forced: true };
        }

        match obs.t_gpu {
            Some(t_gpu) if obs.waited => {
                // Exact gradient. GPU slower (t_gpu > t_cpu) → r falls → the GPU gets less.
                let t_cpu = obs.t_cpu;
                let step = cfg.eta * (t_cpu - t_gpu) / (t_cpu + t_gpu).max(f32::EPSILON);
                self.set_r(self.r + step);
                self.k_probe = 0;
            }
            _ => {
                // GPU had slack of unknown size → probe upward, doubling.
                let quanta = (1usize << self.k_probe).min(MAX_PROBE_QUANTA);
                let q_next = self.q + quanta;
                self.set_q(q_next);
                self.k_probe += 1;
            }
        }
        // Stay inside the bracket: below the lowest waited quantum, at or above the highest
        // unwaited one.
        let mut q = self.q;
        if let Some(hi) = self.hi {
            q = q.min(hi.saturating_sub(1).max(1));
        }
        if let Some(lo) = self.lo {
            q = q.max(lo);
        }
        if q != self.q {
            self.set_q(q);
        }
        SegEvent::None
    }

    /// Median CPU time of the recent Descent tokens: the contention baseline. One sample is
    /// too noisy on sub-millisecond segments.
    fn t_cpu_median(&self) -> f32 {
        let mut v: Vec<f32> = self.hist.iter().map(|h| h.3).collect();
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    }

    /// Converged quantum, if any: the bracket closed (the lower end did not make the CPU wait,
    /// the upper end did — then the faster of the two), or the same quantum three times in a
    /// row (a share held at a clamp).
    fn converged_at(&self) -> Option<usize> {
        if let (Some(lo), Some(hi)) = (self.lo, self.hi)
            && hi == lo + 1
        {
            let best = |q: usize| {
                self.hist
                    .iter()
                    .filter(|h| h.0 == q)
                    .map(|h| h.2)
                    .fold(f32::INFINITY, f32::min)
            };
            return Some(if best(hi) < best(lo) { hi } else { lo });
        }
        if self.hist.len() == STREAK && self.hist.iter().all(|h| h.0 == self.hist[0].0) {
            return Some(self.hist[0].0);
        }
        None
    }

    fn observe_lookup(&mut self, obs: Obs, cfg: &TpConfig) -> SegEvent {
        if self.probing {
            if !obs.waited && !self.probe_confirm {
                // No wait at one more quantum: run the probe once more before believing it.
                self.probe_confirm = true;
                return SegEvent::None;
            }
            self.probing = false;
            self.since_probe = 0;
            if !obs.waited {
                // Two probe tokens in a row without a wait: slack appeared. The probe token is
                // Descent's first observation, so the next step climbs by 2.
                self.probe_confirm = false;
                self.enter_descent(1);
                self.observe_descent(obs, cfg);
                return SegEvent::ProbeHit;
            }
            self.probe_confirm = false;
            self.set_q(self.q_opt);
            return SegEvent::None;
        }

        let ratio = cfg.contention_ratio;
        if obs.t_seg() > ratio * self.t_best {
            self.slow_streak += 1;
        } else {
            self.slow_streak = 0;
        }
        if self.slow_streak >= STREAK {
            self.slow_streak = 0;
            if obs.t_cpu > ratio * self.t_cpu_best {
                self.restart();
                return SegEvent::ContentionCpu;
            }
            // This observation is Descent's first: the gradient step is taken now.
            self.enter_descent(0);
            self.observe_descent(obs, cfg);
            return SegEvent::ContentionGpu;
        }

        // The next token is the probe when it completes a period (7 held + 1 probe).
        self.since_probe += 1;
        if cfg.probe && self.since_probe + 1 >= PROBE_PERIOD && self.q_opt < self.geom.q_max() {
            self.probing = true;
            self.set_q(self.q_opt + 1);
        }
        SegEvent::None
    }
}

/// Per-token aggregate for the `--tbt-log` fields and the end-of-run line (§D6).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TpTokenStats {
    /// Mean ATTN GPU share over layers.
    pub r_attn: f32,
    /// Mean FFN GPU share over layers.
    pub r_ffn: f32,
    /// Segments in Lookup.
    pub lookup: usize,
    /// Contention events so far.
    pub contention: u64,
}

/// Per-token stats of the running partition arm, in decode order, for the `--tbt-log` writer
/// (one entry per plan-executed decode token). Process-global like the partition trace
/// counters: one decode session per process.
static TELEMETRY: std::sync::Mutex<Vec<TpTokenStats>> = std::sync::Mutex::new(Vec::new());

pub fn telemetry_push(stats: TpTokenStats) {
    TELEMETRY
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(stats);
}

/// Drain the recorded per-token stats (empty when partition was off).
pub fn telemetry_take() -> Vec<TpTokenStats> {
    std::mem::take(&mut *TELEMETRY.lock().unwrap_or_else(|e| e.into_inner()))
}

/// All 56 segments of a model plus the thread-count knob they share.
pub struct TpController {
    pub cfg: TpConfig,
    /// `false` = static split: observations are recorded but shares never move.
    pub adaptive: bool,
    /// `[layer][0 = ATTN, 1 = FFN]`.
    pub segs: Vec<[SegState; 2]>,
    /// Total CPU threads (SpinPool workers + dispatch thread) the CPU shares may use.
    pub threads: usize,
    pub threads_initial: usize,
    pub tokens: u32,
    /// Token at which the last segment converged for the first time (Startup + Descent done
    /// everywhere). Later Lookup exits (probe hits, contention) do not move it.
    pub converged_tok: Option<u32>,
    /// Per `(layer, seg)`: converged at least once.
    converged_once: Vec<[bool; 2]>,
    pub forced: u64,
    pub contention: u64,
    pub probe_hits: u64,
    reduce_pending: bool,
}

impl TpController {
    pub fn new(
        n_layers: usize,
        n_heads_q: usize,
        ffn_hidden: usize,
        r0: f32,
        threads: usize,
        adaptive: bool,
        cfg: TpConfig,
    ) -> Self {
        let segs = (0..n_layers)
            .map(|_| {
                [
                    SegState::new(SegGeom::attn(n_heads_q), r0),
                    SegState::new(SegGeom::ffn(ffn_hidden), r0),
                ]
            })
            .collect();
        Self {
            cfg,
            adaptive,
            segs,
            threads,
            threads_initial: threads,
            tokens: 0,
            converged_tok: None,
            converged_once: vec![[false; 2]; n_layers],
            forced: 0,
            contention: 0,
            probe_hits: 0,
            reduce_pending: false,
        }
    }

    /// Whether this segment runs serially this token.
    pub fn serial(&self, layer: usize, seg: usize) -> bool {
        self.adaptive && self.segs[layer][seg].serial()
    }

    /// Quantum index this segment runs at this token.
    pub fn applied(&self, layer: usize, seg: usize) -> usize {
        self.segs[layer][seg].applied()
    }

    /// Feed one segment's observation (static mode ignores it).
    pub fn observe(&mut self, layer: usize, seg: usize, obs: Obs) {
        if !self.adaptive {
            return;
        }
        match self.segs[layer][seg].observe(obs, &self.cfg) {
            SegEvent::Converged { forced } => {
                if forced {
                    self.forced += 1;
                }
                self.converged_once[layer][seg] = true;
            }
            SegEvent::ContentionCpu => {
                self.contention += 1;
                self.reduce_pending = true;
            }
            SegEvent::ContentionGpu => self.contention += 1,
            SegEvent::ProbeHit => self.probe_hits += 1,
            SegEvent::None => {}
        }
    }

    /// Close a token: apply at most one thread step, update the counters. Returns the new thread
    /// count when it changed, for the caller to push into the CPU pool.
    pub fn end_token(&mut self) -> Option<usize> {
        self.tokens += 1;
        if self.converged_tok.is_none()
            && self.adaptive
            && self.converged_once.iter().flatten().all(|&c| c)
        {
            self.converged_tok = Some(self.tokens);
        }
        if !std::mem::take(&mut self.reduce_pending) {
            return None;
        }
        let next = self.threads.saturating_sub(THREAD_STEP).max(MIN_THREADS);
        // The reduction invalidates every segment's CPU baseline, not only the one that tripped.
        for s in self.segs.iter_mut().flatten() {
            s.restart();
        }
        if next == self.threads {
            return None;
        }
        self.threads = next;
        Some(next)
    }

    pub fn stats(&self) -> TpTokenStats {
        let n = self.segs.len().max(1) as f32;
        TpTokenStats {
            r_attn: self.segs.iter().map(|s| s[0].applied_share()).sum::<f32>() / n,
            r_ffn: self.segs.iter().map(|s| s[1].applied_share()).sum::<f32>() / n,
            lookup: self
                .segs
                .iter()
                .flatten()
                .filter(|s| s.phase == Phase::Lookup)
                .count(),
            contention: self.contention,
        }
    }

    /// `[tp] layers=.. segs=..` end-of-run line (§D6).
    pub fn summary_line(&self) -> String {
        let st = self.stats();
        let conv = self
            .converged_tok
            .map_or_else(|| "none".to_string(), |t| t.to_string());
        format!(
            "[tp] layers={} segs={} startup_tok={} converged_tok={} forced={} contention={} \
             probe_hits={} threads={}→{} r_attn={:.3} r_ffn={:.3}",
            self.segs.len(),
            self.segs.len() * 2,
            if self.adaptive { 1 } else { 0 },
            conv,
            self.forced,
            self.contention,
            self.probe_hits,
            self.threads_initial,
            self.threads,
            st.r_attn,
            st.r_ffn,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N_HEADS: usize = 12;
    const FFN: usize = 8960;

    /// Asymmetric observation of a segment with GPU speed `v_g` and CPU speed `v_c` (share per
    /// ms): the GPU time is visible only when the CPU had to wait, or on a serial token.
    fn sim(seg: &SegState, v_g: f32, v_c: f32) -> Obs {
        let r = seg.applied_share();
        let t_gpu = r / v_g;
        let t_cpu = (1.0 - r) / v_c;
        let waited = t_gpu > t_cpu;
        Obs {
            waited,
            t_cpu,
            t_gpu: (waited || seg.serial()).then_some(t_gpu),
        }
    }

    fn r_star(v_g: f32, v_c: f32) -> f32 {
        v_g / (v_g + v_c)
    }

    /// Distance in quanta between the applied share and `r`.
    fn quanta_off(seg: &SegState, r: f32) -> f32 {
        let g = seg.geom;
        (seg.applied_share() - r).abs() * g.total as f32 / g.unit() as f32
    }

    const PAIRS: [(f32, f32); 6] = [
        (1.0, 4.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 2.0),
        (2.0, 1.0),
        (4.0, 1.0),
    ];

    #[test]
    fn tp_controller_converges() {
        let cfg = TpConfig::default();
        for geom in [SegGeom::attn(N_HEADS), SegGeom::ffn(FFN)] {
            for &(v_g, v_c) in &PAIRS {
                for r0 in [0.25f32, 0.75] {
                    let target = r_star(v_g, v_c);
                    let mut seg = SegState::new(geom, r0);
                    let mut shares = vec![seg.applied_share()];
                    let mut steps = 0;
                    while seg.phase != Phase::Lookup {
                        let obs = sim(&seg, v_g, v_c);
                        seg.observe(obs, &cfg);
                        shares.push(seg.applied_share());
                        steps += 1;
                        assert!(
                            steps <= 10,
                            "{geom:?} v=({v_g},{v_c}) r0={r0}: no Lookup after 10 steps, shares {shares:?}"
                        );
                    }
                    assert!(
                        quanta_off(&seg, target) <= 1.0 + 1e-3,
                        "{geom:?} v=({v_g},{v_c}) r0={r0}: settled at {} vs r*={target}",
                        seg.applied_share()
                    );
                    // Sign check: a slower GPU never gets more work on the way down from 0.75.
                    if v_g < v_c && r0 == 0.75 {
                        let band = |s: f32| {
                            (s - target).abs() * geom.total as f32 / geom.unit() as f32 <= 1.0
                        };
                        let first_in_band = shares.iter().position(|&s| band(s)).unwrap();
                        for w in shares[..=first_in_band].windows(2) {
                            assert!(
                                w[1] <= w[0],
                                "{geom:?} v=({v_g},{v_c}): share rose {} → {} before reaching r*",
                                w[0],
                                w[1]
                            );
                        }
                    }
                }
            }
        }
    }

    /// Drive a segment to Lookup under fixed speeds.
    fn settled(geom: SegGeom, v_g: f32, v_c: f32, cfg: &TpConfig) -> SegState {
        let mut seg = SegState::new(geom, 0.5);
        for _ in 0..20 {
            if seg.phase == Phase::Lookup {
                return seg;
            }
            let obs = sim(&seg, v_g, v_c);
            seg.observe(obs, cfg);
        }
        panic!("did not settle");
    }

    #[test]
    fn tp_controller_contention_paths() {
        let cfg = TpConfig {
            probe: false,
            ..TpConfig::default()
        };
        // A segment in Lookup whose converged token had the CPU wait on the GPU (t_gpu visible).
        let lookup = || {
            let mut c = TpController::new(2, N_HEADS, FFN, 0.5, 8, true, cfg);
            for layer in 0..2 {
                let s = &mut c.segs[layer][1];
                s.phase = Phase::Lookup;
                s.q_opt = s.applied();
                s.t_best = 10.0;
                s.t_cpu_best = 9.0;
            }
            c
        };
        let obs = |t_cpu: f32, t_gpu: f32| Obs {
            waited: t_gpu > t_cpu,
            t_cpu,
            t_gpu: (t_gpu > t_cpu).then_some(t_gpu),
        };

        // t_gpu alone 1.3× → Descent, threads unchanged.
        let mut c = lookup();
        for _ in 0..3 {
            c.observe(0, 1, obs(9.0, 13.0));
            assert_eq!(c.end_token(), None);
        }
        assert_eq!(c.segs[0][1].phase, Phase::Descent);
        assert_eq!(c.threads, 8);
        assert_eq!(c.contention, 1);

        // t_cpu alone 1.3× → one worker step down, every segment back to Startup.
        let mut c = lookup();
        for t in 0..3 {
            c.observe(0, 1, obs(13.0, 10.0));
            let changed = c.end_token();
            assert_eq!(changed, (t == 2).then_some(6));
        }
        assert_eq!(c.threads, 6);
        assert!(c.segs.iter().flatten().all(|s| s.phase == Phase::Startup));
        assert!(c.serial(1, 0), "reduction restarts every segment serially");

        // 1.1× is inside the band → nothing.
        let mut c = lookup();
        for _ in 0..10 {
            c.observe(0, 1, obs(9.0, 11.0));
            c.end_token();
        }
        assert_eq!(c.segs[0][1].phase, Phase::Lookup);
        assert_eq!(c.contention, 0);

        // Fewer than three consecutive slow tokens → nothing.
        let mut c = lookup();
        for i in 0..12 {
            let slow = i % 3 != 2;
            c.observe(0, 1, if slow { obs(9.0, 13.0) } else { obs(9.0, 10.0) });
            c.end_token();
        }
        assert_eq!(c.segs[0][1].phase, Phase::Lookup);
        assert_eq!(c.contention, 0);
        assert_eq!(c.threads, 8);
    }

    #[test]
    fn tp_controller_release_probe() {
        let (v_g, v_c) = (1.0f32, 2.0f32);
        let geom = SegGeom::ffn(FFN);

        // Without the probe: after the GPU doubles its speed, the share never moves.
        let no_probe = TpConfig {
            probe: false,
            ..TpConfig::default()
        };
        let mut seg = settled(geom, v_g, v_c, &no_probe);
        let held = seg.applied();
        for _ in 0..64 {
            let obs = sim(&seg, 2.0 * v_g, v_c);
            seg.observe(obs, &no_probe);
            assert_eq!(seg.applied(), held, "no probe → no way back");
            assert_eq!(seg.phase, Phase::Lookup);
        }

        // With the probe: Descent within 9 tokens (7 held + 2 probe), then the new r* within 10
        // steps.
        let cfg = TpConfig::default();
        let mut seg = settled(geom, v_g, v_c, &cfg);
        let mut tokens = 0;
        while seg.phase == Phase::Lookup {
            let obs = sim(&seg, 2.0 * v_g, v_c);
            seg.observe(obs, &cfg);
            tokens += 1;
            assert!(tokens <= 9, "probe did not fire within 9 tokens");
        }
        assert_eq!(seg.phase, Phase::Descent);
        let target = r_star(2.0 * v_g, v_c);
        let mut steps = 0;
        while seg.phase != Phase::Lookup {
            let obs = sim(&seg, 2.0 * v_g, v_c);
            seg.observe(obs, &cfg);
            steps += 1;
            assert!(steps <= 10, "no Lookup within 10 steps after the probe hit");
        }
        assert!(
            quanta_off(&seg, target) <= 1.0 + 1e-3,
            "settled at {} vs new r*={target}",
            seg.applied_share()
        );

        // Speeds unchanged: every probe comes back to the held share, no Descent.
        let mut seg = settled(geom, v_g, v_c, &cfg);
        let held = seg.q_opt;
        for _ in 0..64 {
            let obs = sim(&seg, v_g, v_c);
            seg.observe(obs, &cfg);
            assert_eq!(
                seg.phase,
                Phase::Lookup,
                "a probe without slack left Lookup"
            );
            assert!(seg.applied() == held || seg.applied() == held + 1);
        }
    }

    /// The S25 trace pattern (ticket 021 criterion 6): at the boundary quantum the CPU waits on
    /// most tokens but not all. Descent must still settle without the cap, and a lone unwaited
    /// probe token must not restart Descent.
    #[test]
    fn tp_controller_noisy_boundary() {
        let cfg = TpConfig::default();
        let geom = SegGeom::ffn(FFN);
        // Clean regime: waited above q_b, unwaited below. At q_b every third token is unwaited.
        let q_b = 51usize;
        let mut flips = 0u32;
        let mut obs_at = |seg: &SegState| {
            let q = seg.applied();
            let t_cpu = 1.35 - 0.02 * (q as f32 - 50.0);
            let waited = if q == q_b {
                flips += 1;
                flips % 3 != 0
            } else {
                q > q_b
            };
            let t_gpu = t_cpu + if waited { 0.1 } else { -0.1 };
            Obs {
                waited,
                t_cpu,
                t_gpu: (waited || seg.serial()).then_some(t_gpu),
            }
        };
        for r0 in [0.75f32, 0.5] {
            let mut seg = SegState::new(geom, r0);
            let mut steps = 0;
            let mut forced = false;
            while seg.phase != Phase::Lookup {
                let o = obs_at(&seg);
                if let SegEvent::Converged { forced: f } = seg.observe(o, &cfg) {
                    forced = f;
                }
                steps += 1;
            }
            assert!(!forced, "r0={r0}: hit the 16-token cap");
            assert!(steps <= 12, "r0={r0}: {steps} tokens to settle");
            assert!(
                seg.q_opt.abs_diff(q_b) <= 1,
                "r0={r0}: settled at q={} vs boundary {q_b}",
                seg.q_opt
            );
            let mut hits = 0;
            for _ in 0..1024 {
                let o = obs_at(&seg);
                if seg.observe(o, &cfg) == SegEvent::ProbeHit {
                    hits += 1;
                }
            }
            assert_eq!(hits, 0, "r0={r0}: lone unwaited probes restarted Descent");
        }
    }

    #[test]
    fn tp_quantization_bounds() {
        for r in [0.0f32, 1e-6, 0.04, 0.5, 0.96] {
            let h_g = quantize_attn(r, N_HEADS).expect("split");
            assert!((1..=11).contains(&h_g), "r={r} → h_g={h_g}");
            let s = quantize_ffn(r, FFN).expect("split");
            assert!((128..=8832).contains(&s) && s % 128 == 0, "r={r} → s={s}");
        }
        assert_eq!(quantize_attn(1.0, N_HEADS), None);
        assert_eq!(quantize_ffn(1.0, FFN), None);
        assert_eq!(quantize_attn(0.5, N_HEADS), Some(6));
        assert_eq!(quantize_ffn(0.5, FFN), Some(4480));
    }
}
