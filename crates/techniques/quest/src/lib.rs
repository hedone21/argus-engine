//! Quest read technique crate — query-aware selective KV read.
//!
//! Quest(ICML'24, arXiv 2406.10774): page 별 K 의 채널 min/max 를 유지해, query 와의 **상한 내적**
//! `Σ_d max(q_d·min_d, q_d·max_d)` 로 각 page 의 attention 기여 상한을 추정하고 top-k page 만
//! attention 에서 읽는다. read 축(4번째 plugin 표면 = [`KVReadStage`])의 첫 빌트인.
//!
//! **plan-returning, self-mutating 아님**: plugin 은 page index 목록만 [`KVReadPlan`]
//! 으로 방출하고, 실제 선택 읽기(`attention_into_selected`)는 엔진/format 이 수행한다.
//!
//! **page 메타는 stage 가 보유**: page 별 K min/max 를 `&self` 안 [`QuestState`](Mutex)
//! 에 두고, `read_plan(ctx)` 호출마다 [`StageCtx::tensor`]`(Key)` 로 incremental 갱신한다. 코어
//! (`KVCache`)는 page 를 모른다(무수정). eviction 후 ctx 의 `current_pos` 가 메타 토큰 수보다 작아지면
//! (축소 감지) 전체 재계산.
//!
//! **query 신호원 (현 StageCtx 표면에 실재하는 TensorKind 만 사용)**:
//! Quest 정본은 *현 step 의 query* 이나 [`StageCtx`] 에 query 텐서가 없다. 대신:
//! - `tensor(QueryStats)` 가용(score-active e2e seam) → running **mean** 을 query 근사로 써서
//!   상한 내적 점수(Quest 의 query-aware 선택에 가장 근접).
//! - 미가용(production decode — `read_plan` 이 QueryStats=None 공급, standard_format.rs) → query 가
//!   전혀 없으므로 **query-agnostic proxy**: page 별 K channel magnitude(`max(|min|,|max|)`) 합으로
//!   점수. 이는 Quest 의 *근사의 근사* 이며 query-무관이라 page 활성도 추정의 한계가 있다(보고 명시).
//!
//! 두 경우 모두 **page 0(attention sink) + 최근 window page 는 무조건 포함**(Quest 논문 — 최신
//! 토큰은 항상 attend, sink 보존). 선택 page 가 전체이면 `None` 반환(full read 와 동일 → plan 불요).

use std::sync::Mutex;

use linkme::distributed_slice;
use technique_api::{
    KV_READ_STAGES, KVReadPlan, KVReadStage, KVReadStageReg, ReadGranularity, ReadStageParams,
    StageCtx, TensorKind,
};

/// 한 page 의 채널별 K min/max (per kv_head). `min[kv_head*head_dim + d]` / `max[...]`.
#[derive(Clone)]
struct PageMeta {
    /// `n_kv_heads * head_dim` flat min.
    min: Vec<f32>,
    /// `n_kv_heads * head_dim` flat max.
    max: Vec<f32>,
}

/// Quest 의 incremental page 메타 상태(D2O Mutex 패턴 동형).
struct QuestState {
    /// page index → 채널 min/max.
    pages: Vec<PageMeta>,
    /// 메타가 반영한 토큰 수(= 마지막 갱신 시점의 `current_pos`). eviction(축소) 감지에 사용.
    covered: usize,
    /// 형상 검증용(변하면 전체 재계산).
    n_kv_heads: usize,
    head_dim: usize,
}

impl QuestState {
    fn empty() -> Self {
        Self {
            pages: Vec::new(),
            covered: 0,
            n_kv_heads: 0,
            head_dim: 0,
        }
    }
}

/// Quest read stage — query-aware selective page read.
struct Quest {
    page_size: usize,
    /// 전체 page 의 1/`top_k_ratio_denom` 를 선택(최소 [`MIN_PAGES`]).
    top_k_ratio_denom: usize,
    state: Mutex<QuestState>,
}

/// top-k page 하한(sink + 약간의 context 는 항상 확보).
const MIN_PAGES: usize = 4;
/// 무조건 포함하는 최근 window page 수(최신 토큰은 항상 attend).
const RECENT_WINDOW_PAGES: usize = 1;

impl Quest {
    fn new(params: ReadStageParams) -> Self {
        Self {
            page_size: (params.page_size as usize).max(1),
            top_k_ratio_denom: (params.top_k_ratio_denom as usize).max(1),
            state: Mutex::new(QuestState::empty()),
        }
    }

    /// (a) page 메타 갱신. ctx 의 `tensor(Key)` 로 새로 쓰인 토큰을 읽어 page min/max 를 갱신한다.
    /// eviction(축소) 또는 형상 변화 감지 시 전체 재계산(stage 자기 책임 — 코어 무수정).
    fn refresh_meta(&self, ctx: &dyn StageCtx, st: &mut QuestState) {
        let cur = ctx.current_pos();
        let n_kv_heads = ctx.n_kv_heads().max(1);
        let head_dim = ctx.head_dim();
        let row = n_kv_heads * head_dim;

        // 축소(eviction compaction) 또는 형상 변화 → 메타 stale → 전체 재구축.
        if cur < st.covered || st.n_kv_heads != n_kv_heads || st.head_dim != head_dim {
            st.pages.clear();
            st.covered = 0;
            st.n_kv_heads = n_kv_heads;
            st.head_dim = head_dim;
        }

        let Some(key) = ctx.tensor(TensorKind::Key) else {
            // Key 미가용(value-unaware 엔진) — 메타 갱신 불가. covered 만 동기화(빈 메타).
            st.covered = cur;
            return;
        };

        let n_pages = cur.div_ceil(self.page_size);
        if st.pages.len() < n_pages {
            st.pages.resize(
                n_pages,
                PageMeta {
                    min: vec![f32::INFINITY; row],
                    max: vec![f32::NEG_INFINITY; row],
                },
            );
        }

        // covered..cur 의 새 토큰만 incremental 반영(전체 재구축 시 covered=0 → 전체 스캔).
        let mut buf = vec![0.0f32; head_dim];
        for pos in st.covered..cur {
            let page = pos / self.page_size;
            let pm = &mut st.pages[page];
            for h in 0..n_kv_heads {
                key.read_row(pos, h, &mut buf);
                let base = h * head_dim;
                for (d, &v) in buf.iter().enumerate() {
                    if v < pm.min[base + d] {
                        pm.min[base + d] = v;
                    }
                    if v > pm.max[base + d] {
                        pm.max[base + d] = v;
                    }
                }
            }
        }
        st.covered = cur;
    }

    /// (b+c) page 별 점수 산출. QueryStats 가용 시 상한 내적, 미가용 시 K magnitude proxy.
    /// 반환 `score[page]`(클수록 중요). page 메타가 빈(Key 미가용) 경우 전부 0.
    fn page_scores(&self, ctx: &dyn StageCtx, st: &QuestState, n_pages: usize) -> Vec<f32> {
        let n_kv_heads = st.n_kv_heads.max(1);
        let head_dim = st.head_dim;
        let row = n_kv_heads * head_dim;
        let mut scores = vec![0.0f32; n_pages];

        if st.pages.is_empty() || row == 0 {
            return scores;
        }

        // query 신호: QueryStats running mean(가용 시) — kv_head 별 mean[head_dim].
        let qstats = ctx.tensor(TensorKind::QueryStats);
        let mut q_mean = vec![0.0f32; head_dim];

        for (p, pm) in st.pages.iter().take(n_pages).enumerate() {
            let mut s = 0.0f32;
            for h in 0..n_kv_heads {
                let base = h * head_dim;
                if let Some(qs) = qstats {
                    // row 0 = mean (stage_registry QueryStatsHandle 계약).
                    qs.read_row(0, h, &mut q_mean);
                    // Quest 상한 내적: Σ_d max(q_d·min_d, q_d·max_d).
                    for (d, &q) in q_mean.iter().enumerate() {
                        let lo = q * pm.min[base + d];
                        let hi = q * pm.max[base + d];
                        s += lo.max(hi);
                    }
                } else {
                    // query-agnostic proxy: page 의 채널 magnitude 합(활성도 추정 — 한계는 모듈 doc).
                    for d in 0..head_dim {
                        s += pm.min[base + d].abs().max(pm.max[base + d].abs());
                    }
                }
            }
            scores[p] = s;
        }
        scores
    }

    /// (c+d) top-k page 선택 + sink(page 0) + 최근 window page 무조건 포함 → ascending select.
    /// 선택이 전체이면 `None`(full read 동일).
    fn select_pages(&self, scores: &[f32], n_pages: usize) -> Option<Vec<usize>> {
        if n_pages == 0 {
            return None;
        }
        let top_k = (n_pages / self.top_k_ratio_denom)
            .max(MIN_PAGES)
            .min(n_pages);
        if top_k >= n_pages {
            return None; // 전체 선택 = full read.
        }

        // 점수 desc 정렬로 top_k page 후보(결정론: 동점은 page index asc tie-break).
        let mut order: Vec<usize> = (0..n_pages).collect();
        order.sort_unstable_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let mut selected = vec![false; n_pages];
        for &p in order.iter().take(top_k) {
            selected[p] = true;
        }
        // sink: page 0 무조건.
        selected[0] = true;
        // 최근 window page 무조건.
        for sel in selected
            .iter_mut()
            .take(n_pages)
            .skip(n_pages.saturating_sub(RECENT_WINDOW_PAGES))
        {
            *sel = true;
        }

        let select: Vec<usize> = (0..n_pages).filter(|&p| selected[p]).collect();
        if select.len() >= n_pages {
            None // 무조건 포함으로 전체가 되면 plan 불요.
        } else {
            Some(select)
        }
    }
}

impl KVReadStage for Quest {
    fn name(&self) -> &str {
        "quest"
    }

    fn read_plan(&self, ctx: &dyn StageCtx) -> Option<KVReadPlan> {
        let cur = ctx.current_pos();
        let n_pages = cur.div_ceil(self.page_size);
        // page 가 top_k 하한 이하면 selection 의미 없음 → full read.
        if n_pages <= MIN_PAGES {
            // 메타는 미래 step 을 위해 갱신해 두되 plan 은 None.
            let mut st = self.state.lock().unwrap();
            self.refresh_meta(ctx, &mut st);
            return None;
        }

        let mut st = self.state.lock().unwrap();
        self.refresh_meta(ctx, &mut st);
        let scores = self.page_scores(ctx, &st, n_pages);
        let select = self.select_pages(&scores, n_pages)?;
        Some(KVReadPlan {
            granularity: ReadGranularity::Page {
                page_size: self.page_size as u32,
            },
            select,
        })
    }
}

/// 등록 — 엔진은 `find_read_stage("quest")` 로 이 항목을 찾는다(`KVCacheStageReg` 거울).
#[distributed_slice(KV_READ_STAGES)]
static QUEST: KVReadStageReg = KVReadStageReg {
    name: "quest",
    make: |p: ReadStageParams| Box::new(Quest::new(p)),
};

#[cfg(test)]
mod tests {
    use super::*;
    use technique_api::{TensorHandle, TensorShape, find_read_stage};

    /// 합성 K — `read_row(pos, head, out)` 가 page/head 결정적 값을 반환.
    /// k(pos, head, d) = base(pos) + d. base 는 pos 마다 다르게 두어 page min/max 검증.
    struct MockKey {
        current_pos: usize,
        head_dim: usize,
        n_kv_heads: usize,
        // pos 별 base 값(없으면 pos 자체).
        bases: Vec<f32>,
    }
    impl TensorHandle for MockKey {
        fn shape(&self) -> TensorShape {
            TensorShape {
                rows: self.current_pos,
                cols: self.head_dim,
                per_head: true,
            }
        }
        fn dtype(&self) -> technique_api::TensorDtype {
            technique_api::TensorDtype::F32
        }
        fn read_row(&self, row: usize, _kv_head: usize, out: &mut [f32]) {
            let base = self.bases.get(row).copied().unwrap_or(row as f32);
            for (d, o) in out.iter_mut().enumerate() {
                *o = base + d as f32;
            }
        }
    }

    struct MockCtx {
        cur: usize,
        head_dim: usize,
        n_kv_heads: usize,
        key: Option<MockKey>,
    }
    impl StageCtx for MockCtx {
        fn current_pos(&self) -> usize {
            self.cur
        }
        fn target_len(&self) -> usize {
            self.cur
        }
        fn layer_idx(&self) -> usize {
            0
        }
        fn importance(&self) -> Option<&[f32]> {
            None
        }
        fn n_kv_heads(&self) -> usize {
            self.n_kv_heads
        }
        fn head_dim(&self) -> usize {
            self.head_dim
        }
        fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle> {
            match kind {
                TensorKind::Key => self.key.as_ref().map(|k| k as &dyn TensorHandle),
                _ => None,
            }
        }
    }

    fn params(page_size: u32) -> ReadStageParams {
        ReadStageParams {
            page_size,
            top_k_ratio_denom: 4,
        }
    }

    #[test]
    fn registers_into_slice() {
        let reg = find_read_stage("quest").expect("quest 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "quest");
    }

    /// page 메타 갱신 정확성: page 별 min/max 가 그 page 의 토큰 base 범위와 일치.
    #[test]
    fn page_meta_min_max_correct() {
        let head_dim = 2;
        let n_kv_heads = 1;
        // page_size=2, pos 0..4 → page0={0,1}, page1={2,3}. base = [10, 5, 8, 20].
        let q = Quest::new(params(2));
        let ctx = MockCtx {
            cur: 4,
            head_dim,
            n_kv_heads,
            key: Some(MockKey {
                current_pos: 4,
                head_dim,
                n_kv_heads,
                bases: vec![10.0, 5.0, 8.0, 20.0],
            }),
        };
        let mut st = q.state.lock().unwrap();
        q.refresh_meta(&ctx, &mut st);
        assert_eq!(st.pages.len(), 2);
        assert_eq!(st.covered, 4);
        // page0 토큰 base {10,5}: k(pos,d)=base+d → min over {10,5}+d, max over {10,5}+d.
        // d0: min(10,5)=5, max=10. d1: min(11,6)=6, max=11.
        assert_eq!(st.pages[0].min, vec![5.0, 6.0]);
        assert_eq!(st.pages[0].max, vec![10.0, 11.0]);
        // page1 base {8,20}: d0 min=8 max=20, d1 min=9 max=21.
        assert_eq!(st.pages[1].min, vec![8.0, 9.0]);
        assert_eq!(st.pages[1].max, vec![20.0, 21.0]);
    }

    /// incremental 갱신: 두 번째 호출에서 covered..cur 만 추가 반영(누적 정확).
    #[test]
    fn incremental_extends_meta() {
        let head_dim = 1;
        let q = Quest::new(params(2));
        // 1차: pos 0..2 (page0). base [3, 7].
        let mut ctx = MockCtx {
            cur: 2,
            head_dim,
            n_kv_heads: 1,
            key: Some(MockKey {
                current_pos: 2,
                head_dim,
                n_kv_heads: 1,
                bases: vec![3.0, 7.0],
            }),
        };
        {
            let mut st = q.state.lock().unwrap();
            q.refresh_meta(&ctx, &mut st);
            assert_eq!(st.pages[0].min, vec![3.0]);
            assert_eq!(st.pages[0].max, vec![7.0]);
        }
        // 2차: pos 0..4 (page0 동일 + page1 새). base [3,7,1,9].
        ctx.cur = 4;
        ctx.key = Some(MockKey {
            current_pos: 4,
            head_dim,
            n_kv_heads: 1,
            bases: vec![3.0, 7.0, 1.0, 9.0],
        });
        {
            let mut st = q.state.lock().unwrap();
            q.refresh_meta(&ctx, &mut st);
            // page0 불변(min3 max7), page1 새(min1 max9).
            assert_eq!(st.pages[0].min, vec![3.0]);
            assert_eq!(st.pages[0].max, vec![7.0]);
            assert_eq!(st.pages[1].min, vec![1.0]);
            assert_eq!(st.pages[1].max, vec![9.0]);
            assert_eq!(st.covered, 4);
        }
    }

    /// eviction 후 재구축: cur 가 covered 보다 작아지면 전체 재계산(stale 메타 제거).
    #[test]
    fn eviction_triggers_full_rebuild() {
        let head_dim = 1;
        let q = Quest::new(params(2));
        // 1차: pos 0..6, base [0,1,2,3,4,5].
        let mut ctx = MockCtx {
            cur: 6,
            head_dim,
            n_kv_heads: 1,
            key: Some(MockKey {
                current_pos: 6,
                head_dim,
                n_kv_heads: 1,
                bases: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            }),
        };
        {
            let mut st = q.state.lock().unwrap();
            q.refresh_meta(&ctx, &mut st);
            assert_eq!(st.pages.len(), 3);
            assert_eq!(st.covered, 6);
        }
        // eviction: cur 2 로 축소, base 가 compaction 으로 재배치됨 [100, 200].
        ctx.cur = 2;
        ctx.key = Some(MockKey {
            current_pos: 2,
            head_dim,
            n_kv_heads: 1,
            bases: vec![100.0, 200.0],
        });
        {
            let mut st = q.state.lock().unwrap();
            q.refresh_meta(&ctx, &mut st);
            // 전체 재구축: page 1개, covered=2, 새 base 반영.
            assert_eq!(st.pages.len(), 1);
            assert_eq!(st.covered, 2);
            assert_eq!(st.pages[0].min, vec![100.0]);
            assert_eq!(st.pages[0].max, vec![200.0]);
        }
    }

    /// top-k 선택 결정론: 동일 점수 입력에 항상 같은 select.
    #[test]
    fn select_deterministic() {
        let q = Quest::new(params(2));
        // 8 page, denom=4 → top_k = max(2, 4)=4. scores 임의.
        let scores = vec![1.0, 8.0, 3.0, 2.0, 9.0, 0.5, 7.0, 4.0];
        let s1 = q.select_pages(&scores, 8);
        let s2 = q.select_pages(&scores, 8);
        assert_eq!(s1, s2, "결정론적이어야 함");
        let sel = s1.expect("부분 선택");
        // top4 점수 page = {4(9),1(8),6(7),7(4)} + sink(0) + recent(7) → {0,1,4,6,7} ascending.
        assert_eq!(sel, vec![0, 1, 4, 6, 7]);
        // ascending 보장.
        assert!(sel.windows(2).all(|w| w[0] < w[1]));
    }

    /// sink(page 0) + 최근 window page 는 점수가 낮아도 무조건 포함.
    #[test]
    fn sink_and_recent_always_included() {
        let q = Quest::new(params(2));
        // page 0(sink), 마지막 page 점수를 0 으로 둬도 포함되어야.
        let scores = vec![0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 0.0];
        let sel = q.select_pages(&scores, 8).expect("부분 선택");
        assert!(sel.contains(&0), "sink page 0 포함");
        assert!(sel.contains(&7), "최근 window page 포함");
    }

    /// 전체 page 가 top_k 이하면 None(full read).
    #[test]
    fn small_cache_returns_none() {
        let q = Quest::new(params(16));
        let ctx = MockCtx {
            cur: 10, // 1 page < MIN_PAGES → None.
            head_dim: 4,
            n_kv_heads: 1,
            key: None,
        };
        assert!(q.read_plan(&ctx).is_none());
    }

    /// read_plan: 충분히 긴 캐시 + query-agnostic proxy 경로에서 부분 select Page plan 반환.
    #[test]
    fn read_plan_returns_page_plan_for_long_cache() {
        let head_dim = 2;
        let q = Quest::new(params(2));
        // pos 0..16 → 8 page. base 를 다양하게.
        let bases: Vec<f32> = (0..16).map(|i| (i * 3 % 7) as f32).collect();
        let ctx = MockCtx {
            cur: 16,
            head_dim,
            n_kv_heads: 1,
            key: Some(MockKey {
                current_pos: 16,
                head_dim,
                n_kv_heads: 1,
                bases,
            }),
        };
        let plan = q.read_plan(&ctx).expect("부분 select plan");
        match plan.granularity {
            ReadGranularity::Page { page_size } => assert_eq!(page_size, 2),
            ReadGranularity::Token => panic!("Quest 정본은 Page granularity"),
        }
        assert!(!plan.select.is_empty() && plan.select.len() < 8);
        assert!(plan.select.windows(2).all(|w| w[0] < w[1]));
        assert!(plan.select.contains(&0), "sink");
        assert!(plan.select.contains(&7), "recent");
    }
}
