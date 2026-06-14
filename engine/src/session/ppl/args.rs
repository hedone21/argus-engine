//! `PplRunCtx` — `bin/generate.rs::main()`에서 PPL 모드 진입 시점에
//! 살아있는 모든 outer-scope state. 4-A/4-B 패턴과 동일.

use std::sync::Arc;
use std::time::Instant;

use tokenizers::Tokenizer;

use crate::backend::Backend;
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::inference::skip_config::SkipConfig;
use crate::kv::cache_manager::CacheManager;
use crate::kv::kv_cache::KVCache;
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::qcf::ImportanceTable;
use crate::session::cli::Args;
use crate::weight::decider::SwapDecision;

pub struct PplRunCtx {
    pub args: Args,

    // 백엔드 / 메모리 / 모델
    pub backend: Arc<dyn Backend>,
    pub memory: Arc<dyn Memory>,
    pub model: TransformerModel,
    pub tokenizer: Tokenizer,

    // KV / 상태 객체
    pub kv_caches: Vec<KVCache>,
    pub cache_manager: CacheManager,
    pub score_accumulator: Option<AttentionScoreAccumulator>,
    pub skip_config: Option<SkipConfig>,

    // 파생 상태
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub actual_protected_prefix: usize,
    pub score_based_eviction: bool,

    // QCF prelude state (main()에서 ppl 진입 직전에 계산)
    pub qcf_warmup_importance: Option<ImportanceTable>,
    pub qcf_swap_decision: Option<SwapDecision>,
    pub qcf_workflow_start: Instant,

    // 다른 outer state (ppl_main에서 직접 참조)
    pub auto_eviction: bool,
    pub swap_algorithm: crate::weight::SwapAlgorithm,
}

/// Return value from `run_ppl` for use by the caller (e.g. `--qcf-dump`).
pub struct PplResult {
    pub ppl: f64,
    pub avg_nll: f64,
    pub n_eval_tokens: usize,
    pub wall_time_s: f64,
}
