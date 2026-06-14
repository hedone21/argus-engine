//! `EvalLlRunCtx` — `bin/generate.rs::main()`에서 eval_ll 모드 진입 시점에
//! 살아있는 모든 outer-scope state를 packaging.
//!
//! 4-A의 `BatchRunCtx` 패턴과 동일. `run_eval_ll` 종료 시 process 종료
//! (`return Ok(())`).

use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::backend::Backend;
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::inference::skip_config::SkipConfig;
use crate::kv::cache_manager::CacheManager;
use crate::kv::kv_cache::KVCache;
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::qcf_types::ImportanceFormula;
use crate::session::cli::Args;
use crate::weight::SwapAlgorithm;

pub struct EvalLlRunCtx {
    pub args: Args,

    // 백엔드 / 메모리 / 모델
    pub backend: Arc<dyn Backend>,
    pub memory: Arc<dyn Memory>,
    pub cpu_backend_arc: Arc<dyn Backend>,
    pub gpu_backend_arc: Option<Arc<dyn Backend>>,
    pub model: TransformerModel,
    pub tokenizer: Tokenizer,

    // KV / 상태 객체
    pub kv_caches: Vec<KVCache>,
    pub cache_manager: CacheManager,
    pub score_accumulator: Option<AttentionScoreAccumulator>,
    pub skip_config: Option<SkipConfig>,

    // 파생 상태
    pub prompt: String,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub kv_type: crate::buffer::DType,
    pub actual_protected_prefix: usize,
    pub score_based_eviction: bool,

    // QCF / swap 관련 (qcf_runtime 호출 시 필요)
    pub swap_algorithm: SwapAlgorithm,
    pub importance_formula: ImportanceFormula,
    pub importance_compare: bool,
    pub swap_only_layers: Option<Vec<usize>>,
}
