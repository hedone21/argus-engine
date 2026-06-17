use clap::Args;

#[derive(Args, Debug, Clone)]
pub struct KvModeArgs {
    /// KV cache mode name. Resolved at runtime against the engine KV-mode registry
    /// (`KV_MODES`). Built-ins: standard (default), kivi, offload. An unknown name
    /// fails fast at the build funnel, listing the registered names.
    #[arg(long, default_value = "standard")]
    pub kv_mode: String,

    /// KIVI quantization bits (kv-mode=kivi 한정)
    #[arg(long = "kivi-bits", default_value_t = 2)]
    pub kivi_bits: u8,

    /// KIVI residual buffer length (kv-mode=kivi 한정)
    #[arg(long = "kivi-residual-len", default_value_t = 128)]
    pub kivi_residual_len: usize,

    /// Offload storage backend: raw | disk | mmap | tmpfs | ... (kv-mode=offload 한정)
    #[arg(long = "kv-offload-storage", default_value = "mmap")]
    pub kv_offload_storage: String,

    /// Directory for disk offload files (kv-mode=offload, storage=disk 한정).
    /// 빈 문자열은 system temp dir 사용.
    #[arg(long = "kv-offload-path", default_value = "")]
    pub kv_offload_path: String,

    /// Max adaptive prefetch depth for offload (kv-mode=offload 한정).
    #[arg(long = "kv-max-prefetch-depth", default_value_t = 128)]
    pub kv_max_prefetch_depth: usize,

    /// 선택적 KV read stage 이름. 미지정 = full read(현행). 빌트인: `quest`.
    /// 활성 format 이 SelectiveRead 미지원이면 stderr 1회 경고 후 full read 폴백.
    #[arg(long = "read-stage")]
    pub read_stage: Option<String>,
}

/// Manual default mirroring the clap `default_value`s (the `--kv-mode` string
/// defaults to `"standard"`, not `String::default()` = `""`). Used by any non-clap
/// construction path (`Args` does not derive `Default`, but this keeps the struct
/// self-consistent).
impl Default for KvModeArgs {
    fn default() -> Self {
        Self {
            kv_mode: "standard".to_string(),
            kivi_bits: 2,
            kivi_residual_len: 128,
            kv_offload_storage: "mmap".to_string(),
            kv_offload_path: String::new(),
            kv_max_prefetch_depth: 128,
            read_stage: None,
        }
    }
}
