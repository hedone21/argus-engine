use clap::{Args, ValueEnum};

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvMode {
    #[default]
    Standard,
    Kivi,
    Offload,
}

#[derive(Args, Debug, Clone, Default)]
pub struct KvModeArgs {
    /// KV cache mode (default: standard)
    #[arg(long, value_enum, default_value_t = KvMode::Standard)]
    pub kv_mode: KvMode,

    /// KIVI quantization bits (kv-mode=kivi 한정)
    #[arg(long = "kv-kivi-bits", default_value_t = 2)]
    pub kv_kivi_bits: u8,

    /// KIVI residual buffer length (kv-mode=kivi 한정)
    #[arg(long = "kv-kivi-residual-len", default_value_t = 128)]
    pub kv_kivi_residual_len: usize,

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
