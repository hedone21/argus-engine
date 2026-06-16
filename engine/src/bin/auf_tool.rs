/// auf-tool — AUF (Argus Unified Format) 자산 빌드/검사/수정/검증 CLI 도구.
///
/// ENG-ALG-223 §3.12.17 구현. WSWAP-3.7B-CLI.
///
/// 서브커맨드:
///   build  — GGUF + tokenizer.json → AUF (사전 SOA + Q/K permute 변환)
///   info   — AUF 헤더/섹션/메타 출력
///   strip  — dead variant 섹션 제거 (selective, 백업 옵션)
///   verify — magic/section/required capability/offset 무결성 전수 검증
use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};

use argus_engine::auf::dtype_convert::{build_dtype_candidates, convert_tensor_dtype};
use argus_engine::auf::stripper::strip;
use argus_engine::auf::tensor_index::{
    LAYER_IDX_CROSS, TensorDType, TensorEntry, TensorIndex, TensorKind,
};
use argus_engine::auf::{
    AufError, AufMeta, AufTokenizer, AufWriter, BackendTag, SECTION_STRIPPABLE,
    TAG_WEIGHTS_ADRENO_SOA, TAG_WEIGHTS_CPU_AOS, TAG_WEIGHTS_CUDA_AOS, TOKENIZER_KIND_BPE,
    compute_source_hash, open, q4_0_aos_to_adreno_soa,
};

// ---------------------------------------------------------------------------
// CLI 정의 (clap derive)
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "auf-tool",
    about = "AUF (Argus Unified Format) 자산 빌드/검사/수정/검증 도구",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// GGUF + tokenizer.json → AUF 파일 빌드
    Build(BuildArgs),
    /// AUF 헤더/섹션/메타 정보 출력
    Info(InfoArgs),
    /// dead variant 섹션 제거 (selective strip)
    Strip(StripArgs),
    /// AUF 무결성 전수 검증
    Verify(VerifyArgs),
}

// ── build ──────────────────────────────────────────────────────────────────

#[derive(Args)]
struct BuildArgs {
    /// 원본 GGUF 파일 경로
    #[arg(long, value_name = "FILE")]
    input: PathBuf,

    /// tokenizer.json 경로
    #[arg(long, value_name = "FILE")]
    tokenizer: PathBuf,

    /// (W-AUF-1 C5) HuggingFace `tokenizer_config.json` 경로. AUF TOKENIZER
    /// section의 `bos_id`/`eos_id` 슬롯을 채우는 데 사용한다.
    ///
    /// 미지정 시 `tokenizer.json`와 같은 디렉토리에서 `tokenizer_config.json`을
    /// 자동 탐색한다. 발견되면 거기서 `bos_token`/`eos_token` 또는
    /// `bos_token_id`/`eos_token_id`를 읽고, 못 찾으면 기존 키워드 fallback을
    /// 사용한다. CLI override (`--bos-token-id`/`--eos-token-id`)는 본 파일보다
    /// 우선한다.
    #[arg(long, value_name = "FILE")]
    tokenizer_config: Option<PathBuf>,

    /// (W-AUF-1 C5) bos_token_id 명시 override. tokenizer_config.json 및 기존
    /// 키워드 fallback보다 우선한다.
    #[arg(long, value_name = "ID")]
    bos_token_id: Option<u32>,

    /// (W-AUF-1 C5) eos_token_id 명시 override. tokenizer_config.json 및 기존
    /// 키워드 fallback보다 우선한다.
    #[arg(long, value_name = "ID")]
    eos_token_id: Option<u32>,

    /// 출력 AUF 파일 경로 (atomic rename으로 부분 write 방지)
    #[arg(long, value_name = "FILE")]
    output: PathBuf,

    /// 포함할 backend variant (comma-separated 또는 "all").
    /// 예: "adreno_soa,cpu_aos" 또는 "WEIGHTS_ADRENO_SOA,WEIGHTS_CPU_AOS" 또는 "all"
    #[arg(long, value_name = "VARIANTS")]
    variants: String,

    /// 헤더 created_by 문자열 (최대 32B UTF-8). 기본: "argus_engine auf-tool v<VERSION>"
    #[arg(long, value_name = "STRING")]
    created_by: Option<String>,

    /// lm_head Q4_0 사전 변환 (v0.1.1, Sprint G-1).
    ///
    /// - `auto` (기본값): GGUF lm_head dtype != Q4_0이면 quantize, 이미 Q4_0이면
    ///   AOS bytes를 entry로 그대로 동봉.
    /// - `on`: 강제 quantize (이미 Q4_0이면 AOS bytes 동봉).
    /// - `off`: lm_head Q4_0 entry 미포함 (v0.1.0 byte-level 호환 출력,
    ///   capability bit 2 = 0, format_patch = 0).
    #[arg(long, value_name = "MODE", default_value = "auto")]
    include_lm_head: String,

    /// AUF v0.2 multi-dtype variant — 동봉할 dtype 목록 (Sprint C, INV-139).
    ///
    /// comma-separated. 예: `q4_0,f16` / `Q4_0,F16`.
    /// 미지정 시 source dtype을 따른다 (v0.1.x single-dtype 동작).
    /// 2개 이상 지정 시 capability_optional bit 3 자동 set + format_minor = 2.
    /// 지원 dtype: `q4_0`, `f16`, `f32` (현재). bf16/q4_1/q8_0/u8은 dequant→requant
    /// 파이프라인이 미구현이므로 reject.
    #[arg(long, value_name = "LIST")]
    dtypes: Option<String>,

    /// AUF v0.2 multi-dtype variant — META.default_dtype 명시 (Sprint C, INV-138).
    ///
    /// `--dtypes`에 포함된 값이어야 한다. 미지정 시 `--dtypes`의 첫 번째 값.
    /// 단일 dtype 모드(`--dtypes` 미지정 또는 1개)에서는 무시.
    #[arg(long, value_name = "DTYPE")]
    default_dtype: Option<String>,

    /// 진행 로그 출력 억제
    #[arg(long)]
    quiet: bool,
}

/// `--include-lm-head` 모드.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IncludeLmHeadMode {
    /// lm_head Q4_0 entry 미포함 (v0.1.0 호환).
    Off,
    /// 강제 포함 (이미 Q4_0이면 그대로 동봉, F16/F32 → Q4_0 quantize).
    On,
    /// auto: GGUF lm_head dtype에 따라. 비-Q4_0 → quantize, Q4_0 → 그대로 동봉.
    Auto,
}

impl IncludeLmHeadMode {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(IncludeLmHeadMode::Auto),
            "on" | "true" | "yes" | "1" => Ok(IncludeLmHeadMode::On),
            "off" | "false" | "no" | "0" => Ok(IncludeLmHeadMode::Off),
            other => bail!(
                "Invalid --include-lm-head '{}'. Valid: auto, on, off",
                other
            ),
        }
    }
}

// ── info ──────────────────────────────────────────────────────────────────

#[derive(Args)]
struct InfoArgs {
    /// AUF 파일 경로
    #[arg(value_name = "FILE")]
    file: PathBuf,
}

// ── strip ─────────────────────────────────────────────────────────────────

#[derive(Args)]
struct StripArgs {
    /// AUF 파일 경로 (in-place 수정)
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// 유지할 section tag (comma-separated).
    /// required section (META/TOKENIZER/TENSOR_INDEX)은 자동 보존.
    /// 예: "WEIGHTS_ADRENO_SOA" 또는 "adreno_soa"
    #[arg(long, value_name = "TAGS")]
    keep: String,

    /// 백업 파일 생성 생략 (기본: <file>.bak 자동 생성)
    #[arg(long)]
    no_backup: bool,
}

// ── verify ────────────────────────────────────────────────────────────────

#[derive(Args)]
struct VerifyArgs {
    /// AUF 파일 경로
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// source GGUF 경로 (제공 시 source_hash 재계산하여 비교)
    #[arg(long, value_name = "FILE")]
    source: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Build(args) => cmd_build(args),
        Commands::Info(args) => cmd_info(args),
        Commands::Strip(args) => cmd_strip(args),
        Commands::Verify(args) => cmd_verify(args),
    };
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// build 커맨드
// ---------------------------------------------------------------------------

/// variant 문자열 → section tag 목록으로 변환.
///
/// 입력: "all", "adreno_soa,cpu_aos", "WEIGHTS_ADRENO_SOA,WEIGHTS_CPU_AOS" 등 혼합 허용.
fn parse_variants(variants_str: &str) -> Result<Vec<&'static str>> {
    let s = variants_str.trim();
    if s.eq_ignore_ascii_case("all") {
        return Ok(vec![
            TAG_WEIGHTS_ADRENO_SOA,
            TAG_WEIGHTS_CUDA_AOS,
            TAG_WEIGHTS_CPU_AOS,
        ]);
    }
    let mut tags = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        let tag = match part.to_ascii_uppercase().as_str() {
            "WEIGHTS_ADRENO_SOA" | "ADRENO_SOA" | "ADRENO" => TAG_WEIGHTS_ADRENO_SOA,
            "WEIGHTS_CUDA_AOS" | "CUDA_AOS" | "CUDA" => TAG_WEIGHTS_CUDA_AOS,
            "WEIGHTS_CPU_AOS" | "CPU_AOS" | "CPU" => TAG_WEIGHTS_CPU_AOS,
            other => bail!(
                "Unknown variant '{}'. Valid: adreno_soa, cuda_aos, cpu_aos, all",
                other
            ),
        };
        if !tags.contains(&tag) {
            tags.push(tag);
        }
    }
    if tags.is_empty() {
        bail!("--variants must specify at least one variant");
    }
    Ok(tags)
}

/// dtype 문자열 → `TensorDType` 변환.
///
/// 대소문자 무관 + 일부 동의어 허용. 현재 변환 파이프라인이 지원하는 dtype에 한해 OK.
fn parse_single_dtype(s: &str) -> Result<TensorDType> {
    let upper = s.trim().to_ascii_uppercase();
    match upper.as_str() {
        "F32" => Ok(TensorDType::F32),
        "F16" => Ok(TensorDType::F16),
        "Q4_0" | "Q40" => Ok(TensorDType::Q4_0),
        "BF16" | "Q4_1" | "Q41" | "Q8_0" | "Q80" | "U8" => bail!(
            "dtype '{}' is not supported by AUF v0.2 multi-dtype writer (Sprint C). \
             Supported: f32, f16, q4_0",
            s
        ),
        other => bail!("Unknown dtype '{}'. Supported: f32, f16, q4_0", other),
    }
}

/// `TensorDType` → 표준 문자열 (META.default_dtype 직렬화에 사용).
fn dtype_to_meta_str(dt: TensorDType) -> &'static str {
    match dt {
        TensorDType::F32 => "F32",
        TensorDType::F16 => "F16",
        TensorDType::BF16 => "BF16",
        TensorDType::Q4_0 => "Q4_0",
        TensorDType::Q4_1 => "Q4_1",
        TensorDType::Q8_0 => "Q8_0",
        TensorDType::U8 => "U8",
    }
}

/// `--dtypes` comma-separated 문자열 파싱. 빈 문자열 / 빈 항목 거부.
///
/// 중복 제거 (안정 순서 유지) — 같은 dtype을 여러 번 지정해도 1개만 유지.
fn parse_dtypes(dtypes_str: &str) -> Result<Vec<TensorDType>> {
    let mut out: Vec<TensorDType> = Vec::new();
    for part in dtypes_str.split(',') {
        let p = part.trim();
        if p.is_empty() {
            bail!("--dtypes contains empty entry: '{}'", dtypes_str);
        }
        let dt = parse_single_dtype(p)?;
        if !out.contains(&dt) {
            out.push(dt);
        }
    }
    if out.is_empty() {
        bail!("--dtypes must contain at least one dtype");
    }
    Ok(out)
}

/// tokenizer_config.json 또는 generation_config.json에서 bos/eos id를
/// 추출한다 (W-AUF-1 C5).
///
/// HuggingFace 표준 필드 변형 처리:
/// - `bos_token_id` / `eos_token_id` (integer): 그대로 사용
/// - `bos_token` / `eos_token` (string): `vocab_lookup`을 통해 token → id 변환
/// - `bos_token` / `eos_token` (object `{ "content": "...", "id": ... }`):
///   content/id 둘 다 시도
fn parse_special_ids_from_tokenizer_config(
    path: &std::path::Path,
    vocab_lookup: &dyn Fn(&str) -> Option<i32>,
) -> Result<(Option<i32>, Option<i32>)> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|e| {
        anyhow!(
            "Cannot open tokenizer_config.json {}: {}",
            path.display(),
            e
        )
    })?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)
        .map_err(|e| anyhow!("Cannot read tokenizer_config.json: {}", e))?;
    let json: serde_json::Value = serde_json::from_str(&buf)
        .map_err(|e| anyhow!("Invalid tokenizer_config.json JSON: {}", e))?;
    Ok(resolve_special_ids_from_value(&json, vocab_lookup))
}

/// Pure helper extracted for unit tests: `parse_special_ids_from_tokenizer_config`의
/// 파일 I/O 분리 버전. JSON value와 vocab_lookup만 받아 (bos, eos)를 반환한다.
fn resolve_special_ids_from_value(
    json: &serde_json::Value,
    vocab_lookup: &dyn Fn(&str) -> Option<i32>,
) -> (Option<i32>, Option<i32>) {
    let resolve = |field_id: &str, field_str: &str| -> Option<i32> {
        if let Some(v) = json.get(field_id).and_then(|v| v.as_i64()) {
            return Some(v as i32);
        }
        let token_node = json.get(field_str)?;
        if let Some(s) = token_node.as_str() {
            return vocab_lookup(s);
        }
        if let Some(obj) = token_node.as_object() {
            if let Some(id) = obj.get("id").and_then(|v| v.as_i64()) {
                return Some(id as i32);
            }
            if let Some(content) = obj.get("content").and_then(|v| v.as_str()) {
                return vocab_lookup(content);
            }
        }
        None
    };
    let bos = resolve("bos_token_id", "bos_token");
    let eos = resolve("eos_token_id", "eos_token");
    (bos, eos)
}

/// tokenizer.json에서 AufTokenizer를 구성한다.
///
/// tokenizers crate의 Tokenizer::from_file은 내부 vocab/merges 직접 접근이
/// 어려우므로 JSON을 직접 파싱하여 필드를 추출한다.
///
/// `bos_id` / `eos_id` 결정 우선순위 (W-AUF-1 C5):
/// 1. `bos_override` / `eos_override` — CLI 명시 (`--bos-token-id` /
///    `--eos-token-id`).
/// 2. `tokenizer_config_path`로 지정된 (또는 sibling 자동 탐색된)
///    `tokenizer_config.json`의 `bos_token_id` / `eos_token_id` 정수 또는
///    `bos_token` / `eos_token` 문자열 (후자는 vocab/added_tokens lookup).
/// 3. 기존 키워드 fallback (`begin_of_text`, `eos`, `<|eot_id|>` 등).
fn load_tokenizer_from_json(
    path: &std::path::Path,
    tokenizer_config_path: Option<&std::path::Path>,
    bos_override: Option<u32>,
    eos_override: Option<u32>,
) -> Result<AufTokenizer> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|e| anyhow!("Cannot open tokenizer.json {}: {}", path.display(), e))?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)
        .map_err(|e| anyhow!("Cannot read tokenizer.json: {}", e))?;

    let json: serde_json::Value =
        serde_json::from_str(&buf).map_err(|e| anyhow!("Invalid tokenizer.json JSON: {}", e))?;

    let model = json
        .get("model")
        .ok_or_else(|| anyhow!("tokenizer.json: missing 'model' field"))?;

    // vocab: { "token_string": id, ... }
    let vocab_obj = model
        .get("vocab")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("tokenizer.json: missing model.vocab"))?;

    let vocab_size = vocab_obj.len();
    // id → bytes 매핑 (id 순서로 정렬)
    let mut id_to_token: Vec<(u64, Vec<u8>)> = vocab_obj
        .iter()
        .map(|(k, v)| {
            let id = v.as_u64().unwrap_or(0);
            (id, k.as_bytes().to_vec())
        })
        .collect();
    id_to_token.sort_by_key(|(id, _)| *id);

    // added_tokens에서 ID > vocab_size 인 토큰 추가 (Llama special tokens)
    let mut max_id = id_to_token.last().map(|(id, _)| *id).unwrap_or(0);
    if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
        for tok in added_tokens {
            let id = tok.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
            let content = tok.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if id > max_id {
                // added_tokens에만 있는 token은 gap을 빈 토큰으로 채워야 함
                while max_id + 1 < id {
                    max_id += 1;
                    id_to_token.push((max_id, format!("<pad_{}>", max_id).into_bytes()));
                }
                id_to_token.push((id, content.as_bytes().to_vec()));
                max_id = id;
            }
        }
    }
    id_to_token.sort_by_key(|(id, _)| *id);
    let tokens: Vec<Vec<u8>> = id_to_token.into_iter().map(|(_, t)| t).collect();
    let vocab_size_final = tokens.len();

    // merges: ["Ġ t", "Ġ a", ...] 형태
    let merges: Vec<String> = model
        .get("merges")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_owned()))
                .collect()
        })
        .unwrap_or_default();

    // special token IDs (bos, eos 등)
    // tokenizer.json의 added_tokens에서 special=true인 것들을 키워드로 분류
    let find_special_id = |keywords: &[&str]| -> i32 {
        json.get("added_tokens")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                arr.iter().find(|tok| {
                    let content = tok.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    keywords.iter().any(|kw| content.contains(kw))
                })
            })
            .and_then(|tok| tok.get("id").and_then(|v| v.as_i64()))
            .map(|id| id as i32)
            .unwrap_or(-1)
    };

    // W-AUF-1 C5: tokenizer_config.json + CLI override 처리.
    //
    // 1. vocab_lookup: token string → id 매핑 (tokenizer.json vocab + added_tokens).
    let vocab_obj_ref = vocab_obj.clone();
    let added_tokens_arr = json
        .get("added_tokens")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let vocab_lookup = move |s: &str| -> Option<i32> {
        if let Some(id) = vocab_obj_ref.get(s).and_then(|v| v.as_i64()) {
            return Some(id as i32);
        }
        for tok in &added_tokens_arr {
            if tok.get("content").and_then(|v| v.as_str()) == Some(s)
                && let Some(id) = tok.get("id").and_then(|v| v.as_i64())
            {
                return Some(id as i32);
            }
        }
        None
    };

    // 2. tokenizer_config.json 자동 탐색 또는 명시 경로 사용.
    let auto_config_path = path.parent().map(|p| p.join("tokenizer_config.json"));
    let config_path_used = tokenizer_config_path
        .map(|p| p.to_path_buf())
        .or_else(|| auto_config_path.filter(|p| p.exists()));

    let (cfg_bos, cfg_eos) = if let Some(ref cfg) = config_path_used {
        match parse_special_ids_from_tokenizer_config(cfg, &vocab_lookup) {
            Ok(pair) => {
                eprintln!(
                    "[auf-tool] tokenizer_config.json: {} → bos={:?}, eos={:?}",
                    cfg.display(),
                    pair.0,
                    pair.1,
                );
                pair
            }
            Err(e) => {
                eprintln!(
                    "[auf-tool] Warning: failed to parse tokenizer_config.json ({}), using fallback",
                    e
                );
                (None, None)
            }
        }
    } else {
        (None, None)
    };

    // 3. 우선순위 적용: CLI override → tokenizer_config.json → 키워드 fallback.
    let bos_id = bos_override
        .map(|v| v as i32)
        .or(cfg_bos)
        .unwrap_or_else(|| find_special_id(&["begin_of_text", "bos", "<s>", "[BOS]"]));
    let eos_id = eos_override
        .map(|v| v as i32)
        .or(cfg_eos)
        .unwrap_or_else(|| find_special_id(&["end_of_text", "eos", "</s>", "<|eot_id|>", "[EOS]"]));
    let pad_id = find_special_id(&["pad", "[PAD]", "<pad>"]);
    let unk_id = find_special_id(&["unk", "<unk>", "[UNK]"]);

    // chat_template (선택)
    let chat_template = json
        .get("chat_template")
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned());

    eprintln!(
        "[auf-tool] Tokenizer: vocab={}, merges={}, bos={}, eos={}, chat_template={}",
        vocab_size_final,
        merges.len(),
        bos_id,
        eos_id,
        chat_template.is_some()
    );
    let _ = vocab_size; // suppress warning

    Ok(AufTokenizer {
        kind: TOKENIZER_KIND_BPE,
        tokens,
        merges,
        bos_id,
        eos_id,
        pad_id,
        unk_id,
        chat_template,
    })
}

/// GGUF에서 AufMeta를 구성한다.
fn build_meta_from_gguf(gguf: &argus_engine::models::loader::gguf::GgufFile) -> Result<AufMeta> {
    use argus_engine::models::loader::gguf::parse_model_config;

    let config =
        parse_model_config(gguf).map_err(|e| anyhow!("failed to parse ModelConfig: {}", e))?;

    // max_seq_len: ModelConfig에 없으므로 GGUF에서 직접 읽는다
    let arch_str = gguf.get_str("general.architecture").unwrap_or("llama");
    let max_seq_len = gguf
        .get_u32(&format!("{arch_str}.context_length"))
        .unwrap_or(2048);

    Ok(AufMeta {
        architecture: format!("{:?}", config.arch).to_lowercase(),
        n_layers: config.num_hidden_layers as u32,
        n_heads_q: config.num_attention_heads as u32,
        n_kv_heads: config.num_key_value_heads as u32,
        head_dim: config.head_dim as u32,
        hidden_dim: config.hidden_size as u32,
        ffn_dim: config.intermediate_size as u32,
        vocab_size: config.vocab_size as u32,
        max_seq_len,
        rope_theta: config.rope_theta,
        rotary_dim: config.head_dim as u32,
        rope_scaling: 1.0,
        rms_norm_epsilon: config.rms_norm_eps,
        default_dtype: None, // v0.2 multi-dtype에서만 설정 (Sprint C에서 확장)
    })
}

/// Q4_0 AOS 바이트에 64B 또는 128B 패딩(align)을 추가하여 반환한다.
///
/// CPU AOS: 64B align, CUDA AOS: 128B align.
/// 사전 Q/K permute가 적용된 bytes를 그대로 정렬 패딩만 추가한다.
fn q4_0_aos_with_align(blocks: &[u8], alignment: usize) -> Vec<u8> {
    let size = blocks.len();
    let padded = size.div_ceil(alignment) * alignment;
    let mut out = vec![0u8; padded];
    out[..size].copy_from_slice(blocks);
    out
}

/// build 커맨드: GGUF → AUF
fn cmd_build(args: BuildArgs) -> Result<()> {
    // 1) variant 파싱
    let variant_tags = parse_variants(&args.variants)?;
    let lm_head_mode = IncludeLmHeadMode::parse(&args.include_lm_head)?;
    let quiet = args.quiet;

    // 2) dtype 후보 + default_dtype 파싱 (Sprint C-F).
    //
    // - `--dtypes` 미지정 (None): single-dtype 모드 = source dtype 그대로 동봉. v0.1.x 호환.
    // - `--dtypes` 1개: capability bit 3 미설정 (single-dtype). dtype 변환은 강제되지만
    //   format은 v0.1.x 그대로.
    // - `--dtypes` 2개 이상: multi-dtype 모드 = capability bit 3 set + format_minor = 2.
    let candidate_dtypes: Option<Vec<TensorDType>> = match &args.dtypes {
        Some(s) => Some(parse_dtypes(s)?),
        None => None,
    };
    let multi_dtype_enabled = candidate_dtypes
        .as_ref()
        .map(|v| v.len() >= 2)
        .unwrap_or(false);

    // default_dtype 결정 + validation.
    let default_dtype: Option<TensorDType> = if let Some(cands) = &candidate_dtypes {
        let dd = match &args.default_dtype {
            Some(s) => parse_single_dtype(s)?,
            None => cands[0],
        };
        if !cands.contains(&dd) {
            bail!(
                "--default-dtype {:?} must be one of --dtypes {:?}",
                dd,
                cands
            );
        }
        Some(dd)
    } else {
        if args.default_dtype.is_some() {
            bail!("--default-dtype requires --dtypes to be specified");
        }
        None
    };

    if !quiet {
        eprintln!(
            "[auf-tool] Build: {} → {}",
            args.input.display(),
            args.output.display()
        );
        eprintln!("[auf-tool] Variants: {:?}", variant_tags);
        eprintln!("[auf-tool] include_lm_head: {:?}", lm_head_mode);
        if let Some(cands) = &candidate_dtypes {
            eprintln!(
                "[auf-tool] dtypes: {:?} (default={:?}, multi_dtype={})",
                cands, default_dtype, multi_dtype_enabled
            );
        }
    }

    // 3) source_hash 계산
    if !quiet {
        eprint!("[auf-tool] Computing source_hash...");
    }
    let (source_hash, source_size, source_mtime) = compute_source_hash(&args.input)
        .map_err(|e| anyhow!("failed to compute source_hash: {}", e))?;
    if !quiet {
        eprintln!(" done ({} bytes)", source_size);
    }

    // 4) GGUF 파싱
    if !quiet {
        eprint!("[auf-tool] Parsing GGUF...");
    }
    let gguf = argus_engine::models::loader::gguf::GgufFile::open(&args.input)
        .map_err(|e| anyhow!("failed to open GGUF file: {}", e))?;
    if !quiet {
        eprintln!(" {} tensors", gguf.tensors.len());
    }

    // 5) AufMeta 구성 — multi-dtype이면 default_dtype 필드를 채운다 (INV-138).
    let mut meta = build_meta_from_gguf(&gguf)
        .map_err(|e| anyhow!("failed to extract GGUF metadata: {}", e))?;
    if multi_dtype_enabled {
        meta.default_dtype = default_dtype.map(|d| dtype_to_meta_str(d).to_owned());
    }
    if !quiet {
        eprintln!(
            "[auf-tool] Meta: arch={}, layers={}, vocab={}{}",
            meta.architecture,
            meta.n_layers,
            meta.vocab_size,
            meta.default_dtype
                .as_deref()
                .map(|d| format!(", default_dtype={d}"))
                .unwrap_or_default()
        );
    }

    // 6) Tokenizer 구성 (W-AUF-1 C5: tokenizer_config.json + CLI override 통합)
    let tokenizer = load_tokenizer_from_json(
        &args.tokenizer,
        args.tokenizer_config.as_deref(),
        args.bos_token_id,
        args.eos_token_id,
    )?;

    // 7) ModelConfig (Q/K permute shape 결정용)
    let config = argus_engine::models::loader::gguf::parse_model_config(&gguf)
        .map_err(|e| anyhow!("failed to parse ModelConfig: {}", e))?;

    // 8) weight payload 생성 (각 variant 별)
    let mut writer = AufWriter::new(meta, tokenizer, source_hash, source_size, source_mtime);

    let created_by = args
        .created_by
        .unwrap_or_else(|| format!("argus_engine auf-tool v{}", env!("CARGO_PKG_VERSION")));
    writer = writer.with_created_by(&created_by);

    // Q4_0 tensor raw bytes 추출 (공통, variant마다 재사용)
    // 순서: layer 0..N의 wq, wk, wv, wo, w_gate, w_up, w_down 순
    // permute가 필요한 wq/wk에는 unpermute_qk_rows 적용
    // lm_head는 `lm_head_mode`에 따라 Q4_0 quantize 적용 (Sprint G-1).
    // multi-dtype 모드에서는 candidate dtype별로 변환된 sub-bytes도 함께 채워진다.
    if !quiet {
        eprint!("[auf-tool] Extracting weight tensors...");
    }
    let (tensor_blobs, lm_head_q4_0_present) = extract_weight_blobs(
        &gguf,
        &config,
        lm_head_mode,
        candidate_dtypes.as_deref(),
        quiet,
    )?;
    if !quiet {
        let total_dtype_entries: usize = tensor_blobs.iter().map(|b| b.dtype_bytes.len()).sum();
        eprintln!(
            " {} tensors, {} dtype entries extracted",
            tensor_blobs.len(),
            total_dtype_entries
        );
    }

    // capability_optional bit 2 설정 여부 결정 (Sprint G-1).
    // - lm_head Q4_0 entry가 실제로 추가/유지된 경우만 set.
    // - Off 모드 또는 tied 모델 (lm_head 없음)에서는 unset → v0.1.0 호환.
    writer = writer.with_lm_head_q4_0(lm_head_q4_0_present);
    if !quiet && lm_head_q4_0_present {
        eprintln!(
            "[auf-tool] LM_HEAD_PRECOMPUTED_Q4_0 capability bit 2 = 1 (format_patch=1, v0.1.1)"
        );
    }

    // capability_optional bit 3 설정 (Sprint C, INV-139). multi-dtype 모드에서만 set.
    writer = writer.with_multi_dtype(multi_dtype_enabled);
    if !quiet && multi_dtype_enabled {
        eprintln!("[auf-tool] MULTI_DTYPE_VARIANTS capability bit 3 = 1 (format_minor=2, v0.2)");
    }

    // TensorIndex 구성 (weights payload 추가 전에 offset 계산).
    // INV-138 정렬 키: default_dtype entry가 그룹 첫 번째에 오도록 안정 정렬.
    let tensor_index = build_tensor_index(&tensor_blobs, &variant_tags, default_dtype);
    if !quiet {
        eprintln!(
            "[auf-tool] TensorIndex: {} variants, {} entries",
            tensor_index.variant_tags.len(),
            tensor_index.entries.len()
        );
    }
    writer = writer.with_tensor_index(tensor_index);

    for &tag in &variant_tags {
        if !quiet {
            eprint!("[auf-tool] Building {}...", tag);
        }
        let payload = build_variant_payload(&tensor_blobs, tag)?;
        let size = payload.len();
        writer = writer.add_weights_section(tag, payload);
        if !quiet {
            eprintln!(" {} MB", size / (1024 * 1024));
        }
    }

    // 9) Atomic write
    if !quiet {
        eprint!("[auf-tool] Writing {}...", args.output.display());
    }
    writer
        .write_to_file(&args.output)
        .map_err(|e| anyhow!("failed to write AUF: {}", e))?;

    if !quiet {
        let file_size = std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0);
        eprintln!(
            " done ({:.2} GiB)",
            file_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        eprintln!("[auf-tool] Build complete: {}", args.output.display());
    }
    Ok(())
}

/// 한 tensor의 source-dtype-원본 정보 + dtype별 candidate bytes.
///
/// `shape_logical`은 outermost-first (논리적 순서).
/// `dtype_bytes`는 candidate dtype별 변환된 bytes (Sprint C-A 파이프라인).
///   v0.1.x single-dtype 모드에서는 1개. v0.2 multi-dtype 모드에서는 ≥1개.
#[derive(Debug, Clone)]
struct WeightBlob {
    name: String,
    shape_logical: Vec<u64>,
    /// candidate dtype과 그 bytes — TensorIndex entry 정렬과 무관하게 stable order.
    dtype_bytes: Vec<(TensorDType, Vec<u8>)>,
}

impl WeightBlob {
    /// 단일 dtype (v0.1.x 호환) blob 구성. dtype은 source dtype을 따른다.
    fn single(
        name: String,
        shape_logical: Vec<u64>,
        source_dtype: TensorDType,
        bytes: Vec<u8>,
    ) -> Self {
        WeightBlob {
            name,
            shape_logical,
            dtype_bytes: vec![(source_dtype, bytes)],
        }
    }
}

/// GGUF ggml_type 코드 (subset, lm_head 분기에서 사용).
///
/// `gguf.rs`의 private 상수와 동일하지만 binary crate에서 접근할 수 없으므로 로컬에 둔다.
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;

/// GGUF에서 lm_head를 식별하는 표준 이름 (untied / separate 모델).
const LM_HEAD_SEPARATE_NAME: &str = "output.weight";

/// GGUF에서 token embedding tensor 이름. tied embedding 모델 (Llama 3.2 1B/3B 등)에서
/// lm_head source로 재사용된다.
const LM_HEAD_TIED_SOURCE_NAME: &str = "token_embd.weight";

/// 선택된 lm_head source 정보 (`select_lm_head_source` 반환).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LmHeadSource {
    /// 실제 GGUF tensor 이름 (`output.weight` 또는 `token_embd.weight`).
    source_name: &'static str,
    /// `true`면 tied embedding 모델 (token_embd를 lm_head로 재사용).
    is_tied: bool,
}

/// GGUF에 존재하는 tensor를 기반으로 lm_head source를 선택한다.
///
/// 우선순위:
/// 1. `output.weight` 가 있으면 separate 모델 (Qwen, Llama 3 8B 등) → 그것을 사용.
/// 2. 없으면 `token_embd.weight` 를 tied source로 재사용 (Llama 3.2 1B/3B 등).
/// 3. 둘 다 없으면 `None` (entry 생성 불가).
///
/// 본 함수는 GgufFile 의존성 없이 boolean만 받아 결정하므로 단위 테스트가 용이하다.
fn select_lm_head_source(has_separate: bool, has_token_embd: bool) -> Option<LmHeadSource> {
    if has_separate {
        Some(LmHeadSource {
            source_name: LM_HEAD_SEPARATE_NAME,
            is_tied: false,
        })
    } else if has_token_embd {
        Some(LmHeadSource {
            source_name: LM_HEAD_TIED_SOURCE_NAME,
            is_tied: true,
        })
    } else {
        None
    }
}

/// `WeightBlob` 목록 추출 + lm_head Q4_0 사전 변환 (선택) + multi-dtype 변환.
///
/// GGUF는 innermost-first로 dims를 저장하므로, 여기서 reverse하여 logical order로 반환한다.
/// AUF TensorEntry.shape에는 logical order(outermost-first)로 저장하며,
/// reader(`secondary_mmap.rs`)에서 다시 reverse하여 GGUF order로 복원한다.
///
/// `lm_head_mode`에 따라 lm_head를 다음과 같이 처리한다:
/// - `Off`: separate model에서 `output.weight`이 존재하면 raw bytes를 그대로 추출
///   (기존 동작). tied model에서는 lm_head entry 미생성 (v0.1.0 byte-level 호환).
/// - `On`/`Auto`: lm_head source 선택:
///   - separate model: `output.weight` 사용. dtype 분기:
///     - 이미 Q4_0이면 raw bytes 그대로 동봉 (그리고 multi-dtype 모드면 candidate dtype
///       전부에 대해 추가 변환).
///     - F16/F32 → F32 dequantize → Q4_0 quantize → 18B/block bytes로 교체.
///   - tied model (Llama 3.2 1B 등): `token_embd.weight` 를 lm_head source로 재사용.
///
/// `candidate_dtypes`는 v0.2 multi-dtype 모드에서 동봉할 dtype 후보 목록. v0.1.x single-dtype
/// 모드에서는 None — source dtype 1개만 동봉. lm_head는 multi-dtype 모드에서도 동일한
/// dtype 후보 적용 (Sprint A' 옵션 B). Adreno SOA variant 내 layout 강제는 build_variant_payload에서.
///
/// 반환 튜플 두 번째 값은 lm_head Q4_0 entry가 *실제로* 추가/유지되었는지 (capability bit 2 set
/// 결정에 사용). source가 둘 다 없거나 Off 모드면 false.
fn extract_weight_blobs(
    gguf: &argus_engine::models::loader::gguf::GgufFile,
    config: &argus_engine::model_config::ModelConfig,
    lm_head_mode: IncludeLmHeadMode,
    candidate_dtypes: Option<&[TensorDType]>,
    quiet: bool,
) -> Result<(Vec<WeightBlob>, bool)> {
    let mut blobs: Vec<WeightBlob> = Vec::new();
    let mut lm_head_q4_0_present = false;

    // cross-layer tensors. lm_head ("output.weight")은 모드에 따라 별도 처리.
    let regular_cross = ["token_embd.weight", "output_norm.weight"];
    for &name in &regular_cross {
        if let Some(info) = gguf.find_tensor(name) {
            let data = gguf.tensor_data(info);
            // GGUF dims는 innermost-first → reversed = outermost-first (logical)
            let shape_logical: Vec<u64> = info.dims.iter().rev().copied().collect();
            let src_dtype = ggml_type_to_tensor_dtype(info.ggml_type);
            let dtype_bytes = build_dtype_candidates(
                name,
                data,
                src_dtype,
                &shape_logical,
                candidate_dtypes,
                quiet,
            )?;
            blobs.push(WeightBlob {
                name: name.to_owned(),
                shape_logical,
                dtype_bytes,
            });
        }
    }

    // lm_head — separate (`output.weight`) vs tied (`token_embd.weight` 재사용) 분기.
    //
    // Llama 3.2 1B 등 tied embedding 모델은 GGUF에 `output.weight`을 별도로 저장하지
    // 않고 `token_embd.weight`과 weight를 공유한다. Sprint F의
    // `quantize_lm_head_to_q4_0()`는 tied 모델에서도 별도 Q4_0 GPU 버퍼를 생성하여
    // matmul 효율을 얻으므로, AUF builder도 동일하게 tied source로부터 lm_head Q4_0
    // entry를 생성한다 (Sprint G-1-B fix, 2026-04-26).
    let lm_head_source = select_lm_head_source(
        gguf.find_tensor(LM_HEAD_SEPARATE_NAME).is_some(),
        gguf.find_tensor(LM_HEAD_TIED_SOURCE_NAME).is_some(),
    );

    if let Some(LmHeadSource {
        source_name,
        is_tied,
    }) = lm_head_source
    {
        if lm_head_mode == IncludeLmHeadMode::Off {
            // Off: separate 케이스에서만 raw bytes 동봉. tied에서는 entry 미생성
            // (token_embd는 이미 cross-layer로 추가됨).
            if !is_tied {
                let info = gguf.find_tensor(source_name).expect("source exists");
                let raw = gguf.tensor_data(info);
                let shape_logical: Vec<u64> = info.dims.iter().rev().copied().collect();
                let src_dtype = ggml_type_to_tensor_dtype(info.ggml_type);
                blobs.push(WeightBlob::single(
                    LM_HEAD_SEPARATE_NAME.to_owned(),
                    shape_logical,
                    src_dtype,
                    raw.to_vec(),
                ));
            }
        } else {
            // On / Auto: source dtype 분기. tied/separate 무관하게 동일 처리.
            // 단, tied에서는 entry 이름은 항상 "output.weight" (lm_head 식별).
            let info = gguf
                .find_tensor(source_name)
                .expect("source exists per match arm above");
            let raw = gguf.tensor_data(info);

            // tied 케이스: entry shape은 token_embd shape과 동일 (== [vocab_size, hidden_dim]).
            // separate 케이스: source shape 그대로.
            let shape_logical: Vec<u64> = info.dims.iter().rev().copied().collect();
            let is_q4_0 = info.ggml_type == GGML_TYPE_Q4_0;
            let src_dtype = ggml_type_to_tensor_dtype(info.ggml_type);
            let dtype_str = match info.ggml_type {
                GGML_TYPE_F32 => "F32",
                GGML_TYPE_F16 => "F16",
                GGML_TYPE_Q4_0 => "Q4_0",
                _ => "other",
            };
            let source_label = if is_tied {
                format!("{} (tied)", source_name)
            } else {
                source_name.to_owned()
            };

            // tied 안전망 — config.vocab_size / hidden_size와 일치 검증.
            // tied 모델에서는 token_embd.weight = lm_head이므로 shape이 [vocab, hidden]임이
            // 보장되어야 한다. 미일치 시 이후 quantize / GPU upload가 잘못된 cl_mem 크기를
            // 만들어 silent corruption을 유발하므로 fail-fast.
            if is_tied {
                let expected_rows = config.vocab_size as u64;
                let expected_cols = config.hidden_size as u64;
                if shape_logical.len() != 2
                    || shape_logical[0] != expected_rows
                    || shape_logical[1] != expected_cols
                {
                    bail!(
                        "tied lm_head source '{}' shape {:?} does not match config \
                         [vocab_size={}, hidden_dim={}]",
                        source_name,
                        shape_logical,
                        expected_rows,
                        expected_cols
                    );
                }
            }

            if is_q4_0 {
                // 이미 Q4_0 — quantize 불필요. AOS bytes를 그대로 동봉. multi-dtype 모드면
                // candidate dtype별 추가 변환 (Q4_0 → F16 / F32 등).
                if !quiet {
                    eprintln!(
                        "[auf-tool] lm_head: source={}, dtype={}, raw {} bytes, \
                         entry included as-is",
                        source_label,
                        dtype_str,
                        raw.len()
                    );
                }
                let dtype_bytes = build_dtype_candidates(
                    LM_HEAD_SEPARATE_NAME,
                    raw,
                    src_dtype,
                    &shape_logical,
                    candidate_dtypes,
                    quiet,
                )?;
                // tied: source는 token_embd지만 entry 이름은 lm_head 식별을 위해 output.weight.
                blobs.push(WeightBlob {
                    name: LM_HEAD_SEPARATE_NAME.to_owned(),
                    shape_logical,
                    dtype_bytes,
                });
                lm_head_q4_0_present = true;
            } else if info.ggml_type == GGML_TYPE_F16 || info.ggml_type == GGML_TYPE_F32 {
                // F16/F32 → F32 dequantize → Q4_0 quantize. multi-dtype 모드면 추가 dtype도 동봉.
                if !quiet {
                    eprintln!(
                        "[auf-tool] lm_head: source={}, dtype={}, quantizing to Q4_0...",
                        source_label, dtype_str,
                    );
                }
                if shape_logical.len() != 2 {
                    bail!(
                        "lm_head source '{}' must be 2-D (got shape={:?})",
                        source_name,
                        shape_logical
                    );
                }
                let q4_bytes =
                    quantize_lm_head_to_q4_0(raw, &shape_logical, info.ggml_type, quiet)?;

                // multi-dtype 모드에서는 lm_head도 candidate dtype 후보 전체에 대해 동봉.
                // single-dtype (legacy v0.1.x) 모드에서는 Q4_0 단일 (Sprint G-1 동작).
                let dtype_bytes: Vec<(TensorDType, Vec<u8>)> = if let Some(cands) = candidate_dtypes
                {
                    let mut out: Vec<(TensorDType, Vec<u8>)> = Vec::with_capacity(cands.len());
                    for &dt in cands {
                        let bytes = if dt == TensorDType::Q4_0 {
                            // 이미 quantize_lm_head_to_q4_0로 구한 결과 재사용.
                            q4_bytes.clone()
                        } else {
                            // Q4_0이 아니면 source raw bytes에서 직접 변환 (F16/F32 → 다른 dtype).
                            convert_tensor_dtype(raw, src_dtype, dt, &shape_logical)
                                .map_err(|e| anyhow!("lm_head dtype convert: {}", e))?
                        };
                        out.push((dt, bytes));
                    }
                    out
                } else {
                    vec![(TensorDType::Q4_0, q4_bytes)]
                };

                blobs.push(WeightBlob {
                    name: LM_HEAD_SEPARATE_NAME.to_owned(),
                    shape_logical,
                    dtype_bytes,
                });
                lm_head_q4_0_present = true;
            } else {
                // 미지원 dtype (Q4_K, Q8_0 등) — fallback: raw bytes + bit 미설정.
                if !quiet {
                    eprintln!(
                        "[auf-tool] Warning: lm_head source={}, dtype={} (ggml_type={}) \
                         unsupported for Q4_0 quantize; entry included as raw bytes \
                         (capability bit 2 not set)",
                        source_label, dtype_str, info.ggml_type
                    );
                }
                if !is_tied {
                    blobs.push(WeightBlob::single(
                        LM_HEAD_SEPARATE_NAME.to_owned(),
                        shape_logical,
                        src_dtype,
                        raw.to_vec(),
                    ));
                }
            }
        }
    } else if lm_head_mode != IncludeLmHeadMode::Off && !quiet {
        // separate / tied 양쪽 모두 source가 없는 매우 드문 경우.
        eprintln!(
            "[auf-tool] Note: GGUF has neither 'output.weight' nor 'token_embd.weight'; \
             lm_head Q4_0 entry skipped, capability bit 2 not set."
        );
    }

    // per-layer tensors (모든 레이어)
    let n_layers = config.num_hidden_layers;
    let weight_kinds = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "attn_norm.weight",
        "ffn_norm.weight",
        // Qwen2 등 qkv bias 모델용 — GGUF에 없으면 자동 skip.
        "attn_q.bias",
        "attn_k.bias",
        "attn_v.bias",
    ];

    for layer in 0..n_layers {
        for &kind in &weight_kinds {
            let name = format!("blk.{layer}.{kind}");
            if let Some(info) = gguf.find_tensor(&name) {
                let raw = gguf.tensor_data(info);
                // GGUF dims는 innermost-first → reversed = outermost-first (logical)
                let shape_logical: Vec<u64> = info.dims.iter().rev().copied().collect();

                // Q/K permute (Llama arch only, attn_q/attn_k)
                let bytes = if let Some((n_head, head_dim)) = qk_permute_shape_local(&name, config)
                {
                    let total_rows = n_head * head_dim;
                    let row_size = raw.len() / total_rows;
                    unpermute_qk_rows_local(raw, n_head, head_dim, row_size)
                } else {
                    raw.to_vec()
                };

                let src_dtype = ggml_type_to_tensor_dtype(info.ggml_type);
                let dtype_bytes = build_dtype_candidates(
                    &name,
                    &bytes,
                    src_dtype,
                    &shape_logical,
                    candidate_dtypes,
                    quiet,
                )?;
                blobs.push(WeightBlob {
                    name,
                    shape_logical,
                    dtype_bytes,
                });
            } else if !quiet && kind.ends_with(".weight") && !kind.contains("norm") {
                // weight tensor 누락 경고 (norm/bias는 모델별로 없을 수 있음)
                eprintln!("[auf-tool] Warning: tensor '{}' not found in GGUF", name);
            }
        }
    }

    Ok((blobs, lm_head_q4_0_present))
}

/// GGUF ggml_type 코드 → `TensorDType`.
///
/// 미지원 dtype은 보수적으로 source dtype을 그대로 추정 (raw bytes 동봉용).
fn ggml_type_to_tensor_dtype(ggml_type: u32) -> TensorDType {
    match ggml_type {
        GGML_TYPE_F32 => TensorDType::F32,
        GGML_TYPE_F16 => TensorDType::F16,
        GGML_TYPE_Q4_0 => TensorDType::Q4_0,
        // 기타 (Q4_1 / Q8_0 / BF16 / U8 등) — convert_tensor_dtype은 거부하지만, 변환 없이
        // single source dtype으로 단일 entry를 생성하면 raw bytes로 통과한다 (legacy 호환).
        _ => TensorDType::F16, // placeholder — 실제 변환 시도 시 convert_tensor_dtype이 reject.
    }
}

// `build_dtype_candidates`는 `crate::auf::dtype_convert`에서 export된 라이브러리
// 함수를 사용한다 (Sprint F ISSUE-E-1 fix로 외부 lib로 이동, spec 테스트 가능).

/// GGUF lm_head (F16 or F32) → F32 dequantize → Q4_0 quantize → 18B/block bytes.
///
/// `convert::quantize_q4_0`(layer weight도 동일하게 사용)을 재사용한다 — 신규 변환
/// 함수 작성 금지 (Sprint G-1-A 결정). 결정성은 `quantize_q4_0`의 단순 max-abs scaling
/// 루프가 부동소수점 reduction 순서를 고정하므로 호스트 결정적이다 (ENG-DAT-096.13).
///
/// 변환 시간을 stderr로 기록한다 (Sprint G-1 요구).
fn quantize_lm_head_to_q4_0(
    raw: &[u8],
    shape_logical: &[u64],
    ggml_type: u32,
    quiet: bool,
) -> Result<Vec<u8>> {
    use argus_engine::models::loader::convert::{f16_to_f32, quantize_q4_0};
    use argus_engine::quant::{BlockQ4_0, QK4_0};

    if shape_logical.len() != 2 {
        bail!("lm_head must be 2-D (got shape={:?})", shape_logical);
    }
    let rows = shape_logical[0] as usize; // outermost (vocab)
    let cols = shape_logical[1] as usize; // innermost (hidden)
    if !cols.is_multiple_of(QK4_0) {
        bail!(
            "lm_head inner dim ({}) is not a multiple of QK4_0 ({}); cannot quantize to Q4_0",
            cols,
            QK4_0
        );
    }
    let numel = rows * cols;

    let t0 = std::time::Instant::now();

    // Step 1: dequantize to F32 host buffer.
    let mut f32_data = vec![0.0f32; numel];
    match ggml_type {
        GGML_TYPE_F16 => {
            f16_to_f32(raw, &mut f32_data, numel);
        }
        GGML_TYPE_F32 => {
            // SAFETY: GGUF F32 is little-endian f32 array of length `numel`.
            let src = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, numel) };
            f32_data.copy_from_slice(src);
        }
        other => {
            bail!(
                "quantize_lm_head_to_q4_0: unsupported ggml_type={} (expected F16 or F32)",
                other
            );
        }
    }

    // Step 2: quantize to Q4_0 blocks (layer weight와 동일 함수).
    let blocks = quantize_q4_0(&f32_data, rows, cols);
    let block_bytes_each = std::mem::size_of::<BlockQ4_0>(); // 18B
    let total_bytes = blocks.len() * block_bytes_each;

    // SAFETY: BlockQ4_0 is `#[repr(C)]` with size 18 (asserted at compile time
    // in core::quant). Reading its bytes through `from_raw_parts` is well-defined.
    let block_bytes_view =
        unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, total_bytes) };
    let bytes_out = block_bytes_view.to_vec();

    let elapsed = t0.elapsed();
    if !quiet {
        eprintln!(
            "[auf-tool] lm_head Q4_0 quantize: {:.2}s, output {} MB ({}×{}, {} blocks)",
            elapsed.as_secs_f64(),
            total_bytes / (1024 * 1024),
            rows,
            cols,
            blocks.len(),
        );
    }
    Ok(bytes_out)
}

/// Llama Q/K weight 행 permute가 필요한지 판단하고 (n_head, head_dim) 반환.
///
/// `pub(crate)` 함수 `qk_permute_shape`을 바이너리에서 접근할 수 없으므로
/// 동등한 로직을 로컬에 구현한다.
fn qk_permute_shape_local(
    name: &str,
    config: &argus_engine::model_config::ModelConfig,
) -> Option<(usize, usize)> {
    use argus_engine::model_config::ModelArch;
    if config.arch != ModelArch::Llama {
        return None;
    }
    // "blk.<N>.attn_q.weight" 또는 "blk.<N>.attn_k.weight"
    let stem = name.strip_prefix("blk.")?;
    let (_idx, rest) = stem.split_once('.')?;
    let head_dim = config.head_dim;
    match rest {
        "attn_q.weight" => Some((config.num_attention_heads, head_dim)),
        "attn_k.weight" => Some((config.num_key_value_heads, head_dim)),
        _ => None,
    }
}

/// llama.cpp Q/K 행 permute를 되돌린다 (NeoX RoPE layout 복원).
///
/// `pub(crate)` 함수 `unpermute_qk_rows`를 바이너리에서 접근할 수 없으므로
/// 동등한 로직을 로컬에 구현한다.
fn unpermute_qk_rows_local(
    src: &[u8],
    n_head: usize,
    head_dim: usize,
    row_size_bytes: usize,
) -> Vec<u8> {
    let half = head_dim / 2;
    let total_rows = n_head * head_dim;
    debug_assert_eq!(src.len(), total_rows * row_size_bytes);
    let mut dst = vec![0u8; src.len()];
    for h in 0..n_head {
        let head_base = h * head_dim;
        for j in 0..head_dim {
            let src_in_head = if j < half { 2 * j } else { 2 * (j - half) + 1 };
            let src_row = head_base + src_in_head;
            let dst_row = head_base + j;
            let src_off = src_row * row_size_bytes;
            let dst_off = dst_row * row_size_bytes;
            dst[dst_off..dst_off + row_size_bytes]
                .copy_from_slice(&src[src_off..src_off + row_size_bytes]);
        }
    }
    dst
}

/// 특정 variant tag와 dtype에 맞는 단일 tensor sub-payload 바이트열 생성.
///
/// caller는 blob의 dtype별 bytes를 외부 루프에서 결정하고 이 함수에 명시적으로 전달한다.
/// dtype별 sub-payload를 단조 cursor 배치로 직렬화하기 위함이다.
///
/// **Adreno SOA × layer weight Q4_0 케이스만 SOA 변환을 적용**한다.
/// - lm_head는 dtype에 무관하게 AOS 18B/block layout 강제 (INV-135 v2 / Sprint G-1-F).
///   Adreno OpenCL `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` 한계 초과로 image1d 빠른 path를
///   사용할 수 없기 때문이다.
/// - F16 / F32 dtype은 SOA 변환 불가 — bytes as-is + tag-specific alignment.
///   Adreno SOA variant에서는 alignment 0 (no-op) 그대로 동봉.
fn build_variant_tensor_bytes(
    name: &str,
    bytes: &[u8],
    shape: &[u64],
    dtype: TensorDType,
    tag: &str,
) -> Result<Vec<u8>> {
    let is_lm_head = name == LM_HEAD_SEPARATE_NAME;

    match tag {
        TAG_WEIGHTS_ADRENO_SOA => {
            // Adreno SOA × Q4_0 layer weight: SOA transpose 적용. 그 외는 AOS bytes as-is.
            //
            // **Adreno SOA × F16 layer weight 정책 (INV-135 / Sprint C-D)**:
            //   F16 (또는 F32) layer weight는 Adreno SOA variant에서는 SOA 변환을
            //   적용하지 않는다. SOA layout은 Q4_0 quant block 단위에 의존하므로
            //   F16/F32에 무의미하다. Reader는 backend tag = adreno_soa + dtype = F16
            //   조합을 만나면 layer weight는 fallback (AOS F16 GEMV)으로 처리해야 한다.
            //   본 writer는 sub-payload bytes를 그대로 동봉만 한다.
            if !is_lm_head && dtype == TensorDType::Q4_0 && shape.len() == 2 {
                let ne01 = shape[0] as usize; // rows
                let ne00 = shape[1] as usize; // cols (K dim)
                let expected = (ne01 * ne00 / QK4_0_LOCAL) * 18;
                if expected == bytes.len() && ne00.is_multiple_of(QK4_0_LOCAL) {
                    let (q_buf, d_buf) = q4_0_aos_to_adreno_soa(bytes, ne00, ne01);
                    let mut out = Vec::with_capacity(q_buf.len() + d_buf.len());
                    out.extend_from_slice(&q_buf);
                    out.extend_from_slice(&d_buf);
                    return Ok(out);
                }
                eprintln!(
                    "[auf-tool] Warning: SOA shape guard rejected '{}' (shape={:?}, bytes={}); \
                     emitting AOS bytes (forward path will fall back to AOS GEMV).",
                    name,
                    shape,
                    bytes.len()
                );
            }
            // lm_head (image-limit), F16/F32 tensor, shape-mismatch fallback: bytes as-is.
            Ok(bytes.to_vec())
        }
        TAG_WEIGHTS_CUDA_AOS => Ok(q4_0_aos_with_align(bytes, 128)),
        TAG_WEIGHTS_CPU_AOS => Ok(q4_0_aos_with_align(bytes, 64)),
        _ => bail!("Unknown variant tag: {}", tag),
    }
}

/// QK4_0 = 32 (binary crate에서 core::quant::QK4_0 직접 import 시 build script 회로 우려로
/// 로컬 상수 사용).
const QK4_0_LOCAL: usize = 32;

/// 특정 variant tag에 맞는 payload 바이트열 생성 (dtype-aware multi-dtype).
///
/// 같은 variant 안에서 dtype별 sub-payload를 cursor 단조 배치한다. dtype 순서는
/// blob.dtype_bytes의 등록 순서를 그대로 따른다 (caller가 INV-138 정렬 의무를 만족하도록
/// 미리 정렬해야 하며, build_tensor_index는 동일 정렬을 적용한다).
///
/// dtype별 sub-payload는 `build_variant_tensor_bytes`로 변환된다.
fn build_variant_payload(blobs: &[WeightBlob], tag: &str) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for blob in blobs {
        for (dtype, bytes) in &blob.dtype_bytes {
            let sub =
                build_variant_tensor_bytes(&blob.name, bytes, &blob.shape_logical, *dtype, tag)?;
            out.extend_from_slice(&sub);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// TensorIndex 구성 헬퍼
// ---------------------------------------------------------------------------

/// blob 이름 → `(layer_idx, TensorKind)` 변환.
///
/// cross-layer tensor는 `LAYER_IDX_CROSS` 반환.
/// 인식할 수 없는 이름은 `None` 반환.
fn tensor_name_to_layer_kind(name: &str) -> Option<(u32, TensorKind)> {
    // cross-layer tensors
    match name {
        "token_embd.weight" => return Some((LAYER_IDX_CROSS, TensorKind::Embedding)),
        "output_norm.weight" => return Some((LAYER_IDX_CROSS, TensorKind::FinalNorm)),
        "output.weight" => return Some((LAYER_IDX_CROSS, TensorKind::LmHead)),
        _ => {}
    }

    // per-layer: "blk.<N>.<kind>"
    let rest = name.strip_prefix("blk.")?;
    let (idx_str, kind_str) = rest.split_once('.')?;
    let layer_idx: u32 = idx_str.parse().ok()?;

    let kind = match kind_str {
        "attn_q.weight" => TensorKind::AttnQ,
        "attn_k.weight" => TensorKind::AttnK,
        "attn_v.weight" => TensorKind::AttnV,
        "attn_output.weight" => TensorKind::AttnO,
        "ffn_gate.weight" => TensorKind::FfnGate,
        "ffn_up.weight" => TensorKind::FfnUp,
        "ffn_down.weight" => TensorKind::FfnDown,
        "attn_norm.weight" => TensorKind::AttnNorm,
        "ffn_norm.weight" => TensorKind::FfnNorm,
        "attn_q.bias" => TensorKind::AttnQBias,
        "attn_k.bias" => TensorKind::AttnKBias,
        "attn_v.bias" => TensorKind::AttnVBias,
        _ => return None,
    };

    Some((layer_idx, kind))
}

/// blob bytes가 특정 variant에서 차지하는 payload 크기를 계산한다.
///
/// `build_variant_tensor_bytes`와 동일한 변환 로직을 적용하되, 실제 데이터 복사 없이
/// 크기만 반환한다. SOA 변환은 byte 수를 변경하지 않으므로 입력 길이 그대로.
fn compute_variant_tensor_size(bytes: &[u8], tag: &str) -> usize {
    match tag {
        TAG_WEIGHTS_ADRENO_SOA => {
            // Q4_0 SOA: q_buf(N*16) + d_buf(N*2) = N*18 = bytes.len() (불변).
            // F16/F32 / lm_head AOS / shape-mismatch fallback: bytes.len() 그대로.
            bytes.len()
        }
        TAG_WEIGHTS_CUDA_AOS => bytes.len().div_ceil(128) * 128,
        TAG_WEIGHTS_CPU_AOS => bytes.len().div_ceil(64) * 64,
        _ => bytes.len(),
    }
}

/// `extract_weight_blobs` 결과와 variant tag 목록으로 `TensorIndex`를 구성한다.
///
/// 각 variant마다 payload 내 tensor별 section-local offset을 추적하여
/// `TensorEntry::variant_offsets`/`variant_sizes`를 채운다.
/// `TensorEntry::shape`는 logical order (outermost-first)로 채운다.
///
/// **multi-dtype 모드 (Sprint C-E, INV-138)**: blob에 dtype이 여러 개면 dtype별로 entry를
/// 추가한다. 그러나 같은 (layer_idx, kind) 그룹 안에서는 INV-138 정렬 키로 안정 정렬한다:
/// `(layer_idx ASC, kind ASC, is_default DESC, dtype ASC)`. 이는 v0.1.x reader가 first-match로
/// default_dtype을 자동 선택하도록 보장하는 호환 의무이다.
///
/// `default_dtype`이 None이면 single-dtype 모드 (각 blob에 dtype_bytes가 1개씩) — 정렬 키의
/// is_default 컴포넌트는 무의미.
///
/// payload offset/size는 entry 정렬 후가 아니라 *blob.dtype_bytes의 등록 순서대로* 추적해야
/// 한다 (build_variant_payload가 같은 순서로 sub-payload를 직렬화하기 때문). 따라서
/// 정렬은 entry 자체에만 적용하고 offset/size는 blob 등록 순서로 채운 후, 정렬 후에 entry
/// 안에서 그대로 보존된다.
fn build_tensor_index(
    blobs: &[WeightBlob],
    variant_tags: &[&str],
    default_dtype: Option<TensorDType>,
) -> TensorIndex {
    let variant_count = variant_tags.len();

    // variant_tags → [u8; 24] 배열 변환
    let vt_bytes: Vec<[u8; 24]> = variant_tags
        .iter()
        .map(|tag| {
            let mut buf = [0u8; 24];
            let b = tag.as_bytes();
            buf[..b.len().min(24)].copy_from_slice(&b[..b.len().min(24)]);
            buf
        })
        .collect();

    // variant별 현재 누적 offset (section-local, 0-based)
    let mut variant_cursors: Vec<u64> = vec![0u64; variant_count];

    let mut entries: Vec<TensorEntry> = Vec::new();

    for blob in blobs {
        let recognized = tensor_name_to_layer_kind(&blob.name);

        for (dtype, bytes) in &blob.dtype_bytes {
            // payload cursor는 항상 전진 (인식 불가한 tensor도 sub-payload는 직렬화됨).
            let mut variant_offsets = Vec::with_capacity(variant_count);
            let mut variant_sizes = Vec::with_capacity(variant_count);
            for (vi, &tag) in variant_tags.iter().enumerate() {
                let sz = compute_variant_tensor_size(bytes, tag) as u64;
                variant_offsets.push(variant_cursors[vi]);
                variant_sizes.push(sz);
                variant_cursors[vi] += sz;
            }

            // 인식 불가 tensor는 entry 미등록 (cursor만 전진).
            let Some((layer_idx, kind)) = recognized else {
                continue;
            };

            entries.push(TensorEntry {
                layer_idx,
                kind: kind.as_u32(),
                dtype: dtype.as_u32(),
                // logical order (outermost-first): reader가 .rev()하여 GGUF innermost-first로 복원
                shape: blob.shape_logical.clone(),
                alignment: 64,
                variant_offsets,
                variant_sizes,
            });
        }
    }

    // INV-138 정렬: (layer_idx ASC, kind ASC, is_default DESC, dtype ASC).
    // sort_by_key는 stable이므로 동일 key 내에서는 등록 순서 보존.
    let default_u32 = default_dtype.map(|d| d.as_u32());
    entries.sort_by_key(|e| {
        // is_default DESC == "0이 먼저, 1이 나중" → bool을 그대로 ascending sort하려면 not 적용.
        // is_default = (dtype == default_dtype) → DESC 정렬을 위해 !is_default를 키로 사용.
        let not_default: u8 = match default_u32 {
            Some(d) if e.dtype == d => 0,
            _ => 1,
        };
        (e.layer_idx, e.kind, not_default, e.dtype)
    });

    TensorIndex {
        variant_tags: vt_bytes,
        entries,
    }
}

// ---------------------------------------------------------------------------
// info 커맨드
// ---------------------------------------------------------------------------

fn cmd_info(args: InfoArgs) -> Result<()> {
    let path = &args.file;
    let view = open(path, BackendTag::Any)
        .map_err(|e| anyhow!("failed to open AUF file ({}): {}", path.display(), e))?;

    let file_size = view.file_size();
    println!("File: {}", path.display());
    println!(
        "Size: {} bytes ({:.2} GiB)",
        file_size,
        file_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!();

    // Header
    let h = &view.header;
    let created_by = {
        let end = h.created_by.iter().position(|&b| b == 0).unwrap_or(32);
        String::from_utf8_lossy(&h.created_by[..end]).into_owned()
    };
    let source_hash_hex: String = h.source_hash.iter().map(|b| format!("{:02x}", b)).collect();
    let source_mtime_str = format_unix_time(h.source_mtime);

    println!("Header:");
    println!("  magic            : \"ARGUS_W\\0\"");
    println!(
        "  format           : v{}.{}.{}{}",
        h.format_major,
        h.format_minor,
        h.format_patch,
        if h.format_major == 0 {
            " (experimental)"
        } else {
            ""
        }
    );
    println!("  created_by       : \"{}\"", created_by);
    println!("  source_hash      : {}...", &source_hash_hex[..16]);
    println!("  source_size      : {} bytes", h.source_size);
    println!("  source_mtime     : {}", source_mtime_str);
    println!("  capability_req   : {:#018x}", h.capability_required);
    {
        let mut caps = Vec::new();
        if h.has_lm_head_q4_0() {
            caps.push("LM_HEAD_PRECOMPUTED_Q4_0");
        }
        if h.has_multi_dtype() {
            caps.push("MULTI_DTYPE_VARIANTS");
        }
        let suffix = if caps.is_empty() {
            String::new()
        } else {
            format!(" ({})", caps.join(" + "))
        };
        println!(
            "  capability_opt   : {:#018x}{}",
            h.capability_optional, suffix
        );
    }
    println!("  section_count    : {}", h.section_count);
    println!(
        "  payload_start    : {:#x} ({})",
        h.payload_start_offset, h.payload_start_offset
    );
    println!(
        "  section_table_at : {:#x} ({})",
        h.section_table_offset, h.section_table_offset
    );
    println!();

    // Section table
    println!(
        "{:<26} {:<12} {:<16} {:<12} Version",
        "Tag", "Offset", "Size", "Flags"
    );
    println!("{}", "-".repeat(76));
    let mut variants_present = Vec::new();
    for entry in &view.section_table.entries {
        let tag = entry.tag();
        let flags_str = build_flags_str(entry.flags);
        let size_str = if entry.size >= 1024 * 1024 {
            format!("{:.1} MB", entry.size as f64 / (1024.0 * 1024.0))
        } else if entry.size >= 1024 {
            format!("{:.1} KB", entry.size as f64 / 1024.0)
        } else {
            format!("{} B", entry.size)
        };
        println!(
            "{:<26} {:#010x}   {:<16} {:<12} {}",
            tag, entry.offset, size_str, flags_str, entry.version
        );

        if tag.starts_with("WEIGHTS_") {
            let variant_name = tag.strip_prefix("WEIGHTS_").unwrap_or(tag);
            variants_present.push(variant_name.to_owned());
        }
    }
    println!();

    // META summary
    let m = &view.meta;
    println!("META:");
    println!("  architecture     : {}", m.architecture);
    println!("  n_layers         : {}", m.n_layers);
    println!("  n_heads_q        : {}", m.n_heads_q);
    println!("  n_kv_heads       : {}", m.n_kv_heads);
    println!("  head_dim         : {}", m.head_dim);
    println!("  hidden_dim       : {}", m.hidden_dim);
    println!("  ffn_dim          : {}", m.ffn_dim);
    println!("  vocab_size       : {}", m.vocab_size);
    println!("  max_seq_len      : {}", m.max_seq_len);
    println!("  rope_theta       : {}", m.rope_theta);
    println!("  rms_norm_epsilon : {}", m.rms_norm_epsilon);
    if let Some(dd) = &m.default_dtype {
        println!("  default_dtype    : {}", dd);
    }
    println!();

    // Tokenizer summary
    let t = &view.tokenizer;
    println!("TOKENIZER:");
    println!("  kind             : {} (BPE)", t.kind);
    println!("  vocab_size       : {}", t.tokens.len());
    println!("  merges_count     : {}", t.merges.len());
    println!(
        "  bos_id           : {}",
        if t.bos_id < 0 {
            "N/A".to_owned()
        } else {
            t.bos_id.to_string()
        }
    );
    println!(
        "  eos_id           : {}",
        if t.eos_id < 0 {
            "N/A".to_owned()
        } else {
            t.eos_id.to_string()
        }
    );
    println!(
        "  chat_template    : {}",
        if t.chat_template.is_some() {
            "yes"
        } else {
            "no"
        }
    );
    println!();

    // TENSOR_INDEX summary
    let ti = &view.tensor_index;
    println!("TENSOR_INDEX:");
    println!("  variants         : {:?}", ti.variant_tag_strings());
    println!("  tensor_count     : {}", ti.entries.len());

    // dtype 분포 — BTreeMap<dtype_str, count>를 사용해 결정적 출력 (HashMap 비결정성 회피).
    {
        use std::collections::BTreeMap;
        let mut dtype_count: BTreeMap<&'static str, usize> = BTreeMap::new();
        for e in &ti.entries {
            let label = match TensorDType::from_u32(e.dtype) {
                Some(TensorDType::F32) => "F32",
                Some(TensorDType::F16) => "F16",
                Some(TensorDType::BF16) => "BF16",
                Some(TensorDType::Q4_0) => "Q4_0",
                Some(TensorDType::Q4_1) => "Q4_1",
                Some(TensorDType::Q8_0) => "Q8_0",
                Some(TensorDType::U8) => "U8",
                None => "?",
            };
            *dtype_count.entry(label).or_default() += 1;
        }
        if !dtype_count.is_empty() {
            let parts: Vec<String> = dtype_count
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            println!("  dtype_dist       : {{{}}}", parts.join(", "));
        }
    }

    // multi-dtype 그룹 — 동일 (layer_idx, kind)에 dtype이 2개 이상인 그룹 수.
    if h.has_multi_dtype() {
        use std::collections::BTreeMap;
        let mut group_count: BTreeMap<(u32, u32), usize> = BTreeMap::new();
        for e in &ti.entries {
            *group_count.entry((e.layer_idx, e.kind)).or_default() += 1;
        }
        let multi_groups: usize = group_count.values().filter(|&&c| c >= 2).count();
        println!(
            "  multi_dtype_grps : {} (groups with >=2 dtype candidates)",
            multi_groups
        );
    }
    println!();

    // Variant × dtype × size matrix (multi-dtype 가시화).
    if !ti.entries.is_empty() {
        println!("Variant × Dtype Size Matrix:");
        let variant_strs = ti.variant_tag_strings();
        for (vi, vt) in variant_strs.iter().enumerate() {
            use std::collections::BTreeMap;
            let mut by_dtype: BTreeMap<&'static str, u64> = BTreeMap::new();
            for e in &ti.entries {
                if vi >= e.variant_sizes.len() {
                    continue;
                }
                let label = match TensorDType::from_u32(e.dtype) {
                    Some(TensorDType::F32) => "F32",
                    Some(TensorDType::F16) => "F16",
                    Some(TensorDType::Q4_0) => "Q4_0",
                    _ => "other",
                };
                *by_dtype.entry(label).or_default() += e.variant_sizes[vi];
            }
            let parts: Vec<String> = by_dtype
                .iter()
                .map(|(k, v)| format!("{}={:.1}MB", k, *v as f64 / (1024.0 * 1024.0)))
                .collect();
            println!("  {:<26} {{{}}}", vt, parts.join(", "));
        }
        println!();
    }

    // Variants present
    if variants_present.is_empty() {
        println!("Variants present: (none)");
    } else {
        println!("Variants present: {}", variants_present.join(", "));
    }

    Ok(())
}

fn build_flags_str(flags: u32) -> String {
    use argus_engine::auf::{SECTION_COMPRESSED, SECTION_REQUIRED, SECTION_STRIPPABLE};
    let mut parts = Vec::new();
    if flags & SECTION_REQUIRED != 0 {
        parts.push("REQUIRED");
    }
    if flags & SECTION_STRIPPABLE != 0 {
        parts.push("STRIPPABLE");
    }
    if flags & SECTION_COMPRESSED != 0 {
        parts.push("COMPRESSED");
    }
    if parts.is_empty() {
        "0".to_owned()
    } else {
        parts.join("+")
    }
}

fn format_unix_time(secs: u64) -> String {
    if secs == 0 {
        return "N/A".to_owned();
    }
    // 간단한 UTC 포맷 (외부 crate 없이)
    // secs → 년월일시분초 계산
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // 년월일 계산 (간략, 윤년 근사)
    let mut year = 1970u64;
    let mut days = days_since_epoch;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let months = [
        31u64,
        if is_leap(year) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 0usize;
    for &m in &months {
        if days < m {
            break;
        }
        days -= m;
        month += 1;
    }
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        year,
        month + 1,
        days + 1,
        hours,
        minutes,
        seconds
    )
}

fn is_leap(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

// ---------------------------------------------------------------------------
// strip 커맨드
// ---------------------------------------------------------------------------

fn cmd_strip(args: StripArgs) -> Result<()> {
    let path = &args.file;

    // keep tags 파싱
    let keep_tags_raw: Vec<String> = args
        .keep
        .split(',')
        .map(|s| normalize_tag(s.trim()))
        .collect();

    let keep_tags_str: Vec<&str> = keep_tags_raw.iter().map(|s| s.as_str()).collect();

    eprintln!("[auf-tool] Strip: {}", path.display());
    eprintln!("[auf-tool] Keep tags: {:?}", keep_tags_str);
    if !args.no_backup {
        let bak = path.with_extension("auf.bak");
        eprintln!("[auf-tool] Backup: {}", bak.display());
    }

    strip(path, &keep_tags_str, args.no_backup).map_err(|e| anyhow!("Strip failed: {}", e))?;

    let new_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "[auf-tool] Strip complete: {} ({:.2} GiB)",
        path.display(),
        new_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

/// 사용자 입력 variant tag를 정규화 (대소문자 무관, 축약형 허용).
fn normalize_tag(s: &str) -> String {
    match s.to_ascii_uppercase().as_str() {
        "ADRENO_SOA" | "ADRENO" => TAG_WEIGHTS_ADRENO_SOA.to_owned(),
        "CUDA_AOS" | "CUDA" => TAG_WEIGHTS_CUDA_AOS.to_owned(),
        "CPU_AOS" | "CPU" => TAG_WEIGHTS_CPU_AOS.to_owned(),
        // 이미 전체 tag 형식이거나 required section tag
        other => other.to_owned(),
    }
}

// ---------------------------------------------------------------------------
// verify 커맨드
// ---------------------------------------------------------------------------

fn cmd_verify(args: VerifyArgs) -> Result<()> {
    let path = &args.file;
    let mut all_pass = true;

    println!("Verifying: {}", path.display());
    println!("{}", "=".repeat(60));

    // 파일 읽기 (BackendTag::Any — backend 검증 우회)
    let view_result = open(path, BackendTag::Any);

    match &view_result {
        Ok(_) => {}
        Err(AufError::FileTooSmall) => {
            print_check("File size >= 256B", false, "file is too small (< 256B)");
            all_pass = false;
            print_result(all_pass);
            return if all_pass {
                Ok(())
            } else {
                Err(anyhow!("Verify FAILED"))
            };
        }
        Err(e) => {
            // 특정 에러를 검증 항목별로 분류
            let check_name = classify_error(e);
            print_check(check_name, false, &e.to_string());
            all_pass = false;
            print_result(all_pass);
            return Err(anyhow!("Verify FAILED"));
        }
    }

    let view = view_result.unwrap();

    // INV-132: magic, format_major, capability_required
    print_check("Magic bytes (ARGUS_W\\0)", true, "");
    print_check("format_major <= READER_MAX", true, "");
    print_check("capability_required all known bits", true, "");

    // INV-133: required sections
    let required_sections = ["META", "TOKENIZER", "TENSOR_INDEX"];
    for &tag in &required_sections {
        let found = view.section_table.find(tag).is_some();
        print_check(
            &format!("Required section: {}", tag),
            found,
            if found { "" } else { "missing" },
        );
        if !found {
            all_pass = false;
        }
    }

    // INV-134: section table 무결성 (open()에서 이미 검증됨)
    print_check("Section tags unique", true, "");
    print_check("Section flags valid (no REQUIRED+STRIPPABLE)", true, "");
    print_check("No COMPRESSED sections (v0.1)", true, "");
    print_check("Section offset/size within file bounds", true, "");
    print_check("Section ranges no overlap", true, "");

    // META 파싱 확인
    let meta_ok = !view.meta.architecture.is_empty() && view.meta.n_layers > 0;
    print_check(
        "META JSON parseable",
        meta_ok,
        if meta_ok {
            ""
        } else {
            "parse failed or fields empty"
        },
    );
    if !meta_ok {
        all_pass = false;
    }

    // TOKENIZER 파싱 확인
    let tok_ok = !view.tokenizer.tokens.is_empty();
    print_check(
        "TOKENIZER vocab non-empty",
        tok_ok,
        if tok_ok { "" } else { "vocab is empty" },
    );
    if !tok_ok {
        all_pass = false;
    }

    // Variant sections 정보 출력 (pass/fail 아닌 informational)
    let variant_tags = [
        TAG_WEIGHTS_ADRENO_SOA,
        TAG_WEIGHTS_CUDA_AOS,
        TAG_WEIGHTS_CPU_AOS,
    ];
    let mut found_variants = Vec::new();
    for &tag in &variant_tags {
        if let Some(entry) = view.section_table.find(tag) {
            let strippable = entry.flags & SECTION_STRIPPABLE != 0;
            print_check(
                &format!("Variant section: {}", tag),
                strippable,
                if strippable {
                    "present, STRIPPABLE"
                } else {
                    "present, flag mismatch"
                },
            );
            found_variants.push(tag);
        }
    }
    if found_variants.is_empty() {
        println!("  [INFO] No WEIGHTS_* variant sections present");
    }

    // TENSOR_INDEX 내용 검증: WEIGHTS_* section이 존재하면 TENSOR_INDEX가 cover해야 함
    if !found_variants.is_empty() {
        let ti = &view.tensor_index;
        for &weights_tag in &found_variants {
            let covered = ti.variant_index_for_tag(weights_tag).is_some();
            let check_name = format!("TENSOR_INDEX covers variant: {weights_tag}");
            if covered {
                print_check(&check_name, true, "");
            } else {
                print_check(
                    &check_name,
                    false,
                    &format!(
                        "TENSOR_INDEX does not cover variant '{weights_tag}' — \
                         empty index detected. Rebuild with 'auf-tool build'."
                    ),
                );
                all_pass = false;
            }
        }

        // tensor_count가 0이면 추가 경고
        if ti.entries.is_empty() && !found_variants.is_empty() {
            print_check(
                "TENSOR_INDEX has entries (tensor_count > 0)",
                false,
                "TENSOR_INDEX has 0 entries but WEIGHTS_* sections exist — \
                 rebuild required",
            );
            all_pass = false;
        } else if !ti.entries.is_empty() {
            print_check(
                &format!("TENSOR_INDEX tensor_count={}", ti.entries.len()),
                true,
                "",
            );
        }

        // INV-137: 동일 (layer_idx, kind) 그룹의 모든 dtype candidate은 동일 shape를 가져야 함.
        // BTreeMap으로 결정적 순회 (HashMap 비결정성 회피).
        if view.header.has_multi_dtype() {
            use std::collections::BTreeMap;
            let mut groups: BTreeMap<(u32, u32), Vec<&TensorEntry>> = BTreeMap::new();
            for e in &ti.entries {
                groups.entry((e.layer_idx, e.kind)).or_default().push(e);
            }
            let mut shape_ok = true;
            for ((layer, kind), members) in &groups {
                if members.len() < 2 {
                    continue;
                }
                let first_shape = &members[0].shape;
                for m in members.iter().skip(1) {
                    if &m.shape != first_shape {
                        let layer_str = if *layer == u32::MAX {
                            "cross".to_owned()
                        } else {
                            layer.to_string()
                        };
                        print_check(
                            &format!(
                                "INV-137 multi-dtype shape match (layer={} kind={})",
                                layer_str, kind
                            ),
                            false,
                            &format!(
                                "shape mismatch: dtype={} {:?} vs dtype={} {:?}",
                                members[0].dtype, first_shape, m.dtype, m.shape
                            ),
                        );
                        shape_ok = false;
                        all_pass = false;
                    }
                }
            }
            if shape_ok {
                print_check(
                    "INV-137 multi-dtype shape consistency",
                    true,
                    "all groups have matching shapes",
                );
            }

            // INV-138 (a): META.default_dtype 의무.
            if view.meta.default_dtype.is_none() {
                print_check(
                    "INV-138 META.default_dtype required for multi-dtype",
                    false,
                    "MULTI_DTYPE_VARIANTS bit 3 set but META.default_dtype missing",
                );
                all_pass = false;
            } else {
                print_check(
                    &format!(
                        "INV-138 META.default_dtype = {}",
                        view.meta.default_dtype.as_deref().unwrap_or("?")
                    ),
                    true,
                    "",
                );
            }
        }

        // 각 TensorEntry.shape rank > 0 검증 (빈 shape = swap_executor 차단)
        let mut all_shapes_ok = true;
        for entry in &ti.entries {
            if entry.shape.is_empty() {
                let layer_str = if entry.layer_idx == u32::MAX {
                    "cross".to_owned()
                } else {
                    entry.layer_idx.to_string()
                };
                print_check(
                    &format!(
                        "TensorEntry shape non-empty (layer={} kind={})",
                        layer_str, entry.kind
                    ),
                    false,
                    &format!(
                        "entry layer={} kind={} has empty shape — \
                         TensorEntry.shape must be populated. Rebuild with 'auf-tool build'.",
                        layer_str, entry.kind
                    ),
                );
                all_shapes_ok = false;
                all_pass = false;
            }
        }
        if all_shapes_ok && !ti.entries.is_empty() {
            print_check("All TensorEntry shapes non-empty", true, "");
        }
    }

    // source_hash 비교 (--source 옵션)
    if let Some(source_path) = &args.source {
        eprint!("  Computing source_hash for {}...", source_path.display());
        match compute_source_hash(source_path) {
            Ok((computed_hash, _, _)) => {
                eprintln!(" done");
                let matches = computed_hash == view.header.source_hash;
                if matches {
                    print_check("source_hash matches GGUF", true, "");
                } else {
                    let auf_hash: String = view
                        .header
                        .source_hash
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect();
                    let comp_hash: String =
                        computed_hash.iter().map(|b| format!("{:02x}", b)).collect();
                    println!("  [WARN] source_hash mismatch (informational, not a hard failure)");
                    println!("         AUF says : {}...", &auf_hash[..16]);
                    println!("         Computed : {}...", &comp_hash[..16]);
                }
            }
            Err(e) => {
                eprintln!();
                println!("  [WARN] failed to compute source_hash: {}", e);
            }
        }
    }

    println!("{}", "-".repeat(60));
    print_result(all_pass);

    if all_pass {
        Ok(())
    } else {
        Err(anyhow!("Verify FAILED"))
    }
}

fn print_check(name: &str, pass: bool, detail: &str) {
    let mark = if pass { "PASS" } else { "FAIL" };
    if detail.is_empty() {
        println!("  [{mark}] {name}");
    } else {
        println!("  [{mark}] {name}: {detail}");
    }
}

fn print_result(all_pass: bool) {
    if all_pass {
        println!("Result: PASS — AUF is valid");
    } else {
        println!("Result: FAIL — AUF has integrity issues");
    }
}

fn classify_error(e: &AufError) -> &'static str {
    match e {
        AufError::MagicMismatch => "Magic bytes (ARGUS_W\\0)",
        AufError::UnsupportedFormatMajor { .. } => "format_major <= READER_MAX",
        AufError::UnknownRequiredCapability { .. } => "capability_required all known bits",
        AufError::RequiredSectionMissing { .. } => "Required section present",
        AufError::SectionTableTruncated => "Section table within file bounds",
        AufError::SectionRangeInvalid { .. } => "Section offset/size within file bounds",
        AufError::SectionOverlap { .. } => "Section ranges no overlap",
        AufError::DuplicateSectionTag { .. } => "Section tags unique",
        AufError::ContradictoryFlags { .. } => "Section flags valid",
        AufError::CompressedSectionUnsupported { .. } => "No COMPRESSED sections",
        _ => "AUF parsing",
    }
}

// ---------------------------------------------------------------------------
// 테스트
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_variants_all() {
        let tags = parse_variants("all").unwrap();
        assert_eq!(tags.len(), 3);
        assert!(tags.contains(&TAG_WEIGHTS_ADRENO_SOA));
        assert!(tags.contains(&TAG_WEIGHTS_CUDA_AOS));
        assert!(tags.contains(&TAG_WEIGHTS_CPU_AOS));
    }

    #[test]
    fn parse_variants_comma() {
        let tags = parse_variants("adreno_soa,cpu_aos").unwrap();
        assert_eq!(tags.len(), 2);
        assert!(tags.contains(&TAG_WEIGHTS_ADRENO_SOA));
        assert!(tags.contains(&TAG_WEIGHTS_CPU_AOS));
        assert!(!tags.contains(&TAG_WEIGHTS_CUDA_AOS));
    }

    #[test]
    fn parse_variants_full_tag_name() {
        let tags = parse_variants("WEIGHTS_CUDA_AOS").unwrap();
        assert_eq!(tags.len(), 1);
        assert!(tags.contains(&TAG_WEIGHTS_CUDA_AOS));
    }

    #[test]
    fn parse_variants_unknown_fails() {
        let result = parse_variants("unknown_backend");
        assert!(result.is_err());
    }

    #[test]
    fn parse_variants_dedup() {
        let tags = parse_variants("adreno_soa,adreno_soa,cpu_aos").unwrap();
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn normalize_tag_adreno() {
        assert_eq!(normalize_tag("adreno_soa"), TAG_WEIGHTS_ADRENO_SOA);
        assert_eq!(normalize_tag("ADRENO"), TAG_WEIGHTS_ADRENO_SOA);
        assert_eq!(normalize_tag("WEIGHTS_ADRENO_SOA"), TAG_WEIGHTS_ADRENO_SOA);
    }

    #[test]
    fn normalize_tag_pass_through() {
        assert_eq!(normalize_tag("META"), "META");
        assert_eq!(normalize_tag("TOKENIZER"), "TOKENIZER");
    }

    #[test]
    fn q4_0_aos_with_align_cpu() {
        let data = vec![1u8; 100];
        let out = q4_0_aos_with_align(&data, 64);
        assert_eq!(out.len(), 128); // ceil(100/64)*64
        assert_eq!(&out[..100], data.as_slice());
        assert_eq!(&out[100..], vec![0u8; 28].as_slice());
    }

    #[test]
    fn q4_0_aos_with_align_exact() {
        let data = vec![1u8; 64];
        let out = q4_0_aos_with_align(&data, 64);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn format_unix_time_epoch() {
        let s = format_unix_time(0);
        assert_eq!(s, "N/A");
    }

    #[test]
    fn format_unix_time_known() {
        // 2026-04-25 00:00:00 UTC = 1777075200
        let s = format_unix_time(1777075200);
        assert!(s.contains("2026"), "Expected 2026 in '{}'", s);
        assert!(s.contains("04"), "Expected month 04 in '{}'", s);
        assert!(s.contains("25"), "Expected day 25 in '{}'", s);
    }

    #[test]
    fn parse_include_lm_head_modes() {
        assert_eq!(
            IncludeLmHeadMode::parse("auto").unwrap(),
            IncludeLmHeadMode::Auto
        );
        assert_eq!(
            IncludeLmHeadMode::parse("AUTO").unwrap(),
            IncludeLmHeadMode::Auto
        );
        assert_eq!(
            IncludeLmHeadMode::parse("on").unwrap(),
            IncludeLmHeadMode::On
        );
        assert_eq!(
            IncludeLmHeadMode::parse("off").unwrap(),
            IncludeLmHeadMode::Off
        );
        assert_eq!(
            IncludeLmHeadMode::parse(" off ").unwrap(),
            IncludeLmHeadMode::Off
        );
        assert!(IncludeLmHeadMode::parse("invalid").is_err());
    }

    #[test]
    fn lm_head_q4_0_quantize_deterministic() {
        // 같은 F32 입력으로 두 번 quantize 했을 때 byte-level 동일한지 확인.
        // ENG-DAT-096.13 (writer 결정성) 핵심 invariant.
        let rows = 4usize;
        let cols = 64usize; // 2 blocks/row × 4 rows = 8 blocks → 144 bytes
        let mut f32_buf = vec![0.0f32; rows * cols];
        // pseudo-random 시퀀스 (deterministic seed).
        for (i, v) in f32_buf.iter_mut().enumerate() {
            *v = ((i as f32) * 0.0173).sin();
        }
        // F32 → little-endian byte slice.
        let raw_bytes: Vec<u8> = f32_buf.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape: Vec<u64> = vec![rows as u64, cols as u64];

        let out1 = quantize_lm_head_to_q4_0(&raw_bytes, &shape, GGML_TYPE_F32, true).unwrap();
        let out2 = quantize_lm_head_to_q4_0(&raw_bytes, &shape, GGML_TYPE_F32, true).unwrap();
        assert_eq!(out1, out2, "lm_head Q4_0 quantize must be deterministic");
        // 8 blocks × 18B = 144B
        assert_eq!(out1.len(), rows * cols / 32 * 18);
    }

    #[test]
    fn lm_head_q4_0_quantize_rejects_non_q4_0_aligned_cols() {
        let shape: Vec<u64> = vec![4, 17]; // 17 is not a multiple of 32
        let raw = vec![0u8; 4 * 17 * 4];
        let err = quantize_lm_head_to_q4_0(&raw, &shape, GGML_TYPE_F32, true).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("multiple of QK4_0"), "got: {msg}");
    }

    #[test]
    fn lm_head_q4_0_quantize_rejects_unsupported_dtype() {
        let shape: Vec<u64> = vec![1, 32];
        let raw = vec![0u8; 32 * 4];
        // GGML_TYPE_Q4_0 (already quantized) should NOT be re-quantized via this helper.
        let err = quantize_lm_head_to_q4_0(&raw, &shape, GGML_TYPE_Q4_0, true).unwrap_err();
        assert!(format!("{err}").contains("unsupported ggml_type"));
    }

    #[test]
    fn writer_with_lm_head_q4_0_sets_bit2_and_patch() {
        use argus_engine::auf::header::CAPABILITY_BIT_LM_HEAD_Q4_0;
        use argus_engine::auf::{AufHeader, AufMeta, AufTokenizer, AufWriter};

        let meta = AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 1,
            n_heads_q: 2,
            n_kv_heads: 1,
            head_dim: 4,
            hidden_dim: 8,
            ffn_dim: 16,
            vocab_size: 3,
            max_seq_len: 64,
            rope_theta: 10000.0,
            rotary_dim: 4,
            rope_scaling: 1.0,
            rms_norm_epsilon: 1e-5,
            default_dtype: None,
        };
        let tok = AufTokenizer {
            kind: TOKENIZER_KIND_BPE,
            tokens: vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
            merges: vec![],
            bos_id: 0,
            eos_id: 1,
            pad_id: -1,
            unk_id: -1,
            chat_template: None,
        };

        // (1) with_lm_head_q4_0(true) → bit 2 set, format_patch = 1.
        let bytes_on = AufWriter::new(meta.clone(), tok.clone(), [0u8; 32], 100, 200)
            .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, vec![1u8; 64])
            .with_lm_head_q4_0(true)
            .build()
            .unwrap();
        let hdr_on = AufHeader::from_bytes(&bytes_on).unwrap();
        assert_eq!(hdr_on.format_patch, 1);
        assert_ne!(
            hdr_on.capability_optional & CAPABILITY_BIT_LM_HEAD_Q4_0,
            0,
            "bit 2 must be set"
        );
        assert!(hdr_on.has_lm_head_q4_0());

        // (2) with_lm_head_q4_0(false) (default) → bit 2 clear, format_patch = 0.
        let bytes_off = AufWriter::new(meta, tok, [0u8; 32], 100, 200)
            .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, vec![1u8; 64])
            .build()
            .unwrap();
        let hdr_off = AufHeader::from_bytes(&bytes_off).unwrap();
        assert_eq!(hdr_off.format_patch, 0);
        assert_eq!(hdr_off.capability_optional & CAPABILITY_BIT_LM_HEAD_Q4_0, 0);
        assert!(!hdr_off.has_lm_head_q4_0());
    }

    #[test]
    fn writer_off_byte_level_compatible_with_v0_1_0() {
        // INV-136 회귀 방지: Off 모드 (capability bit 2 = 0)에서 작성된 AUF 헤더는
        // v0.1.0 출력과 동일한 format 필드를 가져야 한다 (format_patch=0, capability=0).
        use argus_engine::auf::{AufHeader, AufMeta, AufTokenizer, AufWriter};

        let meta = AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 1,
            n_heads_q: 2,
            n_kv_heads: 1,
            head_dim: 4,
            hidden_dim: 8,
            ffn_dim: 16,
            vocab_size: 3,
            max_seq_len: 64,
            rope_theta: 10000.0,
            rotary_dim: 4,
            rope_scaling: 1.0,
            rms_norm_epsilon: 1e-5,
            default_dtype: None,
        };
        let tok = AufTokenizer {
            kind: TOKENIZER_KIND_BPE,
            tokens: vec![b"a".to_vec()],
            merges: vec![],
            bos_id: 0,
            eos_id: 0,
            pad_id: -1,
            unk_id: -1,
            chat_template: None,
        };
        // 명시적으로 false 호출 vs 기본값 모두 동일 결과여야 함.
        let bytes_default = AufWriter::new(meta.clone(), tok.clone(), [0u8; 32], 100, 200)
            .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, vec![0xAAu8; 64])
            .build()
            .unwrap();
        let bytes_explicit_off = AufWriter::new(meta, tok, [0u8; 32], 100, 200)
            .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, vec![0xAAu8; 64])
            .with_lm_head_q4_0(false)
            .build()
            .unwrap();
        assert_eq!(
            bytes_default, bytes_explicit_off,
            "with_lm_head_q4_0(false) must match default"
        );
        let hdr = AufHeader::from_bytes(&bytes_default).unwrap();
        assert_eq!(hdr.format_major, 0);
        assert_eq!(hdr.format_minor, 1);
        assert_eq!(hdr.format_patch, 0);
        assert_eq!(hdr.capability_required, 0);
        assert_eq!(hdr.capability_optional, 0);
    }

    // ── Sprint G-1-B fix: tied embedding lm_head source ─────────────────────

    #[test]
    fn tied_model_uses_token_embd_as_lm_head_source() {
        // Llama 3.2 1B: GGUF에 output.weight 없음 (tied) → token_embd.weight를
        // lm_head source로 선택해야 한다. is_tied = true.
        let src = select_lm_head_source(false, true).expect("tied source must be selected");
        assert_eq!(src.source_name, LM_HEAD_TIED_SOURCE_NAME);
        assert_eq!(src.source_name, "token_embd.weight");
        assert!(src.is_tied);
    }

    #[test]
    fn separate_model_uses_output_weight_as_lm_head_source() {
        // Qwen / Llama 3 8B: GGUF에 output.weight 있음 → 그것을 source로 선택.
        // is_tied = false. token_embd 존재 여부와 관계없이 separate 우선.
        let src = select_lm_head_source(true, true).expect("separate source must be selected");
        assert_eq!(src.source_name, LM_HEAD_SEPARATE_NAME);
        assert_eq!(src.source_name, "output.weight");
        assert!(!src.is_tied);

        // token_embd 없어도 separate만 있으면 동작 (이론적 케이스).
        let src2 = select_lm_head_source(true, false).expect("separate-only must work");
        assert_eq!(src2.source_name, LM_HEAD_SEPARATE_NAME);
        assert!(!src2.is_tied);
    }

    #[test]
    fn no_lm_head_source_returns_none() {
        // 둘 다 없는 안전망: None 반환.
        let src = select_lm_head_source(false, false);
        assert!(src.is_none());
    }

    #[test]
    fn tied_model_quantize_deterministic() {
        // tied 케이스에서 token_embd raw bytes (F16) → quantize_lm_head_to_q4_0
        // 두 번 호출 시 byte-level identical 출력.
        // separate 케이스와 동일 helper를 사용하므로 결정성이 자동으로 보장되지만,
        // tied 시나리오에서도 명시적으로 검증한다 (Sprint G-1-B fix 회귀 방지).
        use half::f16;

        let rows: usize = 4; // vocab
        let cols: usize = 64; // hidden, multiple of 32
        // Pseudo-random F16 입력 (deterministic seed; tied source는 일반적으로 F16).
        let f16_data: Vec<u16> = (0..rows * cols)
            .map(|i| f16::from_f32(((i as f32) * 0.0173).cos()).to_bits())
            .collect();
        let raw_bytes: Vec<u8> = f16_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape: Vec<u64> = vec![rows as u64, cols as u64];

        let out1 = quantize_lm_head_to_q4_0(&raw_bytes, &shape, GGML_TYPE_F16, true).unwrap();
        let out2 = quantize_lm_head_to_q4_0(&raw_bytes, &shape, GGML_TYPE_F16, true).unwrap();
        assert_eq!(
            out1, out2,
            "tied lm_head Q4_0 quantize must be deterministic"
        );
        assert_eq!(out1.len(), rows * cols / 32 * 18); // 8 blocks × 18B = 144B
    }

    #[test]
    fn tied_model_lm_head_q4_0_dtype_check() {
        // tied 케이스 quantize 결과의 size invariant: rows*cols / QK4_0 * 18.
        // dtype 자체는 Vec<u8>지만, 크기로 Q4_0 layout (block 18B) 검증.
        let rows: usize = 8;
        let cols: usize = 32;
        let f32_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001 - 0.5).collect();
        let raw: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape: Vec<u64> = vec![rows as u64, cols as u64];

        let out = quantize_lm_head_to_q4_0(&raw, &shape, GGML_TYPE_F32, true).unwrap();
        // 1 block per row × 8 rows = 8 blocks × 18B = 144B.
        assert_eq!(out.len(), 8 * 18);
        // Q4_0 block layout: 2B scale (f16) + 16B nibbles. 첫 block size 검증.
        // (raw 데이터가 0 근처라 scale도 작은 f16. byte content 자체는 결정적이지만 비교는 size로 충분.)
        assert!(
            out.len().is_multiple_of(18),
            "output must be multiple of 18B"
        );
    }

    // ── G-1-F fix: lm_head ADRENO_SOA AOS exception (INV-135 v2) ───────────────

    /// `build_variant_payload(TAG_WEIGHTS_ADRENO_SOA, ...)` 호출 시 lm_head는 SOA
    /// 변환을 skip하고 AOS bytes를 그대로 emit해야 한다 (image1d_buffer_t 한계 회피).
    /// 다른 layer weight는 SOA 변환을 그대로 적용.
    #[test]
    fn build_variant_payload_skips_soa_for_lm_head() {
        use argus_engine::auf::q4_0_aos_to_adreno_soa;
        use argus_engine::auf::section::TAG_WEIGHTS_ADRENO_SOA;

        // VOCAB=64, HIDDEN=128 → 64*128/32 = 256 blocks * 18B = 4608B.
        const VOCAB: usize = 64;
        const HIDDEN: usize = 128;
        const NUM_BLOCKS: usize = VOCAB * HIDDEN / 32;
        let lm_head_aos: Vec<u8> = (0..NUM_BLOCKS * 18).map(|i| (i % 256) as u8).collect();

        // Layer weight (FFN gate) — 32×64 = 64 blocks.
        const LAYER_ROWS: usize = 32;
        const LAYER_COLS: usize = 64;
        const LAYER_BLOCKS: usize = LAYER_ROWS * LAYER_COLS / 32;
        let layer_aos: Vec<u8> = (0..LAYER_BLOCKS * 18)
            .map(|i| ((i * 13 + 7) % 256) as u8)
            .collect();

        let blobs: Vec<WeightBlob> = vec![
            WeightBlob::single(
                "blk.0.ffn_gate.weight".to_owned(),
                vec![LAYER_ROWS as u64, LAYER_COLS as u64],
                TensorDType::Q4_0,
                layer_aos.clone(),
            ),
            WeightBlob::single(
                "output.weight".to_owned(), // = LM_HEAD_SEPARATE_NAME, lm_head 식별
                vec![VOCAB as u64, HIDDEN as u64],
                TensorDType::Q4_0,
                lm_head_aos.clone(),
            ),
        ];

        let payload = build_variant_payload(&blobs, TAG_WEIGHTS_ADRENO_SOA).unwrap();

        // Layer weight 부분: SOA 변환 적용.
        let (layer_q, layer_d) = q4_0_aos_to_adreno_soa(&layer_aos, LAYER_COLS, LAYER_ROWS);
        let layer_soa_len = layer_q.len() + layer_d.len();
        assert_eq!(
            layer_soa_len,
            LAYER_BLOCKS * 18,
            "SOA total size = AOS total size invariant"
        );
        assert_eq!(&payload[..layer_q.len()], &layer_q[..]);
        assert_eq!(
            &payload[layer_q.len()..layer_q.len() + layer_d.len()],
            &layer_d[..]
        );

        // lm_head 부분: AOS bytes 그대로 (SOA 변환 미적용).
        let lm_head_start = layer_soa_len;
        assert_eq!(
            &payload[lm_head_start..lm_head_start + lm_head_aos.len()],
            &lm_head_aos[..],
            "lm_head must be emitted as raw AOS bytes (no SOA transform)"
        );

        // 총 byte 길이는 N*18 invariant 유지.
        assert_eq!(payload.len(), layer_soa_len + lm_head_aos.len());
    }

    /// `build_variant_payload`는 layer weight (kind != lm_head)에 대해서는 SOA 변환을
    /// 그대로 적용. 이는 G-1-F fix가 lm_head 한정 예외임을 검증.
    #[test]
    fn build_variant_payload_applies_soa_for_layer_weights() {
        use argus_engine::auf::q4_0_aos_to_adreno_soa;
        use argus_engine::auf::section::TAG_WEIGHTS_ADRENO_SOA;

        const ROWS: usize = 32;
        const COLS: usize = 64;
        let aos: Vec<u8> = (0..ROWS * COLS / 32 * 18)
            .map(|i| (i % 256) as u8)
            .collect();

        let blobs: Vec<WeightBlob> = vec![WeightBlob::single(
            "blk.5.attn_v.weight".to_owned(),
            vec![ROWS as u64, COLS as u64],
            TensorDType::Q4_0,
            aos.clone(),
        )];

        let payload = build_variant_payload(&blobs, TAG_WEIGHTS_ADRENO_SOA).unwrap();
        let (q, d) = q4_0_aos_to_adreno_soa(&aos, COLS, ROWS);

        assert_eq!(&payload[..q.len()], &q[..]);
        assert_eq!(&payload[q.len()..q.len() + d.len()], &d[..]);
        assert_eq!(payload.len(), q.len() + d.len());
        // SOA-converted bytes는 원본 AOS bytes와 다름이 정상 (sanity).
        assert_ne!(&payload[..aos.len()], &aos[..]);
    }

    #[test]
    fn round_trip_strip_via_auf_api() {
        use argus_engine::auf::{AufMeta, AufTokenizer, AufWriter, strip_bytes};

        let meta = AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 1,
            n_heads_q: 2,
            n_kv_heads: 1,
            head_dim: 4,
            hidden_dim: 8,
            ffn_dim: 16,
            vocab_size: 3,
            max_seq_len: 64,
            rope_theta: 10000.0,
            rotary_dim: 4,
            rope_scaling: 1.0,
            rms_norm_epsilon: 1e-5,
            default_dtype: None,
        };
        let tok = AufTokenizer {
            kind: TOKENIZER_KIND_BPE,
            tokens: vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
            merges: vec![],
            bos_id: 0,
            eos_id: 1,
            pad_id: -1,
            unk_id: -1,
            chat_template: None,
        };
        let original = AufWriter::new(meta, tok, [0u8; 32], 100, 200)
            .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, vec![1u8; 128])
            .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![2u8; 64])
            .build()
            .unwrap();

        // ADRENO만 유지
        let stripped = strip_bytes(&original, &[TAG_WEIGHTS_ADRENO_SOA]).unwrap();

        // stripped에서 CPU_AOS section이 없어야 함
        use argus_engine::auf::header::AufHeader;
        use argus_engine::auf::section::SectionTable;
        let hdr = AufHeader::from_bytes(&stripped).unwrap();
        let tbl = SectionTable::from_bytes(
            &stripped[hdr.section_table_offset as usize..],
            hdr.section_count,
        )
        .unwrap();
        let tags: Vec<_> = tbl.entries.iter().map(|e| e.tag().to_owned()).collect();
        assert!(tags.contains(&"WEIGHTS_ADRENO_SOA".to_owned()));
        assert!(!tags.contains(&"WEIGHTS_CPU_AOS".to_owned()));
        assert_eq!(hdr.section_count, 4); // META + TOKENIZER + TENSOR_INDEX + ADRENO_SOA
    }
}

#[cfg(test)]
mod tokenizer_config_tests {
    //! W-AUF-1 C5: `resolve_special_ids_from_value` 단위 검증.
    use super::resolve_special_ids_from_value;
    use serde_json::json;

    fn vocab_with(pairs: &[(&'static str, i32)]) -> impl Fn(&str) -> Option<i32> {
        let owned: Vec<(String, i32)> = pairs.iter().map(|(s, i)| ((*s).to_owned(), *i)).collect();
        move |needle: &str| owned.iter().find(|(s, _)| s == needle).map(|(_, id)| *id)
    }

    #[test]
    fn integer_fields_take_precedence() {
        let v = json!({
            "bos_token_id": 1,
            "eos_token_id": 2,
            "bos_token": "<bos-str>",  // ignored when *_id present
        });
        let lookup = vocab_with(&[("<bos-str>", 999)]);
        let (bos, eos) = resolve_special_ids_from_value(&v, &lookup);
        assert_eq!(bos, Some(1));
        assert_eq!(eos, Some(2));
    }

    #[test]
    fn string_token_resolved_via_vocab() {
        let v = json!({
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
        });
        let lookup = vocab_with(&[("<|begin_of_text|>", 128000), ("<|eot_id|>", 128009)]);
        let (bos, eos) = resolve_special_ids_from_value(&v, &lookup);
        assert_eq!(bos, Some(128000));
        assert_eq!(eos, Some(128009));
    }

    #[test]
    fn object_token_prefers_inline_id() {
        let v = json!({
            "eos_token": { "content": "<|im_end|>", "id": 151643 },
        });
        // inline id가 있으면 vocab_lookup 없이도 해석돼야 한다.
        let lookup = vocab_with(&[]);
        let (_bos, eos) = resolve_special_ids_from_value(&v, &lookup);
        assert_eq!(eos, Some(151643));
    }

    #[test]
    fn object_token_falls_back_to_content_lookup() {
        let v = json!({
            "eos_token": { "content": "<|im_end|>" },
        });
        let lookup = vocab_with(&[("<|im_end|>", 151645)]);
        let (_bos, eos) = resolve_special_ids_from_value(&v, &lookup);
        assert_eq!(eos, Some(151645));
    }

    #[test]
    fn missing_fields_return_none() {
        let v = json!({});
        let lookup = vocab_with(&[]);
        let (bos, eos) = resolve_special_ids_from_value(&v, &lookup);
        assert!(bos.is_none() && eos.is_none());
    }

    #[test]
    fn unknown_string_token_returns_none() {
        let v = json!({ "eos_token": "<not-in-vocab>" });
        let lookup = vocab_with(&[("<|im_end|>", 1)]);
        let (_bos, eos) = resolve_special_ids_from_value(&v, &lookup);
        assert!(eos.is_none());
    }
}
