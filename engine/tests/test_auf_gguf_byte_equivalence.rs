//! B-3a — AUF primary path가 GGUF primary path와 동일한 (shape, dtype, bytes)를
//! 생성하는지 호스트에서 검증.
//!
//! 본 테스트는 device garbage 이슈를 byte-level diff로 격리하기 위한 회귀 가드.
//! `models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf` (GGUF baseline) 과
//! `models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-aos.auf` (AUF CPU_AOS) 를 둘 다 열어
//! 4종 텐서를 (shape, dtype, byte len, byte content) 모두 비교한다.
//!
//! 미설정 환경(파일 부재)에서는 graceful skip.

use std::path::PathBuf;
use std::sync::Arc;

use argus_engine::backend::Backend;
use argus_engine::backend::cpu::CpuBackend;
use argus_engine::memory::galloc::Galloc;
use argus_engine::models::loader::auf::{AufDtypeChoice, AufSource, AufVariantChoice};
use argus_engine::models::loader::gguf::GgufSource;
use argus_engine::models::loader::{LayerBiasKind, LayerWeightKind, TensorId, TensorSource};
use argus_engine::tensor::Tensor;

const GGUF_REL: &str = "models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf";
const AUF_REL: &str = "models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-aos.auf";

/// crate root(engine/) 의 부모 디렉토리(workspace root)에서 fixture를 찾는다.
/// `CARGO_MANIFEST_DIR`은 컴파일 시점에 engine/ 경로로 고정된다.
fn fixture_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // engine -> workspace root
    p.push(rel);
    p
}

fn fixtures_present() -> (PathBuf, PathBuf, bool) {
    let g = fixture_path(GGUF_REL);
    let a = fixture_path(AUF_REL);
    let ok = g.exists() && a.exists();
    (g, a, ok)
}

fn tensor_bytes(t: &Tensor) -> &[u8] {
    let buf = t.buffer();
    unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.size()) }
}

/// 4종 텐서를 GGUF vs AUF로 로드해 (shape, dtype, bytes) 모두 일치하는지 확인.
/// 첫 mismatch 시 어디서 어긋났는지 명확히 격리 가능하도록 텐서마다 별도 assertion.
#[test]
fn auf_gguf_byte_equivalence_qwen25_1_5b() {
    let (gguf_path, auf_path, present) = fixtures_present();
    if !present {
        eprintln!(
            "[skip] auf_gguf_byte_equivalence: fixtures missing\n  gguf={}\n  auf={}",
            gguf_path.display(),
            auf_path.display(),
        );
        return;
    }

    let gguf = GgufSource::open(&gguf_path).expect("open GGUF");
    let auf = AufSource::open(
        &auf_path,
        AufVariantChoice::CpuAos,
        AufDtypeChoice::Auto,
        None,
    )
    .expect("open AUF CPU_AOS");

    let mem = Galloc;
    let _cpu: Arc<dyn Backend> = Arc::new(CpuBackend::new());

    let probes: &[(&str, TensorId)] = &[
        (
            "blk.0.attn_q.weight (Q4_0)",
            TensorId::LayerWeight {
                layer: 0,
                kind: LayerWeightKind::Wq,
            },
        ),
        (
            "blk.0.attn_norm.weight (F32)",
            TensorId::LayerWeight {
                layer: 0,
                kind: LayerWeightKind::AttentionNorm,
            },
        ),
        ("token_embd.weight (F16/embed)", TensorId::Embed),
        (
            "blk.0.ffn_gate.weight (Q4_0 비대칭)",
            TensorId::LayerWeight {
                layer: 0,
                kind: LayerWeightKind::WGate,
            },
        ),
    ];

    let mut failures: Vec<String> = Vec::new();

    for (label, id) in probes {
        let g = match gguf.load_tensor_cpu(id, true, &mem) {
            Ok(t) => t,
            Err(e) => {
                failures.push(format!("[{label}] GGUF load failed: {e}"));
                continue;
            }
        };
        let a = match auf.load_tensor_cpu(id, true, &mem) {
            Ok(t) => t,
            Err(e) => {
                failures.push(format!("[{label}] AUF load failed: {e}"));
                continue;
            }
        };

        let g_shape = g.shape().dims().to_vec();
        let a_shape = a.shape().dims().to_vec();
        let g_dtype = g.buffer().dtype();
        let a_dtype = a.buffer().dtype();
        let g_bytes = tensor_bytes(&g);
        let a_bytes = tensor_bytes(&a);

        eprintln!(
            "[{label}] GGUF shape={:?} dtype={:?} bytes={} | AUF shape={:?} dtype={:?} bytes={}",
            g_shape,
            g_dtype,
            g_bytes.len(),
            a_shape,
            a_dtype,
            a_bytes.len(),
        );

        if g_shape != a_shape {
            failures.push(format!(
                "[{label}] shape mismatch: gguf={:?} auf={:?}",
                g_shape, a_shape
            ));
            continue;
        }
        if g_dtype != a_dtype {
            failures.push(format!(
                "[{label}] dtype mismatch: gguf={:?} auf={:?}",
                g_dtype, a_dtype
            ));
            continue;
        }
        if g_bytes.len() != a_bytes.len() {
            failures.push(format!(
                "[{label}] byte len mismatch: gguf={} auf={}",
                g_bytes.len(),
                a_bytes.len()
            ));
            continue;
        }

        let mut first_diff: Option<(usize, u8, u8)> = None;
        let mut diff_count: usize = 0;
        for (i, (gb, ab)) in g_bytes.iter().zip(a_bytes.iter()).enumerate() {
            if gb != ab {
                if first_diff.is_none() {
                    first_diff = Some((i, *gb, *ab));
                }
                diff_count += 1;
            }
        }
        if let Some((off, gb, ab)) = first_diff {
            failures.push(format!(
                "[{label}] byte diff at offset {off}: gguf=0x{gb:02x} auf=0x{ab:02x} \
                 ({diff_count} bytes differ of {} total)",
                g_bytes.len()
            ));
        } else {
            eprintln!("[{label}] OK (byte-identical, {} bytes)", g_bytes.len());
        }
    }

    if !failures.is_empty() {
        panic!(
            "auf_gguf_byte_equivalence_qwen25_1_5b: {} probe(s) failed:\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }
}

/// 전 layer × 전 kind + cross-layer 전수 비교. 첫 mismatch에서 panic.
/// 4종 sanity 통과 후에도 device garbage가 지속될 때 범위를 좁히는 진단용.
#[test]
fn auf_gguf_byte_equivalence_qwen25_1_5b_full_sweep() {
    let (gguf_path, auf_path, present) = fixtures_present();
    if !present {
        eprintln!(
            "[skip] full_sweep: fixtures missing\n  gguf={}\n  auf={}",
            gguf_path.display(),
            auf_path.display(),
        );
        return;
    }

    let gguf = GgufSource::open(&gguf_path).expect("open GGUF");
    let auf = AufSource::open(
        &auf_path,
        AufVariantChoice::CpuAos,
        AufDtypeChoice::Auto,
        None,
    )
    .expect("open AUF CPU_AOS");

    let mem = Galloc;
    let n_layers = gguf.config().num_hidden_layers;

    let per_layer = [
        ("Wq", LayerWeightKind::Wq),
        ("Wk", LayerWeightKind::Wk),
        ("Wv", LayerWeightKind::Wv),
        ("Wo", LayerWeightKind::Wo),
        ("WGate", LayerWeightKind::WGate),
        ("WUp", LayerWeightKind::WUp),
        ("WDown", LayerWeightKind::WDown),
        ("AttnNorm", LayerWeightKind::AttentionNorm),
        ("FfnNorm", LayerWeightKind::FfnNorm),
    ];

    let cross: &[(&str, TensorId)] = &[
        ("Embed", TensorId::Embed),
        ("FinalNorm", TensorId::FinalNorm),
        ("LmHead", TensorId::LmHead),
    ];

    let mut first_failure: Option<String> = None;
    let mut compared: usize = 0;

    let mut probe = |label: String, id: &TensorId| {
        if first_failure.is_some() {
            return;
        }
        let g_has = gguf.has_tensor(id);
        let a_has = auf.has_tensor(id);
        if g_has != a_has {
            first_failure = Some(format!(
                "[{label}] has_tensor mismatch: gguf={g_has} auf={a_has}"
            ));
            return;
        }
        if !g_has {
            return; // both missing — OK (e.g. tied lm_head)
        }
        let g = match gguf.load_tensor_cpu(id, true, &mem) {
            Ok(t) => t,
            Err(e) => {
                first_failure = Some(format!("[{label}] GGUF load failed: {e}"));
                return;
            }
        };
        let a = match auf.load_tensor_cpu(id, true, &mem) {
            Ok(t) => t,
            Err(e) => {
                first_failure = Some(format!("[{label}] AUF load failed: {e}"));
                return;
            }
        };
        let g_shape = g.shape().dims().to_vec();
        let a_shape = a.shape().dims().to_vec();
        let g_dtype = g.buffer().dtype();
        let a_dtype = a.buffer().dtype();
        let g_bytes = tensor_bytes(&g);
        let a_bytes = tensor_bytes(&a);
        if g_shape != a_shape || g_dtype != a_dtype || g_bytes.len() != a_bytes.len() {
            first_failure = Some(format!(
                "[{label}] header mismatch: gguf shape={g_shape:?} dtype={g_dtype:?} bytes={} | auf shape={a_shape:?} dtype={a_dtype:?} bytes={}",
                g_bytes.len(),
                a_bytes.len(),
            ));
            return;
        }
        for (i, (gb, ab)) in g_bytes.iter().zip(a_bytes.iter()).enumerate() {
            if gb != ab {
                first_failure = Some(format!(
                    "[{label}] byte diff at offset {i}: gguf=0x{gb:02x} auf=0x{ab:02x} (shape={g_shape:?} dtype={g_dtype:?} total_bytes={})",
                    g_bytes.len()
                ));
                return;
            }
        }
        compared += 1;
    };

    for (name, id) in cross {
        probe(name.to_string(), id);
    }
    for layer in 0..n_layers {
        for (name, kind) in &per_layer {
            probe(
                format!("blk.{layer}.{name}"),
                &TensorId::LayerWeight { layer, kind: *kind },
            );
        }
        // Bias 계열 (Qwen2 등). GGUF에 존재하면 AUF에도 존재해야 한다.
        for (name, kind) in [
            ("Bq", LayerBiasKind::Bq),
            ("Bk", LayerBiasKind::Bk),
            ("Bv", LayerBiasKind::Bv),
        ] {
            probe(
                format!("blk.{layer}.{name}"),
                &TensorId::LayerBias { layer, kind },
            );
        }
    }

    eprintln!(
        "[full_sweep] compared={} tensors (gguf has_tensor && auf has_tensor)",
        compared
    );
    if let Some(msg) = first_failure {
        panic!("full_sweep first mismatch:\n  {msg}");
    }
}
