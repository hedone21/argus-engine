//! Unit tests for the TriAttention faithful core + the per-layer stage scope-equivalence.

use super::*;
use argus_extension_api::{
    CacheHandle, FormatId, TensorDtype, TensorHandle, TensorShape, WeightedMerge,
    find_mutation_stage,
};

// ───────────────────────────── calib ─────────────────────────────

/// Build a flat calibration binary (the P1 format) from a per-(layer,head,field,freq) fill closure.
fn mk_calib_bytes(
    num_layers: usize,
    num_heads: usize,
    freq_count: usize,
    fill: impl Fn(usize, usize, usize, usize) -> f32, // (layer, head, field 0..3, freq) -> value
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"TACALIB1");
    for v in [num_layers as u32, num_heads as u32, freq_count as u32, 0u32] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for l in 0..num_layers {
        for h in 0..num_heads {
            for field in 0..3 {
                for f in 0..freq_count {
                    buf.extend_from_slice(&fill(l, h, field, f).to_le_bytes());
                }
            }
        }
    }
    buf
}

#[test]
fn calib_roundtrip_bit_preserving() {
    let fill = |l: usize, h: usize, field: usize, f: usize| {
        (l as f32) * 100.0 + (h as f32) * 10.0 + (field as f32) + (f as f32) * 0.25
    };
    let raw = mk_calib_bytes(2, 3, 4, fill);
    let c = Calib::from_bytes(&raw).unwrap();
    assert_eq!((c.num_layers, c.num_heads, c.freq_count), (2, 3, 4));
    for l in 0..2 {
        for h in 0..3 {
            assert_eq!(
                c.q_mean_real(l, h),
                &(0..4).map(|f| fill(l, h, 0, f)).collect::<Vec<_>>()[..]
            );
            assert_eq!(
                c.q_mean_imag(l, h),
                &(0..4).map(|f| fill(l, h, 1, f)).collect::<Vec<_>>()[..]
            );
            assert_eq!(
                c.q_abs_mean(l, h),
                &(0..4).map(|f| fill(l, h, 2, f)).collect::<Vec<_>>()[..]
            );
        }
    }
}

#[test]
fn calib_bad_magic_rejected() {
    assert!(Calib::from_bytes(b"NOPE").is_err());
    assert!(Calib::from_bytes(b"TACALIB1\x02\0\0\0\x02\0\0\0\x04\0\0\0\0\0\0\0").is_err()); // body too short
}

// ───────────────────────────── rope / offsets ─────────────────────────────

#[test]
fn inv_freq_and_offsets() {
    let iv = inv_freq(4, 10000.0);
    assert_eq!(iv.len(), 2);
    assert!((iv[0] - 1.0).abs() < 1e-6); // θ^0
    assert!((iv[1] - 0.01).abs() < 1e-6); // 1/10000^(2/4) = 1/100
    let off = build_geometric_offsets(65536);
    assert_eq!(off.len(), 17); // 2^0 .. 2^16
    assert_eq!(off[0], 1.0);
    assert_eq!(*off.last().unwrap(), 65536.0);
}

/// Forward RoPE (engine "half" split) then [`invert_rope_complex`] recovers the base complex key.
#[test]
fn rope_invert_recovers_base() {
    let iv = inv_freq(4, 10000.0);
    let fc = iv.len();
    let base = [0.3f32, -1.2, 0.7, 2.1]; // real=[0.3,-1.2], imag=[0.7,2.1]
    let pos = 5usize;
    // forward: rotated[f]=b[f]cos - b[f+fc]sin ; rotated[f+fc]=b[f]sin + b[f+fc]cos
    let mut rot = [0.0f32; 4];
    for f in 0..fc {
        let (s, c) = ((pos as f32) * iv[f]).sin_cos();
        rot[f] = base[f] * c - base[f + fc] * s;
        rot[f + fc] = base[f] * s + base[f + fc] * c;
    }
    let (real, imag) = invert_rope_complex(&rot, pos, &iv);
    for f in 0..fc {
        assert!((real[f] - base[f]).abs() < 1e-5, "real[{f}]");
        assert!((imag[f] - base[f + fc]).abs() < 1e-5, "imag[{f}]");
    }
}

// ───────────────────────────── selection ─────────────────────────────

#[test]
fn select_union_based_small_hand_case() {
    // 2 heads, 4 candidates, keep_count=2.
    let rows = vec![vec![1.0, 4.0, 2.0, 3.0], vec![5.0, 1.0, 1.0, 1.0]];
    let combined = vec![5.0, 4.0, 2.0, 3.0]; // max over heads
    // head0 top2 -> {1,3}; head1 top2 -> {0,1}; union {0,1,3};
    // top2 of union by combined -> {0(5),1(4)} -> ascending [0,1].
    assert_eq!(select_union_based(&rows, &combined, 2), vec![0, 1]);
}

#[test]
fn select_union_returns_all_when_within_keep() {
    let rows = vec![vec![1.0, 2.0, 3.0]];
    let combined = vec![1.0, 2.0, 3.0];
    assert_eq!(select_union_based(&rows, &combined, 5), vec![0, 1, 2]);
}

#[test]
fn decode_partition_guards() {
    // within budget -> keep all
    assert_eq!(decode_partition(5, 8, 2), Ok((0..5).collect()));
    // decode range present
    assert_eq!(decode_partition(10, 6, 2), Err((2, 8, 4)));
    // budget exhausted by prefill (decode_budget==0) -> first min(budget, prefix)
    assert_eq!(decode_partition(10, 2, 4), Ok(vec![0, 1]));
}

// ───────────────────────────── stage scope-equivalence ─────────────────────────────

/// A Key tensor handle over `keys[kv_head][pos]` (post-RoPE), per_head.
struct KeyHandle<'a> {
    keys: &'a [Vec<Vec<f32>>], // [kv_head][pos][head_dim]
    rows: usize,
    head_dim: usize,
}
impl TensorHandle for KeyHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.rows,
            cols: self.head_dim,
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        out.copy_from_slice(&self.keys[kv_head][row]);
    }
}

struct Ctx<'a> {
    current: usize,
    target_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    keys: &'a [Vec<Vec<f32>>],
}
impl StageCtx for Ctx<'_> {
    fn current_pos(&self) -> usize {
        self.current
    }
    fn target_len(&self) -> usize {
        self.target_len
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
            TensorKind::Key => Some(Box::leak(Box::new(KeyHandle {
                keys: self.keys,
                rows: self.current,
                head_dim: self.head_dim,
            })) as &dyn TensorHandle),
            _ => None,
        }
    }
}

#[derive(Default)]
struct CaptureHandle {
    cur: usize,
    n_kv: usize,
    head_dim: usize,
    kept: Option<Vec<usize>>,
}
impl CacheHandle for CaptureHandle {
    fn current_pos(&self) -> usize {
        self.cur
    }
    fn n_kv_heads(&self) -> usize {
        self.n_kv
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn kv_on_device(&self) -> bool {
        false
    }
    fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
        None
    }
    fn keep(&mut self, keep: &[usize]) -> Result<(), CacheOpError> {
        self.kept = Some(keep.to_vec());
        Ok(())
    }
    fn keep_per_head(&mut self, _keep: &[&[usize]]) -> Result<(), CacheOpError> {
        Ok(())
    }
    fn merge(&mut self, _merges: &[WeightedMerge]) -> Result<(), CacheOpError> {
        Ok(())
    }
    fn reencode(&mut self, _target: FormatId) -> Result<(), CacheOpError> {
        Ok(())
    }
    fn transition_quant_bits(&mut self, _bits: u8) -> Result<(), CacheOpError> {
        Ok(())
    }
    fn offload(&mut self, _prefix_len: usize) -> Result<(), CacheOpError> {
        Ok(())
    }
    fn recall(&mut self) -> Result<(), CacheOpError> {
        Ok(())
    }
}

/// For a SINGLE-layer model the per-layer `on_phase` reproduces the GLOBAL core's keep-set exactly
/// (the cross-layer max is trivially this one layer). Pins the cross-layer decision boundary: the
/// stage's per-layer scope == the global target when n_layers == 1.
#[test]
fn on_phase_single_layer_matches_global() {
    let (num_heads, num_kv, head_dim, l_total) = (2usize, 1usize, 4usize, 8usize);
    let fc = head_dim / 2;
    let (prefix, budget, theta) = (2usize, 5usize, 10000.0f32);

    // calib: 2 heads, distinct stats.
    let raw = mk_calib_bytes(1, num_heads, fc, |_l, h, field, f| {
        0.1 + (h as f32) * 0.7 + (field as f32) * 0.3 + (f as f32) * 0.2
    });
    let calib = Calib::from_bytes(&raw).unwrap();

    // keys[kv_head][pos][head_dim] — deterministic, position-varied (post-RoPE shape; values arbitrary).
    let keys_kv: Vec<Vec<Vec<f32>>> = (0..num_kv)
        .map(|kv| {
            (0..l_total)
                .map(|pos| {
                    (0..head_dim)
                        .map(|d| ((pos * 7 + d * 3 + kv * 5) % 11) as f32 * 0.31 - 1.4)
                        .collect()
                })
                .collect()
        })
        .collect();

    let params = TriParams {
        budget,
        prefix_length: prefix,
        round_start: l_total,
        head_dim,
        num_kv_heads: num_kv,
        num_kv_groups: num_heads / num_kv,
        theta,
        offset_max_length: 8,
        normalize_scores: false,
    };
    let positions: Vec<usize> = (0..l_total).collect();
    let keys_layered = vec![keys_kv.clone()]; // [1 layer][kv][pos][dim]
    let global = compute_keepset_global(&keys_layered, &calib, &params, &positions);

    // The stage path over the same keys/calib/config.
    let stage = TriAttention::with_calib(calib, prefix, 8, false, theta);
    let ctx = Ctx {
        current: l_total,
        target_len: budget,
        n_kv_heads: num_kv,
        head_dim,
        keys: &keys_kv,
    };
    let mut h = CaptureHandle {
        cur: l_total,
        n_kv: num_kv,
        head_dim,
        ..Default::default()
    };
    stage.on_phase(&ctx, &mut h).unwrap();

    assert_eq!(
        h.kept.unwrap(),
        global,
        "per-layer on_phase must match the global core for 1 layer"
    );
    // sanity: prefill pinned + budget respected.
    assert_eq!(global.len(), budget);
    assert_eq!(&global[..prefix], &[0, 1]);
}

#[test]
fn registered_in_slice_with_key_caps() {
    let reg = find_mutation_stage("triattention").expect("triattention registered");
    assert_eq!(reg.name, "triattention");
    assert_eq!(reg.caps.reads, &[TensorKind::Key]);
    assert_eq!(reg.phase, MutationPhase::KvMutate);
}
