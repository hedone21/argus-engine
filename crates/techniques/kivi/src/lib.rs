//! kivi — Backend 축(3번째 axis) capability dlopen plugin 의 **합성/ABI 검증 전용** 구현 (design D8).
//!
//! `register_quant_attn_plugin!` 의 두 번째(그리고 첫 *비-example*) 실사용자로서, backend-cap
//! 축의 ABI round-trip(등록 → category 다리 → vtable 디스패치 → make/drop)을 end-to-end 로 증명한다.
//!
//! **GPU 수학을 하지 않는다(synthetic).** `attention_gen_quant` 는 [`QuantAttnArgs`] 스칼라로 결정적
//! sentinel 을 계산해 `scores_out[0]` 에 기록할 뿐이라, host 게이트가 args struct 의 필드 정렬·값이
//! ABI 경계를 정확히 넘었는지 확인할 수 있다. make 인자(`cl_ctx` 등)는 무시 → GPU 자원 비보유.
//!
//! **실제 KIVI 커널·캐시는 엔진에 그대로 남는다** — `backend/opencl.rs`(kivi_q2/kivi_attn 커널),
//! `kv/kivi_cache.rs`·`kivi_format.rs`·`kivi_forward.rs`, `kv/quantize_handler.rs` 의
//! `target_bits_for_pressure`(spec 보존 진입점)는 이 플러그인과 무관하며 건드리지 않는다.
//!
//! **오용 주의:** `--backend-cap kivi_abi` 는 라이브 attention 백엔드를 *교체*하므로(eviction plugin 이
//! plan-only 인 것과 다름), 이 합성 구현이 decode hot-path 에서 실행되면 sentinel 만 쓰고 실제 attention 을
//! 하지 않아 **출력이 깨진다**. 이름의 `_abi` 접미사가 이 플러그인이 ABI 검증 전용임을 알린다.
//! dlopen-only(엔진 force-link 안 함)라 `--load-plugin` 으로만 opt-in 된다.

use argus_extension_api::{
    QuantAttnArgs, QuantAttnBackend, QuantAttnGatherArgs, QuantAttnMakeArgs,
};

/// 합성 capability — 상태 없음(GPU 자원 비보유). 핸들 lifecycle(make/drop) round-trip 만 운반.
struct KiviAbiBackend;

impl QuantAttnBackend for KiviAbiBackend {
    fn has_quant_attn_kernel(&self, bits: u8) -> bool {
        // synthetic: KIVI 가 지원하는 2/4/8-bit 모두 "보유"한다고 보고(실 커널 없음).
        matches!(bits, 2 | 4 | 8)
    }

    fn is_nosub_device(&self) -> bool {
        false
    }

    fn attention_gen_quant(&self, args: &QuantAttnArgs) -> i32 {
        // mem 포인터가 null 이면 마샬링 실패로 간주(host 가 유효 핸들 패킹 확인).
        if args.q_mem.is_null() || args.out_mem.is_null() {
            return -1;
        }
        // scalar 필드가 정확히 넘어왔는지 host 가 검증하도록 결정적 sentinel 기록.
        if !args.scores_out.is_null() && args.scores_len >= 1 {
            let sentinel = (args.num_heads_q as f32) * 1000.0
                + (args.head_dim as f32)
                + (args.bits as f32) * 0.5
                + args.scale;
            // SAFETY: host 가 scores_len(>=1) 길이의 유효 f32 버퍼를 빌려줌(C5 borrow-for-call).
            unsafe {
                *args.scores_out = sentinel;
            }
        }
        0
    }

    fn gather_update_quant(&self, args: &QuantAttnGatherArgs) -> i32 {
        if args.input_mem.is_null() || args.residual_mem.is_null() {
            return -1;
        }
        0
    }
}

/// make 팩토리 — make 인자(`cl_ctx` 등)는 무시(synthetic, GPU 자원 비보유). closure 대신 named fn 으로
/// 두어 매크로 호출을 한 줄로 유지한다(rustfmt 안정 + 가독성).
fn make_kivi_abi(_args: &QuantAttnMakeArgs) -> Box<dyn QuantAttnBackend> {
    Box::new(KiviAbiBackend)
}

// 정적(linkme 이름 생존) + 동적(cdylib C-ABI vtable) 양쪽 한 줄 등록.
argus_extension_api::register_quant_attn_plugin!("kivi_abi", make_kivi_abi);
// .so 당 1회 — register_kv_stages_v2 / register_kv_formats_v2 / register_backend_caps_v2 엔트리 emit.
argus_extension_api::export_plugin!();

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;

    /// 더미 non-null cl_mem 핸들(synthetic 은 null 검사만 — GPU 안 씀).
    fn dummy_mem(byte: &mut u8) -> *mut c_void {
        (byte as *mut u8) as *mut c_void
    }

    #[test]
    fn has_kernel_gates_supported_bits() {
        let be = KiviAbiBackend;
        assert!(be.has_quant_attn_kernel(2));
        assert!(be.has_quant_attn_kernel(4));
        assert!(be.has_quant_attn_kernel(8));
        assert!(!be.has_quant_attn_kernel(3));
        assert!(!be.has_quant_attn_kernel(16));
    }

    #[test]
    fn nosub_device_is_false() {
        assert!(!KiviAbiBackend.is_nosub_device());
    }

    #[test]
    fn attention_writes_deterministic_sentinel() {
        let be = KiviAbiBackend;
        let mut b = 0u8;
        let mem = dummy_mem(&mut b);
        let mut scores = [0.0f32; 1];
        let args = QuantAttnArgs {
            cl_queue: std::ptr::null_mut(),
            q_mem: mem,
            qk_mem: mem,
            qv_mem: mem,
            res_k_mem: mem,
            res_v_mem: mem,
            out_mem: mem,
            scores_out: scores.as_mut_ptr(),
            scores_len: 1,
            num_heads_q: 32,
            num_heads_kv: 8,
            head_dim: 64,
            q_tokens: 1,
            res_tokens: 16,
            res_cap: 128,
            scale: 0.125,
            bits: 2,
        };
        assert_eq!(be.attention_gen_quant(&args), 0);
        // sentinel = num_heads_q*1000 + head_dim + bits*0.5 + scale (host 게이트와 동일 연산 순서).
        let expected = 32.0_f32 * 1000.0 + 64.0 + 2.0 * 0.5 + 0.125;
        assert_eq!(scores[0], expected);
    }

    #[test]
    fn attention_null_mem_signals_marshalling_failure() {
        let be = KiviAbiBackend;
        let mut b = 0u8;
        let mem = dummy_mem(&mut b);
        let args = QuantAttnArgs {
            cl_queue: std::ptr::null_mut(),
            q_mem: std::ptr::null_mut(),
            qk_mem: mem,
            qv_mem: mem,
            res_k_mem: mem,
            res_v_mem: mem,
            out_mem: mem,
            scores_out: std::ptr::null_mut(),
            scores_len: 0,
            num_heads_q: 32,
            num_heads_kv: 8,
            head_dim: 64,
            q_tokens: 1,
            res_tokens: 16,
            res_cap: 128,
            scale: 0.125,
            bits: 2,
        };
        assert_eq!(be.attention_gen_quant(&args), -1);
    }

    #[test]
    fn gather_round_trip_and_null_guard() {
        let be = KiviAbiBackend;
        let mut b = 0u8;
        let mem = dummy_mem(&mut b);
        let ok = QuantAttnGatherArgs {
            cl_queue: std::ptr::null_mut(),
            input_mem: mem,
            residual_mem: mem,
            kv_heads: 8,
            res_cap: 128,
            head_dim: 64,
            seq_len: 1,
            res_pos: 0,
        };
        assert_eq!(be.gather_update_quant(&ok), 0);
        let bad = QuantAttnGatherArgs {
            input_mem: std::ptr::null_mut(),
            ..ok
        };
        assert_eq!(be.gather_update_quant(&bad), -1);
    }
}
