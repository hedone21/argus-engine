//! example-backend-cap — Backend 축(3번째 axis) capability dlopen plugin 의 synthetic 예제(design D8).
//!
//! **GPU 수학을 하지 않는다** — ABI 마샬링·등록·디스패치 round-trip 만 증명(host-검증 범위, C12 device 제외).
//! `attention_gen_kivi` 는 [`KiviAttnArgs`] 스칼라로 결정적 sentinel 을 계산해 `scores_out[0]` 에 기록,
//! host 게이트가 args struct 가 ABI 경계를 정확히 넘었는지(필드 정렬·값) 확인하게 한다. make 인자(cl_ctx 등)는
//! synthetic 이라 무시. example-kv-format 의 backend 축 짝.

use technique_api::{KiviAttentionBackend, KiviAttnArgs, KiviGatherArgs, KiviMakeArgs};

/// synthetic capability — 상태 없음(GPU 자원 비보유). 핸들 lifecycle(make/drop) round-trip 만 운반.
struct SynthKivi;

impl KiviAttentionBackend for SynthKivi {
    fn has_kivi_attn_kernel(&self, bits: u8) -> bool {
        // synthetic: 2/4/8-bit 모두 "보유"한다고 보고(실 커널 없음).
        matches!(bits, 2 | 4 | 8)
    }

    fn is_nosub_device(&self) -> bool {
        false
    }

    fn attention_gen_kivi(&self, args: &KiviAttnArgs) -> i32 {
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

    fn kivi_gather_update(&self, args: &KiviGatherArgs) -> i32 {
        if args.input_mem.is_null() || args.residual_mem.is_null() {
            return -1;
        }
        0
    }
}

// 정적(linkme 이름 생존) + 동적(cdylib C-ABI vtable) 양쪽 한 줄 등록.
technique_api::register_kivi_attention_plugin!("synth_kivi_attn", |_a: &KiviMakeArgs| -> Box<
    dyn KiviAttentionBackend,
> { Box::new(SynthKivi) });
// .so 당 1회 — register_kv_stages_v2 / register_kv_formats_v2 / register_backend_caps_v2 엔트리 emit.
technique_api::export_plugin!();
