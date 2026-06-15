//! per-`.so` 원자성 롤백 vehicle. 한 봉투에 format 2종: `q4_0`(엔진 빌트인과 충돌) +
//! `rollback_ok`(정상). host `try_register_format` 의 **2-pass**(pass1 전 이름 검사 → pass2 일괄 push)가
//! q4_0 충돌을 pass1 에서 잡아 bail 하므로 `rollback_ok` 는 **등록되지 않는다**(부분 등록 롤백). 게이트는
//! 이후 다른 `.so` 에서 rollback_ok 가 dup 위양성을 내지 않음을 확인한다.
//!
//! 정적 `q4_0` 기여는 이 cdylib 내부에만 있고 엔진 빌트인과 무관(force-link 안 함). 충돌 검사는 host 가
//! 엔진의 `find_kv_format("q4_0")`(빌트인)로 수행한다.

use argus_extension_api::{KVFormat, KVLayoutDesc, Packing, ScaleLayout};

fn q4_like() -> KVLayoutDesc {
    KVLayoutDesc {
        block_elems: 32,
        bits: 4,
        scale_layout: ScaleLayout::PerBlockF16,
        packing: Packing::Nibble,
    }
}

/// 엔진 빌트인 `q4_0` 와 이름 충돌(봉투 순회 중 pass1 에서 reject 유발).
struct CollideQ4;
impl KVFormat for CollideQ4 {
    fn name(&self) -> &str {
        "q4_0"
    }
    fn layout(&self) -> KVLayoutDesc {
        q4_like()
    }
}

/// 정상 format — q4_0 충돌로 bail 되면 **등록되지 않아야** 한다(롤백).
struct RollbackOk;
impl KVFormat for RollbackOk {
    fn name(&self) -> &str {
        "rollback_ok"
    }
    fn layout(&self) -> KVLayoutDesc {
        q4_like()
    }
}

argus_extension_api::register_kv_format!("q4_0", || Box::new(CollideQ4));
argus_extension_api::register_kv_format!("rollback_ok", || Box::new(RollbackOk));
argus_extension_api::export_plugin!();
