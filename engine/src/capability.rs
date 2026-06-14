//! L2 backend capability 레지스트리 — `CapabilityRegistry` (§3.3).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §3.3.
//!
//! 추론 경로는 backend 를 `Arc<dyn Backend>` 추상 핸들 하나로 들고 다니지만(호출지는 backend
//! 종류를 모름), 일부 능력(GPU score accumulator·KIVI fused attention)은 특정 backend 에만
//! 있다. `CapabilityRegistry` 는 그 (backend → capability handle) 매핑을 한 곳(typed anymap)이
//! 담당해, 소비자가 handle 을 construction 시점에 보유하게 한다 → per-forward `as_xxx()` lookup
//! 폐기, hot path 분기 0 (§1.3 Capability over god-trait). registry lookup 은 construction(cold)
//! 에서만 일어나므로 비용 무관.
//!
//! 본 파일은 **Phase α-W 신설**이다 — 자료구조 + sub-trait 모듈 골격까지만. capability sub-trait
//! 의 *물리적 정착*(`KiviAttentionBackend`/`GpuScoreAccess` 를 `backend.rs` god-trait 에서 이리로
//! 이동) + backend factory `register` 배선 + per-forward lookup 제거는 Phase α-W-4 다. 현재
//! `kivi_attention`/`gpu_score` 서브모듈은 `backend.rs` 거주분을 re-export 하는 shim 이라 모든
//! 기존 call site 가 그대로 동작한다(byte-identical).
//!
//! 미생성 capability(`ScoreCollector`/`TierMovable`)는 **첫 소비자 등장 전까지 만들지 않는다**
//! (§3.4/§4.1/§7 deletion-test — 빈 trait 금지, promotion-trigger). `ScoreCollector` 는 §6 score
//! collection sprint, `TierMovable` 은 cross-Format tier move 소비자 등장 시 도입한다.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

pub mod dynamic_backend_registry;
pub mod gpu_score;
pub mod kivi_attention;

/// (backend → capability handle) 매핑을 담는 typed anymap.
///
/// `Arc<dyn Trait>` 를 concrete payload 로 저장한다(unsafe 없음). 새 capability 종류 = 새 trait
/// 으로 `register`/`get` — 공유 struct edit 0 (양 축 open: 새 backend / 새 capability).
///
/// **타입 키 규약**: `register::<C>` 와 `get::<C>` 의 타입 인자 `C` 는 `TypeId` 로 키잉되므로
/// 정확히 동일해야 한다 (보통 `dyn KiviAttentionBackend` 같은 trait object). key mismatch 는
/// panic 이 아니라 `get` 이 `None` 을 돌려주는 silent miss 다 — 등록·조회 양쪽이 같은 trait
/// bound 을 쓰는지 construction 시점에 보장하라(배선은 α-W-4).
#[derive(Default)]
pub struct CapabilityRegistry {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl CapabilityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// capability handle 등록 — backend "자기 모듈" 에서만 호출 (§3.3). `Arc<dyn C>` 를
    /// concrete payload 로 박는다.
    pub fn register<C: ?Sized + Send + Sync + 'static>(&mut self, h: Arc<C>) {
        self.map.insert(TypeId::of::<Arc<C>>(), Box::new(h));
    }

    /// capability handle 조회 — 소비자가 construction 시점에 1회 pull (§3.6). 미등록이면 `None`.
    pub fn get<C: ?Sized + Send + Sync + 'static>(&self) -> Option<Arc<C>> {
        self.map
            .get(&TypeId::of::<Arc<C>>())?
            .downcast_ref::<Arc<C>>()
            .cloned()
    }
}
