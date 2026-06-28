//! `StandardFormat` — `KVCacheFormat` impl wrapping a standard `KVCache` (§4.1, Phase α-K).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §4.1 / §2.1 (guard rail: format impl 은 `kv/`
//! (현 `pressure/`)에, base trait 은 `format/` 에).
//!
//! **purely additive wrapper, now LIVE** — 기존 `KVCache`/`KVCacheOps` 를 1바이트도 건드리지
//! 않는 신규 wrapper 로 출발했으나, 표준 forward 경로가 이제 이 wrapper 로 KV 를 래핑한다
//! (`session/standard_happy.rs`, `session/forward/model_forward::wrap_kv_caches`, `qcf_runtime`).
//! 내부 가변성 = `std::sync::Mutex`(trait `Send+Sync` 요구로 `RefCell` 불가; §4.1 R4 상 cold-path
//! 라 lock 비용 무관).

use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::format::{
    AttnDims, KVCacheFormat, SelectiveRead, SnapshotRestore, dequant_to_f32_tensor,
};
use crate::kv::kv_cache::KVCache;
use crate::memory::host::shared::SharedBuffer;
use crate::shape::Shape;
use crate::tensor::Tensor;
use argus_extension_api::{KVReadPlan, KVReadStage, MergeAxis, WeightedMerge};

/// W-CODEC slice 3 escape hatch: GPU-native q2_0 dequant-attention is ON by default; setting
/// `ARGUS_Q2_GPU_NATIVE_OFF` forces the host descriptor floor (used for A/B TBT comparison and as a
/// production fallback). Read once and cached so the per-attention-call check is free.
fn q2_gpu_native_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| std::env::var("ARGUS_Q2_GPU_NATIVE_OFF").is_err())
}

/// Minimum KV length (cache_seq_len) at which GPU-native q2 attention engages. The slice-3 1a
/// design re-uploads the whole q2 KV and re-dequants its valid [0,cache_seq_len) region every
/// attention call (the whole-buffer upload + per-call f16-mirror alloc are the O(capacity) per-call
/// tax; the kernel decode itself is O(cache_seq_len)), so it only beats the host descriptor floor
/// (host dequant + CPU attention, also O(capacity)) past a break-even. Measured on-device (Adreno
/// 830, qwen2.5-1.5b, head_dim=128): floor wins at
/// cap≈45 (82 vs 97 ms/tok) and cap≈133 (110 vs 121), GPU-native wins at cap≈405 (193 vs 146 ms/tok,
/// 1.32x). Below this threshold the (faster, and absolutely cheap) host floor is kept so the default
/// is never slower than the floor; a persistent device mirror (dequant only the new token — W-CODEC
/// slice-3 follow-up "1b") would remove the per-call tax and lower/erase this threshold.
/// `ARGUS_Q2_GPU_MIN_CTX` overrides it (e.g. `0` to always engage for A/B measurement).
fn q2_gpu_native_min_ctx() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("ARGUS_Q2_GPU_MIN_CTX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256)
    })
}

/// 내부 가변 상태 — `KVCache` 와 비-F32 cast scratch 를 **단일 lock** 으로 묶는다.
///
/// scratch(`k_cast`/`v_cast`)는 비-F32 write 경로의 reusable buffer 로, `forward_gen` 의
/// `ws.k_cast`/`ws.v_cast` 와 같은 역할(토큰마다 재할당 방지). cache 와 한 `Mutex` 안에 두어
/// 별도 lock 으로 인한 동시성 hazard 를 원천 차단한다(write 가 cache+scratch 를 항상 함께 만짐).
pub(crate) struct StandardFormatInner {
    pub(crate) cache: KVCache,
    /// Lazy cast scratch (target dtype). 첫 비-F32 write 에서 inner cache 의 allocator 로 할당.
    k_cast: Option<Tensor>,
    v_cast: Option<Tensor>,
}

/// Standard (F32/F16/Q4_0) KV cache 를 `KVCacheFormat` 으로 노출하는 wrapper.
///
/// 기존 `KVCache` 를 `Mutex` 로 감싸 `&self` 메서드에서 내부 `&mut` 메서드에 위임한다.
/// `KVCache` 자체는 무변.
pub struct StandardFormat {
    idx: usize,
    inner: Mutex<StandardFormatInner>,
}

impl StandardFormat {
    /// `KVCache` 를 layer 인덱스와 함께 wrapping. (표준 forward 경로가 생성 — live.)
    pub fn new(idx: usize, inner: KVCache) -> Self {
        Self {
            idx,
            inner: Mutex::new(StandardFormatInner {
                cache: inner,
                k_cast: None,
                v_cast: None,
            }),
        }
    }

    /// `StandardFormat` 을 소비하여 내부 `KVCache` 를 반환한다.
    ///
    /// prefix cache restore 후 kv_caches 를 재조립할 때 사용 (session::standard_happy 전용).
    /// Mutex 를 into_inner 로 unwrap 하므로 다른 Arc 공유자가 없을 때만 호출할 것.
    pub(crate) fn into_kv_cache(self) -> KVCache {
        self.inner.into_inner().unwrap().cache
    }

    /// 내부 `KVCache` 에 `&mut` 접근하여 `f` 실행 (substep 3c fmt-cache wiring).
    ///
    /// forward(write_kv/attention_into)는 base trait 으로 통과하지만, fmt 활성 시
    /// non-forward 연산(reset_kv 등)이 inner cache 에 도달할 seam 이 필요하다 — base trait 에
    /// method 를 추가하지 않고(`INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC`) concrete inherent 로 제공.
    /// lock guard 안에서 closure 를 실행하므로 호출 종료 시 lock 이 풀린다.
    pub(crate) fn with_cache_mut<R>(&self, f: impl FnOnce(&mut KVCache) -> R) -> R {
        let mut guard = self.inner.lock().unwrap();
        f(&mut guard.cache)
    }

    #[cfg(feature = "opencl")]
    /// plan hot-path geometry 스냅샷 (Phase α-K (3p) ④-a).
    ///
    /// **단일 lock** 으로 `current_pos`/`capacity` 를 묶어 [`PlanGeometry`] 로 반환한다 —
    /// `execute<C>` 가 레이어 진입부에서 호출하던 4개 `KVCacheOps` getter 를 1 lock 으로 통합.
    /// standard 는 residual/quantized partition 부재라 `res_pos`/`q2_tokens` = 0.
    pub(crate) fn plan_geometry(&self) -> crate::backend::opencl::plan::PlanGeometry {
        let g = self.inner.lock().unwrap();
        crate::backend::opencl::plan::PlanGeometry {
            current_pos: g.cache.current_pos(),
            capacity: g.cache.capacity(),
            res_pos: 0,
            q2_tokens: 0,
        }
    }

    #[cfg(feature = "opencl")]
    /// plan hot-path position advance (Phase α-K (3p) ④-a).
    ///
    /// `execute<C>` 의 레이어 끝 `cache.advance_pos(n)` 를 `&self` + interior-mut 로 미러.
    pub(crate) fn plan_advance(&self, n: usize) {
        self.with_cache_mut(|c| c.advance_pos(n));
    }

    #[cfg(feature = "opencl")]
    /// 이 레이어 KV 캐시의 저장 dtype (fused GPU plan 의 F16-only invariant 가드용).
    ///
    /// fused plan 의 KV scatter(`kernel_kv_scatter_f32_to_f16`)/attention 은 F16 전용이라
    /// q4_0 등 비-F16 캐시가 plan 경로를 타면 block 바이트를 F16 으로 오독한다(plan.rs 의 미강제
    /// invariant). `try_build_plan` 이 이 값으로 전 레이어 F16 여부를 확인해 mixed/비-F16 면
    /// dyn forward 폴백시킨다(W-ALLOC per-layer mixed precision correctness).
    pub(crate) fn kv_dtype(&self) -> crate::buffer::DType {
        self.inner.lock().unwrap().cache.kv_dtype()
    }

    #[cfg(feature = "opencl")]
    /// plan 빌드용 lock guard (Phase α-K (3p) ④-a `build_plan`).
    ///
    /// `build_plan` 는 모든 핸들의 guard 를 동시에 잡고 `&KVCache` 슬라이스를 만들어
    /// `build_plan` 본문(byte-identical)을 재사용한다. cl_mem 핸들은 `build_full_plan` 안에서
    /// `set_kernel_arg` 로 즉시 바인딩(클론)되므로 guard 가 그 호출 동안만 살아 있으면 충분하다.
    pub(crate) fn plan_lock(&self) -> std::sync::MutexGuard<'_, StandardFormatInner> {
        self.inner.lock().unwrap()
    }

    /// wrapping 을 해제하고 내부 `KVCache` 를 반환 (Phase α-K ①-c eval transient-wrap round-trip).
    ///
    /// eval 이 forward 1회 동안만 `Vec<KVCache>` → `Arc<StandardFormat>` 로 wrap 한 뒤
    /// `Arc::try_unwrap().into_inner()` 로 concrete cache 를 복귀시키는 seam. cast scratch
    /// (`k_cast`/`v_cast`)는 transient 라 버린다(다음 wrap 에서 lazy 재할당). base trait 무변
    /// (`INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC`).
    pub(crate) fn into_inner(self) -> KVCache {
        self.inner.into_inner().unwrap().cache
    }

    /// Unwrap-Evict-Rewrap (UER) seam (Phase α-K BC (3d)): inner `KVCache` 를 일시적으로 꺼낸다.
    ///
    /// chat 멀티턴 eviction 이 `CacheManager::force_evict(&mut [KVCache])`(연속 슬라이스 요구,
    /// weighted-merge cross-layer 정확성)를 **OLD 경로 그대로** 재사용하도록, fmt_caches 의 inner cache 들을
    /// 연속 `Vec<KVCache>` 로 모으는 용도. `put_inner` 와 페어 호출(단일 lock 구간 sequential).
    /// cast scratch(`k_cast`/`v_cast`)는 guard 에 남아 보존된다(다음 write 재사용). Arc 는 보존
    /// (into_inner 의 try_unwrap 과 달리 self 미소비) — listener phase 무관.
    ///
    /// `KVCache: !Default` 이므로 `mem::take` 불가 → cache 자신의 backend 로 만든 0-size
    /// placeholder 로 `mem::replace`. placeholder 는 `put_inner` 까지 microsecond 만 잔존(eviction
    /// = turn 경계 cold path 라 per-layer 0-byte 할당 무시 가능).
    ///
    /// **β-3 commit B**: `EvictionStage`(stages/kv/eviction.rs)가 동일 UER 로 `force_evict` 를
    /// 적용하고, 등가 integration test 가 stage 산출 byte 를 직접 읽기 위해 `pub` 으로 노출한다
    /// (v1 `try_evict`(model_forward.rs:518-548)와 같은 take/put 페어).
    pub fn take_inner(&self) -> KVCache {
        let mut guard = self.inner.lock().unwrap();
        let backend = guard.cache.k_buffer.backend().clone();
        let buf = Arc::new(SharedBuffer::new(0, DType::F32));
        let ph_k = Tensor::new(Shape::new(vec![1, 0, 1, 1]), buf.clone(), backend.clone());
        let ph_v = Tensor::new(Shape::new(vec![1, 0, 1, 1]), buf, backend);
        std::mem::replace(&mut guard.cache, KVCache::new(ph_k, ph_v, 0))
    }

    /// `take_inner` 의 역연산 — evict 된 `KVCache` 를 다시 넣는다(placeholder 폐기).
    pub fn put_inner(&self, cache: KVCache) {
        self.inner.lock().unwrap().cache = cache;
    }

    /// KV write 흡수 — `forward_gen` 의 KV-update 분기(transformer_layer/forward_gen.rs:330-386)를
    /// format 표면으로 옮긴 것. `is_decode`(seq_len=1)면 GPU fused cast+scatter fast-path 게이팅.
    ///
    /// **host 경로 = correctness fallback** — `CpuBackend` 는 `is_gpu()==false`라 GPU scatter 분기를
    /// 밟지 않으므로 host build+test 가 F32/비-F32 cast 경로를 검증하고, GPU scatter 정확성은
    /// device round(substep (3c))에서 검증한다. **비-F32(F16/Q4_0) cast 경로**(forward_gen 의
    /// `memory.alloc` + `ws.k_cast` scratch)는 inner `KVCache` 의 allocator 로 scratch 를 lazy 할당해
    /// 흡수한다 — write_kv signature 에 `memory` 를 추가하지 않는다(format⊥hardware, KVCache 가 이미
    /// 동일 allocator 보유). 표준 forward 의 write 진입점(`write_kv`/`write_kv_batch` 위임)이다.
    fn write_inner(
        &self,
        new_k: &Tensor,
        new_v: &Tensor,
        backend: &dyn Backend,
        is_decode: bool,
    ) -> Result<()> {
        use crate::kv_cache_ops::KVLayout;

        let mut guard = self.inner.lock().unwrap();
        let kv_dtype = guard.cache.kv_dtype();

        // GPU F16 HeadMajor decode: fused cast+scatter (1 dispatch). host 미진입(is_gpu=false).
        if is_decode
            && backend.is_gpu()
            && kv_dtype == DType::F16
            && guard.cache.layout() == KVLayout::HeadMajor
        {
            let cache = &mut guard.cache;
            let pos = cache.current_pos();
            cache.ensure_capacity(pos + 1)?;
            let cap = cache.capacity();
            let head_dim = cache.head_dim();
            if let Some((k_buf, v_buf)) = cache.get_buffers_mut() {
                backend.kv_scatter_f32_to_f16(new_k, new_v, k_buf, v_buf, head_dim, cap, pos)?;
            }
            cache.advance_pos(1);
            return Ok(());
        }

        // GPU F32 HeadMajor decode: single batched scatter dispatch.
        if is_decode
            && backend.is_gpu()
            && kv_dtype == DType::F32
            && guard.cache.layout() == KVLayout::HeadMajor
            && backend.supports_kv_scatter_f32_batch()
        {
            let cache = &mut guard.cache;
            let pos = cache.current_pos();
            cache.ensure_capacity(pos + 1)?;
            let cap = cache.capacity();
            let n_heads_kv = cache.kv_heads();
            let head_dim = cache.head_dim();
            if let Some((k_buf, v_buf)) = cache.get_buffers_mut() {
                backend.kv_scatter_f32_to_f32_batch(
                    new_k, new_v, k_buf, v_buf, n_heads_kv, head_dim, cap, pos, 1,
                )?;
            }
            cache.advance_pos(1);
            return Ok(());
        }

        // ──────────────────────────────────────────────────────────────────────
        // C3 (§9.1-BC1-CONTRACT ⚠️⚠️ 2차 정정): GPU *prefill batch* scatter fast-path.
        // `write_kv_batch`(decode_fast_path=false, seq_len>1)이 위 decode fast-path 묶음을
        // batch(count=seq_len)로 미러링한다. 게이팅·dtype·position 회계는 decode 분기와 동일하되
        // count 인자만 `1`→실제 seq_len. **bit-identical to cast/update** (kv_scatter_*_batch 의
        // dst_off = h*cap*head_dim + (write_pos_start+s)*head_dim = KVCache::update 의 batch dst_off,
        // advance_pos(seq_len) = update 의 `current_pos += seq_len` + high_water 갱신과 동일).
        // host(CpuBackend, is_gpu=false)는 미진입 → 아래 cast/update 경로가 검증. GPU scatter
        // 정확성은 device round(S25/Jetson)에서 검증. Q4_0 은 GPU fast-path 부재(아래 :131 주석)라
        // 진입하지 않고 cast 경로 유지.
        let seq_len = new_k.shape().dims()[1];

        // GPU F16 HeadMajor batch: fused cast+scatter (1 dispatch over seq_len positions).
        // decode F16(single-pos `kv_scatter_f32_to_f16`)과 달리 batch 변형은 host-pointer
        // fallback 이 device-only 버퍼에서 segfault 하므로 `supports_kv_scatter_batch()` 게이트
        // 필수(미충족 시 아래 cast 경로로 자연 강하 — 동일 출력).
        if !is_decode
            && backend.is_gpu()
            && kv_dtype == DType::F16
            && guard.cache.layout() == KVLayout::HeadMajor
            && backend.supports_kv_scatter_batch()
        {
            let cache = &mut guard.cache;
            let pos = cache.current_pos();
            cache.ensure_capacity(pos + seq_len)?;
            let cap = cache.capacity();
            let n_heads_kv = cache.kv_heads();
            let head_dim = cache.head_dim();
            if let Some((k_buf, v_buf)) = cache.get_buffers_mut() {
                backend.kv_scatter_f32_to_f16_batch(
                    new_k, new_v, k_buf, v_buf, n_heads_kv, head_dim, cap, pos, seq_len,
                )?;
            }
            cache.advance_pos(seq_len);
            return Ok(());
        }

        // GPU F32 HeadMajor batch: single batched scatter dispatch over seq_len positions.
        if !is_decode
            && backend.is_gpu()
            && kv_dtype == DType::F32
            && guard.cache.layout() == KVLayout::HeadMajor
            && backend.supports_kv_scatter_f32_batch()
        {
            let cache = &mut guard.cache;
            let pos = cache.current_pos();
            cache.ensure_capacity(pos + seq_len)?;
            let cap = cache.capacity();
            let n_heads_kv = cache.kv_heads();
            let head_dim = cache.head_dim();
            if let Some((k_buf, v_buf)) = cache.get_buffers_mut() {
                backend.kv_scatter_f32_to_f32_batch(
                    new_k, new_v, k_buf, v_buf, n_heads_kv, head_dim, cap, pos, seq_len,
                )?;
            }
            cache.advance_pos(seq_len);
            return Ok(());
        }

        // 비-F32 cast 경로: F32 입력을 cache dtype(F16/Q4_0)으로 cast 후 update. (forward_gen 의
        // `kv_dtype != F32` 분기 흡수.) `KVCache::update` 는 cast 를 하지 않고 입력이 이미 cache dtype
        // 임을 전제하므로, scatter fast-path 에 안 잡힌 비-F32 write 는 반드시 여기서 cast 해야 한다
        // (Q4_0 은 GPU 에서도 fast-path 부재라 이 경로). dtype 미일치 silent garbage 방지.
        // opaque(.so block-quant): cast 없이 F32 입력을 KVCache::update(=encode+scatter)로 직접 전달
        // kv_dtype=U8 ≠ F32 라 아래 cast 분기로 새지 않도록 여기서 가로챈다.
        if guard.cache.is_opaque() {
            return guard.cache.update(new_k, new_v);
        }

        if kv_dtype != DType::F32 {
            let memory = guard.cache.memory().ok_or_else(|| {
                anyhow::anyhow!(
                    "StandardFormat: non-F32 cast write requires a dynamic KVCache (memory=Some); \
                     fully pre-allocated caches built via KVCache::new() cannot allocate cast scratch"
                )
            })?;
            // scratch lazy 할당 (target dtype). 동일 shape 면 재사용(decode 연속 토큰=seq1 고정),
            // shape 가 바뀌면 재할당한다 — write_kv(decode seq=1)와 write_kv_batch(prefill seq>1)가
            // 같은 format 에서 cast 분기를 공유하므로(K/V 는 KV 불변식상 동일 shape), 첫 write 의
            // 크기로 굳히면 batch↔decode 혼용 시 cast zip 절단·update 오동작. (forward_gen 은
            // decode-only 라 단일 크기였음.)
            let n_elem: usize = new_k.shape().dims().iter().product();
            let buf_size = match kv_dtype {
                DType::F16 => n_elem * 2,
                DType::Q4_0 => {
                    (n_elem / crate::quant::QK4_0) * std::mem::size_of::<crate::quant::BlockQ4_0>()
                }
                _ => n_elem * 4,
            };
            let k_stale = guard
                .k_cast
                .as_ref()
                .is_none_or(|t| t.shape().dims() != new_k.shape().dims());
            if k_stale {
                let buf = memory.alloc(buf_size, kv_dtype)?;
                guard.k_cast = Some(Tensor::new(
                    new_k.shape().clone(),
                    buf,
                    new_k.backend().clone(),
                ));
            }
            let v_stale = guard
                .v_cast
                .as_ref()
                .is_none_or(|t| t.shape().dims() != new_v.shape().dims());
            if v_stale {
                let buf = memory.alloc(buf_size, kv_dtype)?;
                guard.v_cast = Some(Tensor::new(
                    new_v.shape().clone(),
                    buf,
                    new_v.backend().clone(),
                ));
            }
            // 필드별 독립 mutable borrow (cache + scratch 동시 접근).
            let StandardFormatInner {
                cache,
                k_cast,
                v_cast,
            } = &mut *guard;
            let k_cast = k_cast.as_mut().unwrap();
            let v_cast = v_cast.as_mut().unwrap();
            backend.cast(new_k, k_cast)?;
            backend.cast(new_v, v_cast)?;
            return cache.update(k_cast, v_cast);
        }

        // Correctness/CPU F32 경로: `KVCache` 는 GPU-buffer 보유라 `update` 가 내부 backend 로 자체 처리
        // (구 `update_kv_cache` 의 has_gpu_buffers 분기).
        guard.cache.update(new_k, new_v)
    }
}

impl KVCacheFormat for StandardFormat {
    fn idx(&self) -> usize {
        self.idx
    }

    fn current_pos(&self) -> usize {
        self.inner.lock().unwrap().cache.current_pos()
    }

    fn capacity(&self) -> usize {
        self.inner.lock().unwrap().cache.capacity()
    }

    fn write_kv(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        // decode (seq_len=1) — GPU fused cast+scatter fast-path 게이팅 가능.
        self.write_inner(new_k, new_v, backend, true)
    }

    fn write_kv_batch(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        // prefill (seq_len>1) — C3(§9.1-BC1-CONTRACT): GPU prefill batch scatter fast-path 흡수 완료
        // (F32/F16 HeadMajor + supports gate). Q4_0 및 게이트 미충족·CPU 는 cast/update 폴백.
        self.write_inner(new_k, new_v, backend, false)
    }

    fn attention_into(
        &self,
        q: &Tensor,
        backend: &dyn Backend,
        out: &mut Tensor,
        dims: AttnDims,
        scores: Option<&mut [f32]>,
        prefill_scores: Option<(&mut [f32], usize)>,
    ) -> Result<()> {
        let seq_len = q.shape().dims()[1];

        let mut guard = self.inner.lock().unwrap();
        let cache = &mut guard.cache;
        let n_heads_kv = cache.kv_heads();
        let head_dim = cache.head_dim();
        let cache_seq_len = cache.current_pos();

        // opaque(.so block-quant) read = 데이터-구동 floor: descriptor 로 f32 unpack(G3) 후 기존 F32
        // attention 재사용. typed 경로는 아래 무변.
        if cache.is_opaque() {
            // ── W-CODEC slice 3: GPU-native q2_0 dequant-attention ──
            // When the backend has the strict-math q2 dequant kernel AND the format is q2_0 AND the
            // head_dim has an F16 flash kernel, decode the opaque q2 KV into a per-call device F16
            // mirror and run the EXISTING F16 GPU flash attention on it — eliminating the host
            // dequant + CPU-attention floor (no q-readback, no out-writeback). q2 stays opaque and
            // host-resident; only the device F16 mirror is new. Any unmet condition / runtime kernel
            // absence falls through to the unchanged host floor below (byte-identical fallback).
            if q2_gpu_native_enabled()
                && cache_seq_len >= q2_gpu_native_min_ctx()
                && backend.is_gpu()
                && backend.supports_opaque_q2_dequant()
                && (head_dim == 64 || head_dim == 128)
                && cache.layout() == crate::kv_cache_ops::KVLayout::HeadMajor
            {
                let desc = cache.opaque_desc();
                let is_q2 = desc.bits == 2
                    && desc.block_elems == 32
                    && matches!(desc.packing, argus_extension_api::Packing::Quad)
                    && matches!(
                        desc.scale_layout,
                        argus_extension_api::ScaleLayout::PerBlockF16WithMin
                    );
                if is_q2 && let Some(mem) = cache.memory() {
                    let capacity = cache.capacity();
                    let f16_bytes = n_heads_kv * capacity * head_dim * 2;
                    let kv_shape = Shape::new(vec![1, n_heads_kv, capacity, head_dim]);
                    let mut k_dev = Tensor::new(
                        kv_shape.clone(),
                        mem.alloc_kv(f16_bytes, DType::F16)?,
                        q.backend().clone(),
                    );
                    let mut v_dev = Tensor::new(
                        kv_shape,
                        mem.alloc_kv(f16_bytes, DType::F16)?,
                        q.backend().clone(),
                    );
                    // Raw host q2 bytes (whole capacity) — the opaque inner is a host SharedBuffer
                    // (W-DEVKV), so `as_ptr()`/`size()` are valid (mirror apply_weighted_merges_opaque).
                    let k_total = cache.k_buffer.buffer().size();
                    let v_total = cache.v_buffer.buffer().size();
                    let k_q2 = unsafe {
                        std::slice::from_raw_parts(cache.k_buffer.buffer().as_ptr(), k_total)
                    };
                    let v_q2 = unsafe {
                        std::slice::from_raw_parts(cache.v_buffer.buffer().as_ptr(), v_total)
                    };
                    let k_ok = backend.dequant_opaque_q2_to_f16(
                        k_q2,
                        &mut k_dev,
                        n_heads_kv,
                        head_dim,
                        capacity,
                        cache_seq_len,
                    )?;
                    let v_ok = backend.dequant_opaque_q2_to_f16(
                        v_q2,
                        &mut v_dev,
                        n_heads_kv,
                        head_dim,
                        capacity,
                        cache_seq_len,
                    )?;
                    if k_ok && v_ok {
                        if seq_len > 1 {
                            let batch_size = q.shape().dims()[0];
                            let q_start_pos = cache_seq_len - seq_len;
                            let _ = scores; // prefill 은 score 누적 안 함(host floor 동일).
                            return prefill_attention(
                                q,
                                out,
                                &k_dev,
                                &v_dev,
                                dims.n_heads_q,
                                n_heads_kv,
                                head_dim,
                                seq_len,
                                cache_seq_len,
                                capacity,
                                batch_size,
                                cache.layout(),
                                q_start_pos,
                                dims.window,
                                backend,
                                prefill_scores,
                            );
                        }
                        let effective = match dims.window {
                            Some(w) => cache_seq_len.min(w),
                            None => cache_seq_len,
                        };
                        return backend.attention_gen(
                            q,
                            &k_dev,
                            &v_dev,
                            out,
                            dims.n_heads_q,
                            n_heads_kv,
                            head_dim,
                            effective,
                            scores,
                        );
                    }
                    // dequant declined at runtime → fall through to the host floor below.
                }
            }

            let k_f32 = dequant_to_f32_tensor(&cache.k_buffer)?;
            let v_f32 = dequant_to_f32_tensor(&cache.v_buffer)?;
            // W-DEVKV: opaque codec + attention are host-resident f32 (k_f32/v_f32 above are host
            // SharedBuffer on CpuBackend). On a GPU backend the query `q` and `out` are device-resident,
            // so snapshot q→host, run the SAME CpuBackend opaque attention path that the CPU backend
            // uses (proven correct), and upload the result to the device `out`. Mirrors the W-SIGNAL-Q
            // host_snapshot→CpuBackend→write_buffer recipe (see read_plan/attention_into_selected).
            if backend.is_gpu() {
                use crate::backend::cpu::CpuBackend;
                let cpu: Arc<dyn Backend> = Arc::new(CpuBackend::new());
                let mut q_bytes = vec![0u8; q.size()];
                backend.read_buffer(q, &mut q_bytes)?;
                let q_host = Tensor::new(
                    q.shape().clone(),
                    Arc::new(SharedBuffer::from_vec(q_bytes, q.dtype())),
                    cpu.clone(),
                );
                let mut out_host = Tensor::new(
                    out.shape().clone(),
                    Arc::new(SharedBuffer::new(out.size(), out.dtype())),
                    cpu.clone(),
                );
                if seq_len > 1 {
                    let kv_capacity = cache.capacity();
                    let kv_layout = cache.layout();
                    let batch_size = q.shape().dims()[0];
                    let q_start_pos = cache_seq_len - seq_len;
                    let _ = scores; // prefill 은 score 누적 안 함(typed prefill 동일).
                    prefill_attention(
                        &q_host,
                        &mut out_host,
                        &k_f32,
                        &v_f32,
                        dims.n_heads_q,
                        n_heads_kv,
                        head_dim,
                        seq_len,
                        cache_seq_len,
                        kv_capacity,
                        batch_size,
                        kv_layout,
                        q_start_pos,
                        dims.window,
                        &*cpu,
                        prefill_scores,
                    )?;
                } else {
                    let effective = match dims.window {
                        Some(w) => cache_seq_len.min(w),
                        None => cache_seq_len,
                    };
                    cpu.attention_gen(
                        &q_host,
                        &k_f32,
                        &v_f32,
                        &mut out_host,
                        dims.n_heads_q,
                        n_heads_kv,
                        head_dim,
                        effective,
                        scores,
                    )?;
                }
                backend.write_buffer(out, out_host.as_slice::<u8>())?;
                return Ok(());
            }
            if seq_len > 1 {
                let kv_capacity = cache.capacity();
                let kv_layout = cache.layout();
                let batch_size = q.shape().dims()[0];
                let q_start_pos = cache_seq_len - seq_len;
                let _ = scores; // prefill 은 score 누적 안 함(typed prefill 동일).
                return prefill_attention(
                    q,
                    out,
                    &k_f32,
                    &v_f32,
                    dims.n_heads_q,
                    n_heads_kv,
                    head_dim,
                    seq_len,
                    cache_seq_len,
                    kv_capacity,
                    batch_size,
                    kv_layout,
                    q_start_pos,
                    dims.window,
                    backend,
                    prefill_scores,
                );
            }
            let effective = match dims.window {
                Some(w) => cache_seq_len.min(w),
                None => cache_seq_len,
            };
            return backend.attention_gen(
                q,
                &k_f32,
                &v_f32,
                out,
                dims.n_heads_q,
                n_heads_kv,
                head_dim,
                effective,
                scores,
            );
        }

        // ── prefill (seq_len>1): multi-token causal attention (C-1, §9.1-BC1 / ①-b) ──
        // decode delegate(attention_gen / attention_q4_gpu_fallback)는 single-query +
        // causal-mask 부재라 재사용 불가 → forward_prefill(forward.rs:259-585) attention 블록을
        // `prefill_attention` 으로 미러. effective_cache_len clamp 를 **우회**하고(전체 cache_seq_len
        // K + window 를 flash 내부 마스킹에 위임) q_start_pos = cache_seq_len - seq_len. prefill 은
        // score 누적 안 함(scores 무시 — forward_prefill 의 `_need_scores` 와 동일).
        if seq_len > 1 {
            let kv_capacity = cache.capacity();
            let kv_layout = cache.layout();
            let batch_size = q.shape().dims()[0];
            let q_start_pos = cache_seq_len - seq_len;
            let (k_cache, v_cache) = cache.view();
            let _ = scores;
            return prefill_attention(
                q,
                out,
                &k_cache,
                &v_cache,
                dims.n_heads_q,
                n_heads_kv,
                head_dim,
                seq_len,
                cache_seq_len,
                kv_capacity,
                batch_size,
                kv_layout,
                q_start_pos,
                dims.window,
                backend,
                prefill_scores,
            );
        }

        // ── decode (seq_len==1): 기존 경로 (byte-불변) ──
        // Sliding window: 최근 window 토큰으로 제한 (Gemma3 local). global 이면 전체.
        let effective_cache_len = match dims.window {
            Some(w) => cache_seq_len.min(w),
            None => cache_seq_len,
        };

        let (k_cache, v_cache) = cache.view();

        // Q4_0 + GPU: `backend.attention_gen` 은 GPU 에 Q4_0 dequant-attention 커널이 없어
        // BlockQ4_0 raw 바이트를 float 로 오독 → garbage. forward_gen 의 `attention_q4_gpu_fallback`
        // (GPU→CPU readback + dequant + attention + writeback)을 그대로 재사용해 흡수한다 (substep
        // 3c, DRY — 중복 0). `kv_start_pos` = forward_gen.rs:404 와 동일 식(window-clamp 시작 offset).
        // CpuBackend(is_gpu=false)에선 진입 안 함 → host 경로는 아래 attention_gen 유지(Q4_0 CPU arm).
        if cache.kv_dtype() == DType::Q4_0 && backend.is_gpu() {
            let kv_start_pos = cache_seq_len - effective_cache_len;
            let layout = cache.layout();
            let capacity = cache.capacity();
            let need_scores = scores.is_some();
            let mut empty: [f32; 0] = [];
            let scores_buf: &mut [f32] = match scores {
                Some(s) => s,
                None => &mut empty,
            };
            return crate::layers::transformer_layer::TransformerLayer::attention_q4_gpu_fallback(
                q,
                &k_cache,
                &v_cache,
                out,
                scores_buf,
                dims.n_heads_q,
                n_heads_kv,
                head_dim,
                effective_cache_len,
                kv_start_pos,
                layout,
                capacity,
                need_scores,
                backend,
            );
        }

        // typed/F32 경로: backend.attention_gen 에 위임. CPU backend 는 F32/F16/Q4_0 을 dtype-aware
        // 하게 처리(default impl=F32/F16, CpuBackend override 가 Q4_0 등 흡수). GPU backend 는
        // 자기 커널로 dispatch(host 미검증 — device 검증은 substep 3c device round).
        //
        // NOTE: quant_attn-native(get_quant_window_raw_buffers) 분기는 QuantWindowFormat 소관이라 여기 없음.
        backend.attention_gen(
            q,
            &k_cache,
            &v_cache,
            out,
            dims.n_heads_q,
            n_heads_kv,
            head_dim,
            effective_cache_len,
            scores,
        )
    }

    /// `StandardFormat` 은 선택적 읽기를 제공한다 — capability-handle 노출.
    fn as_selective_read(&self) -> Option<&dyn crate::format::SelectiveRead> {
        Some(self)
    }
}

/// (M4-b) [`WeightedMerge`](가중치 baked) 를 `&mut KVCache` 에 in-place 적용한다.
///
/// 구 in-place layer-wide scatter-reduce(이제 가중 merge plugin 의 가중 merge 가 산출)와
/// **bit-identical** 산술이다 — per `WeightedMerge` per head `acc = into_weight·into[d] + Σ w·from[d]`(`into` 먼저, `from` 은 list
/// 순서). K 는 `k_buffer.dtype()`, V 는 `v_buffer.dtype()` 로 독립 디스패치(F32/F16/Q4_0). 위치는
/// compact 적용 직전(pre-compact) 논리 좌표. Q4_0 merge 활성.
///
/// `from` 은 evicted(retain 아님), `into` 는 retained 라 서로/merge 간 겹치지 않아(evicted∉retained)
/// in-place 적용이 안전하다. 빈 `from` 은 skip.
pub(crate) fn apply_weighted_merges(cache: &mut KVCache, merges: &[WeightedMerge]) {
    if merges.is_empty() {
        return;
    }
    // opaque(.so block-quant): descriptor floor 기반 merge. typed 경로 무변.
    if cache.is_opaque() {
        return apply_weighted_merges_opaque(cache, merges);
    }
    use crate::quant::{BlockQ4_0, QK4_0};
    use half::f16;

    let kv_heads = cache.kv_heads();
    let head_dim = cache.head_dim();
    let blocks_per_pos = head_dim / QK4_0; // Q4_0 분기에서만 사용

    for m in merges {
        if m.from.is_empty() {
            continue;
        }
        let into_w = m.into_weight;
        let from_pos: Vec<usize> = m.from.iter().map(|&(p, _)| p).collect();
        let from_w: Vec<f32> = m.from.iter().map(|&(_, w)| w).collect();

        // WeightedKV 축 게이트(KV 로드맵 항목 2). Both=둘 다(구 동작 bit-identical),
        // KeyOnly=K 만 merge·V evict, ValueOnly=V 만 merge·K evict.
        let do_k = m.apply_to != MergeAxis::ValueOnly;
        let do_v = m.apply_to != MergeAxis::KeyOnly;
        for h in 0..kv_heads {
            // ── K (k_buffer.dtype() 디스패치) ──
            if do_k {
                match cache.k_buffer.dtype() {
                    DType::F32 => {
                        let into_off = cache.offset(m.into, h);
                        let from_offs: Vec<usize> =
                            from_pos.iter().map(|&p| cache.offset(p, h)).collect();
                        let k = cache.k_buffer.as_mut_slice::<f32>();
                        merge_row_weighted_f32(k, into_off, &from_offs, &from_w, into_w, head_dim);
                    }
                    DType::F16 => {
                        let into_off = cache.offset(m.into, h);
                        let from_offs: Vec<usize> =
                            from_pos.iter().map(|&p| cache.offset(p, h)).collect();
                        let k = cache.k_buffer.as_mut_slice::<f16>();
                        merge_row_weighted_f16(k, into_off, &from_offs, &from_w, into_w, head_dim);
                    }
                    DType::Q4_0 => {
                        let into_bo = cache.q4_block_offset(m.into, h, blocks_per_pos);
                        let from_bos: Vec<usize> = from_pos
                            .iter()
                            .map(|&p| cache.q4_block_offset(p, h, blocks_per_pos))
                            .collect();
                        let k = cache.k_buffer.as_mut_slice::<BlockQ4_0>();
                        merge_row_weighted_q4(
                            k,
                            into_bo,
                            &from_bos,
                            &from_w,
                            into_w,
                            blocks_per_pos,
                        );
                    }
                    _ => {}
                }
            }

            // ── V (v_buffer.dtype() 독립 디스패치) ──
            if do_v {
                match cache.v_buffer.dtype() {
                    DType::F32 => {
                        let into_off = cache.offset(m.into, h);
                        let from_offs: Vec<usize> =
                            from_pos.iter().map(|&p| cache.offset(p, h)).collect();
                        let v = cache.v_buffer.as_mut_slice::<f32>();
                        merge_row_weighted_f32(v, into_off, &from_offs, &from_w, into_w, head_dim);
                    }
                    DType::F16 => {
                        let into_off = cache.offset(m.into, h);
                        let from_offs: Vec<usize> =
                            from_pos.iter().map(|&p| cache.offset(p, h)).collect();
                        let v = cache.v_buffer.as_mut_slice::<f16>();
                        merge_row_weighted_f16(v, into_off, &from_offs, &from_w, into_w, head_dim);
                    }
                    DType::Q4_0 => {
                        let into_bo = cache.q4_block_offset(m.into, h, blocks_per_pos);
                        let from_bos: Vec<usize> = from_pos
                            .iter()
                            .map(|&p| cache.q4_block_offset(p, h, blocks_per_pos))
                            .collect();
                        let v = cache.v_buffer.as_mut_slice::<BlockQ4_0>();
                        merge_row_weighted_q4(
                            v,
                            into_bo,
                            &from_bos,
                            &from_w,
                            into_w,
                            blocks_per_pos,
                        );
                    }
                    _ => {}
                }
            }
        }
    }
}

#[inline]
fn merge_row_weighted_f32(
    buf: &mut [f32],
    into_off: usize,
    from_offs: &[usize],
    from_w: &[f32],
    into_w: f32,
    head_dim: usize,
) {
    for d in 0..head_dim {
        let mut acc = into_w * buf[into_off + d];
        for (idx, &fo) in from_offs.iter().enumerate() {
            acc += from_w[idx] * buf[fo + d];
        }
        buf[into_off + d] = acc;
    }
}

#[inline]
fn merge_row_weighted_f16(
    buf: &mut [half::f16],
    into_off: usize,
    from_offs: &[usize],
    from_w: &[f32],
    into_w: f32,
    head_dim: usize,
) {
    use half::f16;
    for d in 0..head_dim {
        let mut acc = into_w * buf[into_off + d].to_f32();
        for (idx, &fo) in from_offs.iter().enumerate() {
            acc += from_w[idx] * buf[fo + d].to_f32();
        }
        buf[into_off + d] = f16::from_f32(acc);
    }
}

/// Q4_0 가중 병합 — from 을 head_dim f32 로 dequant(블록 단위) 후, into 블록을 dequant→`*=into_w`→
/// `+= from_w·from` → `BlockQ4_0::quantize`. scatter_reduce_q4 와 동일.
#[inline]
fn merge_row_weighted_q4(
    blocks: &mut [crate::quant::BlockQ4_0],
    into_block_off: usize,
    from_block_offs: &[usize],
    from_w: &[f32],
    into_w: f32,
    blocks_per_pos: usize,
) {
    use crate::quant::{BlockQ4_0, QK4_0};
    // from 을 먼저 full dequant(immutable read) — into write 와 별개 버퍼.
    let from_deq: Vec<Vec<f32>> = from_block_offs
        .iter()
        .map(|&fbo| {
            let mut buf = vec![0.0f32; blocks_per_pos * QK4_0];
            for bi in 0..blocks_per_pos {
                let mut tmp = [0.0f32; QK4_0];
                blocks[fbo + bi].dequantize(&mut tmp);
                buf[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
            }
            buf
        })
        .collect();

    for bi in 0..blocks_per_pos {
        let mut r = [0.0f32; QK4_0];
        blocks[into_block_off + bi].dequantize(&mut r);
        for v in r.iter_mut() {
            *v *= into_w;
        }
        let base = bi * QK4_0;
        for (idx, fbuf) in from_deq.iter().enumerate() {
            for i in 0..QK4_0 {
                r[i] += from_w[idx] * fbuf[base + i];
            }
        }
        blocks[into_block_off + bi] = BlockQ4_0::quantize(&r);
    }
}

/// opaque(.so block-quant) 가중 병합 — `apply_weighted_merges` 의 opaque arm.
///
/// q4_0 분기(`merge_row_weighted_q4`)와 **동일 산술**이나 block dequant/encode 를 descriptor floor
/// (`decode_via_descriptor`/`encode_via_descriptor`, G3/G4)로 수행한다 → q4_0 desc 면 byte-identical.
/// HeadMajor byte offset = `(h*capacity + pos) * bytes_per_head`. K/V 독립 적용.
fn apply_weighted_merges_opaque(cache: &mut KVCache, merges: &[WeightedMerge]) {
    let kv_heads = cache.kv_heads();
    let head_dim = cache.head_dim();
    let capacity = cache.capacity();
    let desc = cache.opaque_desc();
    let bph = cache.opaque_bytes_per_head(); // bytes per (head, pos)

    for m in merges {
        if m.from.is_empty() {
            continue;
        }
        let into_w = m.into_weight;
        let from_pos: Vec<usize> = m.from.iter().map(|&(p, _)| p).collect();
        let from_w: Vec<f32> = m.from.iter().map(|&(_, w)| w).collect();
        // WeightedKV 축 게이트(typed 경로와 동형). Both=K(0)·V(1) 둘 다(구 동작), KeyOnly=K 만,
        // ValueOnly=V 만.
        let do_k = m.apply_to != MergeAxis::ValueOnly;
        let do_v = m.apply_to != MergeAxis::KeyOnly;

        for h in 0..kv_heads {
            let into_off = (h * capacity + m.into) * bph;
            let from_offs: Vec<usize> =
                from_pos.iter().map(|&p| (h * capacity + p) * bph).collect();
            for (buf_idx, buf_t) in [&cache.k_buffer, &cache.v_buffer].iter().enumerate() {
                if (buf_idx == 0 && !do_k) || (buf_idx == 1 && !do_v) {
                    continue;
                }
                let total = buf_t.buffer().size();
                // SAFETY: opaque 버퍼 total 바이트 유효(self 수명 Arc 보유). into/from sub-slice 는
                // evicted∉retained 라 non-overlapping(merge_row_weighted_q4 동일 불변식). interior-mut.
                let all: &mut [u8] =
                    unsafe { std::slice::from_raw_parts_mut(buf_t.buffer().as_mut_ptr(), total) };
                merge_row_weighted_opaque(
                    all, &desc, into_off, &from_offs, &from_w, into_w, head_dim, bph,
                );
            }
        }
    }
}

/// opaque 한 head row 가중 병합: into/from head 를 descriptor floor 로 f32 unpack → `into_w·into +
/// Σ from_w·from` → `encode_via_descriptor` 로 into 에 재인코딩. q4_0 desc 면 `merge_row_weighted_q4`
/// 와 byte-identical(G3 dequant + G4 encode).
#[allow(clippy::too_many_arguments)]
fn merge_row_weighted_opaque(
    buf: &mut [u8],
    desc: &argus_extension_api::KVLayoutDesc,
    into_off: usize,
    from_offs: &[usize],
    from_w: &[f32],
    into_w: f32,
    head_dim: usize,
    bph: usize,
) {
    use crate::format::{decode_via_descriptor, encode_via_descriptor};
    let mut into_f = vec![0.0f32; head_dim];
    decode_via_descriptor(desc, &buf[into_off..into_off + bph], &mut into_f);
    let from_f: Vec<Vec<f32>> = from_offs
        .iter()
        .map(|&fo| {
            let mut f = vec![0.0f32; head_dim];
            decode_via_descriptor(desc, &buf[fo..fo + bph], &mut f);
            f
        })
        .collect();
    for d in 0..head_dim {
        let mut acc = into_w * into_f[d];
        for (idx, ff) in from_f.iter().enumerate() {
            acc += from_w[idx] * ff[d];
        }
        into_f[d] = acc;
    }
    encode_via_descriptor(desc, &into_f, &mut buf[into_off..into_off + bph])
        .expect("opaque weighted-merge merge re-encode (q4_0 family descriptor)");
}

/// prefill multi-token causal attention (C-1, §9.1-BC1 / ①-b).
///
/// `forward_prefill`(transformer_layer/forward.rs:259-585)의 attention 블록을 그대로 미러한다 —
/// **decode delegate(`attention_gen` / `attention_q4_gpu_fallback`)는 single-query + causal-mask
/// 부재라 multi-token prefill 에 재사용 불가**(bit-identical 검증 wfceex20u 정정 B). GPU
/// `flash_attention_prefill` 시도 → 미dispatch(Q4_0 / head_dim 미지원 / CPU)면 dtype별 dequant +
/// `flash_attention_forward_strided`(causal mask 는 `q_start_pos`). prefill 은 score 누적 안 함
/// (forward_prefill 의 `_need_scores` 동일) → scores 인자 없음. `window` 는 flash 내부 마스킹에
/// 위임(decode 진입부의 `effective_cache_len` clamp **우회** — 정정 C). profiler/
/// fallback warn 은 happy-path 미진입·수치-무관이라 생략. forward_prefill 무수정(additive fork) —
/// 중복은 host parity test 로 bit-identical 증명, Step 5(forward_prefill<C> 삭제)에서 자연 해소.
///
/// **Phase α-K ①-e**: `QuantWindowFormat::attention_into` 의 prefill arm 도 이 free fn 을 재사용한다
/// (`pub(crate)`). quant-window 는 multi-token prefill native 커널 부재라 dequantized view(`get_view`) +
/// 본 함수로 처리 — quant-window CPU(SeqMajor F32) / GPU(bits=16 HeadMajor, bits 2/4/8 assembled) 모두
/// `kv_layout`/`kv_capacity` 인자로 분기되므로 별도 경로 불요.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_attention(
    q: &Tensor,
    out: &mut Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    cache_seq_len: usize,
    kv_capacity: usize,
    batch_size: usize,
    kv_layout: crate::kv_cache_ops::KVLayout,
    q_start_pos: usize,
    window: Option<usize>,
    backend: &dyn Backend,
    // R-P1-1 PFA side-channel: `Some((out_scores, q_window))` 면 trailing q_window attention 확률을
    // `out_scores`(caller pre-zeroed, `[n_heads_q * cache_seq_len]`)에 SUM-누적. GPU dispatch 시엔
    // 아래에서 early-return 하므로 자연히 CPU-only. `None`=기존과 byte-identical(producer 미무장).
    prefill_scores: Option<(&mut [f32], usize)>,
) -> Result<()> {
    use crate::kv_cache_ops::KVLayout;

    let is_gpu = backend.is_gpu();
    // GPU flash attention prefill — KV 버퍼가 실제 GPU 버퍼일 때만(CPU-only cache 는 fallback).
    let kv_is_gpu = k_cache.buffer().is_gpu_buffer();
    let gpu_dispatched = if is_gpu && kv_is_gpu {
        backend.flash_attention_prefill(
            q,
            k_cache,
            v_cache,
            out,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            head_dim,
            kv_capacity,
            batch_size,
            kv_layout == KVLayout::HeadMajor,
        )?
    } else {
        false
    };
    if gpu_dispatched {
        return Ok(());
    }

    // CPU attention fallback (GPU 미dispatch 포함).
    let is_device_only = is_gpu && q.as_ptr().is_null();
    let mut out_vec: Vec<f32> = Vec::new();
    {
        fn as_u8_mut(v: &mut [f32]) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
        }

        let mut q_vec = Vec::new();
        let mut k_vec = Vec::new();
        let mut v_vec = Vec::new();

        let (q_data, k_data, v_data, out_ptr) = if is_device_only {
            let read_to_f32 = |t: &Tensor, vec: &mut Vec<f32>| -> Result<()> {
                if t.dtype() == DType::Q4_0 {
                    use crate::quant::{BlockQ4_0, QK4_0};
                    let numel = t.numel();
                    let n_blocks = numel / QK4_0;
                    let byte_size = n_blocks * std::mem::size_of::<BlockQ4_0>();
                    let mut byte_vec = vec![0u8; byte_size];
                    backend.read_buffer(t, &mut byte_vec)?;
                    vec.resize(numel, 0.0);
                    let blocks = unsafe {
                        std::slice::from_raw_parts(byte_vec.as_ptr() as *const BlockQ4_0, n_blocks)
                    };
                    for i in 0..n_blocks {
                        let mut tmp = [0.0f32; QK4_0];
                        blocks[i].dequantize(&mut tmp);
                        vec[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
                    }
                } else if t.dtype() == DType::F16 {
                    let numel = t.numel();
                    let byte_size = numel * 2;
                    let mut byte_vec = vec![0u8; byte_size];
                    backend.read_buffer(t, &mut byte_vec)?;
                    vec.resize(numel, 0.0);
                    unsafe {
                        crate::quant::f16_bulk::bulk_f16_to_f32(
                            byte_vec.as_ptr() as *const u16,
                            vec.as_mut_ptr(),
                            numel,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        let f16_slice = unsafe {
                            std::slice::from_raw_parts(byte_vec.as_ptr() as *const half::f16, numel)
                        };
                        for i in 0..numel {
                            vec[i] = f16_slice[i].to_f32();
                        }
                    }
                } else {
                    vec.resize(t.numel(), 0.0);
                    backend.read_buffer(t, as_u8_mut(vec))?;
                }
                Ok(())
            };

            read_to_f32(q, &mut q_vec)?;
            read_to_f32(k_cache, &mut k_vec)?;
            read_to_f32(v_cache, &mut v_vec)?;

            out_vec.resize(out.numel(), 0.0);

            (&q_vec[..], &k_vec[..], &v_vec[..], &mut out_vec[..])
        } else if k_cache.dtype() == DType::Q4_0 {
            use crate::quant::{BlockQ4_0, QK4_0};
            let n_elems = if kv_layout == KVLayout::HeadMajor {
                n_heads_kv * kv_capacity * head_dim
            } else {
                cache_seq_len * n_heads_kv * head_dim
            };
            let n_blocks = n_elems / QK4_0;
            let k_q4 = unsafe {
                std::slice::from_raw_parts(k_cache.as_ptr() as *const BlockQ4_0, n_blocks)
            };
            let v_q4 = unsafe {
                std::slice::from_raw_parts(v_cache.as_ptr() as *const BlockQ4_0, n_blocks)
            };
            k_vec.resize(n_elems, 0.0f32);
            v_vec.resize(n_elems, 0.0f32);
            for i in 0..n_blocks {
                let mut tmp = [0.0f32; QK4_0];
                k_q4[i].dequantize(&mut tmp);
                k_vec[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
                v_q4[i].dequantize(&mut tmp);
                v_vec[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
            }
            (
                q.as_slice::<f32>(),
                &k_vec[..],
                &v_vec[..],
                out.as_mut_slice::<f32>(),
            )
        } else if k_cache.dtype() == DType::F16 {
            let n_elems = if kv_layout == KVLayout::HeadMajor {
                n_heads_kv * kv_capacity * head_dim
            } else {
                cache_seq_len * n_heads_kv * head_dim
            };
            let k_f16_ptr = k_cache.as_ptr() as *const u16;
            let v_f16_ptr = v_cache.as_ptr() as *const u16;
            k_vec.resize(n_elems, 0.0f32);
            v_vec.resize(n_elems, 0.0f32);
            unsafe {
                crate::quant::f16_bulk::bulk_f16_to_f32(k_f16_ptr, k_vec.as_mut_ptr(), n_elems);
                crate::quant::f16_bulk::bulk_f16_to_f32(v_f16_ptr, v_vec.as_mut_ptr(), n_elems);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let k_f16 =
                    unsafe { std::slice::from_raw_parts(k_f16_ptr as *const half::f16, n_elems) };
                let v_f16 =
                    unsafe { std::slice::from_raw_parts(v_f16_ptr as *const half::f16, n_elems) };
                for i in 0..n_elems {
                    k_vec[i] = k_f16[i].to_f32();
                    v_vec[i] = v_f16[i].to_f32();
                }
            }
            (
                q.as_slice::<f32>(),
                &k_vec[..],
                &v_vec[..],
                out.as_mut_slice::<f32>(),
            )
        } else {
            (
                q.as_slice::<f32>(),
                k_cache.as_slice::<f32>(),
                v_cache.as_slice::<f32>(),
                out.as_mut_slice::<f32>(),
            )
        };

        for x in out_ptr.iter_mut() {
            *x = 0.0;
        }

        use crate::layers::attention::flash_attention_forward_strided;
        let is_head_major_pf = kv_layout == KVLayout::HeadMajor;
        let chunk_q_stride = seq_len * n_heads_q * head_dim;
        let chunk_out_stride = seq_len * n_heads_q * head_dim;
        let chunk_k_stride = kv_capacity * n_heads_kv * head_dim;
        let (k_pos_stride, kv_head_stride) = if is_head_major_pf {
            (head_dim, kv_capacity * head_dim)
        } else {
            (n_heads_kv * head_dim, head_dim)
        };

        for (b, out_batch) in out_ptr.chunks_mut(chunk_out_stride).enumerate() {
            let q_start = b * chunk_q_stride;
            let k_start = b * chunk_k_stride;
            let v_start = b * chunk_k_stride;
            let q_slice = &q_data[q_start..q_start + chunk_q_stride];
            let k_valid_len = if is_head_major_pf {
                n_heads_kv * kv_capacity * head_dim
            } else {
                cache_seq_len * n_heads_kv * head_dim
            };
            let k_slice = &k_data[k_start..k_start + k_valid_len];
            let v_slice = &v_data[v_start..v_start + k_valid_len];

            flash_attention_forward_strided(
                q_slice,
                k_slice,
                v_slice,
                out_batch,
                n_heads_q,
                n_heads_kv,
                seq_len,
                cache_seq_len,
                head_dim,
                n_heads_q * head_dim,
                k_pos_stride,
                k_pos_stride,
                n_heads_q * head_dim,
                kv_head_stride,
                q_start_pos,
                32,
                32,
                window,
            );
        }

        // R-P1-1: PFA side-channel — flash 가 쓰는 동일 dequant K/stride 로 batch 0(단일 시퀀스
        // prefill)의 trailing q_window attention 확률 계산. `out`/`out_ptr` 미접촉(별도 버퍼).
        // GPU dispatch 는 위에서 early-return 했으므로 여기는 CPU-only.
        if let Some((pfa_out, q_window)) = prefill_scores {
            let k_valid_len = if is_head_major_pf {
                n_heads_kv * kv_capacity * head_dim
            } else {
                cache_seq_len * n_heads_kv * head_dim
            };
            let q_slice = &q_data[0..chunk_q_stride];
            let k_slice = &k_data[0..k_valid_len];
            prefill_attention_scores(
                q_slice,
                k_slice,
                n_heads_q,
                n_heads_kv,
                head_dim,
                seq_len,
                cache_seq_len,
                k_pos_stride,
                kv_head_stride,
                q_start_pos,
                q_window,
                window,
                pfa_out,
            );
        }
    }

    if is_device_only {
        let out_bytes =
            unsafe { std::slice::from_raw_parts(out_vec.as_ptr() as *const u8, out_vec.len() * 4) };
        let dst_ptr = out.as_mut_ptr();
        if !dst_ptr.is_null() {
            // UMA / pinned memory: direct memcpy.
            unsafe {
                std::ptr::copy_nonoverlapping(out_bytes.as_ptr(), dst_ptr, out_bytes.len());
            }
        }
        #[cfg(feature = "opencl")]
        {
            // OpenCL device-only buffers need enqueue_write_buffer.
            if dst_ptr.is_null()
                && let Ok(dst_mem) = crate::backend::opencl::get_cl_mem(out.buffer().as_ref())
            {
                if let Some(ocl) = backend
                    .as_any()
                    .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
                {
                    unsafe {
                        ocl::core::enqueue_write_buffer(
                            &ocl.queue,
                            dst_mem,
                            true,
                            0,
                            out_bytes,
                            None::<&ocl::core::Event>,
                            None::<&mut ocl::core::Event>,
                        )?;
                    }
                } else {
                    anyhow::bail!("prefill flash_attn CPU fallback: backend not OpenCL");
                }
            }
        }
    }
    Ok(())
}

/// R-P1-1 PFA side-channel: prefill 의 trailing query window(`q_window` rows)에서 전체 prefix key 로의
/// per-ATTENTION-head(pre-GQA) softmax(q·Kᵀ/√head_dim) 확률을 q_window 에 SUM-accumulate 한다 →
/// `out_scores[h * prefix_len + key_pos]`. flash `out` 은 미접촉(pure scalar CPU, no backend op).
/// `q_data`/`k_data` 는 [`prefill_attention`] 의 CPU 분기가 이미 dequant 한 f32 슬라이스 + 동일 stride
/// (= "모델이 본 attention" 충실). Gate-1 bit-exact 위해 **scalar dot**(SIMD 금지) + eager reference 와
/// 동일 op order(dot→×scale→max→exp/denom→divide). `out_scores` 는 caller 가 pre-zero(SUM 누적, §4.7).
///
/// 사전: `out_scores.len() == n_heads_q * cache_seq_len`; `n_heads_q % n_heads_kv == 0`. batch=0(단일
/// 시퀀스 prefill)만 — KV eviction 은 per-sequence.
// needless_range_loop: 명시적 `key_pos` 인덱스 루프는 Gate-1 bit-exact op-order 계약(§6.1)이다 —
// eager reference(test `pfa_reference`)와 구조가 byte-for-byte 일치해야 bit-exact 비교가 성립한다.
// 반복자 변환은 동일 값이나 계약상 인덱스 형태를 고정한다.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn prefill_attention_scores(
    q_data: &[f32],
    k_data: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    cache_seq_len: usize,
    k_pos_stride: usize,
    kv_head_stride: usize,
    q_start_pos: usize,
    q_window: usize,
    window: Option<usize>,
    out_scores: &mut [f32],
) {
    let prefix_len = cache_seq_len;
    debug_assert_eq!(
        out_scores.len(),
        n_heads_q * prefix_len,
        "PFA out_scores.len must == n_heads_q * prefix_len"
    );
    debug_assert_eq!(
        n_heads_q % n_heads_kv,
        0,
        "n_heads_q must be divisible by n_heads_kv"
    );
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let gqa_group = n_heads_q / n_heads_kv;
    let q_row_stride = n_heads_q * head_dim;
    // trailing query window rows within this chunk: [seq_len - min(q_window, seq_len) .. seq_len).
    let qwin = q_window.min(seq_len);
    let qwin_start = seq_len - qwin;
    // per-(h,r) 재사용 scratch (logits → exp). len = prefix_len. prefill one-shot 라 alloc 1회.
    let mut scratch = vec![0.0f32; prefix_len];

    for h in 0..n_heads_q {
        let kv_head = h / gqa_group; // K group 선택만 — NO GQA reduction.
        let out_row_base = h * prefix_len;
        for r in qwin_start..seq_len {
            let p = q_start_pos + r; // 절대 query 위치 (<= cache_seq_len - 1).
            let lo = match window {
                // SWA band 하한. `w.saturating_sub(1)` — window=0 degenerate 에서 `w-1` underflow 회피
                // (w>=1 인 실제 config 에선 == w-1, bit-exact 무변화; flash 경로보다 strictly robust).
                Some(w) => p.saturating_sub(w.saturating_sub(1)),
                None => 0,
            };
            let q_base = r * q_row_stride + h * head_dim;
            // 1) logits over key_pos in lo..=p (causal + optional SWA band), scalar dot (no SIMD).
            let mut maxv = f32::NEG_INFINITY;
            for key_pos in lo..=p {
                let k_base = key_pos * k_pos_stride + kv_head * kv_head_stride;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_data[q_base + d] * k_data[k_base + d];
                }
                let logit = dot * scale;
                scratch[key_pos] = logit;
                if logit > maxv {
                    maxv = logit;
                }
            }
            // 2) numerically-stable softmax denom (exp 를 scratch 에 저장 → 단일 exp).
            let mut denom = 0.0f32;
            for key_pos in lo..=p {
                let e = (scratch[key_pos] - maxv).exp();
                scratch[key_pos] = e;
                denom += e;
            }
            // 3) SUM-accumulate post-softmax prob over the q_window.
            for key_pos in lo..=p {
                out_scores[out_row_base + key_pos] += scratch[key_pos] / denom;
            }
        }
    }
}

// ── SnapshotRestore capability ────────────────────────────────────

/// `snapshot_prefix` 에서 단일 Tensor의 [0..token_count) 범위만 packed bytes로 추출한다.
///
/// HeadMajor 전제: head_stride = capacity × head_dim(또는 블록). per-head의 [0..token_count)만
/// capacity 패딩 없이 연속 추출. device 버퍼는 backend.read_buffer() 경유(INV-191).
fn extract_packed_bytes(
    t: &crate::tensor::Tensor,
    backend: &dyn Backend,
    kv_heads: usize,
    token_count: usize,
    head_dim: usize,
    capacity: usize,
    dtype: DType,
) -> anyhow::Result<Vec<u8>> {
    use crate::quant::{BlockQ4_0, QK4_0};

    // 전체 버퍼를 read_buffer로 host 쪽으로 읽어온다 (INV-191: as_ptr 금지).
    let total_bytes = t.buffer().size();
    let mut raw = vec![0u8; total_bytes];
    backend.read_buffer(t, &mut raw)?;

    match dtype {
        DType::F32 => {
            // elem_size=4, head_stride = capacity * head_dim elements
            let elem_size = 4usize;
            let head_stride_bytes = capacity * head_dim * elem_size;
            let row_bytes = token_count * head_dim * elem_size;
            let mut out = vec![0u8; kv_heads * row_bytes];
            for h in 0..kv_heads {
                let src_off = h * head_stride_bytes;
                let dst_off = h * row_bytes;
                out[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&raw[src_off..src_off + row_bytes]);
            }
            Ok(out)
        }
        DType::F16 => {
            // elem_size=2, head_stride = capacity * head_dim elements
            let elem_size = 2usize;
            let head_stride_bytes = capacity * head_dim * elem_size;
            let row_bytes = token_count * head_dim * elem_size;
            let mut out = vec![0u8; kv_heads * row_bytes];
            for h in 0..kv_heads {
                let src_off = h * head_stride_bytes;
                let dst_off = h * row_bytes;
                out[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&raw[src_off..src_off + row_bytes]);
            }
            Ok(out)
        }
        DType::Q4_0 => {
            // Q4_0: element ≠ block. 블록 단위 회계 (shrink_to_fit SIGSEGV 동형 함정).
            let bps = head_dim / QK4_0; // blocks per position
            let block_size = std::mem::size_of::<BlockQ4_0>(); // 18 bytes
            // head_stride_blocks = capacity * bps
            let head_stride_bytes = capacity * bps * block_size;
            let row_bytes = token_count * bps * block_size;
            let mut out = vec![0u8; kv_heads * row_bytes];
            for h in 0..kv_heads {
                let src_off = h * head_stride_bytes;
                let dst_off = h * row_bytes;
                out[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&raw[src_off..src_off + row_bytes]);
            }
            Ok(out)
        }
        other => anyhow::bail!(
            "SnapshotRestore: unsupported dtype {:?} (Tier 1 지원: F32/F16/Q4_0)",
            other
        ),
    }
}

/// `restore_prefix` 에서 packed bytes를 현 capacity head_stride로 재배치한다.
///
/// write 후 backend.write_buffer() 로 device에 올린다 (INV-191).
#[allow(clippy::too_many_arguments)]
fn scatter_packed_bytes(
    t: &mut crate::tensor::Tensor,
    backend: &dyn Backend,
    kv_heads: usize,
    token_count: usize,
    head_dim: usize,
    capacity: usize,
    dtype: DType,
    packed: &[u8],
) -> anyhow::Result<()> {
    use crate::quant::{BlockQ4_0, QK4_0};

    // 기존 전체 버퍼를 읽어 host 버퍼를 만든다 (새 capacity이므로 0으로 초기화해도 무방).
    let total_bytes = t.buffer().size();
    let mut raw = vec![0u8; total_bytes];

    match dtype {
        DType::F32 => {
            let elem_size = 4usize;
            let head_stride_bytes = capacity * head_dim * elem_size;
            let row_bytes = token_count * head_dim * elem_size;
            for h in 0..kv_heads {
                let src_off = h * row_bytes;
                let dst_off = h * head_stride_bytes;
                raw[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&packed[src_off..src_off + row_bytes]);
            }
        }
        DType::F16 => {
            let elem_size = 2usize;
            let head_stride_bytes = capacity * head_dim * elem_size;
            let row_bytes = token_count * head_dim * elem_size;
            for h in 0..kv_heads {
                let src_off = h * row_bytes;
                let dst_off = h * head_stride_bytes;
                raw[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&packed[src_off..src_off + row_bytes]);
            }
        }
        DType::Q4_0 => {
            let bps = head_dim / QK4_0;
            let block_size = std::mem::size_of::<BlockQ4_0>();
            let head_stride_bytes = capacity * bps * block_size;
            let row_bytes = token_count * bps * block_size;
            for h in 0..kv_heads {
                let src_off = h * row_bytes;
                let dst_off = h * head_stride_bytes;
                raw[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&packed[src_off..src_off + row_bytes]);
            }
        }
        other => anyhow::bail!(
            "SnapshotRestore: unsupported dtype {:?} (Tier 1 지원: F32/F16/Q4_0)",
            other
        ),
    }

    backend.write_buffer(t, &raw)?;
    Ok(())
}

impl SnapshotRestore for StandardFormat {
    /// `[0..token_count)` K+V를 capacity 패딩 제거 packed bytes로 직렬화.
    ///
    /// pre: `current_pos == token_count`, eviction 미발생 (INV-189).
    /// device는 `backend.read_buffer()` (INV-191).
    /// 반환: K bytes || V bytes (per-layer — 상위 save_prefix가 layer-major concat).
    fn snapshot_prefix(
        &self,
        token_count: usize,
        backend: &dyn Backend,
    ) -> anyhow::Result<Vec<u8>> {
        let guard = self.inner.lock().unwrap();
        let cache = &guard.cache;

        anyhow::ensure!(
            cache.current_pos() == token_count,
            "SnapshotRestore::snapshot_prefix: current_pos({}) != token_count({})",
            cache.current_pos(),
            token_count
        );

        let kv_heads = cache.kv_heads();
        let head_dim = cache.head_dim();
        let capacity = cache.capacity();
        let dtype = cache.kv_dtype();

        let k_bytes = extract_packed_bytes(
            &cache.k_buffer,
            backend,
            kv_heads,
            token_count,
            head_dim,
            capacity,
            dtype,
        )?;
        let v_bytes = extract_packed_bytes(
            &cache.v_buffer,
            backend,
            kv_heads,
            token_count,
            head_dim,
            capacity,
            dtype,
        )?;

        let mut out = k_bytes;
        out.extend_from_slice(&v_bytes);
        Ok(out)
    }

    /// packed bytes에서 KV를 복원.
    ///
    /// pre: `current_pos == 0` (빈 캐시), bytes = 동일 format packed-form.
    /// post: `current_pos == token_count`, KV byte-identical (INV-191).
    fn restore_prefix(
        &self,
        bytes: &[u8],
        token_count: usize,
        backend: &dyn Backend,
    ) -> anyhow::Result<()> {
        let mut guard = self.inner.lock().unwrap();
        let cache = &mut guard.cache;

        anyhow::ensure!(
            cache.current_pos() == 0,
            "SnapshotRestore::restore_prefix: cache is not empty (current_pos={})",
            cache.current_pos()
        );

        // 용량 확보
        cache.ensure_capacity(token_count)?;

        let kv_heads = cache.kv_heads();
        let head_dim = cache.head_dim();
        let capacity = cache.capacity();
        let dtype = cache.kv_dtype();

        // packed bytes를 K / V 로 분리
        let half = bytes.len() / 2;
        let k_packed = &bytes[..half];
        let v_packed = &bytes[half..];

        // scatter: packed → capacity head_stride layout
        scatter_packed_bytes(
            &mut cache.k_buffer,
            backend,
            kv_heads,
            token_count,
            head_dim,
            capacity,
            dtype,
            k_packed,
        )?;
        scatter_packed_bytes(
            &mut cache.v_buffer,
            backend,
            kv_heads,
            token_count,
            head_dim,
            capacity,
            dtype,
            v_packed,
        )?;

        // position 갱신
        cache.set_current_pos(token_count);
        // high_water 갱신 (advance_pos는 current_pos += n이라 부적합, 직접 설정)
        cache.high_water_pos = cache.high_water_pos.max(token_count);

        drop(guard);
        Ok(())
    }

    fn snapshot_format_id(&self) -> u32 {
        let guard = self.inner.lock().unwrap();
        match guard.cache.kv_dtype() {
            DType::F32 => 1,
            DType::F16 => 2,
            DType::Q4_0 => 3,
            other => {
                // 지원하지 않는 dtype은 0으로 표시 (헤더 무효화로 폴백)
                let _ = other;
                0
            }
        }
    }
}

// ── SelectiveRead capability ─────────────────────────────────

/// `ReadGranularity::Page { page_size }` → token pos 목록 전개.
///
/// `page_indices` 의 각 page 를 `[page * page_size .. (page+1)*page_size)` 범위로 전개한 뒤
/// 전체 `current_pos` 로 clamp 한다.
fn page_indices_to_positions(
    page_indices: &[usize],
    page_size: usize,
    current_pos: usize,
) -> Vec<usize> {
    let mut positions = Vec::with_capacity(page_indices.len() * page_size);
    for &pi in page_indices {
        let start = pi * page_size;
        let end = (start + page_size).min(current_pos);
        for pos in start..end {
            positions.push(pos);
        }
    }
    positions
}

/// 선택된 pos 목록(ascending)에서 **임시 KVCache** 를 gather 한다.
///
/// HeadMajor 레이아웃 전제 (CLAUDE.md Production = HeadMajor 고정).
/// head_stride = capacity × head_dim (원소 단위, F32/F16) 또는 capacity × bps (블록 단위, Q4_0).
///
/// **Tier 1 단순 전략**: dtype 3종 처리.
/// - F32/F16: 원소 단위 복사.
/// - Q4_0: dequant → F32 gather (부분 select 는 블록 경계와 안 맞을 수 있어 안전 우선).
///
/// 반환: (gathered_k, gathered_v, n_selected) — F32 텐서, shape [1, 1, kv_heads, head_dim].
/// gathered seq len = `select.len()` 이고, 결과 KVCache 의 `current_pos = select.len()`.
fn gather_selected_kv(cache: &KVCache, select: &[usize]) -> Result<(Vec<f32>, Vec<f32>)> {
    use crate::quant::{BlockQ4_0, QK4_0};
    use half::f16;

    // SelectiveRead gathers via host `as_slice()`; a GPU-resident cache exposes a null
    // host pointer (device buffer), so this path would deref null → segfault. Read-stage
    // selective read is host-only — guard with a clean error rather than UB (full read is
    // the fallback). Host caches report `is_gpu_buffer() == false`, so this is a no-op for
    // the existing CPU SelectiveRead path (byte-identical).
    if cache.k_buffer.buffer().is_gpu_buffer() || cache.v_buffer.buffer().is_gpu_buffer() {
        anyhow::bail!(
            "SelectiveRead (read-stage) gather requires a host-resident KV cache; got a \
             GPU-resident cache. Selective read is not supported on GPU caches — use full read."
        );
    }

    let kv_heads = cache.kv_heads();
    let head_dim = cache.head_dim();
    let capacity = cache.capacity();
    let n_sel = select.len();
    let total = kv_heads * n_sel * head_dim;

    let mut k_out = vec![0.0f32; total];
    let mut v_out = vec![0.0f32; total];

    match cache.kv_dtype() {
        DType::F32 => {
            let k_src = cache.k_buffer.as_slice::<f32>();
            let v_src = cache.v_buffer.as_slice::<f32>();
            for h in 0..kv_heads {
                // HeadMajor: head_stride = capacity * head_dim elements
                let src_head_off = h * capacity * head_dim;
                let dst_head_off = h * n_sel * head_dim;
                for (si, &pos) in select.iter().enumerate() {
                    let src = src_head_off + pos * head_dim;
                    let dst = dst_head_off + si * head_dim;
                    k_out[dst..dst + head_dim].copy_from_slice(&k_src[src..src + head_dim]);
                    v_out[dst..dst + head_dim].copy_from_slice(&v_src[src..src + head_dim]);
                }
            }
        }
        DType::F16 => {
            let k_src = cache.k_buffer.as_slice::<f16>();
            let v_src = cache.v_buffer.as_slice::<f16>();
            for h in 0..kv_heads {
                let src_head_off = h * capacity * head_dim;
                let dst_head_off = h * n_sel * head_dim;
                for (si, &pos) in select.iter().enumerate() {
                    let src = src_head_off + pos * head_dim;
                    let dst = dst_head_off + si * head_dim;
                    for d in 0..head_dim {
                        k_out[dst + d] = k_src[src + d].to_f32();
                        v_out[dst + d] = v_src[src + d].to_f32();
                    }
                }
            }
        }
        DType::Q4_0 => {
            // Q4_0: dequant 경로 — 블록 경계 미정렬 pos select 를 안전하게 처리
            let bps = head_dim / QK4_0; // blocks per position
            let k_blocks = cache.k_buffer.as_slice::<BlockQ4_0>();
            let v_blocks = cache.v_buffer.as_slice::<BlockQ4_0>();
            for h in 0..kv_heads {
                // HeadMajor block offset: h * capacity * bps
                let src_head_block_off = h * capacity * bps;
                let dst_head_off = h * n_sel * head_dim;
                for (si, &pos) in select.iter().enumerate() {
                    let src_block = src_head_block_off + pos * bps;
                    let dst = dst_head_off + si * head_dim;
                    // dequant K
                    for bi in 0..bps {
                        let mut tmp = [0.0f32; QK4_0];
                        k_blocks[src_block + bi].dequantize(&mut tmp);
                        k_out[dst + bi * QK4_0..dst + (bi + 1) * QK4_0].copy_from_slice(&tmp);
                    }
                    // dequant V
                    for bi in 0..bps {
                        let mut tmp = [0.0f32; QK4_0];
                        v_blocks[src_block + bi].dequantize(&mut tmp);
                        v_out[dst + bi * QK4_0..dst + (bi + 1) * QK4_0].copy_from_slice(&tmp);
                    }
                }
            }
        }
        _ => {
            anyhow::bail!("SelectiveRead: unsupported dtype {:?}", cache.kv_dtype());
        }
    }

    Ok((k_out, v_out))
}

impl SelectiveRead for StandardFormat {
    /// `select` 된 KV 위치만 읽는 attention (Tier 1 = gather + 기존 attention 재사용).
    ///
    /// **단순 우선, 성능 주장 없음** — select 된 토큰을 F32 임시 버퍼로 gather 한 뒤
    /// `backend.attention_gen` 에 위임한다. select = 전체 토큰이면 `attention_into` 와 bit-identical
    /// (F32/F16 경우). Q4_0 은 dequant gather 경로라 `attention_into` 와 동일 dequant 경유 비교 가능.
    ///
    /// **softmax 분모**: 선택된 부분집합 위에서 정규화됨 — Quest 의 의도된 근사.
    #[allow(clippy::too_many_arguments)]
    fn attention_into_selected(
        &self,
        q: &Tensor,
        backend: &dyn Backend,
        out: &mut Tensor,
        dims: AttnDims,
        select: &[usize],
        granularity: argus_extension_api::ReadGranularity,
        scores: Option<&mut [f32]>,
    ) -> Result<()> {
        use crate::memory::host::shared::SharedBuffer;

        let guard = self.inner.lock().unwrap();
        let cache = &guard.cache;
        let current_pos = cache.current_pos();

        if current_pos == 0 || select.is_empty() {
            // 빈 캐시 또는 빈 select: out 을 0으로 채우고 반환
            for x in out.as_mut_slice::<f32>() {
                *x = 0.0;
            }
            return Ok(());
        }

        // Page 단위라면 pos 목록으로 전개
        let expanded: Vec<usize>;
        let positions: &[usize] = match granularity {
            argus_extension_api::ReadGranularity::Token => select,
            argus_extension_api::ReadGranularity::Page { page_size } => {
                expanded = page_indices_to_positions(select, page_size as usize, current_pos);
                &expanded
            }
        };

        let n_sel = positions.len();
        if n_sel == 0 {
            for x in out.as_mut_slice::<f32>() {
                *x = 0.0;
            }
            return Ok(());
        }

        let kv_heads = cache.kv_heads();
        let head_dim = cache.head_dim();

        // W-DEVKV: GPU 버퍼면(`is_gpu_buffer()` — `gather_selected_kv` 가드와 동일 predicate) gather(host
        // as_slice 필요)를 위해 host snapshot 위에서 모은다. host-native(CPU) 캐시만 직접(snapshot 없음, 비용
        // 0). ★`as_ptr().is_null()` 이 아니라 `is_gpu_buffer()` 여야 rpcmem/mapped-UMA(host ptr 非null인
        // GPU 버퍼)가 gather guard 와 어긋나지 않는다.
        let needs_snapshot = cache.k_buffer.buffer().is_gpu_buffer();
        let snapshot = if needs_snapshot {
            Some(cache.host_snapshot()?)
        } else {
            None
        };
        let gather_cache = snapshot.as_ref().unwrap_or(cache);
        // gather selected tokens into F32 temporary buffers (SeqMajor [1, n_sel, kv_heads, head_dim]).
        let (k_f32, v_f32) = gather_selected_kv(gather_cache, positions)?;
        drop(snapshot); // host mirror 해제
        drop(guard); // lock 해제 (attention 호출 전)

        // gather 결과를 byte Vec으로 변환
        let k_bytes = {
            let mut b = vec![0u8; k_f32.len() * 4];
            unsafe {
                std::ptr::copy_nonoverlapping(k_f32.as_ptr() as *const u8, b.as_mut_ptr(), b.len());
            }
            b
        };
        let v_bytes = {
            let mut b = vec![0u8; v_f32.len() * 4];
            unsafe {
                std::ptr::copy_nonoverlapping(v_f32.as_ptr() as *const u8, b.as_mut_ptr(), b.len());
            }
            b
        };
        let k_buf = Arc::new(SharedBuffer::from_vec(k_bytes, DType::F32));
        let v_buf = Arc::new(SharedBuffer::from_vec(v_bytes, DType::F32));
        let seq_shape = Shape::new(vec![1, n_sel, kv_heads, head_dim]);

        if needs_snapshot {
            // W-DEVKV Part 2: gathered subset attention 을 CPU(reference-correct)로 계산한 뒤 device `out`
            // 으로 upload. gathered K/V 는 host F32 SeqMajor 라 GPU attention_gen(device cl_mem 전제)에
            // 직접 못 넘긴다 → q(device)를 host 로 읽고 CpuBackend.attention_gen 으로 계산(검증된 정본
            // 경로 — host SelectiveRead 와 동일 커널), 결과를 write_buffer 로 device out 에 쓴다. opt-in
            // read-stage 경로만 진입(production 미진입). 정확성은 host select-all==full read 테스트가 고정.
            use crate::backend::cpu::CpuBackend;
            let cpu: Arc<dyn Backend> = Arc::new(CpuBackend::new());
            let mut q_host_bytes = vec![0u8; q.size()];
            backend.read_buffer(q, &mut q_host_bytes)?;
            let q_host = Tensor::new(
                q.shape().clone(),
                Arc::new(SharedBuffer::from_vec(q_host_bytes, q.dtype())),
                cpu.clone(),
            );
            let k_tmp = Tensor::new(seq_shape.clone(), k_buf, cpu.clone());
            let v_tmp = Tensor::new(seq_shape, v_buf, cpu.clone());
            let mut out_host = Tensor::new(
                out.shape().clone(),
                Arc::new(SharedBuffer::new(out.size(), out.dtype())),
                cpu.clone(),
            );
            cpu.attention_gen(
                &q_host,
                &k_tmp,
                &v_tmp,
                &mut out_host,
                dims.n_heads_q,
                kv_heads,
                head_dim,
                n_sel,
                scores,
            )?;
            backend.write_buffer(out, out_host.as_slice::<u8>())?;
            Ok(())
        } else {
            // host-resident: gathered SeqMajor 텐서를 backend.attention_gen 에 직접 위임(기존 경로).
            let backend_arc = q.backend().clone();
            let k_tmp = Tensor::new(seq_shape.clone(), k_buf, backend_arc.clone());
            let v_tmp = Tensor::new(seq_shape, v_buf, backend_arc);
            backend.attention_gen(
                q,
                &k_tmp,
                &v_tmp,
                out,
                dims.n_heads_q,
                kv_heads,
                head_dim,
                n_sel,
                scores,
            )
        }
    }

    /// read stage 를 자기 `Mutex<KVCache>` 위에서 호출. ctx 구성·borrow 가
    /// 이 메서드에 갇혀 `attention_into_selected`(다시 lock) 와 충돌하지 않는다 — owned `KVReadPlan`
    /// 을 반환해 lock guard 가 함수 종료 시 drop 된다.
    fn read_plan(
        &self,
        rs: &dyn KVReadStage,
        _layer_idx: usize,
        query: Option<&[f32]>,
        query_stats: Option<&[f32]>,
    ) -> Option<KVReadPlan> {
        use crate::stages::kv::mutation::{SnapshotStageCtx, dequant_snapshot};
        let guard = self.inner.lock().unwrap();
        // W-DEVKV: read stage 가 K/V 내용을 읽으려면(`tensor(Key)`→dequantize_k→as_slice / gather)
        // host-resident 버퍼가 필요하다. **GPU 버퍼면(`is_gpu_buffer()` — device-only OpenCLBuffer /
        // UMA UnifiedBuffer(mapped 여부 무관) / rpcmem 모두 포함)** device→host snapshot 을 1회 떠서 그
        // 위에 ctx 를 만든다(geometry 동일 → dequantize_* byte-identical, snapshot 은 host SharedBuffer →
        // gather guard 통과). ★predicate 는 `gather_selected_kv` 의 `is_gpu_buffer()` 가드와 반드시 일치해야
        // 한다 — `as_ptr().is_null()` 로 판정하면 rpcmem/mapped-UMA(host ptr 非null이지만 GPU 버퍼)가 direct
        // 경로로 새서 gather 가 bail(decode abort). host-native(CPU) 캐시만 direct(비용 0). snapshot 실패 시
        // None(full read 폴백). production decode(read_stage=None)는 이 메서드 미진입이라 비용 0.
        let needs_snapshot = guard.cache.k_buffer.buffer().is_gpu_buffer();
        let snapshot = if needs_snapshot {
            Some(guard.cache.host_snapshot().ok()?)
        } else {
            None
        };
        let cache_ref = snapshot.as_ref().unwrap_or(&guard.cache);
        // read stage 는 budget(target_len) 을 읽지 않는다(읽기 범위 결정 ≠ keep budget). importance/
        // scores 미공급(None) — read stage 는 `tensor(Key)`/`tensor(Value)` 로 자기 page 메타를
        // incremental 갱신한다(D5). query_stats 는 dormant fallback(현재 production producer 없음).
        // Build owned K/V dequant snapshots over the (host-resident) cache_ref, then a read ctx that
        // exposes them as `tensor(Key)`/`tensor(Value)` plus the optional faithful current-Q
        // (`tensor(Query)`, Quest's 정본 current-Q; `None` on proxy/offload → `tensor(Query)==None`,
        // byte-identical disabled path) and the dormant `tensor(QueryStats)` fallback.
        let rows = cache_ref.current_pos();
        let n_kv_heads = cache_ref.kv_heads();
        let head_dim = cache_ref.head_dim();
        let key_snap = dequant_snapshot(cache_ref, rows, n_kv_heads, head_dim, true);
        let value_snap = dequant_snapshot(cache_ref, rows, n_kv_heads, head_dim, false);
        let ctx = SnapshotStageCtx::for_read(cache_ref, &key_snap, &value_snap, query, query_stats);
        rs.read_plan(&ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn f32_tensor(dims: Vec<usize>, data: &[f32]) -> Tensor {
        let buf = Arc::new(SharedBuffer::new(data.len() * 4, DType::F32));
        let mut t = Tensor::new(Shape::new(dims), buf, Arc::new(CpuBackend::new()));
        t.as_mut_slice::<f32>().copy_from_slice(data);
        t
    }

    /// Build a SeqMajor F32 KVCache: [1, max_seq, kv_heads, head_dim].
    fn make_cache(max_seq: usize, kv_heads: usize, head_dim: usize) -> KVCache {
        let total = max_seq * kv_heads * head_dim;
        let k = f32_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0; total]);
        let v = f32_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0; total]);
        KVCache::new(k, v, max_seq)
    }

    fn f16_tensor(dims: Vec<usize>, data: &[f32]) -> Tensor {
        use half::f16;
        let buf = Arc::new(SharedBuffer::new(data.len() * 2, DType::F16));
        let mut t = Tensor::new(Shape::new(dims), buf, Arc::new(CpuBackend::new()));
        for (d, &s) in t.as_mut_slice::<f16>().iter_mut().zip(data.iter()) {
            *d = f16::from_f32(s);
        }
        t
    }

    /// Build a SeqMajor F16 *dynamic* KVCache with a real allocator (`memory=Some`),
    /// so the non-F32 cast scratch path can lazily allocate its scratch buffers.
    fn make_f16_dynamic_cache(max_seq: usize, kv_heads: usize, head_dim: usize) -> KVCache {
        use crate::memory::Memory;
        use crate::memory::galloc::Galloc;
        let total = max_seq * kv_heads * head_dim;
        let k = f16_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0f32; total]);
        let v = f16_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0f32; total]);
        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        KVCache::new_dynamic(k, v, max_seq, max_seq, kv_heads, head_dim, mem)
    }

    /// GATE (Stage 1, GATE-B retarget): `DType` 없는 opaque format(synth_q4 layout)이
    /// **production `KVCache`(흡수, D1) + `StandardFormat`** 경로로 write(encode+scatter, grow 포함)+
    /// attention(dequant floor → F32 attention)을 수행한 결과가, 동일 데이터의 q4_0 round-trip
    /// (quantize→dequantize) HeadMajor F32 baseline 과 **bit-identical**. `initial_cap < n_tokens`
    /// 로 opaque grow arm(D2)도 함께 검증한다. (구 `OpaqueKvFormat` 테스트를 KVCache 경로로 이전.)
    #[test]
    fn opaque_kvcache_via_standard_format_bit_identical_to_q4_0_roundtrip() {
        use crate::buffer::Buffer;
        use crate::buffer::opaque::OpaqueBuffer;
        use crate::kv_cache_ops::KVLayout;
        use crate::memory::Memory;
        use crate::memory::galloc::Galloc;
        use crate::quant::{BlockQ4_0, QK4_0};
        use argus_extension_api::{KVLayoutDesc, Packing, ScaleLayout};

        let kv_heads = 2usize;
        let head_dim = 64usize; // 2 blocks/head (block_elems=32)
        let n_heads_q = 2usize; // GQA ratio 1
        let n_tokens = 5usize;
        let initial_cap = 2usize; // < n_tokens → opaque grow 발동 (D2 arm 검증)
        let max_seq = 64usize;
        let desc = KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        };
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());

        let gen_val = |t: usize, h: usize, d: usize, salt: f32| -> f32 {
            (((t * 7 + h * 13 + d * 3) % 17) as f32 - 8.0) * 0.1 + salt
        };
        let mut k_tokens: Vec<Vec<f32>> = Vec::new();
        let mut v_tokens: Vec<Vec<f32>> = Vec::new();
        for t in 0..n_tokens {
            let mut k = vec![0.0f32; kv_heads * head_dim];
            let mut v = vec![0.0f32; kv_heads * head_dim];
            for h in 0..kv_heads {
                for d in 0..head_dim {
                    k[h * head_dim + d] = gen_val(t, h, d, 0.0);
                    v[h * head_dim + d] = gen_val(t, h, d, 0.5);
                }
            }
            k_tokens.push(k);
            v_tokens.push(v);
        }

        // ── opaque KVCache (HeadMajor, synth_q4 desc) via StandardFormat; initial_cap<n_tokens → grow ──
        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        let n0 = kv_heads * initial_cap * head_dim;
        let nbytes = desc.bytes_for_elems(n0).unwrap();
        let shape = Shape::new(vec![1, kv_heads, initial_cap, head_dim]);
        let mk = || -> Tensor {
            let inner = mem.alloc_kv(nbytes, DType::U8).unwrap();
            let op: Arc<dyn Buffer> = Arc::new(OpaqueBuffer::new(inner, desc));
            Tensor::new(shape.clone(), op, backend.clone())
        };
        let cache = KVCache::new_dynamic(
            mk(),
            mk(),
            initial_cap,
            max_seq,
            kv_heads,
            head_dim,
            mem.clone(),
        )
        .with_layout(KVLayout::HeadMajor);
        let fmt = StandardFormat::new(0, cache);
        for t in 0..n_tokens {
            let kt = f32_tensor(vec![1, 1, kv_heads, head_dim], &k_tokens[t]);
            let vt = f32_tensor(vec![1, 1, kv_heads, head_dim], &v_tokens[t]);
            fmt.write_kv(&kt, &vt, backend.as_ref()).unwrap();
        }
        assert_eq!(fmt.current_pos(), n_tokens);
        assert!(
            fmt.capacity() >= n_tokens,
            "grow 가 capacity 를 늘렸어야 함"
        );

        let q_data: Vec<f32> = (0..n_heads_q * head_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.07)
            .collect();
        let q = f32_tensor(vec![1, 1, n_heads_q, head_dim], &q_data);
        let mut out_opaque = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.0; n_heads_q * head_dim],
        );
        fmt.attention_into(
            &q,
            backend.as_ref(),
            &mut out_opaque,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
            None,
        )
        .unwrap();

        // ── reference: q4_0 round-trip(quantize→dequantize) HeadMajor F32 + attention_gen ──
        let cap = fmt.capacity(); // grow 후 실제 capacity (HeadMajor stride)
        let mut ref_k = vec![0.0f32; kv_heads * cap * head_dim];
        let mut ref_v = vec![0.0f32; kv_heads * cap * head_dim];
        for (t, (kt, vt)) in k_tokens.iter().zip(v_tokens.iter()).enumerate() {
            for h in 0..kv_heads {
                for blk in 0..(head_dim / QK4_0) {
                    let lo = h * head_dim + blk * QK4_0;
                    let mut ka = [0.0f32; QK4_0];
                    ka.copy_from_slice(&kt[lo..lo + QK4_0]);
                    let mut ko = [0.0f32; QK4_0];
                    BlockQ4_0::quantize(&ka).dequantize(&mut ko);
                    let mut va = [0.0f32; QK4_0];
                    va.copy_from_slice(&vt[lo..lo + QK4_0]);
                    let mut vo = [0.0f32; QK4_0];
                    BlockQ4_0::quantize(&va).dequantize(&mut vo);
                    let base = (h * cap + t) * head_dim + blk * QK4_0;
                    ref_k[base..base + QK4_0].copy_from_slice(&ko);
                    ref_v[base..base + QK4_0].copy_from_slice(&vo);
                }
            }
        }
        let ref_k_t = f32_tensor(vec![1, kv_heads, cap, head_dim], &ref_k);
        let ref_v_t = f32_tensor(vec![1, kv_heads, cap, head_dim], &ref_v);
        let mut out_ref = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.0; n_heads_q * head_dim],
        );
        backend
            .attention_gen(
                &q,
                &ref_k_t,
                &ref_v_t,
                &mut out_ref,
                n_heads_q,
                kv_heads,
                head_dim,
                n_tokens,
                None,
            )
            .unwrap();

        assert_eq!(
            out_opaque.as_slice::<f32>(),
            out_ref.as_slice::<f32>(),
            "opaque KVCache(StandardFormat, grow 포함) attention != q4_0 round-trip baseline"
        );
    }

    /// Stage 1: opaque KVCache prefill 경로 smoke — `write_kv_batch`(seq>1) + prefill
    /// attention(dequant floor → `prefill_attention`)이 유한 출력 + current_pos 일치.
    #[test]
    fn opaque_kvcache_prefill_smoke() {
        use crate::buffer::Buffer;
        use crate::buffer::opaque::OpaqueBuffer;
        use crate::kv_cache_ops::KVLayout;
        use crate::memory::Memory;
        use crate::memory::galloc::Galloc;
        use argus_extension_api::{KVLayoutDesc, Packing, ScaleLayout};

        let kv_heads = 1usize;
        let head_dim = 32usize; // 1 block/head
        let n_heads_q = 1usize;
        let seq = 4usize;
        let cap = 8usize;
        let desc = KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        };
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        let nbytes = desc.bytes_for_elems(kv_heads * cap * head_dim).unwrap();
        let shape = Shape::new(vec![1, kv_heads, cap, head_dim]);
        let mk = || -> Tensor {
            let inner = mem.alloc_kv(nbytes, DType::U8).unwrap();
            let op: Arc<dyn Buffer> = Arc::new(OpaqueBuffer::new(inner, desc));
            Tensor::new(shape.clone(), op, backend.clone())
        };
        let cache = KVCache::new_dynamic(mk(), mk(), cap, cap, kv_heads, head_dim, mem.clone())
            .with_layout(KVLayout::HeadMajor);
        let fmt = StandardFormat::new(0, cache);

        let kb: Vec<f32> = (0..seq * kv_heads * head_dim)
            .map(|i| (i as f32 % 5.0) - 2.0)
            .collect();
        let vb: Vec<f32> = (0..seq * kv_heads * head_dim)
            .map(|i| (i as f32 % 3.0) - 1.0)
            .collect();
        let kt = f32_tensor(vec![1, seq, kv_heads, head_dim], &kb);
        let vt = f32_tensor(vec![1, seq, kv_heads, head_dim], &vb);
        fmt.write_kv_batch(&kt, &vt, backend.as_ref()).unwrap();
        assert_eq!(fmt.current_pos(), seq);

        let q = f32_tensor(
            vec![1, seq, n_heads_q, head_dim],
            &vec![0.3f32; seq * n_heads_q * head_dim],
        );
        let mut out = f32_tensor(
            vec![1, seq, n_heads_q * head_dim],
            &vec![0.0; seq * n_heads_q * head_dim],
        );
        fmt.attention_into(
            &q,
            backend.as_ref(),
            &mut out,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
            None,
        )
        .unwrap();
        for &x in out.as_slice::<f32>() {
            assert!(
                x.is_finite(),
                "opaque prefill attention 출력은 유한해야 한다"
            );
        }
    }

    /// Stage 2 GATE: opaque KVCache 의 `prune_prefix`(eviction shift arm) + 그로 인한
    /// `shrink_to_fit_opaque`(release_unused_pages 경유) 후 attention 결과가, 생존 토큰의 q4_0
    /// round-trip baseline 과 **bit-identical**. cap=128, prune 후 current_pos=4 → 64 으로 shrink 발동.
    #[test]
    fn opaque_kvcache_prune_prefix_bit_identical_to_q4_0_roundtrip() {
        use crate::buffer::Buffer;
        use crate::buffer::opaque::OpaqueBuffer;
        use crate::kv_cache_ops::KVLayout;
        use crate::memory::Memory;
        use crate::memory::galloc::Galloc;
        use crate::quant::{BlockQ4_0, QK4_0};
        use argus_extension_api::{KVLayoutDesc, Packing, ScaleLayout};

        let kv_heads = 2usize;
        let head_dim = 64usize;
        let n_heads_q = 2usize;
        let n_tokens = 6usize;
        let prune = 2usize; // prune_prefix(2): 토큰 0,1 제거, 2..6 → 위치 0..4
        let remaining = n_tokens - prune;
        let cap = 128usize; // prune 후 shrink_to_fit_opaque(→64) 발동
        let desc = KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        };
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());

        let gen_val = |t: usize, h: usize, d: usize, salt: f32| -> f32 {
            (((t * 7 + h * 13 + d * 3) % 17) as f32 - 8.0) * 0.1 + salt
        };
        let mut k_tokens: Vec<Vec<f32>> = Vec::new();
        let mut v_tokens: Vec<Vec<f32>> = Vec::new();
        for t in 0..n_tokens {
            let mut k = vec![0.0f32; kv_heads * head_dim];
            let mut v = vec![0.0f32; kv_heads * head_dim];
            for h in 0..kv_heads {
                for d in 0..head_dim {
                    k[h * head_dim + d] = gen_val(t, h, d, 0.0);
                    v[h * head_dim + d] = gen_val(t, h, d, 0.5);
                }
            }
            k_tokens.push(k);
            v_tokens.push(v);
        }

        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        let nbytes = desc.bytes_for_elems(kv_heads * cap * head_dim).unwrap();
        let shape = Shape::new(vec![1, kv_heads, cap, head_dim]);
        let mk = || -> Tensor {
            let inner = mem.alloc_kv(nbytes, DType::U8).unwrap();
            let op: Arc<dyn Buffer> = Arc::new(OpaqueBuffer::new(inner, desc));
            Tensor::new(shape.clone(), op, backend.clone())
        };
        let mut cache = KVCache::new_dynamic(mk(), mk(), cap, cap, kv_heads, head_dim, mem.clone())
            .with_layout(KVLayout::HeadMajor);
        for t in 0..n_tokens {
            let kt = f32_tensor(vec![1, 1, kv_heads, head_dim], &k_tokens[t]);
            let vt = f32_tensor(vec![1, 1, kv_heads, head_dim], &v_tokens[t]);
            cache.update(&kt, &vt).unwrap();
        }
        cache.prune_prefix(prune).unwrap();
        assert_eq!(cache.current_pos(), remaining);
        let fmt = StandardFormat::new(0, cache);

        let q_data: Vec<f32> = (0..n_heads_q * head_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.07)
            .collect();
        let q = f32_tensor(vec![1, 1, n_heads_q, head_dim], &q_data);
        let mut out_opaque = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.0; n_heads_q * head_dim],
        );
        fmt.attention_into(
            &q,
            backend.as_ref(),
            &mut out_opaque,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
            None,
        )
        .unwrap();

        // reference: 생존 토큰 [prune..n_tokens) 의 q4_0 round-trip 을 위치 [0..remaining) 에 배치.
        let rcap = fmt.capacity();
        let mut ref_k = vec![0.0f32; kv_heads * rcap * head_dim];
        let mut ref_v = vec![0.0f32; kv_heads * rcap * head_dim];
        for p in 0..remaining {
            let src_t = prune + p;
            for h in 0..kv_heads {
                for blk in 0..(head_dim / QK4_0) {
                    let lo = h * head_dim + blk * QK4_0;
                    let mut ka = [0.0f32; QK4_0];
                    ka.copy_from_slice(&k_tokens[src_t][lo..lo + QK4_0]);
                    let mut ko = [0.0f32; QK4_0];
                    BlockQ4_0::quantize(&ka).dequantize(&mut ko);
                    let mut va = [0.0f32; QK4_0];
                    va.copy_from_slice(&v_tokens[src_t][lo..lo + QK4_0]);
                    let mut vo = [0.0f32; QK4_0];
                    BlockQ4_0::quantize(&va).dequantize(&mut vo);
                    let base = (h * rcap + p) * head_dim + blk * QK4_0;
                    ref_k[base..base + QK4_0].copy_from_slice(&ko);
                    ref_v[base..base + QK4_0].copy_from_slice(&vo);
                }
            }
        }
        let ref_k_t = f32_tensor(vec![1, kv_heads, rcap, head_dim], &ref_k);
        let ref_v_t = f32_tensor(vec![1, kv_heads, rcap, head_dim], &ref_v);
        let mut out_ref = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.0; n_heads_q * head_dim],
        );
        backend
            .attention_gen(
                &q,
                &ref_k_t,
                &ref_v_t,
                &mut out_ref,
                n_heads_q,
                kv_heads,
                head_dim,
                remaining,
                None,
            )
            .unwrap();

        assert_eq!(
            out_opaque.as_slice::<f32>(),
            out_ref.as_slice::<f32>(),
            "opaque KVCache prune_prefix(+shrink) attention != q4_0 round-trip baseline"
        );
    }

    /// Stage 3 GATE: opaque `apply_weighted_merges`(weighted-merge descriptor-generic merge)가 동일
    /// 데이터의 q4_0 round-trip(dequant→weighted sum→requantize→dequant) 과 **bit-identical**.
    /// into=0 ← from=[(1,0.3),(2,0.2)], into_w=0.5. K/V 독립 검증. (구 merge_row_weighted_q4 와 동형.)
    #[test]
    fn opaque_kvcache_weighted_merge_bit_identical_to_q4_0_roundtrip() {
        use crate::buffer::Buffer;
        use crate::buffer::opaque::OpaqueBuffer;
        use crate::format::dequant_to_f32_tensor;
        use crate::kv_cache_ops::KVLayout;
        use crate::memory::Memory;
        use crate::memory::galloc::Galloc;
        use crate::quant::{BlockQ4_0, QK4_0};
        use argus_extension_api::{KVLayoutDesc, MergeAxis, Packing, ScaleLayout, WeightedMerge};

        let kv_heads = 2usize;
        let head_dim = 64usize;
        let n_tokens = 3usize;
        let cap = 8usize;
        let desc = KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        };
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());

        let gen_val = |t: usize, h: usize, d: usize, salt: f32| -> f32 {
            (((t * 7 + h * 13 + d * 3) % 17) as f32 - 8.0) * 0.1 + salt
        };
        let mk_tokens = |salt: f32| -> Vec<Vec<f32>> {
            (0..n_tokens)
                .map(|t| {
                    let mut tok = vec![0.0f32; kv_heads * head_dim];
                    for h in 0..kv_heads {
                        for d in 0..head_dim {
                            tok[h * head_dim + d] = gen_val(t, h, d, salt);
                        }
                    }
                    tok
                })
                .collect()
        };
        let k_tokens = mk_tokens(0.0);
        let v_tokens = mk_tokens(0.5);

        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        let nbytes = desc.bytes_for_elems(kv_heads * cap * head_dim).unwrap();
        let shape = Shape::new(vec![1, kv_heads, cap, head_dim]);
        let mk_buf = || {
            let inner = mem.alloc_kv(nbytes, DType::U8).unwrap();
            let op: Arc<dyn Buffer> = Arc::new(OpaqueBuffer::new(inner, desc));
            Tensor::new(shape.clone(), op, backend.clone())
        };
        let mut cache = KVCache::new_dynamic(
            mk_buf(),
            mk_buf(),
            cap,
            cap,
            kv_heads,
            head_dim,
            mem.clone(),
        )
        .with_layout(KVLayout::HeadMajor);
        for t in 0..n_tokens {
            let kt = f32_tensor(vec![1, 1, kv_heads, head_dim], &k_tokens[t]);
            let vt = f32_tensor(vec![1, 1, kv_heads, head_dim], &v_tokens[t]);
            cache.update(&kt, &vt).unwrap();
        }
        let merge = WeightedMerge {
            into: 0,
            into_weight: 0.5,
            from: vec![(1, 0.3), (2, 0.2)],
            apply_to: MergeAxis::Both,
        };
        apply_weighted_merges(&mut cache, std::slice::from_ref(&merge));

        let k_deq = dequant_to_f32_tensor(&cache.k_buffer).unwrap();
        let v_deq = dequant_to_f32_tensor(&cache.v_buffer).unwrap();
        let k_out = k_deq.as_slice::<f32>();
        let v_out = v_deq.as_slice::<f32>();

        // q4_0 round-trip(quantize→dequantize) of a single block.
        let rt = |blk: &[f32]| -> [f32; QK4_0] {
            let mut a = [0.0f32; QK4_0];
            a.copy_from_slice(blk);
            let mut o = [0.0f32; QK4_0];
            BlockQ4_0::quantize(&a).dequantize(&mut o);
            o
        };
        let check = |tokens: &[Vec<f32>], out: &[f32], label: &str| {
            for h in 0..kv_heads {
                for blk in 0..(head_dim / QK4_0) {
                    let lo = h * head_dim + blk * QK4_0;
                    let rt0 = rt(&tokens[0][lo..lo + QK4_0]);
                    let rt1 = rt(&tokens[1][lo..lo + QK4_0]);
                    let rt2 = rt(&tokens[2][lo..lo + QK4_0]);
                    // into(pos0): 0.5·rt0 + 0.3·rt1 + 0.2·rt2 → requantize → dequant.
                    let mut merged = [0.0f32; QK4_0];
                    for i in 0..QK4_0 {
                        merged[i] = 0.5 * rt0[i] + 0.3 * rt1[i] + 0.2 * rt2[i];
                    }
                    let stored0 = rt(&merged);
                    let base0 = (h * cap) * head_dim + blk * QK4_0;
                    assert_eq!(
                        &out[base0..base0 + QK4_0],
                        &stored0,
                        "{label} into pos0 head{h} blk{blk}"
                    );
                    // from(pos1,2): 불변(roundtrip).
                    let base1 = (h * cap + 1) * head_dim + blk * QK4_0;
                    assert_eq!(
                        &out[base1..base1 + QK4_0],
                        &rt1,
                        "{label} pos1 head{h} blk{blk}"
                    );
                    let base2 = (h * cap + 2) * head_dim + blk * QK4_0;
                    assert_eq!(
                        &out[base2..base2 + QK4_0],
                        &rt2,
                        "{label} pos2 head{h} blk{blk}"
                    );
                }
            }
        };
        check(&k_tokens, k_out, "K");
        check(&v_tokens, v_out, "V");
    }

    #[test]
    fn test_geometry_delegates_to_kvcache() {
        let cache = make_cache(8, 2, 4);
        let fmt = StandardFormat::new(3, cache);
        assert_eq!(fmt.idx(), 3);
        assert_eq!(fmt.capacity(), 8);
        assert_eq!(fmt.current_pos(), 0);
    }

    #[test]
    fn test_take_put_inner_round_trip() {
        // Phase α-K BC (3d) S1: take_inner → put_inner 는 identity. 토큰 write 후 take 한 cache 가
        // 데이터·pos 를 보존하고, put 후 wrapper 가 다시 정상 접근 가능해야 한다(eviction UER seam).
        let kv_heads = 2;
        let head_dim = 4;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));
        let token = vec![7.0f32; kv_heads * head_dim];
        let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 1);

        // take: 꺼낸 cache 가 데이터·pos 보존, wrapper 는 placeholder(pos=0) 보유.
        let taken = fmt.take_inner();
        assert_eq!(taken.current_pos, 1);
        assert_eq!(taken.k_buffer.as_slice::<f32>()[0], 7.0);
        assert_eq!(fmt.current_pos(), 0, "take 후 wrapper 는 placeholder");

        // put: 복귀하면 wrapper 가 원래 cache 를 다시 노출.
        fmt.put_inner(taken);
        assert_eq!(fmt.current_pos(), 1, "put 후 원래 cache 복귀");
        let guard = fmt.inner.lock().unwrap();
        assert_eq!(guard.cache.k_buffer.as_slice::<f32>()[0], 7.0);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_plan_geometry_delegates_and_zeroes_residual() {
        // (3p) ④-a: plan_geometry()가 inner KVCache current_pos/capacity 를 정확히 위임하고
        // standard 의 res_pos/q2_tokens 는 0 이어야 한다.
        let fmt = StandardFormat::new(0, make_cache(8, 2, 4));
        let g = fmt.plan_geometry();
        assert_eq!(g.capacity, 8);
        assert_eq!(g.current_pos, 0);
        assert_eq!(g.res_pos, 0);
        assert_eq!(g.q2_tokens, 0);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_plan_advance_bumps_current_pos() {
        // (3p) ④-a: plan_advance(n) 후 plan_geometry().current_pos 가 증가해야 한다
        // (execute 의 레이어 끝 advance 미러).
        let kv_heads = 2;
        let head_dim = 4;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        // ensure_capacity is not required for advance_pos (pure position bump).
        fmt.plan_advance(1);
        assert_eq!(fmt.plan_geometry().current_pos, 1);
        assert_eq!(
            fmt.current_pos(),
            1,
            "plan_advance must mutate the same cache"
        );

        fmt.plan_advance(2);
        assert_eq!(fmt.plan_geometry().current_pos, 3);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_plan_lock_reads_buffer() {
        // (3p) ④-a: plan_lock() guard seam — build_plan 가 KV buffer(`k_buffer`)에
        // 도달하는 경로(guard 를 잡고 `&KVCache` 슬라이스를 만들어 byte-identical build_plan
        // 본문을 재사용).
        let kv_heads = 1;
        let head_dim = 2;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        // write one token = [7, 7], then read it back through the guard seam.
        let t = vec![7.0f32; kv_heads * head_dim];
        let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
        let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
        fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();

        let guard = fmt.plan_lock();
        assert_eq!(guard.cache.capacity(), 8);
        assert_eq!(guard.cache.k_buffer.as_slice::<f32>()[0], 7.0);
    }

    #[test]
    fn test_write_kv_advances_pos() {
        let kv_heads = 2;
        let head_dim = 4;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        // single-token write: [1, 1, kv_heads, head_dim]
        let token = vec![1.0f32; kv_heads * head_dim];
        let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 1);

        // batch write: 2 tokens
        let batch = vec![2.0f32; 2 * kv_heads * head_dim];
        let kb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        let vb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        fmt.write_kv_batch(&kb, &vb, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 3);
    }

    #[test]
    fn test_write_kv_f16_casts_f32_input() {
        use half::f16;
        // F16 cache + CpuBackend(is_gpu()==false) → 비-F32 cast 분기(GPU scatter fast-path 미진입).
        // F32 입력이 F16 으로 cast 되어 저장되는지 검증 — `KVCache::update` 는 cast 를 안 하므로
        // 이 흡수가 빠지면 dtype 미일치 silent garbage. (forward_gen 의 `kv_dtype != F32` 흡수.)
        let kv_heads = 2;
        let head_dim = 4;
        let row = kv_heads * head_dim;
        let fmt = StandardFormat::new(0, make_f16_dynamic_cache(8, kv_heads, head_dim));

        // F16 로 정확히 표현 가능한 값(0.5 배수).
        let token0: Vec<f32> = (0..row).map(|i| (i as f32) * 0.5).collect();
        let k0 = f32_tensor(vec![1, 1, kv_heads, head_dim], &token0);
        let v0 = f32_tensor(vec![1, 1, kv_heads, head_dim], &token0);
        fmt.write_kv(&k0, &v0, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 1);

        // 두 번째 토큰 — lazy scratch 재사용 경로(k_cast/v_cast 이미 Some).
        let token1: Vec<f32> = (0..row).map(|i| (i as f32) + 1.0).collect();
        let k1 = f32_tensor(vec![1, 1, kv_heads, head_dim], &token1);
        let v1 = f32_tensor(vec![1, 1, kv_heads, head_dim], &token1);
        fmt.write_kv(&k1, &v1, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 2);

        // F16 buffer 검증: SeqMajor 라 pos*row + idx.
        let guard = fmt.inner.lock().unwrap();
        let k16 = guard.cache.k_buffer.as_slice::<f16>();
        let v16 = guard.cache.v_buffer.as_slice::<f16>();
        for (i, &exp) in token0.iter().enumerate() {
            assert!(
                (k16[i].to_f32() - exp).abs() < 1e-3,
                "pos0 K[{i}] expected {exp}, got {}",
                k16[i].to_f32()
            );
            assert!((v16[i].to_f32() - exp).abs() < 1e-3);
        }
        for (i, &exp) in token1.iter().enumerate() {
            assert!(
                (k16[row + i].to_f32() - exp).abs() < 1e-3,
                "pos1 K[{i}] expected {exp}, got {}",
                k16[row + i].to_f32()
            );
            assert!((v16[row + i].to_f32() - exp).abs() < 1e-3);
        }
    }

    #[test]
    fn test_write_kv_f16_batch_then_decode_reallocs_scratch() {
        use half::f16;
        // write_kv_batch(seq=2) 가 cast scratch 를 seq=2 크기로 굳힌 뒤 write_kv(seq=1) 가 와도
        // scratch 가 shape 변화에 맞춰 재할당되어 둘 다 정확해야 한다(가드 부재 시 cast zip 절단).
        let kv_heads = 2;
        let head_dim = 4;
        let row = kv_heads * head_dim;
        let fmt = StandardFormat::new(0, make_f16_dynamic_cache(8, kv_heads, head_dim));

        // prefill batch: 2 tokens. token@pos p = 0.5*(p+1) 균일.
        let batch: Vec<f32> = (0..2 * row)
            .map(|i| if i < row { 0.5 } else { 1.0 })
            .collect();
        let kb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        let vb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        fmt.write_kv_batch(&kb, &vb, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 2);

        // decode single: shape [1,1,...] — scratch 재할당 트리거.
        let dec = vec![2.5f32; row];
        let kd = f32_tensor(vec![1, 1, kv_heads, head_dim], &dec);
        let vd = f32_tensor(vec![1, 1, kv_heads, head_dim], &dec);
        fmt.write_kv(&kd, &vd, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 3);

        let guard = fmt.inner.lock().unwrap();
        let k16 = guard.cache.k_buffer.as_slice::<f16>();
        // pos0 = 0.5, pos1 = 1.0 (batch), pos2 = 2.5 (decode).
        for i in 0..row {
            assert!(
                (k16[i].to_f32() - 0.5).abs() < 1e-3,
                "pos0[{i}]={}",
                k16[i].to_f32()
            );
            assert!(
                (k16[row + i].to_f32() - 1.0).abs() < 1e-3,
                "pos1[{i}]={}",
                k16[row + i].to_f32()
            );
            assert!(
                (k16[2 * row + i].to_f32() - 2.5).abs() < 1e-3,
                "pos2[{i}]={}",
                k16[2 * row + i].to_f32()
            );
        }
    }

    #[test]
    fn test_write_kv_f16_requires_dynamic_cache() {
        // 비-F32 cast 는 inner cache 의 allocator 가 필요. `KVCache::new()`(memory=None)로 만든
        // F16 cache 는 scratch 할당 불가 → 명시적 에러(silent 오동작 금지).
        let kv_heads = 1;
        let head_dim = 4;
        let total = 8 * kv_heads * head_dim;
        let buf_k = Arc::new(SharedBuffer::new(total * 2, DType::F16));
        let buf_v = Arc::new(SharedBuffer::new(total * 2, DType::F16));
        let k = Tensor::new(
            Shape::new(vec![1, 8, kv_heads, head_dim]),
            buf_k,
            Arc::new(CpuBackend::new()),
        );
        let v = Tensor::new(
            Shape::new(vec![1, 8, kv_heads, head_dim]),
            buf_v,
            Arc::new(CpuBackend::new()),
        );
        let fmt = StandardFormat::new(0, KVCache::new(k, v, 8)); // memory=None

        let token = vec![1.0f32; kv_heads * head_dim];
        let kt = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        let vt = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        let err = fmt.write_kv(&kt, &vt, &CpuBackend::new());
        assert!(
            err.is_err(),
            "F16 cast on pre-allocated (memory=None) cache must error"
        );
    }

    #[test]
    fn test_write_kv_batch_f32_matches_sequential_decode() {
        // C3 (§9.1-BC1-CONTRACT): multi-token write_kv_batch must produce a buffer
        // bit-identical to writing the same tokens one-by-one via write_kv (decode).
        // host(CpuBackend, is_gpu=false) → cast/update fallback covers correctness;
        // GPU scatter fast-path is device-verified.
        let kv_heads = 2;
        let head_dim = 4;
        let row = kv_heads * head_dim;
        let seq = 3;

        // distinct per-(token, elem) values, exactly F32-representable.
        let batch: Vec<f32> = (0..seq * row).map(|i| i as f32).collect();

        // (A) batch write.
        let fmt_batch = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));
        let kb = f32_tensor(vec![1, seq, kv_heads, head_dim], &batch);
        let vb = f32_tensor(vec![1, seq, kv_heads, head_dim], &batch);
        fmt_batch
            .write_kv_batch(&kb, &vb, &CpuBackend::new())
            .unwrap();
        assert_eq!(fmt_batch.current_pos(), seq);

        // (B) reference: same tokens written one at a time via write_kv (decode).
        let fmt_seq = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));
        for s in 0..seq {
            let tok = &batch[s * row..(s + 1) * row];
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], tok);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], tok);
            fmt_seq.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        }
        assert_eq!(fmt_seq.current_pos(), seq);

        // K/V buffers must be byte-identical.
        let gb = fmt_batch.inner.lock().unwrap();
        let gs = fmt_seq.inner.lock().unwrap();
        let kb_buf = gb.cache.k_buffer.as_slice::<f32>();
        let ks_buf = gs.cache.k_buffer.as_slice::<f32>();
        let vb_buf = gb.cache.v_buffer.as_slice::<f32>();
        let vs_buf = gs.cache.v_buffer.as_slice::<f32>();
        assert_eq!(kb_buf, ks_buf, "K batch buffer != sequential-decode buffer");
        assert_eq!(vb_buf, vs_buf, "V batch buffer != sequential-decode buffer");
    }

    #[test]
    fn test_write_kv_batch_f16_matches_sequential_decode() {
        use half::f16;
        // F16 cache: batch write goes through the non-F32 cast path on CpuBackend
        // (supports_kv_scatter_batch()==false). Must equal sequential decode writes.
        let kv_heads = 2;
        let head_dim = 4;
        let row = kv_heads * head_dim;
        let seq = 3;

        // 0.5-multiples so values are exactly representable in F16.
        let batch: Vec<f32> = (0..seq * row).map(|i| (i as f32) * 0.5).collect();

        // (A) batch write.
        let fmt_batch = StandardFormat::new(0, make_f16_dynamic_cache(8, kv_heads, head_dim));
        let kb = f32_tensor(vec![1, seq, kv_heads, head_dim], &batch);
        let vb = f32_tensor(vec![1, seq, kv_heads, head_dim], &batch);
        fmt_batch
            .write_kv_batch(&kb, &vb, &CpuBackend::new())
            .unwrap();
        assert_eq!(fmt_batch.current_pos(), seq);

        // (B) reference: sequential decode writes.
        let fmt_seq = StandardFormat::new(0, make_f16_dynamic_cache(8, kv_heads, head_dim));
        for s in 0..seq {
            let tok = &batch[s * row..(s + 1) * row];
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], tok);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], tok);
            fmt_seq.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        }
        assert_eq!(fmt_seq.current_pos(), seq);

        let gb = fmt_batch.inner.lock().unwrap();
        let gs = fmt_seq.inner.lock().unwrap();
        let kb_buf = gb.cache.k_buffer.as_slice::<f16>();
        let ks_buf = gs.cache.k_buffer.as_slice::<f16>();
        let vb_buf = gb.cache.v_buffer.as_slice::<f16>();
        let vs_buf = gs.cache.v_buffer.as_slice::<f16>();
        assert_eq!(
            kb_buf, ks_buf,
            "F16 K batch buffer != sequential-decode buffer"
        );
        assert_eq!(
            vb_buf, vs_buf,
            "F16 V batch buffer != sequential-decode buffer"
        );
    }

    #[test]
    fn test_attention_into_f32_uniform() {
        // current_pos==0 is illegal for softmax; write 2 identical tokens so
        // softmax is uniform and output = the (identical) V row.
        let kv_heads = 1;
        let head_dim = 4;
        let n_heads_q = 1;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        let k_row = vec![0.0f32; head_dim]; // zero K → all scores equal → uniform softmax
        let v_row = vec![5.0f32; head_dim];
        for _ in 0..2 {
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &k_row);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &v_row);
            fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        }
        assert_eq!(fmt.current_pos(), 2);

        let q = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![1.0; head_dim]);
        let mut out = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![0.0; head_dim]);
        let backend = CpuBackend::new();
        let mut scores = vec![0.0f32; n_heads_q * 2];

        fmt.attention_into(
            &q,
            &backend,
            &mut out,
            AttnDims {
                n_heads_q,
                window: None,
            },
            Some(&mut scores),
            None,
        )
        .unwrap();

        // Uniform attention over identical V rows → out == V row.
        let o = out.as_slice::<f32>();
        for &x in o {
            assert!((x - 5.0).abs() < 1e-4, "expected 5.0, got {x}");
        }
        // post-softmax scores: 2 equal weights summing to 1.
        assert!((scores[0] - 0.5).abs() < 1e-4);
        assert!((scores[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_prefill_attention_causal_uniform() {
        // C-1 (①-b): multi-token prefill attention via attention_into(seq_len>1).
        // K=0 → 모든 score 0 → uniform softmax. V[pos]=pos (broadcast). causal mask 로
        // query row r 은 cache pos 0..=r 만 attend → out[r] = mean(0..=r) = r/2.
        // write_kv_batch(prefill write) + attention_into(prefill arm) 합동 검증.
        let kv_heads = 1;
        let head_dim = 4;
        let n_heads_q = 1;
        let seq = 4;
        let fmt = StandardFormat::new(0, make_cache(16, kv_heads, head_dim));
        let backend = CpuBackend::new();

        let k_data = vec![0.0f32; seq * kv_heads * head_dim];
        let mut v_data = vec![0.0f32; seq * kv_heads * head_dim];
        for p in 0..seq {
            for d in 0..head_dim {
                v_data[p * kv_heads * head_dim + d] = p as f32;
            }
        }
        let kb = f32_tensor(vec![1, seq, kv_heads, head_dim], &k_data);
        let vb = f32_tensor(vec![1, seq, kv_heads, head_dim], &v_data);
        fmt.write_kv_batch(&kb, &vb, &backend).unwrap();
        assert_eq!(fmt.current_pos(), seq);

        // q 값은 무관(K=0 → score 0). out = [1, seq, n_heads_q*head_dim].
        let q = f32_tensor(
            vec![1, seq, n_heads_q, head_dim],
            &vec![1.0; seq * n_heads_q * head_dim],
        );
        let mut out = f32_tensor(
            vec![1, seq, n_heads_q * head_dim],
            &vec![0.0; seq * n_heads_q * head_dim],
        );

        fmt.attention_into(
            &q,
            &backend,
            &mut out,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
            None,
        )
        .unwrap();

        let o = out.as_slice::<f32>();
        for r in 0..seq {
            let expected = r as f32 / 2.0; // mean(0..=r)
            for d in 0..head_dim {
                let got = o[r * head_dim + d];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "row {r} d {d}: expected {expected}, got {got}"
                );
            }
        }
    }

    // ───────────────── R-P1-1 PFA producer (prefill_attention_scores) ─────────────────

    /// `prefill_attention_scores` 를 op-for-op 미러하는 eager reference(Gate-1 bit-exact 핀, §6.1).
    /// scalar dot → ×scale → max → exp/denom → divide, 동일 loop 순서. producer 가 spec 에서 벗어나면
    /// 이 reference 와 발산 → 회귀 검출(§6.6: PFA-vs-eager self-referential, flash `out` 무관).
    #[allow(clippy::too_many_arguments)]
    fn pfa_reference(
        q: &[f32],
        k: &[f32],
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        seq_len: usize,
        cache_seq_len: usize,
        k_pos_stride: usize,
        kv_head_stride: usize,
        q_start_pos: usize,
        q_window: usize,
        window: Option<usize>,
    ) -> Vec<f32> {
        let prefix_len = cache_seq_len;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let gqa = n_heads_q / n_heads_kv;
        let q_row = n_heads_q * head_dim;
        let qwin = q_window.min(seq_len);
        let qwin_start = seq_len - qwin;
        let mut out = vec![0.0f32; n_heads_q * prefix_len];
        let mut scratch = vec![0.0f32; prefix_len];
        for h in 0..n_heads_q {
            let kvh = h / gqa;
            let base = h * prefix_len;
            for r in qwin_start..seq_len {
                let p = q_start_pos + r;
                let lo = match window {
                    // 미러: producer 와 동일 saturating_sub(1) (w>=1 인 테스트 config 에선 == w-1).
                    Some(w) => p.saturating_sub(w.saturating_sub(1)),
                    None => 0,
                };
                let qb = r * q_row + h * head_dim;
                let mut m = f32::NEG_INFINITY;
                for kp in lo..=p {
                    let kb = kp * k_pos_stride + kvh * kv_head_stride;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[qb + d] * k[kb + d];
                    }
                    let l = dot * scale;
                    scratch[kp] = l;
                    if l > m {
                        m = l;
                    }
                }
                let mut denom = 0.0f32;
                for kp in lo..=p {
                    let e = (scratch[kp] - m).exp();
                    scratch[kp] = e;
                    denom += e;
                }
                for kp in lo..=p {
                    out[base + kp] += scratch[kp] / denom;
                }
            }
        }
        out
    }

    /// Independent EAGER-softmax reference (NOT op-identical to the producer):
    /// materializes the full causal logit row and runs a textbook softmax in f64,
    /// sharing no float-rounding path with `prefill_attention_scores`. Validates
    /// that the producer's fused f32 softmax is *numerically* correct (approx
    /// equality), complementing the bit-exact op-order pin (`pfa_reference`). This
    /// is the ground-truth guarantee IMP-2's `answer_attention` dump relies on.
    #[allow(clippy::too_many_arguments)]
    fn pfa_eager_reference(
        q: &[f32],
        k: &[f32],
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        seq_len: usize,
        cache_seq_len: usize,
        k_pos_stride: usize,
        kv_head_stride: usize,
        q_start_pos: usize,
        q_window: usize,
        window: Option<usize>,
    ) -> Vec<f32> {
        let prefix_len = cache_seq_len;
        let scale = 1.0f64 / (head_dim as f64).sqrt();
        let gqa = n_heads_q / n_heads_kv;
        let q_row = n_heads_q * head_dim;
        let qwin = q_window.min(seq_len);
        let qwin_start = seq_len - qwin;
        let mut out = vec![0.0f64; n_heads_q * prefix_len];
        for h in 0..n_heads_q {
            let kvh = h / gqa;
            let base = h * prefix_len;
            for r in qwin_start..seq_len {
                let p = q_start_pos + r;
                let lo = match window {
                    Some(w) => p.saturating_sub(w.saturating_sub(1)),
                    None => 0,
                };
                let qb = r * q_row + h * head_dim;
                // Materialize all causal logits in f64 (independent of the producer's
                // fused single-pass online softmax).
                let logits: Vec<f64> = (lo..=p)
                    .map(|kp| {
                        let kb = kp * k_pos_stride + kvh * kv_head_stride;
                        let mut dot = 0.0f64;
                        for d in 0..head_dim {
                            dot += q[qb + d] as f64 * k[kb + d] as f64;
                        }
                        dot * scale
                    })
                    .collect();
                let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let denom: f64 = logits.iter().map(|&l| (l - m).exp()).sum();
                for (i, kp) in (lo..=p).enumerate() {
                    out[base + kp] += (logits[i] - m).exp() / denom;
                }
            }
        }
        out.iter().map(|&x| x as f32).collect()
    }

    #[test]
    fn pfa_matches_eager_softmax_reference() {
        // IMP-2 ground-truth check: the producer's f32 fused softmax must match an
        // independent eager (f64, fully-materialized) softmax within tolerance,
        // over MHA/GQA × SeqMajor/HeadMajor × windowed/full. Approximate (not
        // bit-exact) — the two reference paths share no rounding.
        let head_dim = 8;
        let seq_len = 6;
        let q_start_pos = 0;
        let cache_seq_len = seq_len;
        let q_window = 3;
        let kv_capacity = 16;
        for (nq, nkv, layout) in [(4, 4, "seq"), (4, 2, "seq"), (4, 4, "head"), (4, 2, "head")] {
            for window in [None, Some(4usize)] {
                let q: Vec<f32> = (0..seq_len * nq * head_dim)
                    .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
                    .collect();
                let (k_pos_stride, kv_head_stride, k_len) = if layout == "seq" {
                    (nkv * head_dim, head_dim, cache_seq_len * nkv * head_dim)
                } else {
                    (
                        head_dim,
                        kv_capacity * head_dim,
                        nkv * kv_capacity * head_dim,
                    )
                };
                let k: Vec<f32> = (0..k_len).map(|i| ((i % 17) as f32 - 8.0) * 0.07).collect();

                let mut got = vec![0.0f32; nq * cache_seq_len];
                prefill_attention_scores(
                    &q,
                    &k,
                    nq,
                    nkv,
                    head_dim,
                    seq_len,
                    cache_seq_len,
                    k_pos_stride,
                    kv_head_stride,
                    q_start_pos,
                    q_window,
                    window,
                    &mut got,
                );
                let want = pfa_eager_reference(
                    &q,
                    &k,
                    nq,
                    nkv,
                    head_dim,
                    seq_len,
                    cache_seq_len,
                    k_pos_stride,
                    kv_head_stride,
                    q_start_pos,
                    q_window,
                    window,
                );
                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (g - w).abs() < 1e-4,
                        "PFA != eager softmax at {i} \
                         (nq={nq}, nkv={nkv}, layout={layout}, window={window:?}): {g} vs {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn pfa_bit_exact_matches_eager_reference() {
        let head_dim = 8;
        let seq_len = 6;
        let q_start_pos = 0; // fresh prefill → cache_seq_len == seq_len.
        let cache_seq_len = seq_len;
        let q_window = 3;
        let kv_capacity = 16; // HeadMajor stride 용(>= cache_seq_len).
        // (n_heads_q, n_heads_kv, layout): MHA/GQA × SeqMajor/HeadMajor.
        for (nq, nkv, layout) in [(4, 4, "seq"), (4, 2, "seq"), (4, 4, "head"), (4, 2, "head")] {
            for window in [None, Some(4usize)] {
                let q: Vec<f32> = (0..seq_len * nq * head_dim)
                    .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
                    .collect();
                let (k_pos_stride, kv_head_stride, k_len) = if layout == "seq" {
                    (nkv * head_dim, head_dim, cache_seq_len * nkv * head_dim)
                } else {
                    (
                        head_dim,
                        kv_capacity * head_dim,
                        nkv * kv_capacity * head_dim,
                    )
                };
                let k: Vec<f32> = (0..k_len).map(|i| ((i % 17) as f32 - 8.0) * 0.07).collect();

                let mut got = vec![0.0f32; nq * cache_seq_len];
                prefill_attention_scores(
                    &q,
                    &k,
                    nq,
                    nkv,
                    head_dim,
                    seq_len,
                    cache_seq_len,
                    k_pos_stride,
                    kv_head_stride,
                    q_start_pos,
                    q_window,
                    window,
                    &mut got,
                );
                let want = pfa_reference(
                    &q,
                    &k,
                    nq,
                    nkv,
                    head_dim,
                    seq_len,
                    cache_seq_len,
                    k_pos_stride,
                    kv_head_stride,
                    q_start_pos,
                    q_window,
                    window,
                );
                // bit-exact (not approx) — §6.1 op-order 계약.
                assert_eq!(
                    got, want,
                    "PFA != eager ref (nq={nq}, nkv={nkv}, layout={layout}, window={window:?})"
                );
            }
        }
    }

    #[test]
    fn pfa_softmax_sum_property() {
        // 독립 검증(미러 아님): per-(h, query row) post-softmax 확률은 그 causal/SWA 범위에서 1.0 으로
        // 정규화 → out_scores[h] 의 전체 합 == 그 head 가 SUM-누적한 query row 수(qwin). q_window=1 이면
        // 각 head row 합 == 1.0.
        let head_dim = 4;
        let seq_len = 5;
        let nq = 2;
        let nkv = 1; // GQA(2 q-head → 1 kv-head).
        let cache_seq_len = seq_len;
        let k_pos_stride = nkv * head_dim; // SeqMajor.
        let kv_head_stride = head_dim;
        let q: Vec<f32> = (0..seq_len * nq * head_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.2)
            .collect();
        let k: Vec<f32> = (0..cache_seq_len * nkv * head_dim)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.3)
            .collect();
        for q_window in [1usize, 3] {
            let mut out = vec![0.0f32; nq * cache_seq_len];
            prefill_attention_scores(
                &q,
                &k,
                nq,
                nkv,
                head_dim,
                seq_len,
                cache_seq_len,
                k_pos_stride,
                kv_head_stride,
                0,
                q_window,
                None,
                &mut out,
            );
            let qwin = q_window.min(seq_len) as f32;
            for h in 0..nq {
                let sum: f32 = out[h * cache_seq_len..(h + 1) * cache_seq_len].iter().sum();
                assert!(
                    (sum - qwin).abs() < 1e-4,
                    "head {h} q_window {q_window}: row-sum {sum} != qwin {qwin}"
                );
            }
        }
    }

    #[test]
    fn pfa_side_channel_does_not_touch_out() {
        // Gate-0 armed-identity(§6.2 ii): producer 무장(prefill_scores=Some) 시에도 flash `out` 은
        // 미무장(None)과 byte-identical. 동시에 PFA buffer 는 채워진다(producer 가 실제 발화).
        let kv_heads = 2;
        let head_dim = 4;
        let n_heads_q = 4; // GQA.
        let seq = 5;
        let backend = CpuBackend::new();

        // 비-uniform K/V (PFA 가 trivial 0 이 아니도록).
        let mk = |fmt: &StandardFormat| {
            let mut k_data = vec![0.0f32; seq * kv_heads * head_dim];
            let mut v_data = vec![0.0f32; seq * kv_heads * head_dim];
            for (i, x) in k_data.iter_mut().enumerate() {
                *x = ((i % 11) as f32 - 5.0) * 0.13;
            }
            for (i, x) in v_data.iter_mut().enumerate() {
                *x = ((i % 9) as f32 - 4.0) * 0.21;
            }
            let kb = f32_tensor(vec![1, seq, kv_heads, head_dim], &k_data);
            let vb = f32_tensor(vec![1, seq, kv_heads, head_dim], &v_data);
            fmt.write_kv_batch(&kb, &vb, &backend).unwrap();
        };
        let q_data: Vec<f32> = (0..seq * n_heads_q * head_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();

        // baseline: prefill_scores=None.
        let fmt0 = StandardFormat::new(0, make_cache(16, kv_heads, head_dim));
        mk(&fmt0);
        let q0 = f32_tensor(vec![1, seq, n_heads_q, head_dim], &q_data);
        let mut out0 = f32_tensor(
            vec![1, seq, n_heads_q * head_dim],
            &vec![0.0; seq * n_heads_q * head_dim],
        );
        fmt0.attention_into(
            &q0,
            &backend,
            &mut out0,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
            None,
        )
        .unwrap();

        // armed: prefill_scores=Some.
        let fmt1 = StandardFormat::new(0, make_cache(16, kv_heads, head_dim));
        mk(&fmt1);
        let q1 = f32_tensor(vec![1, seq, n_heads_q, head_dim], &q_data);
        let mut out1 = f32_tensor(
            vec![1, seq, n_heads_q * head_dim],
            &vec![0.0; seq * n_heads_q * head_dim],
        );
        let mut pfa = vec![0.0f32; n_heads_q * seq];
        fmt1.attention_into(
            &q1,
            &backend,
            &mut out1,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
            Some((&mut pfa, 2)),
        )
        .unwrap();

        // `out` byte-identical(side-channel 이 out 미접촉).
        assert_eq!(
            out0.as_slice::<f32>(),
            out1.as_slice::<f32>(),
            "armed PFA changed flash out (must be side-channel)"
        );
        // producer 가 실제 발화(buffer 비어있지 않음).
        assert!(
            pfa.iter().any(|&x| x != 0.0),
            "PFA buffer empty — producer did not fire"
        );
    }

    #[test]
    fn test_attention_into_window_clamps_len() {
        // window=1 must restrict effective_cache_len to 1 (only first token seen
        // by backend.attention_gen). Verify scores buffer reflects single token.
        let kv_heads = 1;
        let head_dim = 4;
        let n_heads_q = 1;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        for p in 0..3 {
            let t = vec![p as f32; head_dim];
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
            fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        }

        let q = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![1.0; head_dim]);
        let mut out = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![0.0; head_dim]);
        let backend = CpuBackend::new();
        let mut scores = vec![0.0f32; n_heads_q * 3];

        fmt.attention_into(
            &q,
            &backend,
            &mut out,
            AttnDims {
                n_heads_q,
                window: Some(1),
            },
            Some(&mut scores),
            None,
        )
        .unwrap();

        // window=1 → only 1 token attended → score[0]=1.0, output = token0 (zeros).
        assert!((scores[0] - 1.0).abs() < 1e-4);
        let o = out.as_slice::<f32>();
        for &x in o {
            assert!(x.abs() < 1e-4, "token0 is all zeros, got {x}");
        }
    }

    // ── SelectiveRead capability 테스트 ──

    /// HeadMajor F32 KVCache: `KVCache::new_dynamic` + `with_layout(HeadMajor)`.
    /// shape [1, kv_heads, capacity, head_dim] 전제.
    fn make_head_major_cache(capacity: usize, kv_heads: usize, head_dim: usize) -> KVCache {
        use crate::kv::kv_cache::KVLayout;
        use crate::memory::Memory;
        use crate::memory::galloc::Galloc;
        let total = kv_heads * capacity * head_dim;
        let k = f32_tensor(vec![1, kv_heads, capacity, head_dim], &vec![0.0f32; total]);
        let v = f32_tensor(vec![1, kv_heads, capacity, head_dim], &vec![0.0f32; total]);
        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        KVCache::new_dynamic(k, v, capacity, capacity, kv_heads, head_dim, mem)
            .with_layout(KVLayout::HeadMajor)
    }

    /// HeadMajor KVCache 에 n_tokens 개의 구분 가능한 토큰을 직접 쓴다.
    /// token i 의 K 값 = pos i + 0.1 * head, V 값 = pos i + 0.5 * head (head 구분용).
    fn write_tokens_headmajor(cache: &mut KVCache, n_tokens: usize) {
        let kv_heads = cache.kv_heads();
        let head_dim = cache.head_dim();
        let capacity = cache.capacity();
        let k = cache.k_buffer.as_mut_slice::<f32>();
        let v = cache.v_buffer.as_mut_slice::<f32>();
        for h in 0..kv_heads {
            let head_off = h * capacity * head_dim;
            for pos in 0..n_tokens {
                let off = head_off + pos * head_dim;
                for d in 0..head_dim {
                    k[off + d] = (pos as f32) + 0.1 * (h as f32);
                    v[off + d] = (pos as f32) + 0.5 * (h as f32);
                }
            }
        }
        // current_pos 갱신
        cache.current_pos = n_tokens;
        cache.high_water_pos = n_tokens;
    }

    /// Area 3 guard: a host-resident cache reports `is_gpu_buffer() == false`, so the
    /// GPU guard at the top of `gather_selected_kv` must NOT trip (no false positive) and
    /// gather must succeed. (The GPU-resident rejection path is on-device-only.)
    #[test]
    fn gather_selected_kv_allows_host_cache() {
        let mut cache = make_head_major_cache(8, 2, 4);
        write_tokens_headmajor(&mut cache, 4);
        let r = gather_selected_kv(&cache, &[0, 1, 2]);
        assert!(r.is_ok(), "host cache gather must succeed: {:?}", r.err());
        let (k, v) = r.unwrap();
        assert_eq!(k.len(), 2 * 3 * 4, "kv_heads * n_sel * head_dim");
        assert_eq!(v.len(), 2 * 3 * 4);
        // head 0, pos 1 → K = 1.0 (pos + 0.1*head); spot-check first element of (h=0, si=1).
        assert!((k[4] - 1.0).abs() < 1e-6, "h0 pos1 K = {}", k[4]);
    }

    /// SelectiveRead: select=전체 토큰 → attention_into 와 bit-identical (F32/HeadMajor).
    ///
    /// Tier 1 게이트: "전체 select == attention_into 와 bit-identical".
    /// NOTE: attention_into 는 SeqMajor kv 경로를 거치고, selective_read 는 HeadMajor→gather→SeqMajor
    /// 후 attention_gen 를 호출한다. 둘 다 최종적으로 동일 gathered F32 데이터로 attention_gen 를 타므로
    /// bit-identical 이어야 한다.
    #[test]
    fn selective_read_full_select_bit_identical_f32() {
        let kv_heads = 1;
        let head_dim = 4;
        let n_heads_q = 1;
        let n_tokens = 4;

        let mut cache_sel = make_head_major_cache(n_tokens + 4, kv_heads, head_dim);
        write_tokens_headmajor(&mut cache_sel, n_tokens);

        // attention_into 결과 기준
        let fmt_ref = StandardFormat::new(0, {
            let mut c = make_head_major_cache(n_tokens + 4, kv_heads, head_dim);
            write_tokens_headmajor(&mut c, n_tokens);
            c
        });

        let q_data = vec![1.0f32; n_heads_q * head_dim];
        let q = f32_tensor(vec![1, 1, n_heads_q, head_dim], &q_data);
        let backend = CpuBackend::new();
        let dims = AttnDims {
            n_heads_q,
            window: None,
        };

        // 기준: attention_into (full)
        let mut out_full = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.0; n_heads_q * head_dim],
        );
        fmt_ref
            .attention_into(&q, &backend, &mut out_full, dims, None, None)
            .unwrap();

        // SelectiveRead: select = 전체 토큰 목록
        let fmt_sel = StandardFormat::new(0, cache_sel);
        let mut out_sel = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.0; n_heads_q * head_dim],
        );
        let select_all: Vec<usize> = (0..n_tokens).collect();
        fmt_sel
            .attention_into_selected(
                &q,
                &backend,
                &mut out_sel,
                dims,
                &select_all,
                argus_extension_api::ReadGranularity::Token,
                None,
            )
            .unwrap();

        let o_full = out_full.as_slice::<f32>();
        let o_sel = out_sel.as_slice::<f32>();
        for (i, (&a, &b)) in o_full.iter().zip(o_sel.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "full[{i}]={a} != selected_full[{i}]={b}"
            );
        }
    }

    /// SelectiveRead: 부분 select(절반) → 에러 없이 완료 + out 유한값.
    #[test]
    fn selective_read_partial_select_completes_finite() {
        let kv_heads = 1;
        let head_dim = 4;
        let n_heads_q = 1;
        let n_tokens = 6;

        let mut cache = make_head_major_cache(n_tokens + 4, kv_heads, head_dim);
        write_tokens_headmajor(&mut cache, n_tokens);

        let fmt = StandardFormat::new(0, cache);
        let q = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.5f32; n_heads_q * head_dim],
        );
        let backend = CpuBackend::new();
        let dims = AttnDims {
            n_heads_q,
            window: None,
        };

        // 앞 절반만 select
        let half = n_tokens / 2;
        let select_half: Vec<usize> = (0..half).collect();

        let mut out = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.0; n_heads_q * head_dim],
        );
        fmt.attention_into_selected(
            &q,
            &backend,
            &mut out,
            dims,
            &select_half,
            argus_extension_api::ReadGranularity::Token,
            None,
        )
        .unwrap();

        // 유한값 확인
        for &x in out.as_slice::<f32>() {
            assert!(x.is_finite(), "출력에 inf/nan 포함: {x}");
        }
    }

    /// SelectiveRead Page 단위: page_size=2, select=[0,1] (2페이지=4토큰) → 부분 선택과 동일 범주.
    #[test]
    fn selective_read_page_granularity_completes() {
        let kv_heads = 1;
        let head_dim = 4;
        let n_heads_q = 1;
        let n_tokens = 8;

        let mut cache = make_head_major_cache(n_tokens + 4, kv_heads, head_dim);
        write_tokens_headmajor(&mut cache, n_tokens);

        let fmt = StandardFormat::new(0, cache);
        let q = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![1.0f32; n_heads_q * head_dim],
        );
        let backend = CpuBackend::new();
        let dims = AttnDims {
            n_heads_q,
            window: None,
        };

        // page_size=2, page 0,1 = pos [0,1,2,3]
        let page_select = vec![0usize, 1];
        let mut out = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.0; n_heads_q * head_dim],
        );
        fmt.attention_into_selected(
            &q,
            &backend,
            &mut out,
            dims,
            &page_select,
            argus_extension_api::ReadGranularity::Page { page_size: 2 },
            None,
        )
        .unwrap();

        for &x in out.as_slice::<f32>() {
            assert!(x.is_finite(), "Page 단위 출력에 inf/nan: {x}");
        }
    }

    // ── S2: read_plan seam (read stage 위임 + capability) ──

    use argus_extension_api::{KVReadPlan, KVReadStage, ReadGranularity, StageCtx};

    /// mock read stage — 생성 시 지정한 select 를 항상 반환. ctx 의 current_pos 로 "전체" plan 도 가능.
    struct MockReadStage {
        granularity: ReadGranularity,
        /// None = ctx.current_pos() 로 전체 select 생성, Some = 그대로.
        fixed_select: Option<Vec<usize>>,
        plan_none: bool,
    }
    impl KVReadStage for MockReadStage {
        fn name(&self) -> &str {
            "mock"
        }
        fn read_plan(&self, ctx: &dyn StageCtx) -> Option<KVReadPlan> {
            if self.plan_none {
                return None;
            }
            let select = match &self.fixed_select {
                Some(s) => s.clone(),
                None => (0..ctx.current_pos()).collect(),
            };
            Some(KVReadPlan {
                granularity: self.granularity,
                select,
            })
        }
    }

    /// `as_selective_read()` capability: StandardFormat 은 Some(자기 자신).
    #[test]
    fn standard_format_exposes_selective_read_capability() {
        let fmt = StandardFormat::new(0, make_head_major_cache(8, 1, 4));
        let cap = (&fmt as &dyn KVCacheFormat).as_selective_read();
        assert!(
            cap.is_some(),
            "StandardFormat 은 SelectiveRead capability 노출"
        );
    }

    /// read_plan 위임: mock 이 전체 select 반환 → 그 plan 으로 attention_into_selected 한 결과가
    /// read stage 부재(attention_into full read) 와 bit-identical (S3 게이트의 read_plan 경유 연장).
    #[test]
    fn read_plan_full_select_routes_bit_identical() {
        let kv_heads = 1;
        let head_dim = 4;
        let n_heads_q = 1;
        let n_tokens = 5;

        // 기준 (read stage 부재 = attention_into full read)
        let fmt_ref = StandardFormat::new(0, {
            let mut c = make_head_major_cache(n_tokens + 4, kv_heads, head_dim);
            write_tokens_headmajor(&mut c, n_tokens);
            c
        });
        // seam (read stage 활성 = read_plan → attention_into_selected)
        let fmt_seam = StandardFormat::new(0, {
            let mut c = make_head_major_cache(n_tokens + 4, kv_heads, head_dim);
            write_tokens_headmajor(&mut c, n_tokens);
            c
        });

        let q = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![1.0f32; head_dim]);
        let backend = CpuBackend::new();
        let dims = AttnDims {
            n_heads_q,
            window: None,
        };

        let mut out_ref = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![0.0; head_dim]);
        fmt_ref
            .attention_into(&q, &backend, &mut out_ref, dims, None, None)
            .unwrap();

        // mock: 전체 select(None → ctx.current_pos() 로 0..n_tokens)
        let rs = MockReadStage {
            granularity: ReadGranularity::Token,
            fixed_select: None,
            plan_none: false,
        };
        let plan = fmt_seam
            .read_plan(&rs, 0, None, None)
            .expect("mock 전체 select plan 반환");
        assert_eq!(plan.select, (0..n_tokens).collect::<Vec<_>>());

        let mut out_seam = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![0.0; head_dim]);
        fmt_seam
            .attention_into_selected(
                &q,
                &backend,
                &mut out_seam,
                dims,
                &plan.select,
                plan.granularity,
                None,
            )
            .unwrap();

        for (i, (&a, &b)) in out_ref
            .as_slice::<f32>()
            .iter()
            .zip(out_seam.as_slice::<f32>().iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-5,
                "full[{i}]={a} != read_plan_routed[{i}]={b}"
            );
        }
    }

    /// read_plan 위임: mock 이 None 반환 → full read 폴백(plan 없음).
    #[test]
    fn read_plan_none_falls_back() {
        let fmt = StandardFormat::new(0, {
            let mut c = make_head_major_cache(8, 1, 4);
            write_tokens_headmajor(&mut c, 4);
            c
        });
        let rs = MockReadStage {
            granularity: ReadGranularity::Token,
            fixed_select: None,
            plan_none: true,
        };
        assert!(
            fmt.read_plan(&rs, 0, None, None).is_none(),
            "mock plan_none=true → read_plan None → 엔진 full read 폴백"
        );
    }

    /// read_plan 위임: 부분 select mock → plan 반환 후 attention_into_selected 완료 + 유한.
    #[test]
    fn read_plan_partial_select_completes_finite() {
        let fmt = StandardFormat::new(0, {
            let mut c = make_head_major_cache(10, 1, 4);
            write_tokens_headmajor(&mut c, 6);
            c
        });
        let q = f32_tensor(vec![1, 1, 1, 4], &vec![0.5f32; 4]);
        let backend = CpuBackend::new();
        let dims = AttnDims {
            n_heads_q: 1,
            window: None,
        };
        let rs = MockReadStage {
            granularity: ReadGranularity::Token,
            fixed_select: Some(vec![0, 2, 4]),
            plan_none: false,
        };
        let plan = fmt.read_plan(&rs, 0, None, None).expect("부분 select plan");
        assert_eq!(plan.select, vec![0, 2, 4]);
        let mut out = f32_tensor(vec![1, 1, 1, 4], &vec![0.0; 4]);
        fmt.attention_into_selected(
            &q,
            &backend,
            &mut out,
            dims,
            &plan.select,
            plan.granularity,
            None,
        )
        .unwrap();
        for &x in out.as_slice::<f32>() {
            assert!(x.is_finite(), "부분 select 출력 유한: {x}");
        }
    }

    // ── WeightedKV 비대칭 merge (KV 로드맵 항목 2) ──────────────────────────────

    /// `make_head_major_cache`(F32) + `write_tokens_headmajor` 로 cache 를 채우고, `into=0`
    /// 에 pos 1,2 를 가중 병합하는 merge 1개를 `apply_to` 별로 적용해 K/V `into` 행이
    /// 변했는지/불변인지 검사한다. `write_tokens_headmajor` 는 K=pos+0.1·h, V=pos+0.5·h.
    fn run_axis_case_f32(axis: MergeAxis) -> (bool, bool) {
        let kv_heads = 2;
        let head_dim = 4;
        let n_tokens = 4;
        let mut cache = make_head_major_cache(n_tokens + 2, kv_heads, head_dim);
        write_tokens_headmajor(&mut cache, n_tokens);

        // into=0 의 원본 K/V (head 0) 스냅샷.
        let off0 = cache.offset(0, 0);
        let k_before: Vec<f32> = cache.k_buffer.as_slice::<f32>()[off0..off0 + head_dim].to_vec();
        let v_before: Vec<f32> = cache.v_buffer.as_slice::<f32>()[off0..off0 + head_dim].to_vec();

        let merge = WeightedMerge {
            into: 0,
            into_weight: 0.5,
            from: vec![(1, 0.3), (2, 0.2)],
            apply_to: axis,
        };
        apply_weighted_merges(&mut cache, std::slice::from_ref(&merge));

        let k_after: &[f32] = &cache.k_buffer.as_slice::<f32>()[off0..off0 + head_dim];
        let v_after: &[f32] = &cache.v_buffer.as_slice::<f32>()[off0..off0 + head_dim];
        let k_changed = k_after != k_before.as_slice();
        let v_changed = v_after != v_before.as_slice();
        (k_changed, v_changed)
    }

    #[test]
    fn weighted_merge_axis_both_updates_k_and_v_f32() {
        let (k_changed, v_changed) = run_axis_case_f32(MergeAxis::Both);
        assert!(k_changed, "Both: K 갱신되어야 함");
        assert!(v_changed, "Both: V 갱신되어야 함");
    }

    #[test]
    fn weighted_merge_axis_key_only_updates_k_not_v_f32() {
        let (k_changed, v_changed) = run_axis_case_f32(MergeAxis::KeyOnly);
        assert!(k_changed, "KeyOnly: K 갱신되어야 함");
        assert!(!v_changed, "KeyOnly: V 불변이어야 함");
    }

    #[test]
    fn weighted_merge_axis_value_only_updates_v_not_k_f32() {
        let (k_changed, v_changed) = run_axis_case_f32(MergeAxis::ValueOnly);
        assert!(!k_changed, "ValueOnly: K 불변이어야 함");
        assert!(v_changed, "ValueOnly: V 갱신되어야 함");
    }

    /// Both 경로가 구 동작과 bit-identical 임을 명시 확인: `apply_to=Both` 의 K·V 결과가
    /// K/V 각각 독립으로 KeyOnly·ValueOnly 를 적용한 결과와 정확히 일치.
    #[test]
    fn weighted_merge_axis_both_equals_keyonly_plus_valueonly_f32() {
        let kv_heads = 2;
        let head_dim = 4;
        let n_tokens = 4;
        let merge = |axis| WeightedMerge {
            into: 0,
            into_weight: 0.5,
            from: vec![(1, 0.3), (2, 0.2)],
            apply_to: axis,
        };
        let off0 = {
            let c = make_head_major_cache(n_tokens + 2, kv_heads, head_dim);
            c.offset(0, 0)
        };

        // Both 한 번.
        let mut both = make_head_major_cache(n_tokens + 2, kv_heads, head_dim);
        write_tokens_headmajor(&mut both, n_tokens);
        apply_weighted_merges(&mut both, std::slice::from_ref(&merge(MergeAxis::Both)));

        // KeyOnly + ValueOnly 따로.
        let mut split = make_head_major_cache(n_tokens + 2, kv_heads, head_dim);
        write_tokens_headmajor(&mut split, n_tokens);
        apply_weighted_merges(&mut split, std::slice::from_ref(&merge(MergeAxis::KeyOnly)));
        apply_weighted_merges(
            &mut split,
            std::slice::from_ref(&merge(MergeAxis::ValueOnly)),
        );

        let k_both = &both.k_buffer.as_slice::<f32>()[off0..off0 + head_dim];
        let v_both = &both.v_buffer.as_slice::<f32>()[off0..off0 + head_dim];
        let k_split = &split.k_buffer.as_slice::<f32>()[off0..off0 + head_dim];
        let v_split = &split.v_buffer.as_slice::<f32>()[off0..off0 + head_dim];
        assert_eq!(k_both, k_split, "Both K == KeyOnly K (bit-identical)");
        assert_eq!(v_both, v_split, "Both V == ValueOnly V (bit-identical)");
    }

    /// F16 비대칭: ValueOnly 면 K f16 비트 불변, V f16 비트 변경. quant round-trip 무관하게
    /// 버퍼 직접 비교(merge 미적용 축은 byte-identical).
    #[test]
    fn weighted_merge_axis_value_only_f16() {
        use crate::kv::kv_cache::KVLayout;
        use crate::memory::Memory;
        use crate::memory::galloc::Galloc;
        let kv_heads = 1;
        let head_dim = 4;
        let cap = 6;
        let n_tokens = 4;
        let total = kv_heads * cap * head_dim;
        let mk = || {
            let bytes = vec![0u8; total * 2]; // f16 = 2 bytes
            let buf = Arc::new(SharedBuffer::from_vec(bytes, DType::F16));
            Tensor::new(
                Shape::new(vec![1, kv_heads, cap, head_dim]),
                buf,
                Arc::new(CpuBackend::new()) as Arc<dyn Backend>,
            )
        };
        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        let mut cache = KVCache::new_dynamic(mk(), mk(), cap, cap, kv_heads, head_dim, mem)
            .with_layout(KVLayout::HeadMajor);
        {
            let k = cache.k_buffer.as_mut_slice::<half::f16>();
            let v = cache.v_buffer.as_mut_slice::<half::f16>();
            for pos in 0..n_tokens {
                for d in 0..head_dim {
                    k[pos * head_dim + d] = half::f16::from_f32(pos as f32 + 1.0);
                    v[pos * head_dim + d] = half::f16::from_f32(pos as f32 + 1.0);
                }
            }
            cache.current_pos = n_tokens;
            cache.high_water_pos = n_tokens;
        }
        let k_before: Vec<half::f16> = cache.k_buffer.as_slice::<half::f16>()[0..head_dim].to_vec();
        let v_before: Vec<half::f16> = cache.v_buffer.as_slice::<half::f16>()[0..head_dim].to_vec();

        let merge = WeightedMerge {
            into: 0,
            into_weight: 0.5,
            from: vec![(1, 0.3), (2, 0.2)],
            apply_to: MergeAxis::ValueOnly,
        };
        apply_weighted_merges(&mut cache, std::slice::from_ref(&merge));

        let k_after = &cache.k_buffer.as_slice::<half::f16>()[0..head_dim];
        let v_after = &cache.v_buffer.as_slice::<half::f16>()[0..head_dim];
        assert_eq!(k_after, k_before.as_slice(), "ValueOnly: K(f16) 불변");
        assert_ne!(v_after, v_before.as_slice(), "ValueOnly: V(f16) 갱신");
    }

    /// Q4_0 비대칭: KeyOnly 면 V q4 블록 byte-identical, K q4 블록 변경.
    #[test]
    fn weighted_merge_axis_key_only_q4() {
        use crate::kv::kv_cache::KVLayout;
        use crate::memory::Memory;
        use crate::memory::galloc::Galloc;
        use crate::quant::{BlockQ4_0, QK4_0};
        let kv_heads = 1;
        let head_dim = QK4_0; // 32 → 1 block/pos
        let cap = 6;
        let n_tokens = 4;
        let blocks = kv_heads * cap * (head_dim / QK4_0);
        let mk = || {
            let bytes = vec![0u8; blocks * std::mem::size_of::<BlockQ4_0>()];
            let buf = Arc::new(SharedBuffer::from_vec(bytes, DType::Q4_0));
            Tensor::new(
                Shape::new(vec![1, kv_heads, cap, head_dim]),
                buf,
                Arc::new(CpuBackend::new()) as Arc<dyn Backend>,
            )
        };
        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        let mut cache = KVCache::new_dynamic(mk(), mk(), cap, cap, kv_heads, head_dim, mem)
            .with_layout(KVLayout::HeadMajor);
        {
            let kb = cache.k_buffer.as_mut_slice::<BlockQ4_0>();
            let vb = cache.v_buffer.as_mut_slice::<BlockQ4_0>();
            for pos in 0..n_tokens {
                let mut row = [0.0f32; QK4_0];
                for (d, r) in row.iter_mut().enumerate() {
                    *r = (pos as f32 + 1.0) * 0.1 + d as f32 * 0.01;
                }
                kb[pos] = BlockQ4_0::quantize(&row);
                vb[pos] = BlockQ4_0::quantize(&row);
            }
            cache.current_pos = n_tokens;
            cache.high_water_pos = n_tokens;
        }
        let k_before = cache.k_buffer.as_slice::<BlockQ4_0>()[0];
        let v_before = cache.v_buffer.as_slice::<BlockQ4_0>()[0];

        let merge = WeightedMerge {
            into: 0,
            into_weight: 0.5,
            from: vec![(1, 0.3), (2, 0.2)],
            apply_to: MergeAxis::KeyOnly,
        };
        apply_weighted_merges(&mut cache, std::slice::from_ref(&merge));

        let k_after = cache.k_buffer.as_slice::<BlockQ4_0>()[0];
        let v_after = cache.v_buffer.as_slice::<BlockQ4_0>()[0];
        // BlockQ4_0 = { d: f16, qs: [u8; 16] } — 필드 직접 비교로 byte-identity 판정.
        let blk_eq = |a: &BlockQ4_0, b: &BlockQ4_0| a.d.to_bits() == b.d.to_bits() && a.qs == b.qs;
        assert!(!blk_eq(&k_after, &k_before), "KeyOnly: K(q4_0) 블록 갱신");
        assert!(
            blk_eq(&v_after, &v_before),
            "KeyOnly: V(q4_0) 블록 byte-identical"
        );
    }
}
