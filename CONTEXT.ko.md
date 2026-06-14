# Argus — KV/Weight Cache Management Context

[English](CONTEXT.md) | **한국어**

decode 루프가 도는 동안 KV·weight 데이터를 *어떻게 다루는가*에 관한 용어집. 세 직교 축 — **stage**(메모리 상주 데이터 조절) ⊥ **format**(연산 표현 precision) ⊥ **hardware**(연산 위치) — 을 다룬다. 세 축은 독립이라 멤버 조합 비용이 곱(M×N×K)이 아니라 합(M+N+K)이 되도록 설계한다.

## Language

### 세 축

**stage 축**:
decode가 도는 동안 *메모리에 상주하는 데이터*(kvcache 토큰, weight layer)를 조절하는 축 — 수정·삭제·로드/언로드. "메모리에 무엇이 얼마나 올라가 있나"를 제어한다. 멤버 예: H2O(토큰 삭제=evict), D2O(토큰 병합=수정), weight swap(layer 로드/언로드).

**format 축**:
각 연산이 *어떤 데이터 표현(precision)으로* 수행되는가의 축. "이 연산을 어떤 포맷으로 계산하나"를 제어한다. 한 멤버 = 하나의 표현 = 바이트 레이아웃 + precision + 그걸 읽는 전용 커널(**paired kernel**). 예: f16, q4_0, q8_0, KIVI(Q2 packed + F32 residual). precision과 레이아웃은 분리된 적 없는 한 좌표다(q4_0 = 4bit 블록 레이아웃이자 precision).

**hardware 축** (신규):
연산이 *어느 물리 연산기*에서 실행되는가의 축. "이 연산을 어디서 계산하나"를 제어한다. 멤버 = backend(CPU NEON / Adreno OpenCL / Jetson CUDA / NPU)이며, 데이터가 사는 memory를 동반한다. 런타임에선 [`Hardware`](#hardware--연산-위치) 객체(read-only resolver)가 좌표를 해석한다.

**직교성 (orthogonality)**:
세 축은 서로 독립이다. 한 축에 멤버를 더해도 다른 축 코드를 안 건드린다 — stage 추가는 커널 0, format 추가는 backend별 opt-in 커널, hardware 추가는 resolver 항목 + opt-in 커널. dispatch·stage·resolver 인터페이스 비용은 합(**M+N+K**)으로 유지된다 (조합 폭발 방지).

**3축의 근거 — precision ⊥ backend 는 분리 가능**:
format(precision)과 hardware(backend)가 같은 개념이면 2축, 분리되면 3축이다. 코드가 분리를 증명한다 — 단일 OpenCL backend 하나가 6 precision(f32/f16/q4_0/q8_0/q6_k/mxfp4) 커널을 갖고, q4_0은 CPU·GPU·CUDA 세 backend에 모두 존재한다. 매핑이 many-to-many라 precision은 backend의 함수가 아니다 → 독립 축.

**복합 동작 = 단일축 primitive 의 합성**:
현실의 여러 동작은 두 축을 동시에 건드린다 — weight swap(stage 로드 + format precision 변경), switch+migrate(hardware 이동), KIVI(stage 수정 + format precision). 이들은 *단일축 primitive 의 합성*으로 분해한다. 새 복합 기능 = 기존 primitive 의 새 조합 → 새 monolith 0줄. 직교성은 primitive 레벨에서 유지되고, 복합 동작은 3-공간의 벡터가 된다.

**(format × hardware) 커널은 M×N — 단 Backend 인터페이스 아래 격리**:
q4-on-Adreno 커널과 q4-on-CPU 커널은 물리적으로 다른 코드라, (format × hardware) 커널 행렬은 환원 불가능한 M×N이다. 그러나 이 행렬은 `backend.matmul(q4_tensor)` 내부 dispatch 에 갇혀 있어 호출자·stage·resolver 는 어느 커널이 뜨는지 모른다. 불가피한 M×N을 leaf 한 곳에 가두는 것이 3축 구조의 핵심 — 성능(특화 커널)과 확장성(가산 인터페이스)을 동시에 잡는 지점이다.

### stage — 메모리 상주 데이터 조절

**Stage**:
decode 루프가 도는 동안 매 단계(phase)마다 끼어들어 *메모리 상주 데이터*(kvcache·weight)를 수정·삭제·로드하는 cross-cutting 동작. 대부분의 Stage 는 표현(format)·위치(hardware)를 안 바꾸고 데이터의 존재·양만 조절한다 (eviction·merge·weight swap). 실행 순서는 통합자 책임, 안전(crash-free)은 프레임워크 책임.
_Avoid_: Handler, hook — 같은 개념의 현 코드 구현 명칭일 뿐, 도메인 용어로는 Stage. (단 코드의 `PipelineStage` *메커니즘* 명칭과 *축* 명칭 "stage"는 층위가 다르다 — → Flagged ambiguities.)

**KVCacheStage** (stage 축 trait):
KV 캐시의 상주 토큰을 *어떻게 조절할지* 결정하는 stage-축 동작의 추상화 trait. 캐시 상태 + scores 를 보고 **`KVCachePlan`**(보존 `keep` + 가중 `merge`)을 산출하며, 버퍼 변형은 엔진이 그 plan 을 실행해 수행한다(plan-returning — 결정은 plugin, 실행은 엔진). **`KVCacheFormat` 의 형제** — `KVCacheFormat`=저장 *표현*(precision/layout) ⊥ `KVCacheStage`=상주 *토큰 조절* (format ⊥ stage 축). 멤버: `Sliding`/`H2O`/`Streaming`(layer-wide keep), `H2O+`(per-head keep), `D2O`(가중 merge). 새 stage = 별도 technique crate + 등록 1줄(엔진 코어 0 edit).

**KVCachePlan** (KVCacheStage 산출물):
한 stage 가 산출하는 *이 KV 캐시가 어떤 상태가 되어야 하는가*의 명세. `keep`(보존 토큰 — layer-wide 또는 per-head) + `merge`(가중 병합 지시)로 구성. 엔진이 `compact` 로 실행한다. "eviction"이 아니라 keep+merge 를 포괄하므로 KVCache 스코프 이름.

**EvictionPolicy** (legacy, KVCacheStage 로 흡수 중):
구 in-place eviction trait — `evict(&mut KVCache)` 로 *직접* 버퍼를 고친다. evict 전용이라 merge(D2O)·per-head(H2O+)를 못 담는다. plan-returning `KVCacheStage` 로 일반화·대체 진행 중 (마이그레이션 중 공존). 그 내부의 "어느 토큰을 버릴지" 규칙 개념은 `KVCacheStage::plan` 결정으로 흡수.

### format — 연산 표현

**format**:
KV 또는 weight 데이터가 *저장·연산되는 표현*. "어떤 포맷으로 계산하나"라는 데이터 좌표다. 한 구체적 표현이 곧 하나의 format 멤버다 — 예: `f16`, `q4_0`, `KIVI`(Q2 packed + F32 residual). 새 format 을 더하면 새 바이트 레이아웃 + 전용 커널(**paired kernel**, backend 별)이 따라온다 — 무겁지만 격리된 확장.
_Note_: precision(q4/f16)과 바이트 레이아웃은 분리된 두 개념이 아니라 한 좌표다. (이전 모델은 "precision 은 Format 안 파라미터"라며 분리했으나 폐기 — q4 와 f16 은 이미 다른 paired kernel 을 쓰므로 precision 이 곧 표현을 가른다.)
_Avoid_: Layer (→ Flagged ambiguities). Precision (precision 은 format 의 다른 이름일 뿐 별도 축이 아니다).

**KVCacheFormat / WeightFormat**:
여러 format 을 추상화하는 base trait. 호출자에게 geometry(위치·용량)·mutation(쓰기·compact)·attention(KV) 또는 dispatch(weight)만 노출하고, dtype·codebook·precision 같은 내부는 숨긴다 (format-agnostic). Stage 는 이 trait 을 통해 데이터를 조작한다.

**Paired kernel**:
한 format 의 바이트 레이아웃을 읽어 연산을 수행하는 전용 커널. format 이 다르면 읽는 커널도 다르고, backend 마다도 달라야 하므로 (format × hardware) M×N 행렬을 이룬다 — 새 format·backend 추가에 따라오는 환원 불가능한 leaf 비용이며 Backend 인터페이스 아래 격리된다.

### hardware — 연산 위치

**hardware (연산 위치)**:
연산이 실행되는 물리 자원. 두 직교 하위차원으로 분해된다 — **compute**(연산기 = backend: CPU NEON / Adreno OpenCL / Jetson CUDA / NPU)와 **data**(메모리 = memory allocator). 둘은 직교다: UMA(ARM SoC)에서는 여러 backend 가 한 memory 를 공유하고(연산기만 바뀜), discrete GPU 에서는 backend 마다 별도 memory(VRAM↔RAM 이동)다. 그래서 (backend, memory) 를 1:1 페어로 묶지 않는다.

**Hardware** (런타임 resolver):
hardware 축 좌표를 해석하는 런타임 객체. 내부에 backend 레지스트리 ⊥ memory 레지스트리를 분리 보유하고, `resolve(target)`이 "이 backend 로 가려면 어느 memory?"의 UMA/discrete 분기를 한 곳에 가둔다. **read-only resolver** 다 — 활성 backend 를 mutable 하게 들고 있지 않으며("현재 device"는 decode-loop local 상태), 어떤 연산도 이 객체를 통해 "현재"를 관찰하지 않는다(실행은 텐서 태그로 storage 에서 backend 를 얻음).

**switch** (hardware 축 이동):
연산 위치를 GPU↔CPU 전환하고(필요시 KV 를 새 backend 의 memory 로 migrate), 표현(format)은 안 바꾼다. 안전한 경계(prefill→decode 등)에서만 실행된다. hardware 축을 따라 데이터를 옮기는 동작이다. 3축 모델에서 switch 는 hardware 축 동작이며 더 이상 Stage 가 아니다.

**partition** (format × hardware 곱):
한 layer 의 forward 를 여러 backend 에 동시 분산하는 것. 슬라이스마다 다른 (format, hardware) 좌표를 가질 수 있다 — 예: GPU 슬라이스는 f16, NPU 슬라이스는 q4 (HeteroLLM). 즉 partition = format(표현) × hardware(위치)의 곱이며, 분산 대상 backend 는 `Hardware.resolve()`에서 받는다. 단일 "현재 위치" 포인터로는 두 좌표를 동시에 못 찍으므로 resolver 가 필요한 정확한 근거다.

**Pressure / PressureSource** (graded system 압력 입력):
memory·thermal·energy 등 여러 system 압력을 **하나의 0–100 scalar(`Pressure`)로 융합**한 입력 신호, 그리고 그 값을 공급하는 pluggable 소스. 어느 축도 아니다 — Stage 가 읽고 *graded 응답*(예: eviction 강도)을 정한다. **의도적 lossy 단일화**: 소비 Stage 는 압력의 *출처*(memory/thermal/…)를 구분하지 않는다 — 단순함·관리 용이를 위해 출처 손실을 수용하므로 carrier 는 단일 scalar 이고 신호 *종류* 확장(anymap)은 불필요하다(닫힌 core 어휘). **단 scalar 로 환원 불가한 *mode* 동작**(switch·suspend = on/off — "73만큼 switch" 는 없음)은 이 스칼라가 아니라 별도 *이산 명령 채널*(→ `EngineCommand`)로 흐른다: Pressure 는 *graded 입력*만 담고 mode 출력은 분리한다. `ManagerPressureSource`(manager 통합값) / `LocalPressureSource`(manager-less 자율, 전 센서 융합) / 3rd-party 가 같은 `fn pressure(&self) -> Pressure` 뒤에 숨고, 소비 Stage 는 어느 소스인지 모른다. 출처 분별 정책이 필요해지면 source-tag 재고(promotion-trigger). 소스는 construction 보유(교체 지점), 값은 매 step `StepInfo` read-only.

### Stage 가 데이터를 잡는 방식 — handle 3종

Stage 가 자신이 조절할 데이터(format)를 보유하는 참조의 정적 타입. 순서 없는 3개 메뉴이며, 4번째는 없다.

**Base-trait-handle**:
`Arc<dyn KVCacheFormat>` 형태. 이를 든 Stage 는 어느 format 인지 *몰라야* 한다. 예: position 만 보는 Sliding.

**Concrete-handle**:
`Arc<StandardFormat>`처럼 특정 format 타입을 든 형태. 그 타입을 *아는 것이 정상*이다(그게 concrete 를 든다는 의미). 예: 원본 K 를 직접 읽어야 하는 D2O.

**Capability-handle**:
`Arc<dyn SomeCapability>` 형태. 이종 format 을 가로지르는 *능력*을 추상화한다. 소비자가 둘 이상일 때만 만든다(하나뿐이면 가설적 seam 이라 만들지 않는다).

## Flagged ambiguities

- **Layer 는 transformer layer 전용**: "Layer"(`LlamaLayer`/`TransformerLayer`, 16개 디코더 블록, `LayerSlot`·`layer_idx` 등)는 *모델 구조*만 가리킨다. KV·weight 의 *표현*은 절대 "Layer"라 부르지 않고 **format**(`KVCacheFormat`/`WeightFormat`)이라 부른다.
- **stage 축 vs `PipelineStage` 메커니즘 vs `KVCacheStage` trait**: "stage"(축)은 메모리 상주 데이터를 조절하는 *개념 차원*이고, 코드의 `PipelineStage`/`CachePressureHandler`는 decode 루프 hook *메커니즘*이며, **`KVCacheStage`는 그 stage 축 동작을 plugin 이 구현하는 정식 trait**(plan-returning)이다. 셋은 층위가 다르다. **세션 `EvictionStage`/`SwapStage`(`session/traits.rs`)는 deprecated** — WHEN-hook 을 plugin 으로 두려던 dormant 추상이었으나, WHEN(decode 루프 통합)은 엔진 소유로, WHAT(보존/병합 결정)은 `KVCacheStage` 로 분리되며 폐기.
- **format 은 표현(precision+layout 융합)**: 구 "precision 은 Format 안 파라미터 / Standard·KIVI 는 별개 바이트 레이아웃"은 폐기. q4/f16/KIVI 는 각각 format 축의 한 좌표다.
- **device 는 이제 축이다 (구 "바탕" 폐기)**: precision(format)과 backend(hardware)가 분리 가능함을 코드로 확인(단일 backend 가 6 precision, q4 가 3 backend) → hardware 는 정식 축. switch = hardware 축 동작, partition = format × hardware 곱. (M×N 커널 행렬은 Backend 인터페이스 아래 격리되므로 축 가산성은 유지.)
- **Eviction 은 stage 이고 KIVI 는 format(+stage 합성)이다**: eviction 은 토큰을 버리는 *상주 데이터 조절*이라 **stage**. KIVI 양자화는 바이트 표현을 바꾸는 *format* 이자 그 변환 동작이 *상주 데이터를 수정*하므로 stage 와의 **합성**이다.

## Example dialogue

> **Dev**: KIVI 를 추가하려는데, Sliding 이나 H2O 처럼 stage 로 넣으면 되나요?
> **Expert**: 부분만 맞아요. Sliding·H2O 는 "어느 토큰을 버리나"라는 **stage** 동작이라 표현을 안 건드려요. KIVI 는 KV 를 Q2 로 *다시 깔고*(상주 데이터 수정 = stage) 그 위에서 Q2 precision 으로 attention(=format)하는 **합성**이에요. 그래서 KVCacheFormat impl + paired kernel(format)이 따라옵니다.
> **Dev**: 그럼 KIVI format 위에서 Sliding 을 돌릴 수도 있나요?
> **Expert**: 네. 그게 세 축이 **직교**한다는 뜻이에요. KIVI(format 좌표) 위에서 Sliding(stage 동작)이 `compact`만 호출하면 됩니다. Sliding 은 그 밑이 Q2 인지 F32 인지 몰라요 — **base-trait-handle**을 들었으니까요.
> **Dev**: GPU 에서 돌리던 걸 CPU 로 옮기는 switch 는 어느 축이죠?
> **Expert**: **hardware** 축이에요. 연산 *위치*를 옮기는 거라 표현(format)도 동작(stage)도 안 바꿔요 — KIVI 는 GPU 든 CPU 든 KIVI 입니다. switch 가 KV 를 새 memory 로 migrate 하는 것도 hardware 축 이동의 일부예요.
> **Dev**: partition 에서 GPU 는 f16, NPU 는 q4 로 두고 싶은데요.
> **Expert**: 그게 format × hardware 곱이에요. 슬라이스마다 (f16, GPU) / (q4, NPU) 라는 서로 다른 좌표를 갖는 거죠. 단일 "현재 device" 포인터로는 두 좌표를 동시에 못 찍으니, `Hardware.resolve()`로 슬라이스별 backend 를 받습니다.
> **Dev**: 그 "format"이 transformer layer 랑 다른 건가요?
> **Expert**: 완전히 달라요. transformer **layer**는 모델 디코더 블록(16개)이고, **format**은 그 layer 들의 KV/weight 가 *어떤 표현으로 연산되나*예요.
