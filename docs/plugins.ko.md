# Argus 플러그인 만들기

[English](plugins.md) | **한국어**

**엔진을 포크하지 않고** Argus에 **KV 캐시 기법**을 더하는 가이드입니다. 새 제거(eviction)
정책, 새 저장 포맷, 쿼리 인지 읽기 전략 같은 것들이죠. 작은 크레이트 하나에 트레이트 하나만
구현하면, 그게 스스로 등록되고, 엔진이 이름으로 찾아 씁니다. 내장 기법들(`q4_0`, `quest`,
`caote` …)이 올라타 있는 바로 그 `argus-extension-api` 표면입니다.

> **처음이세요?** [빠른 시작](#빠른-시작-첫-플러그인)만 하세요(약 10분, 출하된 `argus-cli`에서
> 바로 돕니다). 그 뒤는 필요할 때 한 절씩 돌아와 보면 되는 레퍼런스와 사용법입니다.

빠른 시작을 끝내면 **직접 만든 KV 캐시 포맷**이 자기 크레이트로 컴파일되어
[`linkme`](https://docs.rs/linkme)로 스스로 등록되고, `--load-plugin`으로 실제 추론
바이너리에 올라가, 런타임에 `--kv-format my_format`으로 선택됩니다. **엔진 재빌드는 전혀
없이요.**

---

## 목차

- [60초 멘탈 모델](#60초-멘탈-모델)
- [시작하기 전에](#시작하기-전에)
- [빠른 시작: 첫 플러그인](#빠른-시작-첫-플러그인)
- [플러그인 해부](#플러그인-해부)
- [네 개의 확장 축](#네-개의-확장-축)
- [사용법: 제거/병합 stage 추가](#사용법-제거병합-stage-추가)
- [사용법: read stage 추가](#사용법-read-stage-추가)
- [사용법: 백엔드 capability(KIVI) 추가](#사용법-백엔드-capabilitykivi-추가)
- [생명주기와 성능 계약](#생명주기와-성능-계약)
- [정적 vs 동적: 로드 경로 고르기](#정적-vs-동적-로드-경로-고르기)
- [동적 `.so` 경로 자세히](#동적-so-경로-자세히)
- [정적(force-link) 경로](#정적force-link-경로)
- [문제 해결](#문제-해결)
- [패키징과 배포 체크리스트](#패키징과-배포-체크리스트)
- [다음으로 갈 곳](#다음으로-갈-곳)

---

## 60초 멘탈 모델

이 한 줄이 전부를 떠받칩니다. **결정은 플러그인이, 실행은 엔진이.** 당신 코드는 KV 버퍼나 GPU
명령 큐를 직접 건드리지 않습니다. 캐시의 읽기 전용 뷰를 보고 *계획(plan)*(또는 *서술자
(descriptor)*)을 돌려줄 뿐이고, 모든 변경(mutation)은 엔진이 소유합니다. 이게 기법 크레이트를
엔진 내부와 떼어 놓는 핵심이고, 같은 코드가 정적으로 링크되든 `.so`로 로드되든 통하게 만드는
이유입니다.

Argus는 캐시 관리를 **직교 축(orthogonal axis)** 으로 나눕니다(전체 용어는
[`CONTEXT.ko.md`](../CONTEXT.ko.md)에). 플러그인은 정확히 한 축을 확장합니다.

| 축 | 구현하는 것 | 돌려주거나 서술하는 것 |
|----|-------------|------------------------|
| **stage** | `KVCacheStage` | 어떤 토큰을 남기고 어떻게 병합할지 (제거, H2O, D2O) |
| **format** | `KVFormat` | 캐시를 저장하는 바이트 레이아웃/정밀도 (q4_0, KIVI) |
| **read** | `KVReadStage` | attention에서 어떤 토큰/페이지를 읽을지 (Quest) |
| **backend-capability** | `KiviAttentionBackend` | 특화된 융합 GPU 커널 |

각 축은 서로 독립적입니다. 한 축에 멤버를 더해도 다른 축 코드는 **한 줄도** 안 건드리고, 엔진
코어도 **한 줄도** 안 건드립니다. 그게 보상이고, 이 가이드의 나머지는 그 보상을 챙기는
방법입니다.

> 왜 고쳐야 할 중앙 목록이 없을까요? 각 기법은 **링크 시점**에 `linkme`로 전역 레지스트리
> 슬라이스에 스스로를 제출합니다. 엔진은 그 슬라이스를 읽을 뿐, 추가할 `match` 가지가 없습니다.
> "왜 축이 세 개인가" 같은 더 깊은 모델은 [`CONTEXT.ko.md`](../CONTEXT.ko.md)에 있고, 빠른
> 시작을 끝내는 데는 **필요하지 않습니다.**

---

## 시작하기 전에

필요한 것:

- **안정 버전 Rust 툴체인**과 클론한 `argus-engine` 워크스페이스.
- 환경이 멀쩡한지 확인하는 베이스라인 빌드 한 번:
  ```bash
  cargo build --release
  ```
- 추론을 돌릴 모델(CLI 기본값은 `models/llama3.2-1b`). 엔진이 이미 로드하는 모델이면 무엇이든
  됩니다 — 작성하는 플러그인은 모델과 무관합니다. 레포에는 모델이 동봉돼 있지 않으니, 받거나
  변환하는 방법은 [README](../README.ko.md)를 보세요. **빠른 시작의 1~3단계는 모델이 필요
  없습니다**(등록을 증명하는 단계라서요). 모델이 없어도 빌드·검증 루프는 다 돌 수 있고, 추론은
  4단계에서만 합니다.

빠른 시작은 **CPU 전용이고 호스트에서 돕니다** — GPU도, 기기도 필요 없습니다. (GPU/온디바이스
세부는 여기 범위 밖입니다.)

> **나중을 위한 안전 메모 하나.** 동적 `.so` 경로는 네이티브 코드를 엔진 프로세스에
> **샌드박스 없이** 로드합니다 — 플러그인에서 panic이나 segfault가 나면 엔진이 통째로 죽고,
> ABI는 `argus-extension-api` 버전 간에 안정적이지 않습니다. 빠른 시작에는 해당 없지만(당신은 안전한
> Rust만 씁니다), 남에게 `.so`를 배포하기 전에는 꼭 기억해 두세요.

---

## 빠른 시작: 첫 플러그인

KV 캐시 **포맷**을 더해 봅니다 — 출하된 `argus-cli`에서 끝까지 돌고 엔진 재빌드가 필요 없는
축입니다. 출발점은 관리되는 템플릿 크레이트 `example-kv-format`을 복사해 이름만 바꾸는 것.

### 1단계 — 템플릿 복사

```bash
cp -r crates/techniques/example-kv-format crates/techniques/my-format
```

`crates/techniques/my-format/Cargo.toml`을 열어 패키지 이름만 바꿉니다.

```toml
[package]
name = "my-format"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[lib]
# cdylib = dlopen 가능한 `.so`/`.dylib`; rlib = 정적 force-link 경로.
crate-type = ["cdylib", "rlib"]

[dependencies]
argus-extension-api = { path = "../../argus-extension-api" }
linkme = "0.3"

[features]
# 동적 `.so`가 쓰는 C-ABI export를 켭니다. 기본은 OFF — 로드 가능한 라이브러리를 만들 때만
# 켜세요(정적 빌드에서 꺼 두면 #[no_mangle] 심볼 충돌을 피합니다).
plugin-cdylib = []
```

`crates/techniques/my-format/src/lib.rs`의 내용을 아래 완전한 파일로 통째로 바꿉니다 — 중요한
건 **이름 문자열**이 유일해야 한다는 것뿐입니다.

```rust
//! my-format — 직접 만든 KV 캐시 포맷(q4_0 같은 서술자).

use argus_extension_api::{KVFormat, KVLayoutDesc, Packing, ScaleLayout};

/// 포맷은 *순수 서술자*입니다: 이름 + 바이트 레이아웃. 연산은 여기 없습니다 —
/// 엔진의 범용 리더가 이 레이아웃을 통해 f32로 역양자화합니다.
struct MyFormat;

impl KVFormat for MyFormat {
    fn name(&self) -> &str {
        "my_format"
    }

    fn layout(&self) -> KVLayoutDesc {
        // q4_0 같은 레이아웃: 블록당 32원소, 각 4비트, 블록당 f16 스케일 하나,
        // 니블 패킹. (필드 의미는 KVLayoutDesc 참고.)
        KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        }
    }
}

// 이 한 줄이 MyFormat을 두 경로 모두에 등록합니다: 정적 `KV_FORMATS` linkme 슬라이스와,
// (`--features plugin-cdylib`에서) 동적 C-ABI export.
argus_extension_api::register_kv_format!("my_format", || Box::new(MyFormat));

// 이 크레이트의 `.so` 진입 심볼을 내보냅니다. 동적 경로에만 필요하고, 정적 빌드에선 무해합니다.
argus_extension_api::export_plugin!();

#[cfg(test)]
mod tests {
    use argus_extension_api::find_kv_format;

    #[test]
    fn registers_into_kv_formats() {
        let reg = find_kv_format("my_format").expect("my_format must be registered");
        assert_eq!(reg.name, "my_format");
    }
}
```

워크스페이스 편집은 필요 없습니다: 루트 `Cargo.toml`이 `crates/techniques/*`를 glob하므로,
복사한 폴더는 이미 멤버입니다.

### 2단계 — 로드 가능한 라이브러리로 빌드

```bash
cargo build --release -p my-format --features plugin-cdylib
```

> 플러그인은 **엔진과 같은 프로파일**로 빌드하세요(여기선 둘 다 `--release`). 엔진이 `dlopen`할
> 때 ABI가 맞아야 합니다.

**체크포인트.** 공유 라이브러리가 `target/release/` 아래에 생깁니다.

- Linux / Android: `target/release/libmy_format.so`
- macOS 호스트: `target/release/libmy_format.dylib`

(크레이트 이름의 하이픈이 라이브러리 파일명에선 밑줄로 바뀝니다.) 정확한 경로를 프로그램적으로
찍고 싶다면 — gate 테스트가 하는 것의 변형으로(테스트는 debug로 빌드하고 `.so`만 매칭합니다):

```bash
cargo build --release -p my-format --features plugin-cdylib --message-format=json \
  | rg compiler-artifact | rg -o '"[^"]*libmy_format[^"]*\.(so|dylib)"'
```

### 3단계 — (추론 전에) 등록이 발화했는지 증명

`linkme` 등록은 눈에 안 보입니다. 등록에 실패한 플러그인은 **컴파일 에러가 안 나고**, 그냥
엔진이 못 찾는 이름이 됩니다. 템플릿에는 슬라이스가 채워졌음을 증명하는 단위 테스트가 들어
있으니 돌려 보세요.

```bash
cargo test -p my-format
```

**체크포인트.** `registers_into_kv_formats`가 통과합니다. 이제 당신의 `impl KVFormat`은
`find_kv_format("my_format")`으로 발견됩니다.

### 4단계 — 실제 엔진에서 실행

```bash
cargo run --release -p argus-engine --bin argus-cli -- \
  -m models/llama3.2-1b \
  -p "Hello, world! I am a" \
  -n 20 \
  --load-plugin target/release/libmy_format.dylib \
  --kv-format my_format
```

(macOS에선 빌드된 파일이 `libmy_format.dylib`, Linux/Android에선 `libmy_format.so`입니다 —
2단계에서 나온 경로를 쓰세요.)

**체크포인트.** 디코드가 끝까지 돌고 평소의 `Decode: X ms/tok` 줄을 찍습니다. 엔진이 당신의
`.so`를 `dlopen`해서 동적 레지스트리에서 `my_format`을 찾았다는 뜻입니다. (이 서술자는 내장
`q4_0`과 비트 단위로 동등하므로, 엔진은 이어서 저장을 타입화된 `q4_0` 빠른 경로로 보냅니다. 내장
등가물이 없는 서술자라면 — 예를 들어 `block_elems`가 다르면 — 대신 범용 opaque-서술자 바닥
경로를 탑니다.)

대신 이게 보이면:

```
Unknown --kv-format 'my_format' (not found in either static KV_FORMATS or dynamic
registration — check --load-plugin)
```

…`--kv-format`의 이름이 등록된 포맷과 안 맞거나, `--load-plugin`이 엉뚱한 파일을 가리키는
겁니다. ([문제 해결](#문제-해결) 참고.)

🎉 **방금 플래그 하나로 엔진이 로드하는 KV 캐시 포맷을 더했습니다 — 엔진 재빌드 없이요.** 다음
절들은 각 조각이 무엇을 하고 다른 축들은 어떻게 다른지 설명합니다.

---

## 플러그인 해부

모든 플러그인은 같은 두 부분 모양입니다. **트레이트를 구현**하고, **한 줄로 등록**합니다.

```rust
// 1. 축 트레이트를 구현해 동작을 선언한다.
impl KVFormat for MyFormat {
    fn name(&self) -> &str { "my_format" }
    fn layout(&self) -> KVLayoutDesc { /* ... */ }
}

// 2. 등록한다. 이 한 줄이 배선의 전부다 — 고칠 중앙 파일이 없다.
argus_extension_api::register_kv_format!("my_format", || Box::new(MyFormat));
argus_extension_api::export_plugin!();   // 동적 `.so` 경로에만 필요
```

그 등록 줄이 실제로 하는 일:

- **정적 경로(항상).** 전역 `KV_FORMATS` `#[distributed_slice]`에 `KVFormatReg { name, make }`
  항목을 더합니다. 당신 크레이트가 호스트 바이너리에 링크되면 링커가 그런 항목을 전부 한 배열로
  모으고, 엔진은 `find_kv_format("my_format")`으로 당신 걸 찾습니다. `match` 가지도, 레지스트리
  파일도 없습니다.
- **동적 경로(`--features plugin-cdylib`에서).** 같은 매크로가 피처 게이트 뒤에
  `unsafe extern "C"` vtable도 내보냅니다. `export_plugin!()`은 호스트가 `dlopen` 뒤에 호출하는
  `.so`당 진입 심볼 세 개(`register_kv_stages_v2` / `register_kv_formats_v2` /
  `register_backend_caps_v2`)를 내보냅니다 — 포맷만 있는 크레이트라면 포맷 심볼에만 항목이
  실립니다. 당신은 안전한 Rust만 쓰고, `unsafe` C-ABI 마샬링은 전부 매크로 안에 있습니다.

> **팩토리 인자 수는 축마다 다릅니다.** 포맷 팩토리는 인자가 없고(`|| Box::new(MyFormat)`),
> stage나 read 팩토리는 파라미터 구조체를 받습니다(`|p: StageParams| …` /
> `|p: ReadStageParams| …`).

그래서 **한 줄이 양쪽을 다 배선**하고, 어느 쪽이 활성인지는 크레이트를 어떻게 빌드하느냐(평범한
`rlib` 링크 vs `--features plugin-cdylib` `.so`)로 정해집니다. 이 이중성이
[정적 vs 동적](#정적-vs-동적-로드-경로-고르기) 이야기의 전부입니다.

**포맷**에만 해당하는 메모: `KVFormat`은 *순수 서술자*입니다(`name` + `layout`뿐). 커널을
싣지 않습니다. 새로운 정밀도는 당신이 돌려준 `KVLayoutDesc`로 엔진의 범용 `역양자화 → f32`
바닥을 타고, 손으로 쓰는 짝 커널(paired kernel)은 백엔드가 소유하는 별개 문제입니다
([`CONTEXT.ko.md`](../CONTEXT.ko.md)의 "짝 커널" 참고). 다른 축들(`KVCacheStage`,
`KVReadStage`)은 대신 읽기 전용 [`StageCtx`](#사용법-제거병합-stage-추가)에서 계산한 **계획**을
돌려줍니다.

---

## 네 개의 확장 축

정직한 능력 지도입니다. **"선택자 종류"와 "도는 곳"이 트레이트만큼 중요합니다** — 당신이 만든
플러그인을 지금 가진 바이너리에서 실제로 부를 수 있는지를 결정하니까요.

| 축 | 트레이트 | 돌려주는 것 | 등록 | CLI 선택자 | 선택자 종류 | 동적 `.so`? | 도는 곳 | 성숙도 |
|----|----------|-------------|------|------------|-------------|-------------|---------|--------|
| **format** | `KVFormat` | 레이아웃 서술자 | `register_kv_format!` | `--kv-format <name>` | 자유 문자열 | **예** | `argus-cli`, `argus-bench` | 프로덕션 |
| **read** | `KVReadStage` | `Option<KVReadPlan>` | 직접 `#[distributed_slice(KV_READ_STAGES)]` | `--read-stage <name>` | 자유 문자열 | 아니오(정적 전용) | `argus-cli` | 프로덕션(`quest`) |
| **stage**(제거/병합) | `KVCacheStage` | `Option<KVCachePlan>` | `register_kv_stage!` | `eviction plugin --name <n>` | 자유 문자열 | **예** | `argus-bench`, `argus-eval` | 프로덕션 |
| **backend-cap**(KIVI attn) | `KiviAttentionBackend` | 커널 결과 코드 | `register_kivi_attention_plugin!` | `--backend-cap <name>` | 자유 문자열 | **예**(OpenCL) | `argus-bench`, `argus-eval` | 프로덕션(OpenCL) |

네 축 모두 등록된 기법을 **이름으로, 엔진 수정 0으로** 선택합니다 — 차이는 *어느 바이너리*에서
돌고 플래그 철자가 무엇이냐뿐입니다:

1. **`--kv-format`과 `--read-stage`는 출하된 `argus-cli`에서 쓰는 자유 문자열 플래그입니다.** 가장
   깔끔한 "폴더만 더하고 이름으로 선택" 이야기이고, 빠른 시작이 포맷을 쓴 이유입니다.

2. **stage와 backend-capability는 `argus-bench` / `argus-eval`에서 이름으로 선택합니다.** 플러그인
   제거 stage는 `eviction plugin --name <n>`(`eviction` 서브커맨드의 자유 문자열 탈출구)로, 백엔드
   capability는 `--backend-cap <name>`(OpenCL)로 고릅니다. 둘 다 format과 똑같은 정적→동적 레지스트리
   조회로 풀립니다. `argus-cli`는 v0에서 제거도 KIVI도 안 돌리므로 이 두 축은 bench/eval 바이너리에
   삽니다. 그리고 resilience **manager**는 CLI로 고른 stage를 (evict 타이밍/비율로) 구동합니다 —
   다만 런타임에 *이름으로 기법을 전환*하는 건 아직 안 됩니다(`argus-shared` 프로토콜 추가 필요).
   자세한 건 [stage 사용법](#사용법-제거병합-stage-추가)을 보세요.

> 다섯 번째 **weight** 축(`WeightStage` / `WEIGHT_STAGES`, 런타임 가중치 swap용)도 있습니다.
> 트레이트 표면은 존재하고 단위 테스트도 있으며 지금도 직접
> `#[distributed_slice(WEIGHT_STAGES)]` 경로로 등록할 수 있습니다(편의 매크로는 없음) — 하지만
> **CLI 플래그도 없고 출하된 소비자도 아직 없으니**, 실험적이고 아직 쓸 수 있는 플러그인 표면이
> 아니라고 보세요.

---

## 사용법: 제거/병합 stage 추가

**stage** 축은 가장 풍부한 API이고 "결정은 플러그인, 실행은 엔진" 계약을 이해하기에 가장
좋습니다. stage는 캐시의 읽기 전용 뷰를 보고 어떤 토큰을 **남기고(keep)** 어떻게 **병합(merge)**
할지 계획을 돌려줍니다 — 버퍼를 직접 건드리지 않습니다.

템플릿: [`crates/techniques/example-keep-recent`](../crates/techniques/example-keep-recent)
(가장 최근 `target_len` 토큰만 남김 — 슬라이딩 윈도의 prefix-0 경우).

```rust
use argus_extension_api::{KVCachePlan, KVCacheStage, KeepSpec, StageCtx, StageParams};

struct KeepRecent;

impl KVCacheStage for KeepRecent {
    fn name(&self) -> &str {
        "example_keep_recent"
    }

    /// 엔진이 제거 결정을 필요로 할 때 호출된다. 아무것도 안 하려면 `None`.
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let current = ctx.current_pos();   // 지금 유효한 토큰 수
        let target = ctx.target_len();     // 엔진이 풀어준 예산
        if current <= target {
            return None;                   // 줄일 게 없음
        }
        // 가장 최근 `target`개 토큰을 남긴다(위치 오름차순).
        let keep: Vec<usize> = (current - target..current).collect();
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(keep),
            merges: Vec::new(),            // 가중 병합 없음
        })
    }
}

argus_extension_api::register_kv_stage!(
    "example_keep_recent",
    |_params: StageParams| Box::new(KeepRecent)
);
argus_extension_api::export_plugin!();
```

### 읽을 수 있는 것 — `StageCtx`

`StageCtx`는 캐시를 들여다보는 읽기 전용 창입니다. `plan`이 `&dyn StageCtx`로 받기 때문에 모든
접근자는 객체 안전(object-safe)합니다(출력은 슬라이스 반환이 아니라 `&mut [f32]` out 파라미터로
나갑니다). 접근자들:

| 접근자 | 의미 |
|--------|------|
| `current_pos()` | 유효한 토큰 수 |
| `target_len()` | 엔진이 비율/한도에서 풀어준 예산(토큰 수) |
| `layer_idx()` | 이번 호출이 어느 레이어용인지(레이어별 결정) |
| `n_kv_heads()`, `head_dim()` | 캐시 기하 |
| `importance()` → `Option<&[f32]>` | 토큰별 flat 중요도(H2O류 점수면 `Some`, 점수 없는 정책이면 `None`) |
| `tensor(TensorKind)` → `Option<&dyn TensorHandle>` | 원본 `Key`/`Value`/`AttnWeights`/`Scores`/`QueryStats`를 행 단위로 읽기 |

무거운 건 전부 `tensor()` 하나로 통합되고, `dequant_k`/`dequant_v`/`head_score`/`attn_weight`는
그 위의 기본 슈가입니다. stage는 이 읽기들로 **자기 지표를 직접 계산**합니다 — 예컨대 내장
`caote`([`crates/techniques/caote`](../crates/techniques/caote))는 `Value`를 읽어 엔진이
제공하지 않은 value-인지 criticality 점수를 스스로 계산합니다.

### 돌려주는 것 — `KVCachePlan`

- `keep: KeepSpec` — `LayerWide(Vec<usize>)`(모든 head에 같은 토큰, 오름차순) 또는
  `PerHead(Vec<Vec<usize>>)`(KV head마다 오름차순 목록 하나, 예: H2O+).
- `merges: Vec<WeightedMerge>` — 선택적 가중 병합(D2O는 제거되는 토큰을 크기 보존 가중치로 남는
  슬롯에 접습니다). 제거만 하는 정책은 비웁니다.

엔진은 이 계획을 받아 `compact`로 실제 버퍼 재작성을 수행합니다. 호출 간 상태(EMA 누산기, 페이지
메타데이터)는 `plan`이 `&self`를 받으므로 **`&self`에 내부 가변성(`Mutex`)으로** 둡니다 — 내장
`quest`(`Mutex<QuestState>`)와 엔진 내부 `d2o` stage
(`engine/src/kv/d2o_handler.rs`, 기법 크레이트가 아님)가 어떻게 보관하는지 보세요. (반면 `caote`는
무상태라 매 호출 `ctx`에서 다시 계산합니다.)

### stage 고르기

플러그인 stage는 제거가 가능한 바이너리에서 **이름으로** 선택합니다:

```bash
# stage .so 를 빌드한 뒤(동적 경로 참고), 로드 + 이름으로 선택:
argus-bench ... --load-plugin target/release/libmy_stage.so eviction plugin --name my_stage
# 정적 force-link 했다면 이름만 대면 됨:
argus-eval  ... eviction plugin --name my_stage
```

`eviction plugin --name <name>`은 `eviction` 서브커맨드의 자유 문자열 탈출구로, `--kv-format`의
stage-축 짝입니다. 이름은 `make_stage`(정적 `KV_CACHE_STAGES` 먼저, 그다음 `--load-plugin` 동적
레지스트리)로 풀리므로, 등록된 어떤 stage든 **엔진 수정 0**으로 선택됩니다. (내장 정책은 자기
타입 서브커맨드를 그대로 유지합니다: `eviction sliding`, `eviction h2o`, ….)

범위 메모 둘:

- **bench/eval 전용, `argus-cli` 아님.** 제거는 cache manager가 필요한데 그건 `argus-bench`와
  `argus-eval`만 구성합니다. `argus-cli`(단일 프롬프트 happy path)는 v0에서 제거를 거부합니다.
- **manager는 구동하지만, 런타임 이름 전환은 아직 없음.** resilience manager는 CLI로 고른 stage를
  (evict 타이밍/비율로) 구동합니다 — 다만 런타임에 *다른* 기법을 이름으로 고르는 건 manager IPC에
  새 `argus-shared` 커맨드가 필요해 오늘은 범위 밖입니다.

> 일부 `argus-extension-api` 문서 주석은 선택자를 여전히 `--eviction-policy <name>`이라 부릅니다 — 낡은
> 이름입니다. 실제 CLI 형태는 `eviction plugin --name <name>`(또는 내장 `eviction <policy>`
> 서브커맨드)입니다.

---

## 사용법: read stage 추가

**read** 축은 저장된 것을 바꾸지 않고 *무엇을 읽을지* — 쿼리 인지로 추린 토큰/페이지 — 를
결정합니다. `None`은 "전체 읽기"(정확함, 기본값)라서, read stage는 옵트인 근사일 뿐 절대
정확성을 위협하지 않습니다.

레퍼런스: 내장 [`crates/techniques/quest`](../crates/techniques/quest) (Quest, ICML'24 — 페이지별
K min/max를 유지해 상한 내적으로 top-k 페이지만 읽음).

```rust
use linkme::distributed_slice;
use argus_extension_api::{
    KV_READ_STAGES, KVReadPlan, KVReadStage, KVReadStageReg, ReadGranularity,
    ReadStageParams, StageCtx,
};

struct MyRead { /* 페이지 메타데이터를 Mutex 뒤에 보관 */ }

impl KVReadStage for MyRead {
    fn name(&self) -> &str { "my_read" }

    /// 레이어마다 attention 직전에 한 번 발화. `None` = 이 레이어는 전체 읽기.
    fn read_plan(&self, ctx: &dyn StageCtx) -> Option<KVReadPlan> {
        // ... ctx.tensor(Key)/ctx.current_pos()에서 위치/페이지를 고른다 ...
        Some(KVReadPlan {
            granularity: ReadGranularity::Page { page_size: 16 },
            select: vec![/* 오름차순 페이지 인덱스 */],
        })
    }
}

// read 축에는 편의 매크로가 없습니다 — 등록 항목을 직접 제출합니다.
#[distributed_slice(KV_READ_STAGES)]
static MY_READ: KVReadStageReg = KVReadStageReg {
    name: "my_read",
    make: |p: ReadStageParams| Box::new(MyRead { /* ... */ }),
};
```

알아둘 두 가지:

- **자유 문자열 플래그 `--read-stage my_read`로 선택**하고, 출하된 `argus-cli`에서 디코드까지
  닿습니다. (기본 미설정 = 전체 읽기 = 플러그인 없는 것과 바이트 동일.)
- **정적 전용.** read 축에는 동적 `.so` 진입 심볼이 없으므로, read stage는 **force-link** 해야
  하고([정적 경로](#정적force-link-경로) 참고) `--load-plugin`으로 추가할 수 없습니다. `quest`가
  이렇게 비-옵션 의존성으로 배선돼 있습니다.

> read 계획은 활성 **포맷**이 선택적 읽기를 지원할 때만 효력이 있습니다(기본 `StandardFormat`은
> 지원). `--read-stage`를 opaque한 커스텀 `--kv-format`과 묶으면 한 번 stderr 경고를 내고 전체
> 읽기로 폴백합니다 — 정확하긴 하나 가속은 안 됩니다.

---

## 사용법: 백엔드 capability(KIVI) 추가

> **고급/실험적.** 이 축은 GPU 특화이고 출하된 `argus-cli`에서 도달할 수 없습니다.

**backend-capability** 축은 백엔드 위에 특화 융합 커널을 얹습니다 — in-tree 인스턴스는 KIVI 융합
역양자화+attention입니다. `KiviAttentionBackend`를 구현하고(빌려온 GPU 컨텍스트로 OpenCL 커널을
한 번 빌드), `register_kivi_attention_plugin!`으로 등록합니다. 레퍼런스:
[`crates/techniques/example-backend-cap`](../crates/techniques/example-backend-cap) — 실제 GPU
연산 없이 ABI 왕복만 검증합니다.

capability는 `--backend-cap <name>`으로 이름으로 선택합니다:

```bash
argus-bench ... --kv-mode kivi --load-plugin target/release/libmy_kivi.so --backend-cap my_kivi
```

`--backend-cap <name>`은 `KiviAttentionBackend`를 레지스트리 이름으로 풀어(정적
`KIVI_ATTENTION_REGS` 먼저, 그다음 `--load-plugin` 동적 레지스트리) 엔진 내장 OpenCL 구현 대신
설치합니다. 미설정 = 내장. backend-capability 축의 `--kv-format` 짝입니다.

범위 메모:

- **OpenCL 전용, bench/eval 전용.** capability는 OpenCL 백엔드에만 등록됩니다(CPU/CUDA 분기는 없어
  `--backend-cap`이 경고 후 무시됨). `--kv-mode kivi`는 `argus-cli`가 v0에서 거부합니다.
- **실제 커널은 기기가 필요.** 플러그인 `KiviAttentionBackend`는 진짜 GPU 코드를 돌리므로, 호스트
  CI는 컴파일 + 합성 게이트 테스트(`engine/tests/gate_c_backend_cap_dlopen.rs`)만 검증하고, 실제
  정확성은 온디바이스(Adreno/Mali)에서만 확인됩니다.

---

## 생명주기와 성능 계약

당신의 메서드가 *언제* 도는지를 아는 것이, 잘 도는 플러그인과 디코드 처리량을 조용히 망가뜨리는
플러그인을 가릅니다.

| 메서드 | 도는 시점 | 핫패스? |
|--------|-----------|---------|
| 팩토리 `make(...)` | 세션/캐시 생성 시 한 번 | 아니오 — 무거운 셋업은 여기서 |
| `KVFormat::layout()` | 생성 시 한 번 | 아니오 — 다만 돌려준 서술자가 핫패스에서 읽히니 trivial한 POD 반환으로 유지 |
| `KVCacheStage::plan()` | 엔진이 제거를 트리거할 때(예산/압력) | 따뜻함 — 디코드 중에 돔 |
| `KVReadStage::read_plan()` | 레이어마다 attention 직전 한 번 | **예 — 레이어마다, 토큰마다** |

따뜻한/뜨거운 메서드의 성능 계약:

- **핫패스에서 할당/경합 락/로깅 금지.** 버퍼는 미리 잡고 커널은 팩토리에서 빌드하세요.
- **상태는 `&self`에 내부 가변성(`Mutex<...>`)으로 보관.** `plan`/`read_plan`이 `&self`를
  받으니까요. `quest`의 `Mutex<QuestState>` 페이지 메타데이터를 본떠 쓰세요(엔진 내부 `d2o`
  stage도 EMA를 같은 방식으로 보관합니다).
- **정리는 RAII** — 자원을 소유하면 `Drop`을 구현하세요. 캐시/세션이 끝나면 엔진이 인스턴스를
  떨굽니다.
- **`--profile` 없이 측정.** 레포 정책상 `--profile`은 토큰당 ~54ms 동기 오버헤드를 더하고
  happy-path 바이너리에서 거부됩니다. 실제 처리량은 `Decode: X ms/tok` 로그 줄에서 읽고,
  `--profile`은 *상대적* op별 비교에만 쓰세요.

---

## 정적 vs 동적: 로드 경로 고르기

모든 `register_*!` 매크로는 기법을 **양쪽** 다 배선합니다. 어느 쪽이 활성인지는 빌드 시점에
당신이 고릅니다.

| | **정적**(linkme force-link) | **동적**(`cdylib` `.so` + `dlopen`) |
|--|----------------------------|-------------------------------------|
| 기법 추가 방법 | 당신 크레이트를 링크해 호스트 재컴파일 | 시작 시 `--load-plugin foo.so`, **호스트 재빌드 없음** |
| 핫스왑 | 아니오 | 예 |
| 런타임 오버헤드 | 0(직접 호출) | `dlopen` 1회 + 간접 C-ABI 호출 |
| 타입 풍부함 | 완전한 Rust 타입 | flat C-ABI 구조체(매크로 안에서 처리됨) |
| ABI 안정성 | 해당 없음(같은 컴파일) | **취약** — `argus-extension-api` 버전이 맞아야 함 |
| 충돌 격리 | 프로세스 공유 | 프로세스 공유(샌드박스 없음) |
| 배포 주체 | 엔진 메인테이너 | 누구나, 독립 `.so`로 |
| 지원 축 | 네 축 모두 | format, stage, backend-cap — **read 제외** |

**컴파일 타임 레지스트리가 어떻게 런타임 플러그인을 집어가는가**(겉보기 역설): 두 경로는 *서로
다른 시점*에 등록합니다. 정적 항목은 **링크 시점**에 링커 섹션으로 모이고, `.so`의 능력은
**`dlopen`이 도는 시점**에 등록됩니다. 조회 시점에 엔진은 정적 슬라이스를 먼저 보고 동적
레지스트리로 폴백하므로, 둘 다 하나의 출처 무관 `find_*` / `make_*` 솔기로 풀립니다. 한 줄
(`register_*!` + `export_plugin!()`)이, `plugin-cdylib` 피처와 `crate-type = ["cdylib", "rlib"]`로
게이트되어, 양쪽을 다 섬깁니다.

경험칙: **개발과 내장 출하는 정적으로**, 엔진 재빌드 없이 기법을 배포하고 싶을 때 동적
경로를(오늘은 자유 문자열 플래그로 선택되는 **format** 축이 가장 깔끔합니다).

---

## 동적 `.so` 경로 자세히

`.so`를 빌드하고 로드하는 전 과정:

```bash
# 빌드(엔진과 같은 프로파일):
cargo build --release -p my-format --features plugin-cdylib
# -> target/release/libmy_format.{so|dylib}

# 시작 시 로드(반복 가능; 각 .so는 한 번 dlopen되어 export한 모든 축으로 라우팅)하고 선택:
argus-cli ... --load-plugin target/release/libmy_format.so --kv-format my_format
```

내부에서 벌어지는 일:

- `export_plugin!()`은 `.so`당 동일한 세 개의 `#[no_mangle] extern "C"` 진입 심볼을 항상
  내보냅니다(동적 축마다 하나): `register_kv_stages_v2`, `register_kv_formats_v2`,
  `register_backend_caps_v2`. (read 축 심볼은 **없습니다** — read는 정적으로 남습니다.) 안 쓴
  축은 그냥 개수 0을 export하고, 부재는 에러가 아닙니다.
- `--load-plugin`은 각 경로를 **한 번** `dlopen`해 그 핸들을 세 축 등록기에 모두 라우팅합니다.
  그래서 한 `.so`가 여러 축을 묶을 수 있습니다 —
  [`example-bundle`](../crates/techniques/example-bundle)(stage + format를 한 `.so`에),
  [`example-multi-format`](../crates/techniques/example-multi-format)(포맷 둘) 참고.
- **ABI 핸드셰이크.** 각 envelope가 `abi_version`을 싣고, 호스트는 불일치 시 `.so`를 빠르게
  거부합니다(`KV_STAGE`/`KV_FORMAT_ABI_VERSION = 2`, `BACKEND_CAP = 1`). 플러그인을 엔진의
  `argus-extension-api` 버전에 맞춰 재빌드하세요.
- **이름 충돌.** 내장(또는 다른 로드된 플러그인)과 충돌하는 동적 이름은 거부됩니다 — 내장이
  이깁니다. 한 `.so`의 등록은 2-pass 원자적이라, 부분/충돌 envelope는 아무것도 등록하지 않습니다
  ([`example-rollback`](../crates/techniques/example-rollback) 참고).
- **`export_plugin!()`을 깜빡했다면?** `.so`가 진입 심볼을 안 내보내 능력 0을 기여하고, 로더가
  bail합니다([`example-no-export`](../crates/techniques/example-no-export) 참고).

**안전.** `.so`는 샌드박스 없이 프로세스 안에서 돕니다 — panic이나 segfault는 엔진을 죽입니다.
정적 빌드에선 `plugin-cdylib` 피처를 **꺼 두세요**. 안 그러면 `#[no_mangle]` C-ABI 심볼이
force-link된 사본과 충돌합니다.

플랫폼 메모: `crate-type = ["cdylib"]`는 Linux/Android에선 `lib<name>.so`를, macOS 호스트에선
`lib<name>.dylib`를 만듭니다.

---

## 정적(force-link) 경로

내장(`quest`, `synth-q4-format`)이 항상-링크로 배선되는 방식이자, **read** stage(동적 경로가
없음)나 엔진에 컴파일해 넣고 싶은 어떤 기법에든 쓰는 경로입니다. (`caote` stage도 같은 방식이지만
cargo 피처 뒤에 있습니다 — [stage 사용법](#사용법-제거병합-stage-추가) 참고.)

1. 크레이트를 `engine/Cargo.toml`에 경로 의존성으로 추가:
   ```toml
   my-format = { path = "../crates/techniques/my-format" }
   ```
2. **force-link 하기.** Rust는 참조 안 된 rlib을 떨구고 — 그러면 당신의 `#[distributed_slice]`
   등록도 같이 사라집니다 — 그래서 그 축의 지정된 force-link 자리에 참조 줄을 더합니다(예:
   `use my_format as _;`). (포맷: `engine/src/format/builtin_kv_formats.rs`; read stage:
   `engine/src/kv/read/read_stage_registry.rs`.)
3. 엔진의 시작 자가 점검이 당신 이름이 있는지 단언합니다(fat-LTO / `--gc-sections`가 등록을 조용히
   떨구는 걸 막음). 이름이 없으면 기법을 조용히 끄는 대신 시작 때 빠르게 실패합니다.

정적 기법은 `--features plugin-cdylib`도 `--load-plugin`도 필요 없고, linkme 슬라이스에서
`find_kv_format` / `find_read_stage` / `find_stage`로 곧장 풀립니다.

---

## 문제 해결

뚜렷한 컴파일 에러가 안 나는, 이 시스템의 *대표적인* 실패 양상들입니다.

| 증상 | 원인 | 해결 |
|------|------|------|
| 플러그인은 컴파일되는데 엔진이 이름을 못 찾음 | 크레이트가 호스트 링크 그래프에 없거나, 등록이 dead-code로 제거됨 | 정적 경로면 경로 의존성 **과** `use my_crate as _;` force-link 줄을 추가; 동적 경로면 `--load-plugin <.so>` 전달 |
| `Unknown --kv-format '…' (… check --load-plugin)` | 이름 오타, 또는 `--load-plugin`이 잘못된/오래된 `.so`를 가리킴 | `name()` 문자열이 플래그 값과 같은지 확인; 재빌드 후 갓 나온 `target/<profile>/lib*.so`를 가리키게 |
| `.so`는 로드되나 `0 capabilities registered (… export_plugin! missing?)` | `argus_extension_api::export_plugin!()`을 깜빡함 | `.so`당 한 번 추가([`example-no-export`](../crates/techniques/example-no-export)) |
| `.so` 거부: `abi_version … != expected …` | 다른 `argus-extension-api` 버전으로 빌드된 플러그인 | 엔진 버전에 맞춰 재빌드(ABI는 버전 간 안정적이지 않음) |
| 로드 시 이름 중복 거부 | 이름이 내장이나 다른 플러그인과 충돌 | 기법 이름 변경([`example-rollback`](../crates/techniques/example-rollback)이 원자적 롤백 동작을 보여줌) |
| 링커 `#[no_mangle]` 심볼 충돌 | 어떤 크레이트가 force-link **되면서** `--features plugin-cdylib`로도 빌드됨 | 정적 빌드에선 `plugin-cdylib`를 끄고, `.so`를 만들 때만 켜기 |
| 커스텀 제거 stage가 선택 안 됨 | `argus-cli` 사용(v0엔 제거 없음), 또는 이름 오타 | `argus-bench`/`argus-eval`에서 `eviction plugin --name <name>`으로 선택; 이름이 `name()`과 일치하는지 확인 — [stage 고르기](#사용법-제거병합-stage-추가) 참고 |
| stage/read 플러그인을 더했더니 디코드가 느려짐 | 핫패스에서 할당/락/로깅 | 셋업을 팩토리로 옮기고, 상태는 `&self`에 두고, `--profile` 말고 `Decode: X ms/tok`로 측정 |

---

## 패키징과 배포 체크리스트

"트레이트를 구현했다"를 "누가 로드할 수 있는 크레이트를 출하했다"로 바꾸기:

- [ ] `Cargo.toml`: `crate-type = ["cdylib", "rlib"]`과 `plugin-cdylib = []` 피처(기본 off).
- [ ] 의존성은 정확히 `argus-extension-api`(경로 또는 버전) + `linkme = "0.3"` — 엔진 의존성 없음.
- [ ] 등록 발화를 단언하는 단위 테스트(`find_kv_format(...).is_some()` 등).
- [ ] `.so`를 빌드한 `argus-extension-api` 버전을 명시 — 동적 ABI는 버전 간 안정적이지 않음.
- [ ] 라이선스: 이 레포 기여는 `MIT OR Apache-2.0` 이중 라이선스 — in-tree 크레이트는 맞추세요.
- [ ] 유일하고 서술적인 `name()` 문자열을 고르세요(CLI 선택자이자 레지스트리 키입니다).

---

## 다음으로 갈 곳

- **레퍼런스** — 트레이트/매크로 표면 전체와 문서 주석:
  [`crates/argus-extension-api/src/lib.rs`](../crates/argus-extension-api/src/lib.rs) (`cargo doc
  -p argus-extension-api --open`)과 크레이트 자체
  [`README.md`](../crates/argus-extension-api/README.md).
- **개념/용어** — [`CONTEXT.ko.md`](../CONTEXT.ko.md): stage ⊥ format ⊥ hardware 모델, *포맷*이
  왜 절대 *레이어*가 아닌지, 복합 연산이 어떻게 단일 축 원시 연산으로 분해되는지.
- **템플릿, 축별 매핑:**

  | …의 템플릿으로 쓰기 | 크레이트 |
  |---------------------|----------|
  | **stage**(제거/병합) | [`example-keep-recent`](../crates/techniques/example-keep-recent) |
  | **format** | [`example-kv-format`](../crates/techniques/example-kv-format) |
  | 지표를 직접 계산하는 실제 stage | [`caote`](../crates/techniques/caote) |
  | 실제 read stage | [`quest`](../crates/techniques/quest) |
  | 한 `.so`에 여러 축 | [`example-bundle`](../crates/techniques/example-bundle) |
  | 한 `.so`에 여러 포맷 | [`example-multi-format`](../crates/techniques/example-multi-format) |
  | 백엔드 capability(KIVI) | [`example-backend-cap`](../crates/techniques/example-backend-cap) |
  | 실패 경로(export 누락, 이름 충돌) | [`example-no-export`](../crates/techniques/example-no-export), [`example-rollback`](../crates/techniques/example-rollback) |
