# Argus Engine

[![CI](https://github.com/hedone21/argus-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/hedone21/argus-engine/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#라이선스)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue.svg)](Cargo.toml)
[![Release](https://img.shields.io/github/v/release/hedone21/argus-engine)](https://github.com/hedone21/argus-engine/releases)

[English](README.md) | **한국어**

**ARM64 엣지·모바일 기기를 위한 온디바이스 LLM 추론 엔진**, Rust로 작성되었습니다.

Argus는 유연한 백엔드 추상화와 zero-copy 메모리 아키텍처로 Android/Linux ARM64 SoC를
타깃합니다. HuggingFace Safetensors 및 GGUF 형식의 Llama 계열·Qwen/Gemma 모델을
Q4_0/Q8_0 블록 양자화와 OpenCL / CUDA GPU 가속으로 구동합니다.

이 저장소는 **엔진**입니다. Argus는 세 개의 저장소로 구성됩니다:

| 저장소 | 역할 |
|------|------|
| [`argus-engine`](https://github.com/hedone21/argus-engine) | LLM 추론 엔진 (본 저장소) |
| [`argus-shared`](https://github.com/hedone21/argus-shared) | IPC 프로토콜 타입 (manager ↔ engine) |
| [`argus-manager`](https://github.com/hedone21/argus-manager) | 시스템 리소스 관리 서비스 |

## 핵심 기능

- **ARM64 최적화** — Android/Linux ARM64 SoC를 위한 NEON + dotprod 인트린식.
- **Zero-copy 메모리** — UMA SoC에서 `CL_MEM_ALLOC_HOST_PTR` / DMA-BUF로 GPU 버퍼를
  CPU 포인터에 매핑하여 CPU↔GPU memcpy를 제거.
- **플러그형 백엔드** — CPU(NEON) / OpenCL(Adreno) / CUDA를 아우르는 `Backend` 트레이트.
- **양자화** — Q4_0 / Q8_0 블록 양자화, F16/BF16. GGUF 사전 양자화 모델은 바로 로드되고,
  Safetensors F16/BF16은 로드 시 변환.
- **KV-cache eviction** — Sliding Window / H2O / H2O+ / D2O(merge 보상) / StreamingLLM을
  조합 가능한 `KVCacheStage` 플러그인으로 제공.
- **KIVI KV-cache 양자화** — 메모리 절감을 위한 동적 Q4/Q8 KV 양자화.
- **Flash attention** — GQA 인식 GPU flash attention (strided).
- **Tensor partition** — FFN gate/up matmul을 GPU + CPU에 동시 분할.
- **Adaptive resilience** — 메모리/발열 압력 하의 런타임 적응(eviction, 백엔드 전환, throttle)을
  위한 `argus-manager`와의 선택적 연동.
- **Zero-compile 확장 표면** — KV-cache stage / format / read-stage를 별도 크레이트로 추가하면
  엔진 코어 수정 없이 `linkme`로 자기 등록. 세 직교 축(stage ⊥ format ⊥ hardware)은
  [`CONTEXT.ko.md`](CONTEXT.ko.md), 확장 작성법은
  [`crates/technique-api/README.md`](crates/technique-api/README.md) 참고.

## 지원 모델 · 하드웨어

### 모델

| 패밀리 | 아키텍처 | 소스 포맷 | 양자화 |
|--------|----------|-----------|--------|
| Llama | Llama 계열 (`LlamaForCausalLM`) | GGUF, Safetensors | Q4_0, Q8_0, F16, BF16 |
| Qwen | Qwen2 / Qwen2.5 | GGUF, Safetensors | Q4_0, Q8_0, F16, BF16 |
| Gemma | Gemma / Gemma 2 / Gemma 3 (text) | GGUF, Safetensors | Q4_0, Q8_0, F16, BF16 |

GGUF 권장(dtype 자동 감지, 로드 시 변환 없음); Safetensors F16/BF16은 로드 시 변환.

### 하드웨어 백엔드

| 백엔드 | 빌드 | 하드웨어 / 타깃 |
|--------|------|------------------|
| CPU (NEON + dotprod) | 기본 | ARM64 — Android / Linux |
| CPU (AVX2 + FMA) | 기본 | x86_64 — Linux (호스트 / 개발) |
| OpenCL | 기본 (`opencl`) | Adreno GPU — Android ARM64 (프로덕션 경로) |
| CUDA | `--no-default-features --features cuda` | NVIDIA 외장 GPU / Jetson Orin |
| CUDA (임베디드 UMA) | `--features cuda-embedded` | Jetson Xavier |

크로스 컴파일 타깃: `aarch64-linux-android`, `aarch64-unknown-linux-gnu`,
`aarch64-unknown-linux-musl`, `x86_64-unknown-linux-gnu` (`.cargo/config.toml` 참고).

## 사전 요구사항

- **Rust** (stable): `rustup install stable`.
- **OpenCL 헤더** — 기본 빌드는 `opencl` 기능을 활성화합니다. Linux:
  `sudo apt-get install ocl-icd-opencl-dev`. macOS는 OpenCL 프레임워크를 기본 제공합니다.
  빌드에 GPU는 *필요하지 않습니다* (GPU 백엔드 실행 시에만 필요).

## 설치 / 소스에서 빌드

Argus는 소스 형태로 배포됩니다. [`argus-shared`](https://github.com/hedone21/argus-shared)를
git 의존성으로 사용하므로 crates.io에 게시되지 않습니다(`cargo install argus-engine`은 동작하지
않습니다). git 체크아웃에서 직접 빌드하세요.

```bash
git clone https://github.com/hedone21/argus-engine.git
cd argus-engine
cargo build --release

# CPU
./target/release/argus_cli -m model.gguf --prompt "Hello" -n 50 -b cpu

# GPU (OpenCL, Adreno 프로덕션 경로)
./target/release/argus_cli -m model.gguf --prompt "Hello" -n 50 -b opencl
```

GGUF가 권장 모델 형식입니다(dtype 자동 감지, 로드 시 변환 없음). `tokenizer.json`은 `.gguf`
파일 옆에 두거나 `--tokenizer-path`로 지정해야 합니다.

```bash
# CUDA (NVIDIA 외장 GPU / Jetson) — opencl과 상호배타
cargo build --release --no-default-features --features cuda
```

### 모델 변환

Argus는 GGUF를 직접 로드합니다. HuggingFace Safetensors 모델로부터 GGUF를 생성하거나
AUF(Argus Unified Format) 자산을 빌드하려면 [`scripts/`](scripts/)의 도구를 사용하세요:

```bash
pip install -r scripts/requirements.txt

# Safetensors → GGUF (기본 Q4_0)
python scripts/convert_safetensors_to_gguf.py models/qwen2.5-1.5b out.gguf

# Safetensors 또는 GGUF → AUF (원샷; 필요 시 auf_tool 바이너리를 빌드)
scripts/convert_to_auf.sh --input models/qwen2.5-1.5b/ --output model.auf
```

전체 변환 가이드는 [`scripts/README.md`](scripts/README.md)를 참고하세요.

### Android (크로스 컴파일 + 배포)

Android NDK로 `aarch64-linux-android`를 빌드합니다. Adreno 프로덕션 경로는 `-b opencl`이며,
기기에는 벤더 `libOpenCL.so`가 필요합니다(여기서 배포하지 않음 — 기기의 `/vendor/lib64`에서
가져오세요).

`scripts/run_device.py`는 adb(및 Jetson용 ssh)로 빌드 → 푸시 → 실행을 자동화하며, 두 개의
로컬 설정 파일로 구동됩니다(템플릿은 커밋되어 있고, 채워 넣은 사본은 gitignore 처리됨):

```bash
cp hosts.toml.example hosts.toml        # NDK 경로 설정
# 또는: python scripts/device_registry.py bootstrap-host   # NDK 자동 탐지
cp devices.toml.example devices.toml    # 기기 등록
# 또는: python scripts/device_registry.py discover         # 연결된 adb 기기 자동 탐지

python scripts/run_device.py --list-devices
python scripts/run_device.py -d android argus_cli \
    --model-path /data/local/tmp/models/model.gguf -b opencl --prompt "Hello"
```

원시 타깃 플래그는 `.cargo/config.toml`을, 디바이스 러너 및 평가 도구는
[`scripts/README.md`](scripts/README.md)를 참고하세요.

## Cargo 기능 (features)

| Feature | 기본 | 설명 |
|---------|------|------|
| `opencl` | ✅ | OpenCL GPU 백엔드 (Adreno) |
| `profile` | ✅ | 연산별 프로파일링 계측 |
| `cuda` | | CUDA 백엔드 (외장 GPU / Jetson) |
| `cuda-embedded` | | 임베디드 UMA용 CUDA (Jetson Xavier) |
| `resilience` | | `argus-manager`와의 D-Bus IPC 연동 |
| `caote` | | CAOTE value-aware eviction 플러그인 |
| `rkv` | | R-KV 결합 eviction 측정 프로토타입 (실험적, 기본 off) |

> `opencl`과 `cuda`/`cuda-embedded`는 상호배타입니다 — GPU 백엔드를 정확히 하나만 선택하세요.
> 위 CUDA 빌드는 `--no-default-features`로 기본 `opencl`을 제거하고 `--features cuda`를
> 추가합니다. GPU 백엔드를 **하나도** 켜지 않는 빌드는 현재 지원하지 않습니다.

## 문서

- [`CONTEXT.ko.md`](CONTEXT.ko.md) — 도메인 용어집: stage ⊥ format ⊥ hardware 축과 캐시 관리 어휘.
- [`AGENTS.md`](AGENTS.md) — AI 코딩 에이전트 및 기여자를 위한 가이드.

## 라이선스

[Apache-2.0](LICENSE-APACHE) 또는 [MIT](LICENSE-MIT) 중 선택하여 사용할 수 있습니다.
별도로 명시하지 않는 한, 기여물은 위와 동일하게 이중 라이선스로 제공됩니다.

이 엔진의 일부는 [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)과
[jquesnelle/yarn](https://github.com/jquesnelle/yarn)(모두 MIT)에서 이식했습니다. 전체 출처
표기는 [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md)를 참고하세요.
