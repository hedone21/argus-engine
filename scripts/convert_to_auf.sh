#!/usr/bin/env bash
# convert_to_auf.sh — Safetensors/GGUF에서 한 번에 AUF 자산을 만든다.
#
# 두 변환 단계를 통합:
#   1) Safetensors 디렉토리 → GGUF (scripts/convert_safetensors_to_gguf.py)
#   2) GGUF → AUF (target/release/auf_tool build, v0.1.1)
#
# 사용 예:
#   # Safetensors → AUF (가장 일반적)
#   scripts/convert_to_auf.sh \
#       --input  models/llama-3.2-1b/ \
#       --output models/llama-3.2-1b.auf
#
#   # 이미 GGUF가 있으면 단계 1 건너뜀
#   scripts/convert_to_auf.sh \
#       --input  models/llama-3.2-1b-q4_0.gguf \
#       --output models/llama-3.2-1b.auf
#
#   # F16 GGUF 경유 + 강제 lm_head 사전 양자화 (Sprint G-1 권장)
#   scripts/convert_to_auf.sh \
#       --input  models/llama-3.2-1b/ \
#       --output models/llama-3.2-1b.auf \
#       --outtype f16 --include-lm-head on
#
#   # 디바이스 변종만 동봉 (mobile 배포)
#   scripts/convert_to_auf.sh \
#       --input  models/llama-3.2-1b/ \
#       --output models/llama-3.2-1b-mobile.auf \
#       --variants adreno_soa
#
#   # 중간 GGUF 보존
#   scripts/convert_to_auf.sh \
#       --input  models/llama-3.2-1b/ \
#       --output models/llama-3.2-1b.auf \
#       --keep-gguf models/llama-3.2-1b-q4_0.gguf

set -euo pipefail

usage() {
    cat <<EOF
convert_to_auf.sh — Safetensors/GGUF에서 한 번에 AUF 빌드

필수:
  --input <PATH>           Safetensors 디렉토리 또는 .gguf 파일
  --output <PATH>          출력 .auf 경로

선택:
  --outtype <q4_0|f16>     Safetensors→GGUF 변환 dtype. 기본 q4_0
                           (입력이 .gguf면 무시됨)
  --variants <LIST>        AUF backend variant. comma-separated 또는 all.
                           기본 all (예: adreno_soa,cpu_aos)
  --include-lm-head <MODE> auto|on|off. 기본 auto
                           (v0.1.1 lm_head Q4_0 entry; off = v0.1.0 호환)
  --dtypes <LIST>          AUF v0.2 multi-dtype variant — 동봉할 dtype 목록.
                           comma-separated. 예: q4_0,f16
                           2개 이상 지정 시 capability bit 3 자동 set + format_minor=2.
                           미지정 시 single-dtype (source dtype 그대로, v0.1.x 호환).
  --default-dtype <DT>     v0.2 multi-dtype 모드에서 META.default_dtype 명시.
                           --dtypes에 포함된 값이어야 한다. 미지정 시 첫 번째.
  --tokenizer <PATH>       tokenizer.json 경로. 미지정 시 자동 탐색:
                             - Safetensors 입력: <input>/tokenizer.json
                             - GGUF 입력: <gguf-dir>/tokenizer.json
                                          → <gguf-stem>.tokenizer.json
  --keep-gguf <PATH>       Safetensors 입력일 때 중간 GGUF를 이 경로에 보존.
                           미지정 시 임시 디렉토리에 만들고 종료 시 삭제.
  --created-by <STR>       AUF 헤더 created_by 문자열 (32B UTF-8).
  --quiet                  진행 로그 억제.
  -h, --help               이 메시지 출력.

요구 사항:
  - python3 + safetensors + numpy
  - cargo (auf_tool 빌드용; 이미 빌드돼 있으면 재사용)
EOF
}

# 인자 기본값
INPUT=""
OUTPUT=""
OUTTYPE="q4_0"
VARIANTS="all"
INCLUDE_LM_HEAD="auto"
DTYPES=""
DEFAULT_DTYPE=""
TOKENIZER=""
KEEP_GGUF=""
CREATED_BY=""
QUIET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)            INPUT="$2"; shift 2 ;;
        --output)           OUTPUT="$2"; shift 2 ;;
        --outtype)          OUTTYPE="$2"; shift 2 ;;
        --variants)         VARIANTS="$2"; shift 2 ;;
        --include-lm-head)  INCLUDE_LM_HEAD="$2"; shift 2 ;;
        --dtypes)           DTYPES="$2"; shift 2 ;;
        --default-dtype)    DEFAULT_DTYPE="$2"; shift 2 ;;
        --tokenizer)        TOKENIZER="$2"; shift 2 ;;
        --keep-gguf)        KEEP_GGUF="$2"; shift 2 ;;
        --created-by)       CREATED_BY="$2"; shift 2 ;;
        --quiet)            QUIET=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        *)                  echo "ERROR: unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "ERROR: --input and --output are required." >&2
    usage; exit 1
fi
if [[ ! -e "$INPUT" ]]; then
    echo "ERROR: input does not exist: $INPUT" >&2; exit 1
fi

log() { [[ $QUIET -eq 1 ]] || echo "[convert_to_auf] $*" >&2; }

# 프로젝트 루트 자동 탐색 (스크립트가 어디서 호출되어도 동작)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
PY_CONVERT="$SCRIPT_DIR/convert_safetensors_to_gguf.py"
AUF_TOOL="$PROJECT_ROOT/target/release/auf_tool"

# 1) auf_tool 바이너리 확보
if [[ ! -x "$AUF_TOOL" ]]; then
    log "auf_tool 바이너리가 없음 — cargo build로 생성"
    (cd "$PROJECT_ROOT" && cargo build --release -p argus-engine --bin auf_tool ${QUIET:+--quiet})
fi
[[ -x "$AUF_TOOL" ]] || { echo "ERROR: auf_tool 빌드 실패" >&2; exit 1; }

# 2) 입력 분기 — Safetensors 디렉토리인지 GGUF 단일 파일인지
GGUF_PATH=""
TMP_GGUF=""
cleanup() {
    if [[ -n "$TMP_GGUF" && -f "$TMP_GGUF" ]]; then
        log "임시 GGUF 정리: $TMP_GGUF"
        rm -f "$TMP_GGUF"
    fi
}
trap cleanup EXIT

if [[ -d "$INPUT" ]]; then
    # Safetensors 디렉토리
    [[ -f "$PY_CONVERT" ]] || { echo "ERROR: $PY_CONVERT 없음" >&2; exit 1; }

    if [[ -n "$KEEP_GGUF" ]]; then
        GGUF_PATH="$KEEP_GGUF"
        mkdir -p "$(dirname -- "$GGUF_PATH")"
    else
        TMP_GGUF="$(mktemp --suffix=.gguf -t convert_to_auf.XXXXXX)"
        GGUF_PATH="$TMP_GGUF"
    fi

    log "단계 1: Safetensors → GGUF (--outtype $OUTTYPE)"
    log "  input : $INPUT"
    log "  output: $GGUF_PATH"
    python3 "$PY_CONVERT" --outtype "$OUTTYPE" "$INPUT" "$GGUF_PATH"

    # Safetensors 입력 시 tokenizer 자동 탐색
    if [[ -z "$TOKENIZER" ]]; then
        if [[ -f "$INPUT/tokenizer.json" ]]; then
            TOKENIZER="$INPUT/tokenizer.json"
        fi
    fi
elif [[ -f "$INPUT" ]]; then
    # GGUF 단일 파일
    case "$INPUT" in
        *.gguf|*.GGUF) GGUF_PATH="$INPUT" ;;
        *) echo "ERROR: 입력은 디렉토리(Safetensors) 또는 .gguf 파일이어야 함: $INPUT" >&2; exit 1 ;;
    esac
    log "단계 1 건너뜀: 입력이 이미 GGUF ($GGUF_PATH)"

    # GGUF 입력 시 tokenizer 자동 탐색
    if [[ -z "$TOKENIZER" ]]; then
        GGUF_DIR="$(dirname -- "$GGUF_PATH")"
        GGUF_STEM="$(basename -- "$GGUF_PATH" .gguf)"
        # 1순위: <gguf-dir>/<gguf-stem>.tokenizer.json
        if [[ -f "$GGUF_DIR/$GGUF_STEM.tokenizer.json" ]]; then
            TOKENIZER="$GGUF_DIR/$GGUF_STEM.tokenizer.json"
        # 2순위: quant suffix 제거 후 ('-q4_0' 등)
        elif [[ -f "$GGUF_DIR/${GGUF_STEM%-q4_0}.tokenizer.json" ]]; then
            TOKENIZER="$GGUF_DIR/${GGUF_STEM%-q4_0}.tokenizer.json"
        elif [[ -f "$GGUF_DIR/${GGUF_STEM%-f16}.tokenizer.json" ]]; then
            TOKENIZER="$GGUF_DIR/${GGUF_STEM%-f16}.tokenizer.json"
        # 3순위: legacy fallback
        elif [[ -f "$GGUF_DIR/tokenizer.json" ]]; then
            TOKENIZER="$GGUF_DIR/tokenizer.json"
        fi
    fi
else
    echo "ERROR: --input 은 디렉토리(Safetensors) 또는 파일(.gguf)이어야 함: $INPUT" >&2
    exit 1
fi

if [[ -z "$TOKENIZER" || ! -f "$TOKENIZER" ]]; then
    echo "ERROR: tokenizer.json을 찾지 못했습니다. --tokenizer <PATH>로 명시하세요." >&2
    exit 1
fi

# 3) GGUF → AUF
log "단계 2: GGUF → AUF"
log "  input    : $GGUF_PATH"
log "  tokenizer: $TOKENIZER"
log "  output   : $OUTPUT"
log "  variants : $VARIANTS"
log "  lm_head  : $INCLUDE_LM_HEAD"

mkdir -p "$(dirname -- "$OUTPUT")"

AUF_ARGS=(
    build
    --input "$GGUF_PATH"
    --tokenizer "$TOKENIZER"
    --output "$OUTPUT"
    --variants "$VARIANTS"
    --include-lm-head "$INCLUDE_LM_HEAD"
)
[[ -n "$DTYPES" ]] && AUF_ARGS+=(--dtypes "$DTYPES")
[[ -n "$DEFAULT_DTYPE" ]] && AUF_ARGS+=(--default-dtype "$DEFAULT_DTYPE")
[[ -n "$CREATED_BY" ]] && AUF_ARGS+=(--created-by "$CREATED_BY")
[[ $QUIET -eq 1 ]] && AUF_ARGS+=(--quiet)

"$AUF_TOOL" "${AUF_ARGS[@]}"

log "완료: $OUTPUT"
