#!/usr/bin/env python3
"""Convert safetensors model to GGUF (Q4_0 hybrid or F16) compatible with Argus and llama.cpp.

Usage:
    # Q4_0 (default, 2D weights quantized, embed F16, norm F32)
    python scripts/convert_safetensors_to_gguf.py \
        models/qwen2.5-1.5b \
        models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf

    # F16 (all 2D weights + embed F16, norm F32)
    python scripts/convert_safetensors_to_gguf.py \
        --outtype f16 \
        models/qwen2.5-1.5b \
        models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf
"""
import sys, json, struct, os, argparse
import numpy as np
from pathlib import Path
from safetensors import safe_open

QK4_0 = 32  # Q4_0 block size

# ggml_type constants
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2

# GGUF value type constants
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10

ALIGNMENT = 32


def quantize_q4_0(data_f32: np.ndarray) -> bytes:
    """Quantize F32 array to Q4_0 blocks (matches Argus/ggml BlockQ4_0)."""
    assert data_f32.size % QK4_0 == 0, f"Size {data_f32.size} not divisible by {QK4_0}"
    n_blocks = data_f32.size // QK4_0
    data = data_f32.reshape(n_blocks, QK4_0)

    result = bytearray()
    for block in data:
        amax = np.max(np.abs(block))
        d = amax / 7.0
        id_ = 1.0 / d if d != 0 else 0.0

        # Quantize to 4-bit signed [-8, 7], store as unsigned [0, 15]
        q = np.clip(np.round(block * id_), -8, 7).astype(np.int8) + 8

        # Pack d as f16 (2 bytes)
        d_f16 = np.float16(d)
        result.extend(d_f16.tobytes())

        # Pack 32 nibbles into 16 bytes (lower 16 in low nibble, upper 16 in high nibble)
        qs = bytearray(16)
        for i in range(16):
            qs[i] = int(q[i]) | (int(q[i + 16]) << 4)
        result.extend(qs)

    return bytes(result)


# ---------- GGUF writers ----------

def write_gguf_string(buf: bytearray, s: str):
    """Write GGUF string: u64 len + bytes."""
    encoded = s.encode('utf-8')
    buf.extend(struct.pack('<Q', len(encoded)))
    buf.extend(encoded)


def write_gguf_kv(buf: bytearray, key: str, value, vtype: int):
    """Write a single KV pair."""
    write_gguf_string(buf, key)
    buf.extend(struct.pack('<I', vtype))

    if vtype == GGUF_TYPE_UINT32:
        buf.extend(struct.pack('<I', value))
    elif vtype == GGUF_TYPE_INT32:
        buf.extend(struct.pack('<i', value))
    elif vtype == GGUF_TYPE_FLOAT32:
        buf.extend(struct.pack('<f', value))
    elif vtype == GGUF_TYPE_BOOL:
        buf.extend(struct.pack('<B', 1 if value else 0))
    elif vtype == GGUF_TYPE_STRING:
        write_gguf_string(buf, value)
    elif vtype == GGUF_TYPE_UINT64:
        buf.extend(struct.pack('<Q', value))


def write_gguf_kv_array(buf: bytearray, key: str, element_type: int, elements):
    """Write a KV pair with array value."""
    write_gguf_string(buf, key)
    buf.extend(struct.pack('<I', GGUF_TYPE_ARRAY))
    buf.extend(struct.pack('<I', element_type))
    buf.extend(struct.pack('<Q', len(elements)))

    for elem in elements:
        if element_type == GGUF_TYPE_STRING:
            write_gguf_string(buf, elem)
        elif element_type == GGUF_TYPE_FLOAT32:
            buf.extend(struct.pack('<f', elem))
        elif element_type == GGUF_TYPE_INT32:
            buf.extend(struct.pack('<i', elem))
        elif element_type == GGUF_TYPE_UINT32:
            buf.extend(struct.pack('<I', elem))


# ---------- GGUF tensor name mapping ----------

GGUF_NAMES = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
for i in range(256):
    GGUF_NAMES.update({
        f"model.layers.{i}.self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
        f"model.layers.{i}.self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
        f"model.layers.{i}.self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
        f"model.layers.{i}.self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
        f"model.layers.{i}.mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
        f"model.layers.{i}.mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
        f"model.layers.{i}.mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
        f"model.layers.{i}.input_layernorm.weight": f"blk.{i}.attn_norm.weight",
        f"model.layers.{i}.post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
        # QKV bias (Qwen2)
        f"model.layers.{i}.self_attn.q_proj.bias": f"blk.{i}.attn_q.bias",
        f"model.layers.{i}.self_attn.k_proj.bias": f"blk.{i}.attn_k.bias",
        f"model.layers.{i}.self_attn.v_proj.bias": f"blk.{i}.attn_v.bias",
        # Gemma3 extra norms
        f"model.layers.{i}.self_attn.q_norm.weight": f"blk.{i}.attn_q_norm.weight",
        f"model.layers.{i}.self_attn.k_norm.weight": f"blk.{i}.attn_k_norm.weight",
        f"model.layers.{i}.pre_feedforward_layernorm.weight": f"blk.{i}.pre_ffn_norm.weight",
        f"model.layers.{i}.post_feedforward_layernorm.weight": f"blk.{i}.post_ffn_norm.weight",
    })


# ---------- Tokenizer ----------

# Token types matching ggml/llama.cpp
LLAMA_TOKEN_TYPE_NORMAL       = 1
LLAMA_TOKEN_TYPE_UNKNOWN      = 2
LLAMA_TOKEN_TYPE_CONTROL      = 3
LLAMA_TOKEN_TYPE_USER_DEFINED = 4
LLAMA_TOKEN_TYPE_UNUSED       = 5
LLAMA_TOKEN_TYPE_BYTE         = 6


def load_tokenizer(model_dir: Path):
    """Load tokenizer data from tokenizer.json and tokenizer_config.json.

    Returns dict with keys: model_type, tokens, scores, token_types, merges,
    bos_id, eos_id, pad_id, add_bos, add_eos, chat_template.
    """
    tokenizer_path = model_dir / "tokenizer.json"
    config_path = model_dir / "tokenizer_config.json"
    model_config_path = model_dir / "config.json"

    if not tokenizer_path.exists():
        print(f"WARNING: {tokenizer_path} not found — skipping tokenizer metadata")
        return None

    tok = json.loads(tokenizer_path.read_text())
    tok_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    model_config = json.loads(model_config_path.read_text()) if model_config_path.exists() else {}

    # Detect tokenizer model type
    tok_model_type = tok.get("model", {}).get("type", "BPE")
    if tok_model_type == "BPE":
        ggml_model = "gpt2"
    elif tok_model_type == "Unigram":
        ggml_model = "llama"
    elif tok_model_type == "WordPiece":
        ggml_model = "bert"
    else:
        ggml_model = "gpt2"
    print(f"Tokenizer: {tok_model_type} → ggml model '{ggml_model}'")

    # Build vocab: need token string for each ID 0..vocab_size-1
    vocab = tok.get("model", {}).get("vocab", {})
    added_tokens = tok.get("added_tokens", [])
    vocab_size = model_config.get("vocab_size", len(vocab) + len(added_tokens))

    # Create id→token mapping
    id_to_token = {}
    for token_str, token_id in vocab.items():
        id_to_token[token_id] = token_str

    # Added tokens override
    special_ids = set()
    for at in added_tokens:
        tid = at["id"]
        id_to_token[tid] = at["content"]
        if at.get("special", False):
            special_ids.add(tid)

    # Build ordered arrays
    tokens = []
    scores = []
    token_types = []
    for i in range(vocab_size):
        if i in id_to_token:
            tokens.append(id_to_token[i])
        else:
            tokens.append(f"[UNUSED_{i}]")

        # Scores: for BPE, use negative index as score (lower = more frequent)
        scores.append(float(-i))

        # Token type
        if i in special_ids:
            token_types.append(LLAMA_TOKEN_TYPE_CONTROL)
        elif i not in id_to_token:
            token_types.append(LLAMA_TOKEN_TYPE_UNUSED)
        else:
            token_types.append(LLAMA_TOKEN_TYPE_NORMAL)

    # Merges
    merges = tok.get("model", {}).get("merges", [])

    # Special token IDs
    bos_id = model_config.get("bos_token_id")
    eos_id = model_config.get("eos_token_id")

    # Try to resolve from token strings if IDs not in config
    if bos_id is None and tok_config.get("bos_token"):
        bos_str = tok_config["bos_token"]
        if isinstance(bos_str, dict):
            bos_str = bos_str.get("content", "")
        bos_id = vocab.get(bos_str)
    if eos_id is None and tok_config.get("eos_token"):
        eos_str = tok_config["eos_token"]
        if isinstance(eos_str, dict):
            eos_str = eos_str.get("content", "")
        eos_id = vocab.get(eos_str)

    # Padding token
    pad_id = None
    if tok_config.get("pad_token"):
        pad_str = tok_config["pad_token"]
        if isinstance(pad_str, dict):
            pad_str = pad_str.get("content", "")
        pad_id = vocab.get(pad_str)

    add_bos = tok_config.get("add_bos_token", False)
    add_eos = tok_config.get("add_eos_token", False)

    # Chat template
    chat_template = tok_config.get("chat_template")

    print(f"  Vocab: {len(tokens)} tokens, {len(merges)} merges")
    print(f"  BOS={bos_id}, EOS={eos_id}, PAD={pad_id}")
    print(f"  Special tokens: {len(special_ids)}")

    return {
        "model_type": ggml_model,
        "tokens": tokens,
        "scores": scores,
        "token_types": token_types,
        "merges": merges,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "add_bos": add_bos,
        "add_eos": add_eos,
        "chat_template": chat_template,
    }


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Convert safetensors to GGUF (Q4_0 hybrid or F16).")
    parser.add_argument("model_dir", type=Path, help="Directory containing *.safetensors + config.json")
    parser.add_argument("output_path", type=str, help="Output .gguf path")
    parser.add_argument("--outtype", choices=["q4_0", "f16"], default="q4_0",
                        help="Output weight type (default: q4_0)")
    args = parser.parse_args()

    model_dir = args.model_dir
    output_path = args.output_path
    outtype = args.outtype
    print(f"Output type: {outtype}")

    # Load config.json
    config = json.loads((model_dir / "config.json").read_text())

    # Detect architecture from model_type
    model_type = config.get("model_type", "llama").lower()
    ARCH_MAP = {"llama": "llama", "qwen2": "qwen2", "gemma": "gemma", "gemma2": "gemma2",
                 "gemma3": "gemma3", "gemma3_text": "gemma3"}
    arch = ARCH_MAP.get(model_type, "llama")
    print(f"Architecture: {arch} (model_type={model_type})")

    # Load tokenizer
    tokenizer = load_tokenizer(model_dir)

    # Find safetensors files
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"No .safetensors files found in {model_dir}")
        sys.exit(1)

    # Open all safetensors
    handles = []
    tensor_names = {}
    for f in st_files:
        h = safe_open(str(f), framework="numpy")
        for name in h.keys():
            tensor_names[name] = len(handles)
        handles.append(h)

    print(f"Found {len(tensor_names)} tensors in {len(st_files)} file(s)")

    # Classify tensors: 2D weights (outtype-dependent) vs norm/embed (F32/F16)
    weight_2d = set()  # 2D weight tensors → Q4_0 or F16 depending on outtype
    norm_1d = set()    # 1D norm/bias → F32, embed → F16

    for name in tensor_names:
        h = handles[tensor_names[name]]
        shape = h.get_slice(name).get_shape()
        if len(shape) == 1:
            norm_1d.add(name)
        elif "embed_tokens" in name:
            # embed_tokens stays F16: used by gather (embedding lookup) which
            # needs direct indexing. Also serves as lm_head via weight tying.
            norm_1d.add(name)
        else:
            if outtype == "q4_0" and shape[-1] % QK4_0 != 0:
                norm_1d.add(name)  # can't quantize, keep as F32
            else:
                weight_2d.add(name)

    print(f"  2D-weight ({outtype}): {len(weight_2d)}, norm/embed: {len(norm_1d)}")

    # Prepare tensor data
    tensors = []  # (gguf_name, dims_gguf, ggml_type, data_bytes)

    for hf_name in sorted(tensor_names.keys()):
        gguf_name = GGUF_NAMES.get(hf_name)
        if gguf_name is None:
            print(f"  SKIP: {hf_name} (no GGUF mapping)")
            continue

        h = handles[tensor_names[hf_name]]
        # Handle BF16: numpy doesn't support bf16 natively
        try:
            tensor = h.get_tensor(hf_name)
        except TypeError:
            # BF16 fallback: read raw uint16 bytes, convert to f32 via bit manipulation
            sl = h.get_slice(hf_name)
            shape_orig = sl.get_shape()
            st_path = st_files[tensor_names[hf_name]]
            with open(str(st_path), 'rb') as raw_f:
                header_size = struct.unpack('<Q', raw_f.read(8))[0]
                header_json = json.loads(raw_f.read(header_size))
                info = header_json[hf_name]
                offsets = info['data_offsets']
                raw_f.seek(8 + header_size + offsets[0])
                raw_bytes = raw_f.read(offsets[1] - offsets[0])
            bf16 = np.frombuffer(raw_bytes, dtype=np.uint16)
            f32_bits = bf16.astype(np.uint32) << 16
            tensor = np.frombuffer(f32_bits.tobytes(), dtype=np.float32).reshape(shape_orig)
        shape = tensor.shape

        # GGUF dims are reversed (innermost first)
        dims_gguf = list(reversed(shape))

        if hf_name in weight_2d:
            if outtype == "q4_0":
                f32 = tensor.astype(np.float32).flatten()
                data = quantize_q4_0(f32)
                tensors.append((gguf_name, dims_gguf, GGML_Q4_0, data))
            else:  # f16
                f16 = tensor.astype(np.float32).astype(np.float16)
                data = f16.tobytes()
                tensors.append((gguf_name, dims_gguf, GGML_F16, data))
        elif "embed_tokens" in hf_name:
            # Embedding: store as F16 (saves 50% vs F32, used by gather lookup)
            f16 = tensor.astype(np.float32).astype(np.float16)
            data = f16.tobytes()
            tensors.append((gguf_name, dims_gguf, GGML_F16, data))
        else:
            f32 = tensor.astype(np.float32)
            data = f32.tobytes()
            tensors.append((gguf_name, dims_gguf, GGML_F32, data))

    print(f"Prepared {len(tensors)} tensors for GGUF")

    # ========== Build GGUF file ==========

    # 1. Header
    header = bytearray()
    header.extend(struct.pack('<I', 0x46554747))  # magic: GGUF LE
    header.extend(struct.pack('<I', 3))            # version: 3
    header.extend(struct.pack('<Q', len(tensors))) # tensor_count

    # 2. KV metadata
    kv = bytearray()
    kv_count = 0

    def add_str(key, val):
        nonlocal kv_count; write_gguf_kv(kv, key, val, GGUF_TYPE_STRING); kv_count += 1

    def add_u32(key, val):
        nonlocal kv_count; write_gguf_kv(kv, key, val, GGUF_TYPE_UINT32); kv_count += 1

    def add_f32(key, val):
        nonlocal kv_count; write_gguf_kv(kv, key, val, GGUF_TYPE_FLOAT32); kv_count += 1

    def add_bool(key, val):
        nonlocal kv_count; write_gguf_kv(kv, key, val, GGUF_TYPE_BOOL); kv_count += 1

    def add_array_str(key, vals):
        nonlocal kv_count; write_gguf_kv_array(kv, key, GGUF_TYPE_STRING, vals); kv_count += 1

    def add_array_f32(key, vals):
        nonlocal kv_count; write_gguf_kv_array(kv, key, GGUF_TYPE_FLOAT32, vals); kv_count += 1

    def add_array_i32(key, vals):
        nonlocal kv_count; write_gguf_kv_array(kv, key, GGUF_TYPE_INT32, vals); kv_count += 1

    # --- Model architecture metadata ---
    add_str("general.architecture", arch)
    add_str("general.name", config.get("_name_or_path", model_dir.name))
    # GGML_FTYPE: MOSTLY_F16=1, MOSTLY_Q4_0=2
    add_u32("general.file_type", 2 if outtype == "q4_0" else 1)

    add_u32(f"{arch}.embedding_length", config["hidden_size"])
    add_u32(f"{arch}.block_count", config["num_hidden_layers"])
    add_u32(f"{arch}.attention.head_count", config["num_attention_heads"])
    add_u32(f"{arch}.attention.head_count_kv",
            config.get("num_key_value_heads", config["num_attention_heads"]))
    add_u32(f"{arch}.feed_forward_length", config["intermediate_size"])
    add_u32(f"{arch}.vocab_size", config["vocab_size"])
    add_u32(f"{arch}.context_length", config.get("max_position_embeddings", 2048))
    add_f32(f"{arch}.attention.layer_norm_rms_epsilon", config.get("rms_norm_eps", 1e-5))
    add_f32(f"{arch}.rope.freq_base", config.get("rope_theta", 10000.0))

    # Gemma3 extra metadata
    if "head_dim" in config:
        add_u32(f"{arch}.attention.head_dim", config["head_dim"])
    if "sliding_window" in config:
        add_u32(f"{arch}.attention.sliding_window", config["sliding_window"])
    if "sliding_window_pattern" in config:
        add_u32(f"{arch}.attention.sliding_window_pattern", config["sliding_window_pattern"])
    if "query_pre_attn_scalar" in config:
        add_f32(f"{arch}.attention.query_pre_attn_scalar", config["query_pre_attn_scalar"])

    # --- Tokenizer metadata (required by llama.cpp) ---
    if tokenizer:
        add_str("tokenizer.ggml.model", tokenizer["model_type"])
        add_str("tokenizer.ggml.pre", "default")
        add_array_str("tokenizer.ggml.tokens", tokenizer["tokens"])
        add_array_f32("tokenizer.ggml.scores", tokenizer["scores"])
        add_array_i32("tokenizer.ggml.token_type", tokenizer["token_types"])

        if tokenizer["merges"]:
            add_array_str("tokenizer.ggml.merges", tokenizer["merges"])

        if tokenizer["bos_id"] is not None:
            add_u32("tokenizer.ggml.bos_token_id", tokenizer["bos_id"])
        if tokenizer["eos_id"] is not None:
            add_u32("tokenizer.ggml.eos_token_id", tokenizer["eos_id"])
        if tokenizer["pad_id"] is not None:
            add_u32("tokenizer.ggml.padding_token_id", tokenizer["pad_id"])

        add_bool("tokenizer.ggml.add_bos_token", tokenizer["add_bos"])
        add_bool("tokenizer.ggml.add_eos_token", tokenizer["add_eos"])

        if tokenizer.get("chat_template"):
            add_str("tokenizer.chat_template", tokenizer["chat_template"])

    # Finalize header with kv_count
    header.extend(struct.pack('<Q', kv_count))

    # 3. Tensor info section
    tensor_info = bytearray()
    current_offset = 0

    for gguf_name, dims, ggml_type, data in tensors:
        write_gguf_string(tensor_info, gguf_name)
        tensor_info.extend(struct.pack('<I', len(dims)))
        for d in dims:
            tensor_info.extend(struct.pack('<Q', d))
        tensor_info.extend(struct.pack('<I', ggml_type))
        tensor_info.extend(struct.pack('<Q', current_offset))
        current_offset += len(data)
        padding = (ALIGNMENT - (current_offset % ALIGNMENT)) % ALIGNMENT
        current_offset += padding

    # 4. Compute data start (aligned)
    meta_size = len(header) + len(kv) + len(tensor_info)
    data_start_padding = (ALIGNMENT - (meta_size % ALIGNMENT)) % ALIGNMENT

    # 5. Write file
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(kv)
        f.write(tensor_info)
        f.write(b'\x00' * data_start_padding)

        for i, (gguf_name, dims, ggml_type, data) in enumerate(tensors):
            f.write(data)
            padding = (ALIGNMENT - (len(data) % ALIGNMENT)) % ALIGNMENT
            if padding and i < len(tensors) - 1:
                f.write(b'\x00' * padding)

    file_size = os.path.getsize(output_path)
    print(f"\nWritten {output_path}: {file_size / 1024 / 1024:.1f} MB ({len(tensors)} tensors, {kv_count} metadata)")


if __name__ == "__main__":
    main()
