# Third-Party Licenses & Attribution

argus-engine is dual licensed **MIT OR Apache-2.0** (see [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE)). It also contains code adapted from the third-party
projects listed below, each used under its own license, reproduced in full here.

Where practical, derived source files also carry an `SPDX-License-Identifier` header
pointing here, but the attribution in this file is authoritative. The lists below name
representative files; they are not exhaustive.

---

## llama.cpp / ggml

- Upstream: <https://github.com/ggml-org/llama.cpp> (and <https://github.com/ggml-org/ggml>)
- License: MIT

Many of the OpenCL compute kernels under `engine/kernels/` are ported or adapted from
llama.cpp's OpenCL backend (`ggml-opencl`), and parts of the CPU (NEON / AVX2) and CUDA
backends under `engine/src/backend/` follow ggml's `Q4_0` / `Q8_0` quantization block
formats and kernel structure. Representative derived files:

- `engine/kernels/gemv_noshuffle_q4_0.cl`
- `engine/kernels/mul_mm_q4_0_f32_l4_lm.cl`
- `engine/kernels/rope.cl`
- `engine/kernels/argsort.cl`, `engine/kernels/simple_ops.cl`,
  `engine/kernels/flash_attn_f32_f16.cl`, `engine/kernels/mul_mv_f16_f32_1row_simple.cl`,
  `engine/kernels/mul_mat_Ab_Bi_8x4.cl`, and the `mul_mv_q4_0_*` / `mul_mv_q8_0_*` /
  `mul_mv_id_*` matrix-vector kernel family
- the `Q4_0` / `Q8_0` (and `Q4_K`) block layouts and dequantization paths shared across
  the CPU and CUDA backends — including `engine/src/quant.rs`,
  `engine/src/format/dtype_layout.rs`, `engine/src/models/loader/gguf.rs`,
  `engine/src/models/loader/convert.rs`, `engine/src/backend/cpu/neon.rs`,
  `engine/src/backend/cpu/x86.rs`, `engine/src/backend/cuda_embedded/kernels.cu`, and
  `engine/src/backend/cuda_pc/kernels.cu`

```
MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## YaRN (jquesnelle/yarn)

- Upstream: <https://github.com/jquesnelle/yarn>
- License: MIT

The YaRN RoPE scaling math in `engine/kernels/rope.cl` is based on
`LlamaYaRNScaledRotaryEmbedding.py` from the YaRN reference implementation.

```
MIT License

Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
