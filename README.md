# ggml-mlir

Translates [llama.cpp](https://github.com/ggml-org/llama.cpp)'s operation graph into [MLIR](https://mlir.llvm.org), the compiler representation shared by many hardware toolchains.

llama.cpp runs a model by executing a graph of operations on hand-written kernels, one set per backend. ggml-mlir is a llama.cpp backend that hands the graph to a compiler instead. It writes each run of operations as one MLIR function (`linalg` on tensors). Any MLIR-based compiler can then optimise that function and generate code for its target.

**Status: early prototype.** The plugin translates 16 operations, enough to cover 91% of the graph nodes of a small language model. It compiles them for the CPU through upstream MLIR and LLVM and runs inside unmodified llama.cpp with results that match llama.cpp's own CPU backend. A second compiler (AMD's for Ryzen AI NPUs) reads the same MLIR, but nothing has been compiled all the way to an NPU binary or run on an NPU.

## How it works

```
llama.cpp graph
      |
  ggml-mlir plugin: one MLIR function per run of operations
      |
      +-- CPU: upstream MLIR -> LLVM -> native code   (works, used to check results)
      +-- export: .mlir files for any MLIR compiler    (works)
      +-- other targets, e.g. AMD's NPU compilers      (the MLIR is accepted; not compiled yet)
```

- The plugin is a normal llama.cpp backend, loaded at runtime. llama.cpp itself is not modified.
- The plugin takes the operations it can translate. llama.cpp runs the rest on its own backends.
- Inside a function, each operation is a tensor value that feeds the next, so a compiler sees the data flow and can fuse or reorder across operations. [examples/transformer-layer.mlir](examples/transformer-layer.mlir) is a real example: the 20 operations of one transformer layer in one function.
- Each function is compiled once per shape and cached.
- The CPU path exists to check correctness: the emitted MLIR is run on the CPU and compared with llama.cpp's own results.

## What works

| | |
|---|---|
| Plugin loads into unmodified llama.cpp | yes |
| Operations translated | `MUL_MAT`, `ADD`, `SUB`, `MUL`, `DIV`, `SCALE`, `RMS_NORM`, `SOFT_MAX` (with mask), `ROPE`, `GET_ROWS`, `SILU`, `SIGMOID`, `RELU`, `NEG`, SwiGLU, `CONT` |
| llama.cpp's operation tests (`test-backend-ops`) | 1583/1583 claimed cases pass, including multi-operation cases |
| Weight types | f32, f16, bf16 for `MUL_MAT`; f32 elsewhere |
| Views, permuted and transposed inputs | yes, without copies when they come from outside the function |
| A real model (SmolLM2-135M, f16) | output matches the CPU backend (see below) |
| Export of the emitted MLIR to files | yes (`GGML_MLIR_EXPORT`) |
| Emitted MLIR is read by a second compiler (AMD's `air-opt`, built on a different LLVM version) | yes, unchanged |

Not yet:

- Quantised weights (Q8_0, Q4_0, Q4_K).
- KV cache writes (`SET_ROWS`) and llama.cpp's single fused attention operation (`FLASH_ATTN_EXT`). In a model run these stay on llama.cpp's own backends. Attention built from separate operations (`-fa off`) is translated.
- RoPE variants beyond the standard and NeoX layouts (YaRN scaling, multi-axis positions), and softmax with sinks or ALiBi bias.
- Fast CPU code. The CPU path is unoptimised loops.
- Compiling the emitted MLIR to an NPU binary, and running on NPU hardware.

## Build

Requirements: CMake, Ninja, a C++17 compiler, and LLVM with MLIR (tested with LLVM 23 from Homebrew on macOS, Apple Silicon).

```sh
git clone --recursive https://github.com/salaheldinsoliman/ggml-mlir
cd ggml-mlir

# 1. build llama.cpp with dynamically loaded backends
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF
cmake --build third_party/llama.cpp/build --target test-backend-ops llama-perplexity

# 2. build the plugin
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm
cmake --build build
```

## Test

llama.cpp's `test-backend-ops` runs each operation on the plugin and on llama.cpp's CPU backend and compares the results:

```sh
cd third_party/llama.cpp/build/bin
GGML_BACKEND_PATH=$PWD/../../../../build/libggml-mlir.so \
    ./test-backend-ops test -b MLIR
```

Environment variables:

| | |
|---|---|
| `GGML_MLIR_EXPORT=<dir>` | write each emitted function to `<dir>/kernel_<n>.mlir` |
| `GGML_MLIR_DUMP=1` | print each emitted function (`2`: also its lowered form) |
| `GGML_MLIR_STATS=1` | print how many functions were compiled and called |

### Whole-model check

`llama-perplexity` can compare a model's output distribution between two runs. The reference run uses only llama.cpp's CPU backend. The second run adds the plugin:

```sh
# reference: CPU backend only
./llama-perplexity -m model-f16.gguf -f text.txt -c 128 -b 128 --chunks 8 -ngl 0 \
    --kl-divergence-base base.kld

# with the plugin
GGML_MLIR_STATS=1 GGML_BACKEND_PATH=/path/to/libggml-mlir.so \
./llama-perplexity -m model-f16.gguf -f text.txt -c 128 -b 128 --chunks 8 -ngl 0 \
    --kl-divergence-base base.kld --kl-divergence
```

Result for SmolLM2-135M-Instruct (f16) on 8 chunks of 128 tokens, macOS on Apple Silicon. Both runs add `-fa off`, so that attention is built from separate operations:

| | |
|---|---|
| Graph nodes run through MLIR | 607 of 669 (91%). The rest are 61 KV cache writes and one copy. |
| MLIR functions compiled | 8, holding 92 operations |
| Largest function | 20 operations (one whole transformer layer) |
| Mean KL divergence against the CPU backend | 0.000000 |
| Maximum KL divergence | 0.000045 |
| Same top token | 100% |
| Perplexity, plugin / CPU | 5.3583 / 5.3563 |

With llama.cpp's default fused attention operation, 487 of 579 nodes (84%) run through MLIR and the output matches equally well.

For the reference run, keep other accelerator backends (BLAS, Metal) out of the way, for example by running from a directory that contains only `libggml-cpu.so`.

The plugin run takes about 40 times longer than the CPU backend here (87 s against 2 s), because the generated code is unoptimised loops.

## Other compilers

The emitted MLIR uses only upstream dialects (`linalg`, `tensor`, `arith`, `math`, `bufferization`), so it is not tied to one compiler.

`scripts/amd-toolchain.sh` sets up AMD's open NPU compilers (MLIR-AIR, MLIR-AIE, llvm-aie) in an x86-64 container. In that container:

- AMD's own bf16 matmul example compiles for the Ryzen AI 300 NPU without hardware.
- AMD's `air-opt` parses [examples/transformer-layer.mlir](examples/transformer-layer.mlir) unchanged.

Parsing is as far as this goes today. Lowering our functions to an NPU binary needs a tiling recipe for AMD's pipeline and has not been done.

## Layout

- `src/ggml-mlir.cpp`: the llama.cpp backend interface.
- `src/mlir-emit.cpp`: turns a run of ggml operations into MLIR text. It does not depend on the MLIR libraries.
- `src/mlir-jit.cpp`: compiles that MLIR for the CPU and caches the result.
- `examples/`: MLIR emitted from a real model.
- `scripts/amd-toolchain.sh`: container with AMD's NPU compilers.
- `third_party/llama.cpp`: llama.cpp, pinned to the commit the plugin is tested against.

## Roadmap

1. The remaining operations of a dense model: KV cache writes and the fused attention operation.
2. Quantised weights, with a custom operation for llama.cpp's block-quantised formats.
3. A CPU pipeline with tiling and vectorisation, to show the compiler doing real optimisation work.
4. A second target end to end: the same functions compiled to an AMD NPU binary, then run and measured on hardware.
5. An on-disk cache of compiled functions.

## Licence

MIT
