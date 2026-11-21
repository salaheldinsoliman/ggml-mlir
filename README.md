# ggml-mlir

A compiler backend for [llama.cpp](https://github.com/ggml-org/llama.cpp). It translates llama.cpp's operation graph into [MLIR](https://mlir.llvm.org), so that existing MLIR-based compilers can generate the code that runs it.

The goal is to run open AI models on laptop NPUs (the low-power AI chips in recent laptops). The first target is the AMD Ryzen AI NPU, the only laptop NPU with an open compiler that accepts MLIR.

**Status: early prototype.** One operation (matrix multiplication) works end to end on the CPU. Nothing has run on an NPU yet.

## How it works

```
llama.cpp graph
      |
  ggml-mlir plugin: writes each operation as MLIR (linalg)
      |
      +-- CPU:     upstream MLIR -> LLVM -> native code     (works today)
      +-- AMD NPU: AMD's open MLIR compilers -> NPU binary  (planned)
```

- The plugin is a normal llama.cpp backend, loaded at runtime. llama.cpp itself is not modified.
- The plugin takes the operations it can compile. llama.cpp runs the rest on its own CPU or GPU backends.
- Each operation is compiled once per shape and cached.
- The CPU path exists to check correctness: the same MLIR that will go to the NPU compiler is run on the CPU and compared with llama.cpp's own results.

## What works

| | |
|---|---|
| Plugin loads into unmodified llama.cpp | yes |
| `MUL_MAT`, f32, contiguous inputs, batched and broadcast | 149/149 `test-backend-ops` cases pass |
| f16 / bf16 weights, views and permuted inputs | not yet |
| Quantised weights (Q8_0, Q4_0, Q4_K) | not yet |
| Other operations (norm, softmax, RoPE, ...) | not yet |
| AMD NPU compile path | not yet |
| Run on NPU hardware | not yet |

The generated CPU code is plain loops and is slow. Speed on the CPU is not a goal.

## Build

Requirements: CMake, Ninja, a C++17 compiler, and LLVM with MLIR (tested with LLVM 23 from Homebrew on macOS, Apple Silicon).

```sh
git clone --recursive https://github.com/salaheldinsoliman/ggml-mlir
cd ggml-mlir

# 1. build llama.cpp with dynamically loaded backends
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF
cmake --build third_party/llama.cpp/build --target test-backend-ops

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
    ./test-backend-ops test -b MLIR -o MUL_MAT
```

Set `GGML_MLIR_DUMP=1` to print the MLIR generated for each operation.

## Layout

- `src/ggml-mlir.cpp`: the llama.cpp backend interface.
- `src/mlir-emit.cpp`: turns a ggml operation into MLIR text. It does not depend on the MLIR libraries.
- `src/mlir-jit.cpp`: compiles that MLIR for the CPU and caches the result.
- `third_party/llama.cpp`: llama.cpp, pinned to the commit the plugin is tested against.

## Roadmap

1. f16 / bf16 weights and non-contiguous inputs.
2. A small model with all its matrix multiplications running through the plugin, checked against the CPU.
3. The same MLIR compiled to an AMD NPU binary.
4. Run and measure on NPU hardware.
5. Quantised weights, more operations, and whole blocks compiled as one kernel.

## Licence

MIT
