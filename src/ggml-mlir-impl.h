#pragma once

#include "ggml.h"

#include <string>

// --- emitter (mlir-emit.cpp): ggml node -> MLIR text, no MLIR library dependency ---

// true if the emitter can translate this node
bool ggml_mlir_can_emit(const struct ggml_tensor * node);

// cache key: everything the emitted code depends on (op, types, shapes)
std::string ggml_mlir_node_key(const struct ggml_tensor * node);

// MLIR text of a module with one function @kernel(src0, src1, ..., dst).
// Every argument is a rank-4 memref in ggml order reversed: ne[3] x ne[2] x ne[1] x ne[0].
std::string ggml_mlir_emit(const struct ggml_tensor * node);

// --- JIT (mlir-jit.cpp): MLIR text -> native code on the host CPU, cached by key ---

struct ggml_mlir_jit;

ggml_mlir_jit * ggml_mlir_jit_new(void);
void            ggml_mlir_jit_free(ggml_mlir_jit * jit);

// compile (or fetch from cache) and run the kernel for this node
void            ggml_mlir_jit_compute(ggml_mlir_jit * jit, struct ggml_tensor * node);
