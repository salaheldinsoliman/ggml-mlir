#pragma once

#include "ggml.h"

#include <string>
#include <vector>

// --- emitter (mlir-emit.cpp): ggml graph -> MLIR text, no MLIR library dependency ---

// true if the emitter can translate this node
bool ggml_mlir_can_emit(const struct ggml_tensor * node);

// One MLIR function for a whole run of ggml nodes.
//
// The function is @kernel(args...). Every argument is a rank-4 memref over a ggml tensor's
// data, in ggml order reversed: ne[3] x ne[2] x ne[1] x ne[0]. Arguments are the tensors the
// nodes read from outside the function, followed by one destination per computed node, in the
// order they are first needed.
struct ggml_mlir_func {
    std::string                             text;   // MLIR module text; identical graphs give identical text
    std::vector<const struct ggml_tensor *> args;   // tensors to pass, in argument order
    int                                     n_ops = 0; // computed nodes in the function
};

ggml_mlir_func ggml_mlir_emit(struct ggml_tensor ** nodes, int n_nodes);

// --- JIT (mlir-jit.cpp): MLIR text -> native code on the host CPU, cached by text ---

struct ggml_mlir_jit;

ggml_mlir_jit * ggml_mlir_jit_new(void);
void            ggml_mlir_jit_free(ggml_mlir_jit * jit);

// compile (or fetch from cache) one function for these nodes and run it
void            ggml_mlir_jit_compute(ggml_mlir_jit * jit, struct ggml_tensor ** nodes, int n_nodes);
