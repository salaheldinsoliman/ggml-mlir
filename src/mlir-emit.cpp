// ggml node -> MLIR text (linalg on tensors).
//
// Conventions:
// - ggml ne[] is fastest-first, MLIR shapes are slowest-first: ne=[K,N,B2,B3] -> 4-D shape BxB2xNxK.
// - kernel arguments are rank-4 memrefs pointing at ggml's data; the body converts them to tensors,
//   computes with linalg, and writes the result into the dst memref in place.

#include "ggml-mlir-impl.h"

#include <cinttypes>
#include <cstdio>
#include <sstream>

// "d0xd1x...xelem"
static std::string shape_str(std::initializer_list<int64_t> dims, const char * elem) {
    std::ostringstream os;
    for (int64_t d : dims) {
        os << d << "x";
    }
    os << elem;
    return os.str();
}

static std::string tensor_ty(std::initializer_list<int64_t> dims, const char * elem) {
    return "tensor<" + shape_str(dims, elem) + ">";
}

static std::string memref_ty(std::initializer_list<int64_t> dims, const char * elem) {
    return "memref<" + shape_str(dims, elem) + ">";
}

static const char * mlir_elem_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "f32";
        default:            return nullptr;
    }
}

// --- MUL_MAT ---
//
// ggml_mul_mat(src0, src1) contracts ne[0] of both inputs:
//   src0 ne=[K, N, B2a, B3a]  (weights)
//   src1 ne=[K, M, B2,  B3 ]  (activations), B2 = B2a*r2, B3 = B3a*r3
//   dst  ne=[N, M, B2,  B3 ]
//   dst[b3,b2,m,n] = sum_k src1[b3,b2,m,k] * src0[b3/r3, b2/r2, n, k]
// i.e. dst = src1 * src0^T per batch. src0 batches are broadcast over groups of r2 (r3) consecutive src1 batches.

static bool mul_mat_supported(const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    if (!mlir_elem_type(src0->type) || src1->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(node)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (src0->ne[i] == 0 || src1->ne[i] == 0) {
            return false;
        }
    }
    return src1->ne[2] % src0->ne[2] == 0 && src1->ne[3] % src0->ne[3] == 0;
}

static std::string emit_mul_mat(const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    const char * t0 = mlir_elem_type(src0->type);
    const char * t1 = "f32";
    const char * td = "f32";

    const int64_t K   = src0->ne[0];
    const int64_t N   = src0->ne[1];
    const int64_t B2a = src0->ne[2];
    const int64_t B3a = src0->ne[3];
    const int64_t M   = src1->ne[1];
    const int64_t B2  = src1->ne[2];
    const int64_t B3  = src1->ne[3];
    const int64_t r2  = B2 / B2a;
    const int64_t r3  = B3 / B3a;

    const std::string w4 = tensor_ty({B3a, B2a, N, K}, t0);
    const std::string x4 = tensor_ty({B3,  B2,  M, K}, t1);
    const std::string y4 = tensor_ty({B3,  B2,  M, N}, td);
    const std::string w4m = memref_ty({B3a, B2a, N, K}, t0);
    const std::string x4m = memref_ty({B3,  B2,  M, K}, t1);
    const std::string y4m = memref_ty({B3,  B2,  M, N}, td);

    std::ostringstream os;
    os << "func.func @kernel(%src0: " << w4m << ", %src1: " << x4m << ", %dst: " << y4m << ") attributes {llvm.emit_c_interface} {\n";
    os << "  %w4 = bufferization.to_tensor %src0 restrict : " << w4m << " to " << w4 << "\n";
    os << "  %x4 = bufferization.to_tensor %src1 restrict : " << x4m << " to " << x4 << "\n";
    os << "  %d4 = bufferization.to_tensor %dst restrict writable : " << y4m << " to " << y4 << "\n";
    os << "  %zero = arith.constant 0.0 : " << td << "\n";

    // multiply-accumulate body, shared by both forms
    std::ostringstream body;
    body << "  ^bb0(%a: " << t1 << ", %b: " << t0 << ", %acc: " << td << "):\n";
    body << "    %p = arith.mulf %a, %b : " << td << "\n";
    body << "    %s = arith.addf %acc, %p : " << td << "\n";
    body << "    linalg.yield %s : " << td << "\n";

    if (B2 == 1 && B3 == 1) {
        // plain 2-D matmul: y[m,n] = sum_k x[m,k] * w[n,k]
        const std::string w = tensor_ty({N, K}, t0);
        const std::string x = tensor_ty({M, K}, t1);
        const std::string y = tensor_ty({M, N}, td);
        os << "  %w = tensor.collapse_shape %w4 [[0, 1, 2], [3]] : " << w4 << " into " << w << "\n";
        os << "  %x = tensor.collapse_shape %x4 [[0, 1, 2], [3]] : " << x4 << " into " << x << "\n";
        os << "  %d = tensor.collapse_shape %d4 [[0, 1, 2], [3]] : " << y4 << " into " << y << "\n";
        os << "  %init = linalg.fill ins(%zero : " << td << ") outs(%d : " << y << ") -> " << y << "\n";
        os << "  %y = linalg.generic {\n"
              "      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,\n"
              "                       affine_map<(d0, d1, d2) -> (d1, d2)>,\n"
              "                       affine_map<(d0, d1, d2) -> (d0, d1)>],\n"
              "      iterator_types = [\"parallel\", \"parallel\", \"reduction\"]}\n";
        os << "      ins(%x, %w : " << x << ", " << w << ") outs(%init : " << y << ") {\n";
        os << body.str();
        os << "  } -> " << y << "\n";
        os << "  %y4 = tensor.expand_shape %y [[0, 1, 2], [3]] output_shape [1, 1, " << M << ", " << N << "] : " << y << " into " << y4 << "\n";
    } else {
        // batched with broadcast: split B3 -> (B3a, r3), B2 -> (B2a, r2)
        // loops: d0=b3a d1=j3 d2=b2a d3=j2 d4=m d5=n d6=k
        const std::string x6 = tensor_ty({B3a, r3, B2a, r2, M, K}, t1);
        const std::string y6 = tensor_ty({B3a, r3, B2a, r2, M, N}, td);
        os << "  %x = tensor.expand_shape %x4 [[0, 1], [2, 3], [4], [5]] output_shape ["
           << B3a << ", " << r3 << ", " << B2a << ", " << r2 << ", " << M << ", " << K << "] : " << x4 << " into " << x6 << "\n";
        os << "  %d = tensor.expand_shape %d4 [[0, 1], [2, 3], [4], [5]] output_shape ["
           << B3a << ", " << r3 << ", " << B2a << ", " << r2 << ", " << M << ", " << N << "] : " << y4 << " into " << y6 << "\n";
        os << "  %init = linalg.fill ins(%zero : " << td << ") outs(%d : " << y6 << ") -> " << y6 << "\n";
        os << "  %y = linalg.generic {\n"
              "      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d6)>,\n"
              "                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d5, d6)>,\n"
              "                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5)>],\n"
              "      iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"parallel\", \"parallel\", \"parallel\", \"reduction\"]}\n";
        os << "      ins(%x, %w4 : " << x6 << ", " << w4 << ") outs(%init : " << y6 << ") {\n";
        os << body.str();
        os << "  } -> " << y6 << "\n";
        os << "  %y4 = tensor.collapse_shape %y [[0, 1], [2, 3], [4], [5]] : " << y6 << " into " << y4 << "\n";
    }

    os << "  bufferization.materialize_in_destination %y4 in restrict writable %dst : (" << y4 << ", " << y4m << ") -> ()\n";
    os << "  return\n";
    os << "}\n";
    return os.str();
}

// --- public API ---

bool ggml_mlir_can_emit(const struct ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_MUL_MAT: return mul_mat_supported(node);
        default:              return false;
    }
}

std::string ggml_mlir_node_key(const struct ggml_tensor * node) {
    std::ostringstream os;
    os << ggml_op_name(node->op) << ":" << ggml_type_name(node->type);
    for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
        const struct ggml_tensor * src = node->src[s];
        os << "|" << ggml_type_name(src->type) << "[" << src->ne[0] << "," << src->ne[1] << "," << src->ne[2] << "," << src->ne[3] << "]";
    }
    return os.str();
}

std::string ggml_mlir_emit(const struct ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_MUL_MAT: return emit_mul_mat(node);
        default:              return "";
    }
}
