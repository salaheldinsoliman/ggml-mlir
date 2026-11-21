// ggml graph -> MLIR text (linalg on tensors).
//
// Conventions:
// - ggml ne[] is fastest-first, MLIR shapes are slowest-first: ne=[K,N,B2,B3] -> shape B3xB2xNxK.
//   Every ggml tensor is a rank-4 MLIR tensor.
// - A run of ggml nodes becomes one function. Inside it each node is an SSA tensor value, so a
//   compiler sees the data flow between operations. Each computed node is also written to its
//   ggml buffer, because other backends may read it.
// - Tensors that come from outside the function, and views the emitter does not model, are read
//   through a memref argument over ggml's data (with ggml's strides).

#include "ggml-mlir-impl.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <unordered_map>

static const char * mlir_elem_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_BF16: return "bf16";
        default:             return nullptr;
    }
}

// "d0xd1x...xelem"
static std::string shape_str(const std::vector<int64_t> & dims, const char * elem) {
    std::ostringstream os;
    for (int64_t d : dims) {
        os << d << "x";
    }
    os << elem;
    return os.str();
}

static std::string tensor_ty(const std::vector<int64_t> & dims, const char * elem) {
    return "tensor<" + shape_str(dims, elem) + ">";
}

static std::vector<int64_t> dims4(const struct ggml_tensor * t) {
    return { t->ne[3], t->ne[2], t->ne[1], t->ne[0] };
}

static std::string tensor_ty(const struct ggml_tensor * t) {
    return tensor_ty(dims4(t), mlir_elem_type(t->type));
}

// rank-4 memref type of a ggml tensor. Views and permutes get an explicit strided layout.
static std::string memref_ty(const struct ggml_tensor * t) {
    std::ostringstream os;
    os << "memref<" << shape_str(dims4(t), mlir_elem_type(t->type));
    if (!ggml_is_contiguous(t)) {
        const int64_t es = (int64_t) ggml_type_size(t->type);
        os << ", strided<[" << t->nb[3] / es << ", " << t->nb[2] / es << ", " << t->nb[1] / es << ", " << t->nb[0] / es << "], offset: 0>";
    }
    os << ">";
    return os.str();
}

// exact f32 constant, as its bit pattern
static std::string f32_const(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    char buf[16];
    snprintf(buf, sizeof(buf), "0x%08X", bits);
    return buf;
}

static float op_param_f32(const struct ggml_tensor * t, int i) {
    float v;
    memcpy(&v, (const char *) t->op_params + i * sizeof(float), sizeof(float));
    return v;
}

static int32_t op_param_i32(const struct ggml_tensor * t, int i) {
    int32_t v;
    memcpy(&v, (const char *) t->op_params + i * sizeof(int32_t), sizeof(int32_t));
    return v;
}

// can this tensor be passed as a memref argument?
static bool is_readable(const struct ggml_tensor * t) {
    if (!mlir_elem_type(t->type)) {
        return false;
    }
    const size_t es = ggml_type_size(t->type);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t->ne[i] == 0 || t->nb[i] % es != 0) {
            return false;
        }
    }
    return true;
}

static bool is_f32(const struct ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && is_readable(t);
}

static bool same_shape(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    return a->ne[0] == b->ne[0] && a->ne[1] == b->ne[1] && a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3];
}

static const char * k_id4  = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>";
static const char * k_row4 = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>"; // one value per row (all but ne[0])
static const char * k_par4 = "[\"parallel\", \"parallel\", \"parallel\", \"parallel\"]";
static const char * k_red4 = "[\"parallel\", \"parallel\", \"parallel\", \"reduction\"]";

//
// supported ops
//

static bool unary_supported(const struct ggml_tensor * node) {
    switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_NEG:
            return is_f32(node->src[0]);
        default:
            return false;
    }
}

bool ggml_mlir_can_emit(const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    // the result is written to a plain contiguous buffer
    if (!is_readable(node) || !ggml_is_contiguous(node)) {
        return false;
    }

    switch (node->op) {
        case GGML_OP_MUL_MAT:
            // weights may be half precision, activations and result are f32
            return is_readable(src0) && is_f32(src1) && node->type == GGML_TYPE_F32 &&
                   src1->ne[2] % src0->ne[2] == 0 && src1->ne[3] % src0->ne[3] == 0;

        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (!is_f32(src0) || !is_f32(src1) || node->type != GGML_TYPE_F32 || !same_shape(node, src0)) {
                return false;
            }
            // src1 is either full size or 1 along each dim (no partial repeats)
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (src1->ne[i] != 1 && src1->ne[i] != src0->ne[i]) {
                    return false;
                }
            }
            return true;

        case GGML_OP_SCALE:
        case GGML_OP_RMS_NORM:
            return is_f32(src0) && node->type == GGML_TYPE_F32;

        case GGML_OP_SOFT_MAX:
            // plain softmax only: no mask, no sinks, no ALiBi bias
            return is_f32(src0) && node->type == GGML_TYPE_F32 && src1 == nullptr && node->src[2] == nullptr && op_param_f32(node, 1) == 0.0f;

        case GGML_OP_UNARY:
            return unary_supported(node) && node->type == GGML_TYPE_F32;

        case GGML_OP_GLU:
            // split form only: silu(src0) * src1
            return op_param_i32(node, 0) == GGML_GLU_OP_SWIGLU && src1 != nullptr && is_f32(src0) && is_f32(src1) && node->type == GGML_TYPE_F32;

        case GGML_OP_CONT:
            return is_readable(src0) && src0->type == node->type;

        default:
            return false;
    }
}

//
// function builder
//

struct builder {
    std::ostringstream body;
    std::ostringstream sig;
    std::vector<const struct ggml_tensor *> args;
    std::unordered_map<const struct ggml_tensor *, std::string> vals; // tensor -> SSA tensor value
    int n_vals = 0;
    int n_ops  = 0;

    std::string fresh() {
        return "%v" + std::to_string(n_vals++);
    }

    // memref argument over the tensor's ggml buffer
    std::string arg(const struct ggml_tensor * t) {
        const std::string name = "%a" + std::to_string(args.size());
        sig << (args.empty() ? "" : ", ") << name << ": " << memref_ty(t);
        args.push_back(t);
        return name;
    }

    // SSA value of a tensor: computed in this function, or read from ggml's buffer
    std::string get(const struct ggml_tensor * t) {
        auto it = vals.find(t);
        if (it != vals.end()) {
            return it->second;
        }
        const std::string a = arg(t);
        const std::string v = fresh();
        body << "  " << v << " = bufferization.to_tensor " << a << " restrict : " << memref_ty(t) << " to " << tensor_ty(t) << "\n";
        vals[t] = v;
        return v;
    }

    std::string empty(const std::string & ty) {
        const std::string v = fresh();
        body << "  " << v << " = tensor.empty() : " << ty << "\n";
        return v;
    }

    std::string filled(const std::string & ty, const std::string & elem, const std::string & value) {
        const std::string e = empty(ty);
        const std::string c = fresh();
        const std::string v = fresh();
        body << "  " << c << " = arith.constant " << value << " : " << elem << "\n";
        body << "  " << v << " = linalg.fill ins(" << c << " : " << elem << ") outs(" << e << " : " << ty << ") -> " << ty << "\n";
        return v;
    }

    // write a computed node to its ggml buffer
    void store(const struct ggml_tensor * node, const std::string & v) {
        const std::string a = arg(node);
        body << "  bufferization.materialize_in_destination " << v << " in restrict writable " << a
             << " : (" << tensor_ty(node) << ", " << memref_ty(node) << ") -> ()\n";
        vals[node] = v;
        n_ops++;
    }
};

// elementwise op on same-shaped f32 tensors: `expr` computes %y from %in0, %in1, ...
static std::string emit_map(builder & b, const struct ggml_tensor * node, const std::vector<std::string> & ins, const std::string & expr) {
    const std::string ty = tensor_ty(node);
    const std::string e  = b.empty(ty);
    const std::string v  = b.fresh();

    b.body << "  " << v << " = linalg.generic {indexing_maps = [";
    for (size_t i = 0; i <= ins.size(); i++) {
        b.body << (i ? ", " : "") << k_id4;
    }
    b.body << "], iterator_types = " << k_par4 << "}\n      ins(";
    for (size_t i = 0; i < ins.size(); i++) {
        b.body << (i ? ", " : "") << ins[i];
    }
    b.body << " : ";
    for (size_t i = 0; i < ins.size(); i++) {
        b.body << (i ? ", " : "") << ty;
    }
    b.body << ") outs(" << e << " : " << ty << ") {\n  ^bb0(";
    for (size_t i = 0; i < ins.size(); i++) {
        b.body << "%in" << i << ": f32, ";
    }
    b.body << "%out: f32):\n" << expr << "    linalg.yield %y : f32\n  } -> " << ty << "\n";
    return v;
}

// silu(x) = x / (1 + exp(-x)), as body text computing `res` from `x`
static std::string silu_expr(const char * x, const char * res) {
    std::ostringstream os;
    os << "    %one = arith.constant 1.0 : f32\n";
    os << "    %neg = arith.negf " << x << " : f32\n";
    os << "    %exp = math.exp %neg : f32\n";
    os << "    %den = arith.addf %one, %exp : f32\n";
    os << "    " << res << " = arith.divf " << x << ", %den : f32\n";
    return os.str();
}

// --- MUL_MAT ---
//
// ggml_mul_mat(src0, src1) contracts ne[0] of both inputs:
//   src0 ne=[K, N, B2a, B3a]  (weights)
//   src1 ne=[K, M, B2,  B3 ]  (activations), B2 = B2a*r2, B3 = B3a*r3
//   dst  ne=[N, M, B2,  B3 ]
//   dst[b3,b2,m,n] = sum_k src1[b3,b2,m,k] * src0[b3/r3, b2/r2, n, k]
// i.e. dst = src1 * src0^T per batch. src0 batches are broadcast over groups of r2 (r3) consecutive src1 batches.

static std::string emit_mul_mat(builder & b, const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    const char * t0 = mlir_elem_type(src0->type);

    const int64_t K   = src0->ne[0];
    const int64_t N   = src0->ne[1];
    const int64_t B2a = src0->ne[2];
    const int64_t B3a = src0->ne[3];
    const int64_t M   = src1->ne[1];
    const int64_t B2  = src1->ne[2];
    const int64_t B3  = src1->ne[3];
    const int64_t r2  = B2 / B2a;
    const int64_t r3  = B3 / B3a;

    const std::string w4t = tensor_ty(src0);
    const std::string x4t = tensor_ty(src1);
    const std::string y4t = tensor_ty(node);

    const std::string w4 = b.get(src0);
    const std::string x4 = b.get(src1);

    // multiply-accumulate body, shared by both forms
    std::ostringstream mac;
    mac << "  ^bb0(%a: f32, %b: " << t0 << ", %acc: f32):\n";
    if (src0->type == GGML_TYPE_F32) {
        mac << "    %p = arith.mulf %a, %b : f32\n";
    } else {
        // half-precision weights: widen to f32, accumulate in f32
        mac << "    %bw = arith.extf %b : " << t0 << " to f32\n";
        mac << "    %p = arith.mulf %a, %bw : f32\n";
    }
    mac << "    %s = arith.addf %acc, %p : f32\n";
    mac << "    linalg.yield %s : f32\n";

    std::ostringstream & os = b.body;
    const std::string y4 = b.fresh();

    if (B2 == 1 && B3 == 1) {
        // plain 2-D matmul: y[m,n] = sum_k x[m,k] * w[n,k]
        const std::string wt = tensor_ty({N, K}, t0);
        const std::string xt = tensor_ty({M, K}, "f32");
        const std::string yt = tensor_ty({M, N}, "f32");
        const std::string w = b.fresh(), x = b.fresh(), y = b.fresh();
        os << "  " << w << " = tensor.collapse_shape " << w4 << " [[0, 1, 2], [3]] : " << w4t << " into " << wt << "\n";
        os << "  " << x << " = tensor.collapse_shape " << x4 << " [[0, 1, 2], [3]] : " << x4t << " into " << xt << "\n";
        const std::string init = b.filled(yt, "f32", "0.0");
        os << "  " << y << " = linalg.generic {\n"
              "      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,\n"
              "                       affine_map<(d0, d1, d2) -> (d1, d2)>,\n"
              "                       affine_map<(d0, d1, d2) -> (d0, d1)>],\n"
              "      iterator_types = [\"parallel\", \"parallel\", \"reduction\"]}\n";
        os << "      ins(" << x << ", " << w << " : " << xt << ", " << wt << ") outs(" << init << " : " << yt << ") {\n";
        os << mac.str();
        os << "  } -> " << yt << "\n";
        os << "  " << y4 << " = tensor.expand_shape " << y << " [[0, 1, 2], [3]] output_shape [1, 1, " << M << ", " << N << "] : " << yt << " into " << y4t << "\n";
    } else {
        // batched with broadcast: split B3 -> (B3a, r3), B2 -> (B2a, r2)
        // loops: d0=b3a d1=j3 d2=b2a d3=j2 d4=m d5=n d6=k
        const std::string x6t = tensor_ty({B3a, r3, B2a, r2, M, K}, "f32");
        const std::string y6t = tensor_ty({B3a, r3, B2a, r2, M, N}, "f32");
        const std::string x = b.fresh(), y = b.fresh();
        os << "  " << x << " = tensor.expand_shape " << x4 << " [[0, 1], [2, 3], [4], [5]] output_shape ["
           << B3a << ", " << r3 << ", " << B2a << ", " << r2 << ", " << M << ", " << K << "] : " << x4t << " into " << x6t << "\n";
        const std::string init = b.filled(y6t, "f32", "0.0");
        os << "  " << y << " = linalg.generic {\n"
              "      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d6)>,\n"
              "                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d5, d6)>,\n"
              "                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5)>],\n"
              "      iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"parallel\", \"parallel\", \"parallel\", \"reduction\"]}\n";
        os << "      ins(" << x << ", " << w4 << " : " << x6t << ", " << w4t << ") outs(" << init << " : " << y6t << ") {\n";
        os << mac.str();
        os << "  } -> " << y6t << "\n";
        os << "  " << y4 << " = tensor.collapse_shape " << y << " [[0, 1], [2, 3], [4], [5]] : " << y6t << " into " << y4t << "\n";
    }
    return y4;
}

// --- ADD / SUB / MUL / DIV ---
// src1 is broadcast along every dim where it has size 1.

static std::string emit_binary(builder & b, const struct ggml_tensor * node, const char * op) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    const std::string ty = tensor_ty(node);
    const std::string x  = b.get(src0);
    std::string       y  = b.get(src1);
    std::string       yt = tensor_ty(src1);
    std::string       ymap = k_id4;

    // drop the broadcast dims of src1, so its indexing map stays a plain projection
    const std::vector<int64_t> d0 = dims4(src0);
    const std::vector<int64_t> d1 = dims4(src1);
    std::vector<int> kept;
    for (int i = 0; i < 4; i++) {
        if (d1[i] == d0[i]) {
            kept.push_back(i);
        }
    }
    if (kept.size() < 4) {
        std::ostringstream reassoc, map;
        std::vector<int64_t> dims;
        reassoc << "[";
        map << "affine_map<(d0, d1, d2, d3) -> (";
        for (size_t g = 0; g < kept.size(); g++) {
            const int first = g == 0 ? 0 : kept[g - 1] + 1;
            const int last  = g + 1 == kept.size() ? 3 : kept[g];
            reassoc << (g ? ", " : "") << "[";
            for (int i = first; i <= last; i++) {
                reassoc << (i > first ? ", " : "") << i;
            }
            reassoc << "]";
            map << (g ? ", " : "") << "d" << kept[g];
            dims.push_back(d1[kept[g]]);
        }
        reassoc << "]";
        map << ")>";

        const std::string ct = tensor_ty(dims, "f32");
        const std::string c  = b.fresh();
        b.body << "  " << c << " = tensor.collapse_shape " << y << " " << reassoc.str() << " : " << yt << " into " << ct << "\n";
        y = c; yt = ct; ymap = map.str();
    }

    const std::string e = b.empty(ty);
    const std::string v = b.fresh();
    b.body << "  " << v << " = linalg.generic {indexing_maps = [" << k_id4 << ", " << ymap << ", " << k_id4 << "], iterator_types = " << k_par4 << "}\n";
    b.body << "      ins(" << x << ", " << y << " : " << ty << ", " << yt << ") outs(" << e << " : " << ty << ") {\n";
    b.body << "  ^bb0(%in0: f32, %in1: f32, %out: f32):\n";
    b.body << "    %y = arith." << op << " %in0, %in1 : f32\n";
    b.body << "    linalg.yield %y : f32\n  } -> " << ty << "\n";
    return v;
}

// --- SCALE: y = x*s + b ---

static std::string emit_scale(builder & b, const struct ggml_tensor * node) {
    std::ostringstream expr;
    expr << "    %s = arith.constant " << f32_const(op_param_f32(node, 0)) << " : f32\n";
    expr << "    %b = arith.constant " << f32_const(op_param_f32(node, 1)) << " : f32\n";
    expr << "    %m = arith.mulf %in0, %s : f32\n";
    expr << "    %y = arith.addf %m, %b : f32\n";
    return emit_map(b, node, { b.get(node->src[0]) }, expr.str());
}

// --- UNARY ---

static std::string emit_unary(builder & b, const struct ggml_tensor * node) {
    std::string expr;
    switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_SILU:
            expr = silu_expr("%in0", "%y");
            break;
        case GGML_UNARY_OP_SIGMOID:
            expr = "    %one = arith.constant 1.0 : f32\n"
                   "    %neg = arith.negf %in0 : f32\n"
                   "    %exp = math.exp %neg : f32\n"
                   "    %den = arith.addf %one, %exp : f32\n"
                   "    %y = arith.divf %one, %den : f32\n";
            break;
        case GGML_UNARY_OP_RELU:
            expr = "    %zero = arith.constant 0.0 : f32\n"
                   "    %pos = arith.cmpf ogt, %in0, %zero : f32\n"
                   "    %y = arith.select %pos, %in0, %zero : f32\n";
            break;
        case GGML_UNARY_OP_NEG:
            expr = "    %y = arith.negf %in0 : f32\n";
            break;
        default:
            break;
    }
    return emit_map(b, node, { b.get(node->src[0]) }, expr);
}

// --- GLU (SwiGLU, split): y = silu(src0) * src1 ---

static std::string emit_swiglu(builder & b, const struct ggml_tensor * node) {
    const std::string expr = silu_expr("%in0", "%act") + "    %y = arith.mulf %act, %in1 : f32\n";
    return emit_map(b, node, { b.get(node->src[0]), b.get(node->src[1]) }, expr);
}

// --- RMS_NORM: y = x / sqrt(mean(x^2) + eps), mean over ne[0] ---

static std::string emit_rms_norm(builder & b, const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const std::string ty   = tensor_ty(node);
    const std::string rowt = tensor_ty({ node->ne[3], node->ne[2], node->ne[1] }, "f64");
    const std::string x    = b.get(src0);

    // sum of squares per row, accumulated in f64 like ggml does
    const std::string init = b.filled(rowt, "f64", "0.0");
    const std::string sum  = b.fresh();
    b.body << "  " << sum << " = linalg.generic {indexing_maps = [" << k_id4 << ", " << k_row4 << "], iterator_types = " << k_red4 << "}\n";
    b.body << "      ins(" << x << " : " << ty << ") outs(" << init << " : " << rowt << ") {\n";
    b.body << "  ^bb0(%in: f32, %acc: f64):\n";
    b.body << "    %sq = arith.mulf %in, %in : f32\n";
    b.body << "    %sqd = arith.extf %sq : f32 to f64\n";
    b.body << "    %s = arith.addf %acc, %sqd : f64\n";
    b.body << "    linalg.yield %s : f64\n  } -> " << rowt << "\n";

    const std::string e = b.empty(ty);
    const std::string v = b.fresh();
    b.body << "  " << v << " = linalg.generic {indexing_maps = [" << k_id4 << ", " << k_row4 << ", " << k_id4 << "], iterator_types = " << k_par4 << "}\n";
    b.body << "      ins(" << x << ", " << sum << " : " << ty << ", " << rowt << ") outs(" << e << " : " << ty << ") {\n";
    b.body << "  ^bb0(%in: f32, %sum: f64, %out: f32):\n";
    b.body << "    %n = arith.constant " << node->ne[0] << ".0 : f64\n";
    b.body << "    %eps = arith.constant " << f32_const(op_param_f32(node, 0)) << " : f32\n";
    b.body << "    %one = arith.constant 1.0 : f32\n";
    b.body << "    %meand = arith.divf %sum, %n : f64\n";
    b.body << "    %mean = arith.truncf %meand : f64 to f32\n";
    b.body << "    %me = arith.addf %mean, %eps : f32\n";
    b.body << "    %rt = math.sqrt %me : f32\n";
    b.body << "    %scale = arith.divf %one, %rt : f32\n";
    b.body << "    %y = arith.mulf %in, %scale : f32\n";
    b.body << "    linalg.yield %y : f32\n  } -> " << ty << "\n";
    return v;
}

// --- SOFT_MAX over ne[0]: y = exp(x*scale - max) / sum(exp(x*scale - max)) ---

static std::string emit_soft_max(builder & b, const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const std::string ty    = tensor_ty(node);
    const std::string maxt  = tensor_ty({ node->ne[3], node->ne[2], node->ne[1] }, "f32");
    const std::string sumt  = tensor_ty({ node->ne[3], node->ne[2], node->ne[1] }, "f64");
    const std::string x     = b.get(src0);
    const std::string scale = "    %scale = arith.constant " + f32_const(op_param_f32(node, 0)) + " : f32\n"
                              "    %w = arith.mulf %in, %scale : f32\n";

    // row maximum
    const std::string max_init = b.filled(maxt, "f32", "0xFF800000"); // -inf
    const std::string max      = b.fresh();
    b.body << "  " << max << " = linalg.generic {indexing_maps = [" << k_id4 << ", " << k_row4 << "], iterator_types = " << k_red4 << "}\n";
    b.body << "      ins(" << x << " : " << ty << ") outs(" << max_init << " : " << maxt << ") {\n";
    b.body << "  ^bb0(%in: f32, %acc: f32):\n" << scale;
    b.body << "    %gt = arith.cmpf ogt, %w, %acc : f32\n";
    b.body << "    %m = arith.select %gt, %w, %acc : f32\n";
    b.body << "    linalg.yield %m : f32\n  } -> " << maxt << "\n";

    // row sum of exp(x - max), accumulated in f64 like ggml does
    const std::string sum_init = b.filled(sumt, "f64", "0.0");
    const std::string sum      = b.fresh();
    b.body << "  " << sum << " = linalg.generic {indexing_maps = [" << k_id4 << ", " << k_row4 << ", " << k_row4 << "], iterator_types = " << k_red4 << "}\n";
    b.body << "      ins(" << x << ", " << max << " : " << ty << ", " << maxt << ") outs(" << sum_init << " : " << sumt << ") {\n";
    b.body << "  ^bb0(%in: f32, %max: f32, %acc: f64):\n" << scale;
    b.body << "    %d = arith.subf %w, %max : f32\n";
    b.body << "    %e = math.exp %d : f32\n";
    b.body << "    %ed = arith.extf %e : f32 to f64\n";
    b.body << "    %s = arith.addf %acc, %ed : f64\n";
    b.body << "    linalg.yield %s : f64\n  } -> " << sumt << "\n";

    const std::string e = b.empty(ty);
    const std::string v = b.fresh();
    b.body << "  " << v << " = linalg.generic {indexing_maps = [" << k_id4 << ", " << k_row4 << ", " << k_row4 << ", " << k_id4 << "], iterator_types = " << k_par4 << "}\n";
    b.body << "      ins(" << x << ", " << max << ", " << sum << " : " << ty << ", " << maxt << ", " << sumt << ") outs(" << e << " : " << ty << ") {\n";
    b.body << "  ^bb0(%in: f32, %max: f32, %sum: f64, %out: f32):\n" << scale;
    b.body << "    %d = arith.subf %w, %max : f32\n";
    b.body << "    %e = math.exp %d : f32\n";
    b.body << "    %oned = arith.constant 1.0 : f64\n";
    b.body << "    %invd = arith.divf %oned, %sum : f64\n";
    b.body << "    %inv = arith.truncf %invd : f64 to f32\n";
    b.body << "    %y = arith.mulf %e, %inv : f32\n";
    b.body << "    linalg.yield %y : f32\n  } -> " << ty << "\n";
    return v;
}

//
// views: modelled as tensor ops when their source is a value of this function.
// Otherwise the view is read through its own memref argument when something uses it.
//

static void emit_view(builder & b, const struct ggml_tensor * node) {
    const struct ggml_tensor * src = node->src[0];
    if (!src || !mlir_elem_type(node->type) || !b.vals.count(src) || ggml_nelements(node) == 0) {
        return;
    }
    const std::string s  = b.vals[src];
    const std::string st = tensor_ty(src);
    const std::string nt = tensor_ty(node);
    const char *      et = mlir_elem_type(node->type);

    if (node->op == GGML_OP_RESHAPE) {
        const std::string flat_t = tensor_ty({ ggml_nelements(node) }, et);
        const std::string flat   = b.fresh();
        const std::string v      = b.fresh();
        b.body << "  " << flat << " = tensor.collapse_shape " << s << " [[0, 1, 2, 3]] : " << st << " into " << flat_t << "\n";
        b.body << "  " << v << " = tensor.expand_shape " << flat << " [[0, 1, 2, 3]] output_shape ["
               << node->ne[3] << ", " << node->ne[2] << ", " << node->ne[1] << ", " << node->ne[0] << "] : " << flat_t << " into " << nt << "\n";
        b.vals[node] = v;
        return;
    }

    // PERMUTE / TRANSPOSE: ggml dim axis[i] of the result is dim i of the source
    int axis[4] = { 1, 0, 2, 3 };
    if (node->op == GGML_OP_PERMUTE) {
        for (int i = 0; i < 4; i++) {
            axis[i] = op_param_i32(node, i);
        }
    }
    int perm[4]; // MLIR result dim j takes MLIR source dim perm[j]
    for (int i = 0; i < 4; i++) {
        perm[3 - axis[i]] = 3 - i;
    }
    const std::string e = b.empty(nt);
    const std::string v = b.fresh();
    b.body << "  " << v << " = linalg.transpose ins(" << s << " : " << st << ") outs(" << e << " : " << nt << ") permutation = ["
           << perm[0] << ", " << perm[1] << ", " << perm[2] << ", " << perm[3] << "]\n";
    b.vals[node] = v;
}

static std::string emit_op(builder & b, const struct ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_MUL_MAT:  return emit_mul_mat(b, node);
        case GGML_OP_ADD:      return emit_binary(b, node, "addf");
        case GGML_OP_SUB:      return emit_binary(b, node, "subf");
        case GGML_OP_MUL:      return emit_binary(b, node, "mulf");
        case GGML_OP_DIV:      return emit_binary(b, node, "divf");
        case GGML_OP_SCALE:    return emit_scale(b, node);
        case GGML_OP_UNARY:    return emit_unary(b, node);
        case GGML_OP_GLU:      return emit_swiglu(b, node);
        case GGML_OP_RMS_NORM: return emit_rms_norm(b, node);
        case GGML_OP_SOFT_MAX: return emit_soft_max(b, node);
        case GGML_OP_CONT:     return b.get(node->src[0]);
        default:               return "";
    }
}

ggml_mlir_func ggml_mlir_emit(struct ggml_tensor ** nodes, int n_nodes) {
    builder b;

    for (int i = 0; i < n_nodes; i++) {
        const struct ggml_tensor * node = nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_VIEW:
                break;
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                emit_view(b, node);
                break;
            default:
                b.body << "  // " << ggml_op_desc(node) << "\n";
                b.store(node, emit_op(b, node));
                break;
        }
    }

    ggml_mlir_func f;
    f.n_ops = b.n_ops;
    if (b.n_ops == 0) {
        return f; // nothing to compute (views only)
    }
    f.text = "func.func @kernel(" + b.sig.str() + ") {\n" + b.body.str() + "  return\n}\n";
    f.args = std::move(b.args);
    return f;
}
